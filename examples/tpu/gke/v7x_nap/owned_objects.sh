#!/usr/bin/env bash
# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ===================================================================================
# Ownership-checked creation and deletion of the JobSets of one launcher invocation (sourced by
# run_ray_jobset_v7x_nap.sh; tested by tests/special_tpu/test_v7x_nap_launcher_on_cpu.py with a fake kubectl).
#
# The launcher must delete every object this invocation created, and nothing else:
#   * owned_create creates ONE JobSet per call (so a partial failure is tracked exactly) and records the UID the API
#     server returns. kubectl's stderr (e.g. admission warnings) is logged but never parsed. When the create's outcome
#     is not clear (kubectl succeeded but printed something unexpected, or failed with anything but AlreadyExists),
#     the object is looked up by name and adopted only if it carries this invocation's verl-nap/launch-id
#     annotation; if that cannot be confirmed either, the name is kept as "unconfirmed" and re-checked at cleanup.
#   * owned_delete deletes with a UID precondition (DeleteOptions.preconditions.uid): an object that was replaced
#     by someone else under the same name is left alone, without a check-then-delete race.
#
# Needs: KUBE_CONTEXT, RAY_NAMESPACE, LAUNCH_ID, OWNED_LEDGER (append-only record of created/deleted objects), and
# a log() function. OWNED_LOOKUP_ATTEMPTS / OWNED_LOOKUP_SLEEP_S tune the launch-id lookups.
# ===================================================================================
# shellcheck disable=SC2034  # OWNED_LAST_UID is read by the sourcing launcher

OWNED_OBJECTS=()      # "<name> <uid>" of JobSets this invocation created and has not deleted yet
OWNED_UNCONFIRMED=()  # names whose create outcome is unknown; re-checked by launch id at cleanup
OWNED_LAST_UID=""     # UID recorded by the last successful owned_create

_owned_ledger() { echo "$(date -u +%FT%TZ) $*" >>"${OWNED_LEDGER}"; }

# owned_lookup NAME: print the UID of JobSet NAME if it carries this invocation's launch id.
# Returns 0 (ours, UID printed), 1 (absent or not ours) or 2 (lookup failed, e.g. API error).
owned_lookup() {
  local name="$1" out rc=0
  out="$(kubectl --context "${KUBE_CONTEXT}" -n "${RAY_NAMESPACE}" get jobset "${name}" -o json 2>&1)" || rc=$?
  if ((rc != 0)); then
    case "${out}" in
      *NotFound* | *"not found"*) return 1 ;;
      *) return 2 ;;
    esac
  fi
  python3 -c '
import json, sys
try:
    meta = json.loads(sys.stdin.read())["metadata"]
except (ValueError, KeyError, TypeError):
    sys.exit(2)
if meta.get("annotations", {}).get("verl-nap/launch-id") != sys.argv[1] or not meta.get("uid"):
    sys.exit(1)
print(meta["uid"])
' "${LAUNCH_ID}" <<<"${out}"
}

# owned_lookup_retry NAME: owned_lookup, retried while the lookup itself fails (not while the object is absent).
owned_lookup_retry() {
  local name="$1" i rc=2
  for ((i = 0; i < ${OWNED_LOOKUP_ATTEMPTS:-5}; i++)); do
    owned_lookup "${name}" && return 0
    rc=$?
    ((rc == 1)) && return 1
    sleep "${OWNED_LOOKUP_SLEEP_S:-2}"
  done
  return "${rc}"
}

_owned_record() {  # NAME UID HOW
  OWNED_OBJECTS+=("$1 $2")
  OWNED_LAST_UID="$2"
  _owned_ledger "$3 jobset $1 uid $2"
  log "$3 JobSet $1 (uid $2)"
}

# owned_create FILE: create the single JobSet in FILE. On success sets OWNED_LAST_UID and records ownership.
# Must not run in a subshell (it updates OWNED_OBJECTS / OWNED_UNCONFIRMED).
owned_create() {
  local file="$1" name out err_file err rc=0 kind created uid
  OWNED_LAST_UID=""
  name="$(python3 -c '
import sys
for line in open(sys.argv[1]):
    if line.startswith("  name:"):
        print(line.split(":", 1)[1].strip().strip("\"")); break
' "${file}")"
  err_file="$(mktemp)"
  out="$(kubectl --context "${KUBE_CONTEXT}" -n "${RAY_NAMESPACE}" create -f "${file}" \
    -o jsonpath='{.kind} {.metadata.name} {.metadata.uid}' 2>"${err_file}")" || rc=$?
  err="$(cat "${err_file}")"
  rm -f "${err_file}"
  if ((rc == 0)); then
    [[ -z "${err}" ]] || log "kubectl create ${name} (succeeded) said on stderr: ${err}"
    read -r kind created uid <<<"${out}"
    if [[ "${kind}" == "JobSet" && "${created}" == "${name}" && "${uid}" =~ ^[0-9a-fA-F-]{8,}$ ]]; then
      _owned_record "${name}" "${uid}" "created"
      return 0
    fi
    log "WARNING: kubectl create ${name} succeeded but printed '${out}'; confirming it by launch id"
  else
    case "${err}" in
      *AlreadyExists*)
        log "ERROR: JobSet ${name} already exists and is not ours; not touching it"
        return 1
        ;;
    esac
    log "WARNING: kubectl create ${name} failed: ${err}; checking by launch id whether it reached the server"
  fi
  # The outcome is unclear: adopt the object only if it carries our launch id.
  if uid="$(owned_lookup_retry "${name}")"; then
    _owned_record "${name}" "${uid}" "adopted"
    return 0
  fi
  OWNED_UNCONFIRMED+=("${name}")
  _owned_ledger "unconfirmed jobset ${name} launch-id ${LAUNCH_ID}"
  log "ERROR: could not confirm whether JobSet ${name} exists with launch id ${LAUNCH_ID}; it is re-checked at cleanup"
  return 1
}

# owned_delete NAME UID: delete JobSet NAME only if it still has UID; 0 when deleted, gone or not ours.
owned_delete() {
  local name="$1" uid="$2" opts out rc=0
  opts="$(mktemp)"
  printf '{"kind":"DeleteOptions","apiVersion":"v1","propagationPolicy":"Background","preconditions":{"uid":"%s"}}' \
    "${uid}" >"${opts}"
  out="$(kubectl --context "${KUBE_CONTEXT}" delete --raw \
    "/apis/jobset.x-k8s.io/v1alpha2/namespaces/${RAY_NAMESPACE}/jobsets/${name}" -f "${opts}" 2>&1)" || rc=$?
  rm -f "${opts}"
  if ((rc == 0)); then
    _owned_ledger "deleted jobset ${name} uid ${uid}"
    log "deleted JobSet ${name} (uid ${uid})"
    return 0
  fi
  case "${out}" in
    *NotFound* | *"not found"*)
      _owned_ledger "gone jobset ${name} uid ${uid}"
      log "JobSet ${name} (uid ${uid}) is already gone"
      ;;
    *Precondition* | *Conflict*)
      _owned_ledger "skipped jobset ${name} uid ${uid} (replaced)"
      log "WARNING: JobSet ${name} no longer has uid ${uid} (replaced by someone else); left alone"
      ;;
    *)
      log "WARNING: could not delete JobSet ${name} (uid ${uid}): ${out}"
      return 1
      ;;
  esac
}

# owned_delete_all: delete every owned JobSet (newest first) and resolve the unconfirmed names; keeps what failed.
owned_delete_all() {
  local i entry name uid rc left=() unresolved=()
  for name in "${OWNED_UNCONFIRMED[@]}"; do
    rc=0
    uid="$(owned_lookup_retry "${name}")" || rc=$?
    if ((rc == 0)); then
      _owned_record "${name}" "${uid}" "adopted"
    elif ((rc == 1)); then
      _owned_ledger "absent jobset ${name} launch-id ${LAUNCH_ID}"
      log "no JobSet ${name} with launch id ${LAUNCH_ID} (absent or not ours); nothing to delete"
    else
      unresolved+=("${name}")
    fi
  done
  OWNED_UNCONFIRMED=("${unresolved[@]}")
  for ((i = ${#OWNED_OBJECTS[@]} - 1; i >= 0; i--)); do
    entry="${OWNED_OBJECTS[i]}"
    owned_delete "${entry% *}" "${entry#* }" || left=("${entry}" "${left[@]}")
  done
  OWNED_OBJECTS=("${left[@]}")
}

# owned_list: print the owned and unconfirmed JobSets with the commands that resolve each one safely.
owned_list() {
  local entry name
  for entry in "${OWNED_OBJECTS[@]}"; do
    log "  JobSet ${entry% *} (uid ${entry#* }): delete only if it still has that uid:"
    log "    kubectl --context ${KUBE_CONTEXT} -n ${RAY_NAMESPACE} get jobset ${entry% *} -o jsonpath='{.metadata.uid}'"
    log "    kubectl --context ${KUBE_CONTEXT} -n ${RAY_NAMESPACE} delete jobset ${entry% *}"
  done
  for name in "${OWNED_UNCONFIRMED[@]}"; do
    log "  JobSet ${name}: unconfirmed; it is this run's only if annotation verl-nap/launch-id is ${LAUNCH_ID}:"
    log "    kubectl --context ${KUBE_CONTEXT} -n ${RAY_NAMESPACE} get jobset ${name} -o jsonpath='{.metadata.annotations}'"
  done
}
