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
# The launcher must only ever delete objects that this invocation created:
#   * owned_create creates ONE JobSet per call and records the UID the API server returns, so a partial failure
#     (e.g. the second of two placeholders already exists) is tracked exactly. When `kubectl create` fails, nothing
#     is recorded, unless an object of that name exists AND carries this invocation's verl-nap/launch-id annotation
#     (a create that reached the server but whose response was lost).
#   * owned_delete deletes with a UID precondition (DeleteOptions.preconditions.uid): an object that was replaced
#     by someone else under the same name is left alone, without a check-then-delete race.
#
# Needs: KUBE_CONTEXT, RAY_NAMESPACE, LAUNCH_ID, OWNED_LEDGER (append-only record of created/deleted objects), and
# a log() function.
# ===================================================================================
# shellcheck disable=SC2034  # OWNED_LAST_UID is read by the sourcing launcher

OWNED_OBJECTS=()  # "<name> <uid>" of the JobSets this invocation created and has not deleted yet
OWNED_LAST_UID=""  # UID recorded by the last successful owned_create

_owned_ledger() { echo "$(date -u +%FT%TZ) $*" >>"${OWNED_LEDGER}"; }

# owned_jobset_uid NAME: print the UID of JobSet NAME if it carries this invocation's launch id (else nothing).
owned_jobset_uid() {
  kubectl --context "${KUBE_CONTEXT}" -n "${RAY_NAMESPACE}" get jobset "$1" -o json 2>/dev/null | python3 -c '
import json, sys
try:
    meta = json.load(sys.stdin)["metadata"]
except (ValueError, KeyError):
    sys.exit(0)
if meta.get("annotations", {}).get("verl-nap/launch-id") == sys.argv[1]:
    print(meta.get("uid", ""))
' "${LAUNCH_ID}"
}

# owned_create FILE: create the single JobSet in FILE. On success sets OWNED_LAST_UID and records ownership.
# Must not run in a subshell (it updates OWNED_OBJECTS).
owned_create() {
  local file="$1" out kind name uid
  OWNED_LAST_UID=""
  name="$(python3 -c '
import sys
for line in open(sys.argv[1]):
    if line.startswith("  name:"):
        print(line.split(":", 1)[1].strip().strip("\"")); break
' "${file}")"
  if out="$(kubectl --context "${KUBE_CONTEXT}" -n "${RAY_NAMESPACE}" create -f "${file}" \
      -o jsonpath='{.kind} {.metadata.name} {.metadata.uid}' 2>&1)"; then
    read -r kind name uid <<<"${out}"
    if [[ "${kind}" != "JobSet" || -z "${uid}" ]]; then
      log "ERROR: unexpected output creating ${file}: ${out}"
      return 1
    fi
  else
    case "${out}" in
      *AlreadyExists*) log "ERROR: JobSet ${name} already exists and is not ours; not touching it"; return 1 ;;
    esac
    # The request may have reached the server: adopt the object only if it carries our launch id.
    uid="$(owned_jobset_uid "${name}")"
    if [[ -z "${uid}" ]]; then
      log "ERROR: creating ${file} failed: ${out}"
      return 1
    fi
    log "WARNING: creating ${name} reported '${out}', but the object exists with this run's launch id; adopting it"
  fi
  OWNED_OBJECTS+=("${name} ${uid}")
  OWNED_LAST_UID="${uid}"
  _owned_ledger "created jobset ${name} uid ${uid}"
  log "created JobSet ${name} (uid ${uid})"
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
  if (( rc == 0 )); then
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

# owned_delete_all: delete every owned JobSet (newest first); keeps the ones whose deletion failed.
owned_delete_all() {
  local i entry left=()
  for ((i = ${#OWNED_OBJECTS[@]} - 1; i >= 0; i--)); do
    entry="${OWNED_OBJECTS[i]}"
    owned_delete "${entry% *}" "${entry#* }" || left=("${entry}" "${left[@]}")
  done
  OWNED_OBJECTS=("${left[@]}")
}

# owned_list: print the owned JobSets with the command that deletes each one safely.
owned_list() {
  local entry
  for entry in "${OWNED_OBJECTS[@]}"; do
    log "  JobSet ${entry% *} (uid ${entry#* }): delete with"
    log "    kubectl --context ${KUBE_CONTEXT} -n ${RAY_NAMESPACE} delete jobset ${entry% *}"
    log "    (after checking: kubectl ... get jobset ${entry% *} -o jsonpath='{.metadata.uid}' == ${entry#* })"
  done
}
