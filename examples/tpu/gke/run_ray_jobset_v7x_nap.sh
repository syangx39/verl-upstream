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
# Ray-on-JobSet launcher for the shared GKE TPU 7x (Ironwood) NAP cluster `bodaborg-tpu7x-nap`
# (Kueue + JobSet, node auto-provisioning, no KubeRay operator).
#
# One run = ONE JobSet (a CPU Ray head + one replicatedJob per TPU slice) pinned to ONE commit:
#   1. Pins the code: `git archive` of CODE_REF (default HEAD) -> <run dir>/src-<sha>.tar.gz on GCS. The manifest is
#      rendered from the same archive, and every pod runs its bootstrap (examples/tpu/gke/v7x_nap/) from it.
#   2. Writes <run dir>/job_spec.json: `bash $JOB_SCRIPT "$@"` plus the job's environment (JOBENV_<NAME>=<value> sets
#      <NAME>; `@RUN_DIR@` in arguments and values becomes the run directory as the pods see it).
#   3. Renders the JobSet (v7x_nap/render.py), checks it against the cluster's rules and validates it with the
#      cluster's admission webhooks (`kubectl create --dry-run=server`). DRY_RUN=1 stops here: nothing is uploaded,
#      nothing is created.
#   4. PREWARM=1: node auto-provisioning grows at most one node group per capacity request, so a job that needs two
#      new multi-host slices is never provisioned. One placeholder JobSet per slice is created first (same chips and
#      priority as the run); once their pods run they are deleted and the real JobSet is created right away. This is
#      best effort: the freed nodes are not reserved, and another job can take them before the run is admitted.
#   5. Creates the JobSet, waits for Kueue admission and the pods, follows the head log into LOG_DIR, waits for the
#      JobSet to finish and exits with the job's verdict (<run dir>/verdict.json).
#   6. Deletes only objects this invocation created: every JobSet is created on its own, its UID recorded
#      (LOG_DIR/owned_objects.log), and deleted with a UID precondition (v7x_nap/owned_objects.sh).
#
# Exit, Ctrl-C, SIGTERM and SIGHUP all delete the run's JobSet (and placeholders), which stops the run. To let a run
# continue without the launcher, set KEEP_JOBSET=1: the JobSet then ends with its head (success or failure) and is
# removed JOBSET_TTL_SECONDS later; checking and cleaning up is then yours. A launcher killed without running its
# exit handler (SIGKILL, lost machine) leaves the run going the same way.
#
# Reruns: a JobSet restart after a TPU pod failure (at most JOBSET_MAX_RESTARTS; a failed job is never restarted) or a
# Kueue eviction and re-admission (preemption, reclaim, node failure, pods-ready timeout) reruns the job from the start.
# The head enforces the total: the job runs at most MAX_ATTEMPTS = 1 + JOBSET_MAX_RESTARTS + ALLOW_REQUEUE times (each
# head registers under <run dir>/attempts/ and refuses to run beyond the budget, which fails the JobSet), also with
# KEEP_JOBSET=1 or without the launcher. Kueue can still re-admit the JobSet after that; its pods then start and stop
# again without running the job. The launcher additionally gives up (and deletes the JobSet) once the workload's
# eviction count exceeds ALLOW_REQUEUE. RUN_LIMIT_MIN is the cluster's limit for one run, counted from the pods
# becoming ready; queueing and pre-warm time come on top.
#
# Usage:
#   # Validate only (render + server-side dry-run; creates nothing):
#   DRY_RUN=1 GCS_BUCKET=<bucket> RAY_NAMESPACE=<namespace> bash examples/tpu/gke/run_ray_jobset_v7x_nap.sh
#
#   # Qwen3-30B-A3B GRPO (examples/tpu/grpo/run_qwen3_30b_a3b_torchtitan.sh) on 1 trainer + 1 rollout 2x2x4 slice,
#   # inputs (assets/hf/Qwen3-30B-A3B, data/gsm8k) under gs://<bucket>/<inputs prefix>:
#   GCS_BUCKET=<bucket> RUN_ROOT=<runs prefix> INPUTS_ROOT=<inputs prefix> RAY_NAMESPACE=<namespace> PREWARM=1 \
#   IMAGE=<image@sha256:...> JOBSET_MAX_RESTARTS=1 ALLOW_REQUEUE=0 \
#   JOBENV_MODEL_PATH=/dev/shm/models/Qwen3-30B-A3B \
#   JOBENV_LIBTPU_INIT_ARGS="--xla_tpu_use_enhanced_launch_barrier=false --xla_tpu_bf16_emission_mode=PROMOTE_F32" \
#   bash examples/tpu/gke/run_ray_jobset_v7x_nap.sh \
#     trainer.rollout_data_dir=@RUN_DIR@/rollout trainer.validation_data_dir=@RUN_DIR@/validation
#
# Do not pass credentials through JOBENV_*: the job spec is stored in the run directory.
# ===================================================================================
# shellcheck source-path=SCRIPTDIR
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
IMAGE_ENV_FILE="${REPO_ROOT}/tests/special_tpu/gke/tpu-ci-image.env"
if [[ -z "${IMAGE:-}" && -f "${IMAGE_ENV_FILE}" ]]; then
  # shellcheck source=/dev/null
  source "${IMAGE_ENV_FILE}"
fi

log() { echo "[v7x-nap $(date -u +%H:%M:%S)] $*"; }
die() {
  log "ERROR: $*" >&2
  exit 1
}

# --- Cluster & queue ---
KUBE_CONTEXT="${KUBE_CONTEXT:-gke_cloud-tpu-shared-capacity_us-central1_bodaborg-tpu7x-nap}"
RAY_NAMESPACE="${RAY_NAMESPACE:-default}"
KUEUE_QUEUE="${KUEUE_QUEUE:-multislice-queue}"
# medium is the highest class the cluster accepts for these queues; lower classes get preempted quickly.
PRIORITY_CLASS="${PRIORITY_CLASS:-medium}"
SERVICE_ACCOUNT="${SERVICE_ACCOUNT:-default}"
HEAD_NODEPOOL="${HEAD_NODEPOOL:-cpu-np}"
# Pin images by digest (`...@sha256:...`) so a run can be reproduced. Default: the TPU CI image
# (TPU_CI_IMAGE in tests/special_tpu/gke/tpu-ci-image.env).
IMAGE="${IMAGE:-${TPU_CI_IMAGE:-}}"

# --- Storage (GCS bucket mounted with GCS FUSE at DATA_MOUNT on every pod) ---
GCS_BUCKET="${GCS_BUCKET:-}"
DATA_MOUNT="${DATA_MOUNT:-/data}"
RUN_OWNER="$(echo "${RUN_OWNER:-${USER:-user}}" | tr '[:upper:]' '[:lower:]' | tr -cd 'a-z0-9' | cut -c1-12)"
RUN_ROOT="${RUN_ROOT:-verl-nap-runs/${RUN_OWNER}}"   # object prefix of the run directories
INPUTS_ROOT="${INPUTS_ROOT:-}"                       # object prefix exported to the job as RAY_DATA_HOME
GCSFUSE_MOUNT_OPTIONS="${GCSFUSE_MOUNT_OPTIONS:-implicit-dirs}"

# --- Slices (TPU 7x: 4 chips = 8 TensorCore devices per host) ---
TPU_TOPOLOGY="${TPU_TOPOLOGY:-2x2x4}"
TRAINER_SLICES="${TRAINER_SLICES:-1}"
ROLLOUT_SLICES="${ROLLOUT_SLICES:-1}"
RAY_TPU_PER_HOST="${RAY_TPU_PER_HOST:-8}"
POD_TPU_ACCELERATOR_TYPE="${POD_TPU_ACCELERATOR_TYPE:-tpu7x}"
HEAD_NUM_CPUS="${HEAD_NUM_CPUS:-12}"
HEAD_OBJECT_STORE_BYTES="${HEAD_OBJECT_STORE_BYTES:-32000000000}"
HEAD_DSHM_SIZE="${HEAD_DSHM_SIZE:-48Gi}"
# Holds the staged checkpoint (MODEL_PATH=/dev/shm/...) and Ray's object store (<= 200 GB by default).
TPU_DSHM_SIZE="${TPU_DSHM_SIZE:-400Gi}"
WORKER_NUM_CPUS="${WORKER_NUM_CPUS:-}"
# Whole-JobSet restarts after TPU pod failures (e.g. a reclaimed or failed node); a failed job (head exit != 0) is final.
JOBSET_MAX_RESTARTS="${JOBSET_MAX_RESTARTS:-0}"
JOBSET_TTL_SECONDS="${JOBSET_TTL_SECONDS:-1800}"
REQUIRE_NODE_RANK="${REQUIRE_NODE_RANK:-1}"
# Rank of a host inside its slice (Ray node label verl-tpu-node-rank, which verl orders a slice's hosts by):
# job-index = Kubernetes job completion index (= GKE TPU_WORKER_ID, torch_tpu 0.1.x's launcher order); probe = torch_tpu's
# topological rank (v7x_nap/tpu_node_rank.py, only for torch_tpu builds with calculate_topological_node_rank).
NODE_RANK_SOURCE="${NODE_RANK_SOURCE:-job-index}"
NODE_RANK_TIMEOUT_S="${NODE_RANK_TIMEOUT_S:-600}"

# --- Run ---
RUN_NAME="${RUN_NAME:-v7x-${RUN_OWNER:0:8}-$(date -u +%m%d-%H%M)}"
CODE_REF="${CODE_REF:-HEAD}"
JOB_SCRIPT="${JOB_SCRIPT:-examples/tpu/grpo/run_qwen3_30b_a3b_torchtitan.sh}"
# Run-time limit the cluster enforces from the pods becoming ready (shorter in the dev and regression namespaces).
case "${RAY_NAMESPACE}" in
  default) DEFAULT_RUN_LIMIT_MIN=180 ;;
  *) DEFAULT_RUN_LIMIT_MIN=120 ;;
esac
RUN_LIMIT_MIN="${RUN_LIMIT_MIN:-${DEFAULT_RUN_LIMIT_MIN}}"
TEARDOWN_MARGIN_MIN="${TEARDOWN_MARGIN_MIN:-10}"
DRIVER_TIMEOUT_MIN="${DRIVER_TIMEOUT_MIN:-$((RUN_LIMIT_MIN - TEARDOWN_MARGIN_MIN))}"
NODE_WAIT_TIMEOUT_MIN="${NODE_WAIT_TIMEOUT_MIN:-20}"
QUEUE_TIMEOUT_MIN="${QUEUE_TIMEOUT_MIN:-240}"
DRY_RUN="${DRY_RUN:-0}"
PREWARM="${PREWARM:-0}"
PREWARM_TIMEOUT_MIN="${PREWARM_TIMEOUT_MIN:-30}"
PREWARM_HOLD_S="${PREWARM_HOLD_S:-1800}"
KEEP_JOBSET="${KEEP_JOBSET:-0}"
# Kueue evictions (preemption, reclaim, node-failure requeue, pods-ready timeout) put the workload back in the queue
# and the run restarts from scratch on re-admission: number of evictions to wait through before giving up and
# deleting the JobSet (default 0: the first eviction ends the run).
ALLOW_REQUEUE="${ALLOW_REQUEUE:-0}"
POLL_S="${POLL_S:-20}"
LOG_DIR="${LOG_DIR:-/tmp/verl_nap_logs/${RUN_NAME}}"

# --- Derived ---
[[ -n "${GCS_BUCKET}" ]] || die "GCS_BUCKET is required (bucket mounted at ${DATA_MOUNT} on every pod)"
[[ -n "${IMAGE}" ]] || die "IMAGE is required"
if [[ "${IMAGE}" != *@sha256:* ]]; then
  log "WARNING: IMAGE ${IMAGE} is not pinned by digest; the run record will not identify the exact image"
fi
for tool in kubectl git python3; do
  command -v "${tool}" >/dev/null 2>&1 || die "${tool} not found on PATH"
done
if [[ "${DRY_RUN}" != "1" ]]; then
  command -v gcloud >/dev/null 2>&1 || die "gcloud not found on PATH (needed to stage the run directory)"
fi
CODE_SHA="$(git -C "${REPO_ROOT}" rev-parse --verify "${CODE_REF}^{commit}")" || die "unknown CODE_REF ${CODE_REF}"
git -C "${REPO_ROOT}" cat-file -e "${CODE_SHA}:${JOB_SCRIPT}" 2>/dev/null \
  || die "JOB_SCRIPT ${JOB_SCRIPT} is not in commit ${CODE_SHA}"
git -C "${REPO_ROOT}" cat-file -e "${CODE_SHA}:examples/tpu/gke/v7x_nap/worker.sh" 2>/dev/null \
  || die "commit ${CODE_SHA} does not contain the launcher bootstrap (examples/tpu/gke/v7x_nap/)"
if [[ -n "$(git -C "${REPO_ROOT}" status --porcelain --untracked-files=no)" ]]; then
  log "WARNING: uncommitted changes in ${REPO_ROOT} are NOT part of the run (it uses commit ${CODE_SHA})"
fi

RUN_SUBDIR="${RUN_ROOT%/}/${RUN_NAME}"
RUN_SUBDIR="${RUN_SUBDIR#/}"
RUN_GCS="gs://${GCS_BUCKET}/${RUN_SUBDIR}"
RUN_DIR="${DATA_MOUNT%/}/${RUN_SUBDIR}"
SRC_NAME="src-${CODE_SHA:0:12}.tar.gz"
SRC_TARBALL="${RUN_DIR}/${SRC_NAME}"
NODE_WAIT_TIMEOUT_S=$((NODE_WAIT_TIMEOUT_MIN * 60))
DRIVER_TIMEOUT_S=$((DRIVER_TIMEOUT_MIN * 60))
MAX_ATTEMPTS=$((1 + JOBSET_MAX_RESTARTS + ALLOW_REQUEUE))

mkdir -p "${LOG_DIR}"
MANIFEST="${LOG_DIR}/jobset.yaml"
PREWARM_MANIFEST="${LOG_DIR}/prewarm.yaml"
JOB_SPEC="${LOG_DIR}/job_spec.json"
RUN_RECORD="${LOG_DIR}/run_record.json"
HEAD_LOG="${LOG_DIR}/head.log"
SRC_DIR="${LOG_DIR}/src"
RENDER="${SRC_DIR}/examples/tpu/gke/v7x_nap/render.py"

export RUN_NAME RAY_NAMESPACE KUEUE_QUEUE PRIORITY_CLASS SERVICE_ACCOUNT RUN_OWNER CODE_SHA IMAGE
export POD_TPU_ACCELERATOR_TYPE GCS_BUCKET GCSFUSE_MOUNT_OPTIONS DATA_MOUNT RUN_DIR SRC_TARBALL INPUTS_ROOT
export HEAD_NODEPOOL HEAD_NUM_CPUS HEAD_OBJECT_STORE_BYTES HEAD_DSHM_SIZE TPU_DSHM_SIZE TPU_TOPOLOGY
export TRAINER_SLICES ROLLOUT_SLICES RAY_TPU_PER_HOST WORKER_NUM_CPUS JOBSET_MAX_RESTARTS JOBSET_TTL_SECONDS
export NODE_WAIT_TIMEOUT_S NODE_RANK_SOURCE NODE_RANK_TIMEOUT_S DRIVER_TIMEOUT_S REQUIRE_NODE_RANK PREWARM_HOLD_S KUBE_CONTEXT
export MAX_ATTEMPTS

k() { kubectl --context "${KUBE_CONTEXT}" -n "${RAY_NAMESPACE}" "$@"; }

WARM_NAMES=()
WARM_DIR="${LOG_DIR}/prewarm"
FOLLOWER_PID=""
JOBSET_UID=""
LAUNCH_ID="$(python3 -c 'import uuid; print(uuid.uuid4().hex[:16])')"
OWNED_LEDGER="${LOG_DIR}/owned_objects.log"  # read by owned_objects.sh
export LAUNCH_ID OWNED_LEDGER
# shellcheck source=v7x_nap/owned_objects.sh
source "${SCRIPT_DIR}/v7x_nap/owned_objects.sh"

stop_follower() {
  if [[ -n "${FOLLOWER_PID}" ]] && kill -0 "${FOLLOWER_PID}" 2>/dev/null; then
    kill "${FOLLOWER_PID}" 2>/dev/null || true
    wait "${FOLLOWER_PID}" 2>/dev/null || true
  fi
  FOLLOWER_PID=""
}

cleanup() {
  local rc=$?
  trap - INT TERM HUP
  stop_follower
  if (( ${#OWNED_OBJECTS[@]} + ${#OWNED_UNCONFIRMED[@]} > 0 )); then
    if [[ "${KEEP_JOBSET}" == "1" ]]; then
      log "KEEP_JOBSET=1: leaving the run's JobSets in place (the head still caps the job at ${MAX_ATTEMPTS} attempt(s)):"
      owned_list
    else
      owned_delete_all
      if (( ${#OWNED_OBJECTS[@]} + ${#OWNED_UNCONFIRMED[@]} > 0 )); then
        log "WARNING: some JobSets of this run could not be deleted or confirmed (see ${OWNED_LEDGER}):"
        owned_list
      fi
    fi
  fi
  log "local files: ${LOG_DIR}; run directory: ${RUN_GCS} (exit ${rc})"
  exit "${rc}"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
trap 'exit 129' HUP

# --- 1. Pin the code and render from the pinned tree ---
git -C "${REPO_ROOT}" archive --format=tar.gz -o "${LOG_DIR}/${SRC_NAME}" "${CODE_SHA}"
rm -rf "${SRC_DIR}"
mkdir -p "${SRC_DIR}"
tar -xzf "${LOG_DIR}/${SRC_NAME}" -C "${SRC_DIR}"

# --- 2. Job spec and run record ---
python3 - "${JOB_SPEC}" "${JOB_SCRIPT}" "$@" <<'PY'
import json
import os
import sys

out, script, *args = sys.argv[1:]
run_dir = os.environ["RUN_DIR"]


def sub(value: str) -> str:
    return value.replace("@RUN_DIR@", run_dir)


env = {
    "PYTHONPATH": ".",
    "PYTHONUNBUFFERED": "1",
    "VERL_PLATFORM": "tpu",
    "TPU_ACCELERATOR_TYPE": os.environ["POD_TPU_ACCELERATOR_TYPE"],
    "RAY_memory_monitor_refresh_ms": "0",
    "RAY_memory_usage_threshold": "0.99",
    "RAY_EXPERIMENTAL_NOSET_TPU_VISIBLE_CHIPS": "1",
    "RAY_OVERRIDE_JOB_RUNTIME_ENV": "1",
    "TENSORBOARD_DIR": f"{run_dir}/tensorboard",
    "VERL_NAP_RUN_DIR": run_dir,
}
if os.environ.get("INPUTS_ROOT", "").strip("/"):
    env["RAY_DATA_HOME"] = os.path.join(os.environ["DATA_MOUNT"], os.environ["INPUTS_ROOT"].strip("/"))
for name, value in sorted(os.environ.items()):
    if name.startswith("JOBENV_") and len(name) > len("JOBENV_"):
        env[name[len("JOBENV_") :]] = sub(value)
spec = {"entrypoint": ["bash", script, *(sub(a) for a in args)], "env_vars": env}
with open(out, "w") as f:
    json.dump(spec, f, indent=1, sort_keys=True)
    f.write("\n")
PY

python3 - "${RUN_RECORD}" <<'PY'
import json
import os
import sys
import time

keys = (
    "RUN_NAME RUN_OWNER CODE_SHA IMAGE KUBE_CONTEXT RAY_NAMESPACE KUEUE_QUEUE PRIORITY_CLASS TPU_TOPOLOGY "
    "TRAINER_SLICES ROLLOUT_SLICES RAY_TPU_PER_HOST POD_TPU_ACCELERATOR_TYPE GCS_BUCKET RUN_DIR INPUTS_ROOT "
    "DRIVER_TIMEOUT_S NODE_WAIT_TIMEOUT_S NODE_RANK_SOURCE REQUIRE_NODE_RANK JOBSET_MAX_RESTARTS"
).split()
record = {k.lower(): os.environ.get(k, "") for k in keys}
record["created_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
with open(sys.argv[1], "w") as f:
    json.dump(record, f, indent=1, sort_keys=True)
    f.write("\n")
PY

# --- 3. Render, check, validate ---
python3 "${RENDER}" manifest --out "${MANIFEST}"
SLICE_PLAN="$(python3 "${RENDER}" plan)"
TOTAL_HOSTS="$(python3 -c 'import json, sys; p = json.loads(sys.argv[1]); print(p["hosts_per_slice"] * len(p["slices"]))' "${SLICE_PLAN}")"
TOTAL_CHIPS=$((TOTAL_HOSTS * 4))

log "==================================================================================="
log "run ${RUN_NAME} (owner ${RUN_OWNER}) on ${KUBE_CONTEXT} / ${RAY_NAMESPACE} / queue ${KUEUE_QUEUE} / ${PRIORITY_CLASS}"
log "  code      ${CODE_SHA} (${CODE_REF})"
log "  image     ${IMAGE}"
log "  slices    ${SLICE_PLAN}"
log "  chips     ${TOTAL_CHIPS} (${TOTAL_HOSTS} hosts) + CPU head on ${HEAD_NODEPOOL}"
log "  run dir   ${RUN_GCS} (pods: ${RUN_DIR})"
log "  job       bash ${JOB_SCRIPT} $* (driver deadline ${DRIVER_TIMEOUT_MIN} min after head start)"
log "  reruns    at most ${MAX_ATTEMPTS} attempt(s): JOBSET_MAX_RESTARTS=${JOBSET_MAX_RESTARTS} ALLOW_REQUEUE=${ALLOW_REQUEUE} (head-enforced)"
log "  manifest  ${MANIFEST}"
log "==================================================================================="
python3 - "${JOB_SPEC}" <<'PY'
import json
import sys

spec = json.load(open(sys.argv[1]))
markers = ("TOKEN", "SECRET", "PASSWORD", "CREDENTIAL", "AUTH", "_KEY")
for name, value in sorted(spec["env_vars"].items()):
    shown = "<redacted>" if any(m in name.upper() for m in markers) else value
    print(f"[v7x-nap]   env {name}={shown}")
PY

kubectl config get-contexts "${KUBE_CONTEXT}" >/dev/null 2>&1 \
  || die "kube context ${KUBE_CONTEXT} not found (gcloud container clusters get-credentials ...)"
[[ "$(k auth can-i create jobsets.jobset.x-k8s.io 2>/dev/null || true)" == "yes" ]] \
  || die "no permission to create JobSets in namespace ${RAY_NAMESPACE}"
CLUSTER_QUEUE="$(k get localqueue "${KUEUE_QUEUE}" -o jsonpath='{.spec.clusterQueue}' 2>/dev/null || true)"
[[ -n "${CLUSTER_QUEUE}" ]] || die "LocalQueue ${KUEUE_QUEUE} not found in namespace ${RAY_NAMESPACE}"
if k get jobset "${RUN_NAME}" >/dev/null 2>&1; then
  die "JobSet ${RUN_NAME} already exists; every run needs a new RUN_NAME"
fi
CQ_JSON="${LOG_DIR}/clusterqueue.json"
if kubectl --context "${KUBE_CONTEXT}" get clusterqueue "${CLUSTER_QUEUE}" -o json >"${CQ_JSON}" 2>/dev/null; then
  python3 - "${CQ_JSON}" "${TOTAL_CHIPS}" <<'PY' || true
import json
import sys

# Admission is decided on quota *reservations* (taken at QuotaReserved, before the capacity admission check),
# not on the admitted usage, so the free room within the nominal quota is nominal - reserved.
cq = json.load(open(sys.argv[1]))
need = int(sys.argv[2])
reserved = {}
for entry in cq.get("status", {}).get("flavorsReservation", []):
    for r in entry.get("resources", []):
        if r["name"] == "google.com/tpu":
            reserved[entry["name"]] = (int(r.get("total", 0) or 0), int(r.get("borrowed", 0) or 0))
for rg in cq.get("spec", {}).get("resourceGroups", []):
    for flavor in rg.get("flavors", []):
        for r in flavor.get("resources", []):
            if r["name"] != "google.com/tpu":
                continue
            nominal = int(r.get("nominalQuota", 0) or 0)
            if not nominal:
                continue
            total, borrowed = reserved.get(flavor["name"], (0, 0))
            free = nominal - total
            verdict = "fits within the nominal quota" if need <= free else f"needs {need - max(free, 0)} chips of cohort borrowing or preemption, or a wait"
            print(f"[v7x-nap]   ClusterQueue {cq['metadata']['name']} / {flavor['name']}: TPU nominal {nominal}, "
                  f"reserved {total} (borrowed {borrowed}), free {free}; this run needs {need}: {verdict}")
PY
fi
rm -f "${CQ_JSON}"

log "server-side dry-run of ${MANIFEST}"
k create --dry-run=server -f "${MANIFEST}" -o name
if [[ "${PREWARM}" == "1" || "${DRY_RUN}" == "1" ]]; then
  python3 "${RENDER}" prewarm --out "${PREWARM_MANIFEST}" --out-dir "${WARM_DIR}"
  mapfile -t WARM_NAMES < <(python3 "${RENDER}" prewarm-names)
  for name in "${WARM_NAMES[@]}"; do
    if k get jobset "${name}" >/dev/null 2>&1; then
      die "pre-warm placeholder ${name} already exists; every run needs a new RUN_NAME"
    fi
  done
  log "server-side dry-run of ${PREWARM_MANIFEST}"
  k create --dry-run=server -f "${PREWARM_MANIFEST}" -o name
fi
if [[ "${DRY_RUN}" == "1" ]]; then
  log "DRY_RUN=1: manifest rendered and accepted by the API server; nothing was uploaded or created."
  exit 0
fi

# --- 4. Stage the run directory ---
if [[ -n "$(gcloud storage ls "${RUN_GCS}/" 2>/dev/null | head -n 1)" ]]; then
  die "${RUN_GCS}/ already has objects; every run needs a new RUN_NAME"
fi
log "staging ${RUN_GCS}/"
gcloud storage cp --quiet "${LOG_DIR}/${SRC_NAME}" "${RUN_GCS}/${SRC_NAME}"
gcloud storage cp --quiet "${JOB_SPEC}" "${RUN_GCS}/job_spec.json"
gcloud storage cp --quiet "${RUN_RECORD}" "${RUN_GCS}/run_record.json"
gcloud storage cp --quiet "${MANIFEST}" "${RUN_GCS}/jobset.yaml"

PODS_JSON="${LOG_DIR}/.pods.json"
pod_summary() {  # $1 = label selector; prints "<pods> <running> <fatal reason or ->", then one summary line
  k get pods -l "$1" -o json >"${PODS_JSON}" 2>/dev/null || echo '{}' >"${PODS_JSON}"
  python3 - "${PODS_JSON}" <<'PY'
import json
import sys

try:
    items = json.load(open(sys.argv[1])).get("items", [])
except ValueError:
    items = []
fatal_reasons = ("ImagePullBackOff", "ErrImagePull", "CreateContainerConfigError", "InvalidImageName")
running, fatal, parts = 0, "-", []
for pod in items:
    name = pod["metadata"]["name"]
    status = pod.get("status", {})
    phase = status.get("phase", "Pending")
    running += phase == "Running"
    for cs in status.get("containerStatuses", []) + status.get("initContainerStatuses", []):
        waiting = cs.get("state", {}).get("waiting", {})
        if waiting.get("reason") in fatal_reasons:
            fatal = "_".join([name, cs.get("name", ""), waiting.get("reason", ""), waiting.get("message", "")])
            fatal = fatal.replace(" ", "_")
    labels = pod["metadata"].get("labels", {})
    group = labels.get("jobset.sigs.k8s.io/replicatedjob-name", "?")
    if labels.get("verl-nap/prewarm") == "true":
        group = labels.get("jobset.sigs.k8s.io/jobset-name", "?").rsplit("-", 1)[-1]
    index = labels.get("batch.kubernetes.io/job-completion-index", "0")
    parts.append(f"{group}/{index}:{phase}")
print(len(items), running, fatal)
print(" ".join(sorted(parts)))
PY
}

# --- 5. Optional pre-warm ---
if [[ "${PREWARM}" == "1" ]]; then
  log "pre-warm: creating ${#WARM_NAMES[@]} placeholder JobSet(s): ${WARM_NAMES[*]}"
  for name in "${WARM_NAMES[@]}"; do  # one object per create, so a partial failure is tracked exactly
    owned_create "${WARM_DIR}/${name}.yaml" || die "could not create pre-warm placeholder ${name}"
  done
  WARM_START=$(date +%s)
  LAST=""
  while true; do
    mapfile -t S < <(pod_summary "verl-nap/run=${RUN_NAME},verl-nap/prewarm=true")
    read -r W_TOTAL W_RUNNING W_FATAL <<<"${S[0]:-0 0 -}"
    [[ "${W_FATAL}" == "-" ]] || die "pre-warm pod failed: ${W_FATAL}"
    if [[ "${S[1]:-}" != "${LAST}" ]]; then
      log "pre-warm: ${W_RUNNING}/${TOTAL_HOSTS} running (${S[1]:-no pods yet})"
      LAST="${S[1]:-}"
    fi
    if (( W_TOTAL == TOTAL_HOSTS && W_RUNNING == TOTAL_HOSTS )); then
      break
    fi
    (( $(date +%s) - WARM_START < PREWARM_TIMEOUT_MIN * 60 )) || die "pre-warm placeholders not running after ${PREWARM_TIMEOUT_MIN} min"
    sleep 10
  done
  log "pre-warm: all ${TOTAL_HOSTS} hosts held; swapping the placeholders for the run"
  owned_delete_all
  (( ${#OWNED_OBJECTS[@]} + ${#OWNED_UNCONFIRMED[@]} == 0 )) || die "could not delete the pre-warm placeholders"
  sleep 2  # let Kueue release the placeholders' quota before the run asks for it
fi

# --- 6. Create and watch ---
owned_create "${MANIFEST}" || die "could not create JobSet ${RUN_NAME}"
JOBSET_UID="${OWNED_LAST_UID}"
START_TS=$(date +%s)
ADMITTED_TS=""
LAST_WL=""
LAST_PODS=""
RESULT=""
LAST_EVICTIONS=0
LAST_RESTARTS="0"
HEAD_UIDS=""
TOTAL_PODS=$((TOTAL_HOSTS + 1))
while true; do
  NOW=$(date +%s)
  CONDS="$(k get jobset "${RUN_NAME}" -o jsonpath='{range .status.conditions[*]}{.type}={.status};{end}' 2>/dev/null || true)"
  if [[ "${CONDS}" == *"Completed=True"* ]]; then RESULT="Completed"; break; fi
  if [[ "${CONDS}" == *"Failed=True"* ]]; then RESULT="Failed"; break; fi
  if [[ "$(k get jobset "${RUN_NAME}" -o jsonpath='{.metadata.uid}' 2>/dev/null || true)" != "${JOBSET_UID}" ]]; then
    RESULT="Deleted"  # gone, or replaced by another object of that name (which is not ours to touch)
    break
  fi

  WL="$(k get workloads.kueue.x-k8s.io -l "kueue.x-k8s.io/job-uid=${JOBSET_UID}" \
    -o jsonpath='{range .items[0].status.conditions[*]}{.type}={.status}({.reason}) {end}' 2>/dev/null || true)"
  if [[ "${WL}" != "${LAST_WL}" ]]; then
    log "workload: ${WL:-not created yet}"
    LAST_WL="${WL}"
  fi
  if [[ -z "${ADMITTED_TS}" && "${WL}" == *"Admitted=True"* ]]; then
    ADMITTED_TS=${NOW}
    log "admitted after $((NOW - START_TS)) s"
  elif [[ -n "${ADMITTED_TS}" && "${WL}" != *"Admitted=True"* ]]; then
    ADMITTED_TS=""
    START_TS=${NOW}
    stop_follower
  fi
  # Evictions (preemption, reclaim, node failure, pods-ready timeout) as counted by Kueue itself, so an eviction and
  # re-admission between two polls is not missed; each one reruns the job on re-admission.
  EVICTIONS="$(k get workloads.kueue.x-k8s.io -l "kueue.x-k8s.io/job-uid=${JOBSET_UID}" \
    -o jsonpath='{range .items[0].status.schedulingStats.evictions[*]}{.count}{" "}{.reason}{";"}{end}' 2>/dev/null || true)"
  EVICTION_COUNT="$(python3 -c 'import sys; print(sum(int(e.split()[0]) for e in sys.argv[1].split(";") if e.strip()))' "${EVICTIONS}")"
  if (( EVICTION_COUNT != LAST_EVICTIONS )); then
    log "workload evictions: ${EVICTION_COUNT} (${EVICTIONS%;}; ALLOW_REQUEUE=${ALLOW_REQUEUE})"
    LAST_EVICTIONS=${EVICTION_COUNT}
    (( EVICTION_COUNT <= ALLOW_REQUEUE )) || die "workload evicted ${EVICTION_COUNT} time(s) > ALLOW_REQUEUE=${ALLOW_REQUEUE}"
  fi
  RESTARTS="$(k get jobset "${RUN_NAME}" -o jsonpath='{.status.restarts}' 2>/dev/null || true)"
  if [[ -n "${RESTARTS}" && "${RESTARTS}" != "${LAST_RESTARTS}" ]]; then
    log "JobSet restarts: ${RESTARTS} (JOBSET_MAX_RESTARTS=${JOBSET_MAX_RESTARTS})"
    LAST_RESTARTS="${RESTARTS}"
  fi
  for uid in $(k get pods -l "verl-nap/run=${RUN_NAME},verl-nap/role=head" -o jsonpath='{.items[*].metadata.uid}' 2>/dev/null || true); do
    if [[ " ${HEAD_UIDS} " != *" ${uid} "* ]]; then
      HEAD_UIDS="${HEAD_UIDS} ${uid}"
      log "head attempt $(wc -w <<<"${HEAD_UIDS}") of at most ${MAX_ATTEMPTS}: pod uid ${uid} (attempts/${uid:0:8}/)"
    fi
  done

  mapfile -t S < <(pod_summary "verl-nap/run=${RUN_NAME},verl-nap/prewarm!=true")
  read -r _ P_RUNNING P_FATAL <<<"${S[0]:-0 0 -}"
  [[ "${P_FATAL}" == "-" ]] || die "pod failed: ${P_FATAL}"
  if [[ "${S[1]:-}" != "${LAST_PODS}" ]]; then
    log "pods ${P_RUNNING}/${TOTAL_PODS} running: ${S[1]:-none}"
    LAST_PODS="${S[1]:-}"
  fi

  HEAD_POD="$(k get pods -l "verl-nap/run=${RUN_NAME},verl-nap/role=head" \
    --field-selector=status.phase=Running -o jsonpath='{.items[0].metadata.name}' 2>/dev/null || true)"
  if [[ -n "${HEAD_POD}" ]] && { [[ -z "${FOLLOWER_PID}" ]] || ! kill -0 "${FOLLOWER_PID}" 2>/dev/null; }; then
    SINCE=()
    [[ -s "${HEAD_LOG}" ]] && SINCE=("--since=$((POLL_S + 10))s")
    k logs -f "${HEAD_POD}" -c ray-head "${SINCE[@]}" >>"${HEAD_LOG}" 2>&1 &
    FOLLOWER_PID=$!
    log "following ${HEAD_POD} into ${HEAD_LOG}"
  fi

  if [[ -z "${ADMITTED_TS}" ]] && (( QUEUE_TIMEOUT_MIN > 0 && NOW - START_TS > QUEUE_TIMEOUT_MIN * 60 )); then
    die "not admitted after ${QUEUE_TIMEOUT_MIN} min"
  fi
  if [[ -n "${ADMITTED_TS}" ]] && (( NOW - ADMITTED_TS > (RUN_LIMIT_MIN + 20) * 60 )); then
    die "JobSet still running $((RUN_LIMIT_MIN + 20)) min after admission"
  fi
  sleep "${POLL_S}"
done

sleep 10
stop_follower
log "JobSet ${RUN_NAME}: ${RESULT} after $(($(date +%s) - START_TS)) s"
VERDICT="$(gcloud storage cat "${RUN_GCS}/verdict.json" 2>/dev/null || true)"
if [[ -n "${VERDICT}" ]]; then
  echo "${VERDICT}" >"${LOG_DIR}/verdict.json"
  log "verdict: $(python3 -c 'import json, sys; v = json.loads(sys.argv[1]); print(v.get("status"), "-", v.get("message", ""))' "${VERDICT}")"
else
  log "no verdict.json in ${RUN_GCS}/ (the head did not get that far; see ${HEAD_LOG})"
fi
[[ "${RESULT}" == "Completed" && "${VERDICT}" == *'"status": "SUCCEEDED"'* ]] || exit 1
log "SUCCESS"
