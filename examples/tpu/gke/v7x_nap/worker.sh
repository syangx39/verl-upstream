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
# Bootstrap of one TPU 7x host (pod) of a slice in the JobSet rendered by run_ray_jobset_v7x_nap.sh.
#
#   1. Multi-host slices: determine the host's rank inside the slice, by default its Kubernetes job completion index
#      (= GKE's TPU_WORKER_ID), or with VERL_NAP_NODE_RANK_SOURCE=probe its topological rank (tpu_node_rank.py).
#   2. Unset the multi-slice MEGASCALE_* variables GKE injects into JobSets, so every slice is its own libtpu mesh.
#      The other TPU_* variables stay: Ray derives its TPU node labels from them, and verl sets the per-process
#      TPU environment of its workers explicitly.
#   3. Record the pod's allowlisted environment and, in the background, that of its Ray worker processes
#      (pod_env_snapshot.py) under $VERL_NAP_RUN_DIR/env/.
#   4. Join the Ray head with resources {"TPU": <devices per host>, "<slice resource>": 1} and the Ray node label
#      verl-tpu-node-rank=<rank>, which verl uses to order the slice's hosts.
#
# Configured by the VERL_NAP_* variables of the pod spec (ray-jobset-v7x-nap-slice.yaml).
# ===================================================================================
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
: "${VERL_NAP_RUN_DIR:?}" "${VERL_NAP_HEAD_HOST:?}" "${VERL_NAP_SLICE_RESOURCE:?}" "${VERL_NAP_SLICE_TOPOLOGY:?}"
: "${VERL_NAP_HOSTS_PER_SLICE:?}" "${VERL_NAP_RAY_TPU_PER_HOST:?}"
POD="${POD_NAME:-$(hostname)}"
log() { echo "[verl-nap ${VERL_NAP_ROLE:-worker} ${POD}] $*"; }

mkdir -p "${VERL_NAP_RUN_DIR}/env" || log "WARNING: cannot create ${VERL_NAP_RUN_DIR}/env"

# 1. The host's rank inside its slice (published as the Ray node label verl-tpu-node-rank in step 4).
#    job-index (default): the Kubernetes job completion index. GKE exports it as TPU_WORKER_ID (inherited by every
#      TPU process) and lists TPU_WORKER_HOSTNAMES in that order, and torch_tpu 0.1.x's own multi-host launcher uses
#      it as NODE_RANK; verl's host order then agrees with all of them (verl orders by IP string otherwise).
#    probe: the topological rank from torch_tpu's calculate_topological_node_rank (only in builds that have it),
#      computed on every host of the slice at about the same time, before Ray starts.
NODE_RANK=""
NODE_RANK_OUT=""
NODE_RANK_SOURCE="${VERL_NAP_NODE_RANK_SOURCE:-job-index}"
if (( VERL_NAP_HOSTS_PER_SLICE == 1 )); then
  NODE_RANK=0
elif [[ "${NODE_RANK_SOURCE}" == "job-index" ]]; then
  NODE_RANK="${JOB_COMPLETION_INDEX:-}"
  if [[ -n "${TPU_WORKER_ID:-}" && "${TPU_WORKER_ID}" != "${NODE_RANK}" ]]; then
    # The premise of this rank source is that both orders agree; if they do not, publish no rank so that the head
    # (REQUIRE_NODE_RANK=1) stops the run for investigation instead of guessing.
    log "ERROR: JOB_COMPLETION_INDEX=${NODE_RANK} differs from GKE TPU_WORKER_ID=${TPU_WORKER_ID}; no node rank label"
    NODE_RANK=""
  fi
  log "node rank ${NODE_RANK:-MISSING} (source: job completion index)"
elif [[ "${NODE_RANK_SOURCE}" == "probe" ]]; then
  NODE_RANK_OUT="$(timeout "${VERL_NAP_NODE_RANK_TIMEOUT_S:-600}" python3 "${HERE}/tpu_node_rank.py" \
    "${VERL_NAP_SLICE_TOPOLOGY}" 2>/tmp/verl-nap-node-rank.err | grep '^NODE_RANK=' | tail -n 1 || true)"
  NODE_RANK="$(sed -n 's/^NODE_RANK=\([0-9][0-9]*\) .*/\1/p' <<<"${NODE_RANK_OUT}")"
  log "topological node rank: ${NODE_RANK_OUT:-FAILED} (job completion index ${JOB_COMPLETION_INDEX:-?}," \
    "GKE TPU_WORKER_ID ${TPU_WORKER_ID:-?})"
  if [[ -z "${NODE_RANK}" ]]; then
    tail -n 20 /tmp/verl-nap-node-rank.err 2>/dev/null | sed "s/^/[verl-nap ${POD}] node rank probe: /" || true
  fi
  # A probe killed by the timeout can leave libtpu's lock file behind, which would block the TPU for Ray's workers.
  rm -f /tmp/libtpu_lockfile
else
  log "WARNING: unknown VERL_NAP_NODE_RANK_SOURCE=${NODE_RANK_SOURCE}; no node rank label"
fi

# 2. Multi-slice variables off.
for var in $(compgen -e | grep '^MEGASCALE_' || true); do
  log "unset ${var}=${!var}"
  unset "${var}"
done

# 3. Environment evidence.
python3 "${HERE}/pod_env_snapshot.py" pod --out "${VERL_NAP_RUN_DIR}/env/${POD}.json" \
  --set "role=${VERL_NAP_ROLE:-}" --set "slice=${VERL_NAP_SLICE_RESOURCE}" --set "node_rank=${NODE_RANK}" \
  --set "node_rank_source=${NODE_RANK_SOURCE}" --set "node_rank_probe=${NODE_RANK_OUT}" \
  --set "job_completion_index=${JOB_COMPLETION_INDEX:-}" --set "gke_tpu_worker_id=${TPU_WORKER_ID:-}" \
  || log "WARNING: pod environment snapshot failed"
nohup python3 "${HERE}/pod_env_snapshot.py" procs --out "${VERL_NAP_RUN_DIR}/env/${POD}-procs.jsonl" \
  --interval 60 >/tmp/verl-nap-procs.log 2>&1 &

# 4. Join the Ray cluster.
for _ in $(seq 1 360); do
  if python3 -c 'import socket, sys; socket.create_connection((sys.argv[1], 6379), 3).close()' \
      "${VERL_NAP_HEAD_HOST}" 2>/dev/null; then
    break
  fi
  sleep 5
done
RESOURCES="{\"TPU\": ${VERL_NAP_RAY_TPU_PER_HOST}, \"${VERL_NAP_SLICE_RESOURCE}\": 1}"
LABEL_ARGS=()
if [[ -n "${NODE_RANK}" ]]; then
  if python3 -c 'from ray._private.label_utils import parse_node_labels_string' 2>/dev/null; then
    LABEL_ARGS=("--labels=verl-tpu-node-rank=${NODE_RANK}")
  else  # older Ray: JSON form, values must be strings
    LABEL_ARGS=("--labels={\"verl-tpu-node-rank\": \"${NODE_RANK}\"}")
  fi
fi
CPU_ARGS=()
if [[ -n "${VERL_NAP_WORKER_NUM_CPUS:-}" ]]; then
  CPU_ARGS=("--num-cpus=${VERL_NAP_WORKER_NUM_CPUS}")
fi
log "ray start -> ${VERL_NAP_HEAD_HOST}:6379 resources=${RESOURCES} ${LABEL_ARGS[*]:-(no node rank label)}"
exec ray start --address="${VERL_NAP_HEAD_HOST}:6379" --resources="${RESOURCES}" "${LABEL_ARGS[@]}" \
  "${CPU_ARGS[@]}" --disable-usage-stats --block
