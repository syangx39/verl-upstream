#!/usr/bin/env bash
# Multi-node (N x 4-GPU NVIDIA GB200; default NUM_NODES=2) Ray cluster launcher on GKE with GCSFuse /data and /dev/shm model staging.
#
# SECURITY & ISOLATION BOUNDARY:
# - Pods in this launcher use `securityContext.privileged: true` (required by the NVIDIA DRA driver to inject
#   `/dev/nvidia-caps-imex-channels` for multi-GPU NVLink5 / ComputeDomain IMEX P2P) and mount the shared GCSFuse
#   bucket at `/data`.
# - Meanwhile, the default loopback `sandbox_fusion_server.py` executes model-generated Python snippets via local
#   subprocesses (dropping privileges to `nobody:nogroup` 65534:65534 and hiding GPUs via `CUDA_VISIBLE_DEVICES=""`,
#   but sharing the container filesystem and kernel namespaces). The 4-case check in `run_qwen3_30b_a3b_dapo_torchtitan_gb200.sh`
#   only verifies functional API compatibility, NOT security isolation.
# - Therefore, this launcher with the in-Pod loopback tool server MUST ONLY be used in single-tenant, trusted
#   benchmark environments with trusted datasets/models. For untrusted prompts/models or shared environments,
#   point `SANDBOX_FUSION_HOST` to an external, non-privileged, sandboxed (`nsjail`/`gVisor`) tool execution service
#   that does not mount `/data` or host devices.
#
# Required environment variables (kept external so no cluster/bucket names are hardcoded):
#   RUN_NAME       Globally unique run identifier across all concurrent runs sharing `gs://${GCS_BUCKET}/${RUN_ROOT}`
#                  (including across different K8s namespaces or clusters; e.g., g1-retool-rl-smoke-01)
#   KUBE_CONTEXT   Kubernetes context name
#   NODEPOOL       Target 4xGB200 GKE nodepool name
#   GCS_BUCKET     Shared GCS bucket name mounted via GCSFuse CSI at /data
#   RUN_ROOT       Bucket-relative directory for run outputs (e.g., exp/retool-gb200/runs)
#   INPUTS_ROOT    Bucket-relative directory for datasets (e.g., exp/retool-v7x/inputs)
#   IMAGE          Container image (preferably immutable @sha256:... digest)

set -euo pipefail

: "${RUN_NAME:?RUN_NAME is required}"
: "${KUBE_CONTEXT:?KUBE_CONTEXT is required}"
: "${NODEPOOL:?NODEPOOL is required}"
: "${GCS_BUCKET:?GCS_BUCKET is required}"
: "${RUN_ROOT:?RUN_ROOT is required}"
: "${INPUTS_ROOT:?INPUTS_ROOT is required}"
: "${IMAGE:?IMAGE is required}"

K8S_NAMESPACE="${K8S_NAMESPACE:-default}"
K8S_SA="${K8S_SA:-default}"
JOB_SCRIPT="${JOB_SCRIPT:-examples/tpu/retool/gb200/run_qwen3_30b_a3b_dapo_torchtitan_gb200.sh}"
KEEP_PODS="${KEEP_PODS:-0}"
ALLOW_OVERWRITE_RUN_DIR="${ALLOW_OVERWRITE_RUN_DIR:-0}"
LAUNCH_ID="${LAUNCH_ID:-$(python3 -c 'import uuid; print(uuid.uuid4().hex[:12])')}"

NUM_NODES="${NUM_NODES:-2}"
GPUS_PER_NODE="${GPUS_PER_NODE:-4}"
if ! [[ "${NUM_NODES}" =~ ^[0-9]+$ ]] || [[ "${NUM_NODES}" -lt 2 ]]; then
    echo "ERROR: NUM_NODES must be an integer >= 2 (got NUM_NODES='${NUM_NODES}')" >&2
    exit 1
fi
if ! [[ "${GPUS_PER_NODE}" =~ ^[0-9]+$ ]] || [[ "${GPUS_PER_NODE}" -lt 1 ]] || [[ "${GPUS_PER_NODE}" -gt 4 ]]; then
    echo "ERROR: GPUS_PER_NODE must be an integer between 1 and 4 on 4xGB200 nodes (got GPUS_PER_NODE='${GPUS_PER_NODE}')" >&2
    exit 1
fi
NUM_WORKERS=$((NUM_NODES - 1))
TOTAL_GPUS=$((NUM_NODES * GPUS_PER_NODE))
COMPUTE_DOMAIN_NUM_NODES="${COMPUTE_DOMAIN_NUM_NODES:-${NUM_NODES}}"
if ! [[ "${COMPUTE_DOMAIN_NUM_NODES}" =~ ^[0-9]+$ ]] || [[ "${COMPUTE_DOMAIN_NUM_NODES}" -lt 1 ]]; then
    echo "ERROR: COMPUTE_DOMAIN_NUM_NODES must be an integer >= 1 (got COMPUTE_DOMAIN_NUM_NODES='${COMPUTE_DOMAIN_NUM_NODES}')" >&2
    exit 1
fi

SANDBOX_FUSION_HOST="${SANDBOX_FUSION_HOST:-${JOBENV_SANDBOX_FUSION_HOST:-127.0.0.1}}"
SANDBOX_FUSION_PORT="${SANDBOX_FUSION_PORT:-${JOBENV_SANDBOX_FUSION_PORT:-18080}}"
SANDBOX_FUSION_URL="${SANDBOX_FUSION_URL:-${JOBENV_SANDBOX_FUSION_URL:-http://${SANDBOX_FUSION_HOST}:${SANDBOX_FUSION_PORT}/run_code}}"

RUN_DIR="/data/${RUN_ROOT}/${RUN_NAME}"
INPUTS_DIR="/data/${INPUTS_ROOT}"
TENSORBOARD_DIR="${TENSORBOARD_DIR:-${JOBENV_TENSORBOARD_DIR:-${RUN_DIR}/tensorboard}}"
EXPECT_CHECKPOINT_SHA256="${EXPECT_CHECKPOINT_SHA256:-${JOBENV_EXPECT_CHECKPOINT_SHA256:-59247a00c13f2364a12aab5ab6b88b6d971d59ff643c57aad8ac4e55a71231b0}}"
EXPECT_GIT_SHA="${EXPECT_GIT_SHA:-${JOBENV_EXPECT_GIT_SHA:-}}"
WANDB_SECRET_NAME="${WANDB_SECRET_NAME:-}"
WANDB_SECRET_KEY="${WANDB_SECRET_KEY:-WANDB_API_KEY}"

WANDB_EXTRA_ENV=""
if [[ -n "${WANDB_SECRET_NAME}" ]]; then
    if ! [[ "${WANDB_SECRET_NAME}" =~ ^[a-z0-9]([-a-z0-9.]*[a-z0-9])?$ ]] || [[ "${#WANDB_SECRET_NAME}" -gt 253 ]]; then
        echo "ERROR: WANDB_SECRET_NAME='${WANDB_SECRET_NAME}' is not a valid Kubernetes Secret name" >&2
        exit 1
    fi
    if ! [[ "${WANDB_SECRET_KEY}" =~ ^[-._a-zA-Z0-9]+$ ]]; then
        echo "ERROR: WANDB_SECRET_KEY='${WANDB_SECRET_KEY}' is not a valid Kubernetes Secret key" >&2
        exit 1
    fi
    WANDB_EXTRA_ENV+="    - name: WANDB_API_KEY"$'\n'
    WANDB_EXTRA_ENV+="      valueFrom:"$'\n'
    WANDB_EXTRA_ENV+="        secretKeyRef:"$'\n'
    WANDB_EXTRA_ENV+="          name: \"${WANDB_SECRET_NAME}\""$'\n'
    WANDB_EXTRA_ENV+="          key: \"${WANDB_SECRET_KEY}\""$'\n'
fi

EXTRA_ARGS=()
for arg in "$@"; do
    resolved="${arg//@RUN_DIR@/${RUN_DIR}}"
    resolved="${resolved//@INPUTS_DIR@/${INPUTS_DIR}}"
    EXTRA_ARGS+=("${resolved}")
done

JOBENV_YAML=""
while IFS='=' read -r name val; do
    if [[ "${name}" == JOBENV_* ]]; then
        env_key="${name#JOBENV_}"
        if [[ "${env_key}" == "WANDB_API_KEY" ]]; then
            echo "ERROR: refusing to inject plaintext JOBENV_WANDB_API_KEY into Pod specs; use WANDB_SECRET_NAME=<k8s-secret-name> instead." >&2
            exit 1
        fi
        if [[ "${env_key}" == "SANDBOX_FUSION_HOST" || "${env_key}" == "SANDBOX_FUSION_PORT" || "${env_key}" == "SANDBOX_FUSION_URL" || "${env_key}" == "TENSORBOARD_DIR" || "${env_key}" == "EXPECT_CHECKPOINT_SHA256" || "${env_key}" == "EXPECT_GIT_SHA" ]]; then
            continue
        fi
        escaped_val="${val//\"/\\\"}"
        JOBENV_YAML+="    - name: ${env_key}"$'\n'"      value: \"${escaped_val}\""$'\n'
    fi
done < <(env | sort)

CREATED_CD_UID=""
CREATED_SVC_UID=""
CREATED_HEAD_POD_UID=""
declare -a CREATED_WORKER_POD_UIDS=()
declare -a WORKER_POD_NAMES=()

ATTEMPTED_CD=""
ATTEMPTED_SVC=""
ATTEMPTED_HEAD_POD=""
declare -a ATTEMPTED_WORKER_PODS=()

for ((w = 0; w < NUM_WORKERS; w++)); do
    WORKER_POD_NAMES[w]="${RUN_NAME}-worker-${w}"
    CREATED_WORKER_POD_UIDS[w]=""
    ATTEMPTED_WORKER_PODS[w]=""
done

resolve_attempted_uid() {
    local k8s_res="$1"
    local recorded_uid="$2"
    local attempted="$3"
    if [[ -n "${recorded_uid}" ]]; then
        printf '%s' "${recorded_uid}"
        return 0
    fi
    if [[ "${attempted}" != "1" ]]; then
        return 0
    fi
    local get_out get_rc
    set +e
    get_out="$(kubectl --context="${KUBE_CONTEXT}" -n "${K8S_NAMESPACE}" get "${k8s_res}" -o jsonpath='{.metadata.labels.retool-launch-id}:{.metadata.uid}' 2>&1)"
    get_rc=$?
    set -e
    if [[ "${get_rc}" -eq 0 ]]; then
        local found_launch_id="${get_out%%:*}"
        local found_uid="${get_out#*:}"
        if [[ "${found_launch_id}" == "${LAUNCH_ID}" && -n "${found_uid}" ]]; then
            printf '%s' "${found_uid}"
            return 0
        fi
        return 0
    fi
    if [[ "${get_out}" == *"NotFound"* || "${get_out}" == *"not found"* ]]; then
        return 0
    fi
    echo "ERROR: cleanup failed to inspect ${k8s_res} after ambiguous create (launch_id=${LAUNCH_ID}, rc=${get_rc}): ${get_out}" >&2
    return 1
}

atomic_delete_by_uid() {
    local proxy_addr="$1"
    local api_path="$2"
    local expected_uid="$3"
    if [[ -z "${expected_uid}" ]]; then
        return 0
    fi
    local payload resp http_code curl_rc
    payload="$(printf '{"apiVersion":"meta.k8s.io/v1","kind":"DeleteOptions","propagationPolicy":"Background","preconditions":{"uid":"%s"}}' "${expected_uid}")"
    set +e
    resp="$(curl -sS -w '\n%{http_code}' -X DELETE \
        -H "Content-Type: application/json" \
        -d "${payload}" \
        "http://${proxy_addr}${api_path}" 2>&1)"
    curl_rc=$?
    set -e
    if [[ "${curl_rc}" -ne 0 ]]; then
        echo "ERROR: curl DELETE failed for ${api_path} (uid=${expected_uid}, rc=${curl_rc}): ${resp}" >&2
        return 1
    fi
    http_code="$(printf '%s\n' "${resp}" | tail -n1)"
    if [[ "${http_code}" == "200" || "${http_code}" == "202" || "${http_code}" == "404" ]]; then
        return 0
    fi
    echo "ERROR: Kubernetes DELETE failed for ${api_path} (uid=${expected_uid}, http=${http_code}): ${resp}" >&2
    return 1
}

cleanup() {
    local exit_rc=$?
    local cleanup_failed=0
    if [[ "${KEEP_PODS}" != "1" ]]; then
        local any_worker_attempted=""
        local w
        for ((w = 0; w < NUM_WORKERS; w++)); do
            if [[ -n "${ATTEMPTED_WORKER_PODS[$w]:-}" ]]; then
                any_worker_attempted="1"
                break
            fi
        done
        if [[ -n "${ATTEMPTED_HEAD_POD}${any_worker_attempted}${ATTEMPTED_SVC}${ATTEMPTED_CD}" ]]; then
            echo "[gb200-launcher $(date -u +%H:%M:%S)] atomically cleaning up resources by DeleteOptions.preconditions.uid for ${RUN_NAME} (launch_id=${LAUNCH_ID})"
            local head_uid="" svc_uid="" cd_uid="" any_worker_uid=""
            local -a worker_uids=()
            head_uid="$(resolve_attempted_uid "pod/${RUN_NAME}-head" "${CREATED_HEAD_POD_UID}" "${ATTEMPTED_HEAD_POD}")" || cleanup_failed=1
            for ((w = 0; w < NUM_WORKERS; w++)); do
                worker_uids[w]="$(resolve_attempted_uid "pod/${WORKER_POD_NAMES[$w]}" "${CREATED_WORKER_POD_UIDS[$w]:-}" "${ATTEMPTED_WORKER_PODS[$w]:-}")" || cleanup_failed=1
                if [[ -n "${worker_uids[$w]}" ]]; then
                    any_worker_uid="1"
                fi
            done
            svc_uid="$(resolve_attempted_uid "svc/${RUN_NAME}-head" "${CREATED_SVC_UID}" "${ATTEMPTED_SVC}")" || cleanup_failed=1
            cd_uid="$(resolve_attempted_uid "computedomain.resource.nvidia.com/${RUN_NAME}-cd" "${CREATED_CD_UID}" "${ATTEMPTED_CD}")" || cleanup_failed=1

            if [[ -n "${head_uid}${any_worker_uid}${svc_uid}${cd_uid}" ]]; then
                local proxy_log proxy_pid proxy_addr
                proxy_log="$(mktemp)"
                kubectl --context="${KUBE_CONTEXT}" proxy --port=0 >"${proxy_log}" 2>&1 &
                proxy_pid=$!
                proxy_addr=""
                for _ in $(seq 1 50); do
                    if grep -q "Starting to serve on" "${proxy_log}" 2>/dev/null; then
                        proxy_addr="$(sed -n 's/.*Starting to serve on //p' "${proxy_log}" | head -n1)"
                        break
                    fi
                    sleep 0.1
                done
                if [[ -n "${proxy_addr}" ]]; then
                    atomic_delete_by_uid "${proxy_addr}" "/api/v1/namespaces/${K8S_NAMESPACE}/pods/${RUN_NAME}-head" "${head_uid}" || cleanup_failed=1
                    for ((w = 0; w < NUM_WORKERS; w++)); do
                        atomic_delete_by_uid "${proxy_addr}" "/api/v1/namespaces/${K8S_NAMESPACE}/pods/${WORKER_POD_NAMES[$w]}" "${worker_uids[$w]}" || cleanup_failed=1
                    done
                    atomic_delete_by_uid "${proxy_addr}" "/api/v1/namespaces/${K8S_NAMESPACE}/services/${RUN_NAME}-head" "${svc_uid}" || cleanup_failed=1
                    atomic_delete_by_uid "${proxy_addr}" "/apis/resource.nvidia.com/v1beta1/namespaces/${K8S_NAMESPACE}/computedomains/${RUN_NAME}-cd" "${cd_uid}" || cleanup_failed=1
                else
                    echo "ERROR: kubectl proxy failed to start during cleanup; unable to delete resources for ${RUN_NAME} (launch_id=${LAUNCH_ID})." >&2
                    cleanup_failed=1
                fi
                kill "${proxy_pid}" 2>/dev/null || true
                wait "${proxy_pid}" 2>/dev/null || true
                rm -f "${proxy_log}"
            fi
        fi
    fi
    if [[ "${cleanup_failed}" -ne 0 ]]; then
        exit 1
    fi
    exit "${exit_rc}"
}
trap cleanup EXIT

# 1. Fail fast if any target K8s resource name is already in use so we never clobber an active run.
PREFLIGHT_RES_SPECS=("pod/${RUN_NAME}-head")
for ((w = 0; w < NUM_WORKERS; w++)); do
    PREFLIGHT_RES_SPECS+=("pod/${WORKER_POD_NAMES[$w]}")
done
PREFLIGHT_RES_SPECS+=(
    "svc/${RUN_NAME}-head"
    "computedomain.resource.nvidia.com/${RUN_NAME}-cd"
)
for res_spec in "${PREFLIGHT_RES_SPECS[@]}"; do
    if kubectl --context="${KUBE_CONTEXT}" -n "${K8S_NAMESPACE}" get "${res_spec}" >/dev/null 2>&1; then
        echo "ERROR: ${res_spec} already exists in namespace ${K8S_NAMESPACE} for RUN_NAME=${RUN_NAME}; refusing to delete or overwrite existing workload." >&2
        exit 1
    fi
done
if [[ -n "${WANDB_SECRET_NAME}" ]]; then
    SECRET_HAS_KEY="$(kubectl --context="${KUBE_CONTEXT}" -n "${K8S_NAMESPACE}" get secret "${WANDB_SECRET_NAME}" -o jsonpath="{.data.${WANDB_SECRET_KEY}}" 2>/dev/null || true)"
    if [[ -z "${SECRET_HAS_KEY}" ]]; then
        echo "ERROR: Secret ${WANDB_SECRET_NAME} (key ${WANDB_SECRET_KEY}) not found or empty in namespace ${K8S_NAMESPACE}" >&2
        exit 1
    fi
    unset SECRET_HAS_KEY
    echo "[gb200-launcher] verified Secret ${WANDB_SECRET_NAME}[${WANDB_SECRET_KEY}] exists in namespace ${K8S_NAMESPACE}"
fi

# 2. Refuse to reuse an existing GCS run directory or stale G0 precheck files unless ALLOW_OVERWRITE_RUN_DIR=1.
#    Fail closed if `gsutil stat` fails for any reason other than genuine object-not-found (e.g. permission/auth/network error).
#    NOTE: RUN_NAME must be globally unique across all concurrent launches sharing `gs://${GCS_BUCKET}/${RUN_ROOT}`
#    (including across different K8s namespaces or clusters). To close the TOCTOU window between `gsutil stat` and
#    pod startup, when ALLOW_OVERWRITE_RUN_DIR!=1 we also atomically create `${GCS_RUN_URI}/.launch_lock.json` with
#    GCS precondition `x-goog-if-generation-match:0`.
GCS_RUN_URI="gs://${GCS_BUCKET}/${RUN_ROOT}/${RUN_NAME}"
set +e
gcs_stat_out="$(gsutil stat "${GCS_RUN_URI}/**" 2>&1)"
gcs_stat_rc=$?
set -e
if [[ "${gcs_stat_rc}" -eq 0 ]]; then
    if [[ "${ALLOW_OVERWRITE_RUN_DIR}" != "1" ]]; then
        echo "ERROR: ${GCS_RUN_URI}/ already contains artifacts from a previous run; refusing to reuse stale RUN_DIR or G0 precheck files. Choose a globally unique RUN_NAME or set ALLOW_OVERWRITE_RUN_DIR=1." >&2
        exit 1
    fi
    gsutil -m rm -f \
        "${GCS_RUN_URI}/exit_code.txt" \
        "${GCS_RUN_URI}/env/*-g0-verify.json" >/dev/null 2>&1 || true
elif [[ "${gcs_stat_out}" == *"No URLs matched"* || "${gcs_stat_out}" == *"One or more URLs matched no objects"* || "${gcs_stat_out}" == *"matched no objects or files"* ]]; then
    if [[ "${ALLOW_OVERWRITE_RUN_DIR}" != "1" ]]; then
        set +e
        gcs_lock_out="$(printf '{"run_name":"%s","launch_id":"%s","namespace":"%s"}\n' "${RUN_NAME}" "${LAUNCH_ID}" "${K8S_NAMESPACE}" | gsutil -h "x-goog-if-generation-match:0" cp - "${GCS_RUN_URI}/.launch_lock.json" 2>&1)"
        gcs_lock_rc=$?
        set -e
        if [[ "${gcs_lock_rc}" -ne 0 ]]; then
            echo "ERROR: failed to acquire atomic GCS launch lock ${GCS_RUN_URI}/.launch_lock.json (rc=${gcs_lock_rc}); another concurrent run may have claimed RUN_NAME=${RUN_NAME}: ${gcs_lock_out}" >&2
            exit 1
        fi
    fi
else
    echo "ERROR: failed to check GCS run directory ${GCS_RUN_URI}/ (rc=${gcs_stat_rc}); failing closed to prevent accidental overwrite: ${gcs_stat_out}" >&2
    exit 1
fi

EXTRA_ARGS_STR=""
for a in "${EXTRA_ARGS[@]}"; do
    EXTRA_ARGS_STR+=" $(printf '%q' "${a}")"
done

create_k8s_object_and_record_uid() {
    local var_name="$1"
    local attempted_var_name="$2"
    printf -v "${attempted_var_name}" '1'
    local uid=""
    if uid="$(kubectl --context="${KUBE_CONTEXT}" -n "${K8S_NAMESPACE}" create -f - -o jsonpath='{.metadata.uid}')" && [[ -n "${uid}" ]]; then
        printf -v "${var_name}" '%s' "${uid}"
        return 0
    fi
    echo "ERROR: failed to create or capture metadata.uid from Kubernetes object (${var_name}, launch_id=${LAUNCH_ID})" >&2
    exit 1
}

create_k8s_object_and_record_uid CREATED_CD_UID ATTEMPTED_CD <<EOF
apiVersion: resource.nvidia.com/v1beta1
kind: ComputeDomain
metadata:
  name: ${RUN_NAME}-cd
  labels:
    retool-run: ${RUN_NAME}
    retool-launch-id: ${LAUNCH_ID}
spec:
  numNodes: ${COMPUTE_DOMAIN_NUM_NODES}
  channel:
    allocationMode: Single
    resourceClaimTemplate:
      name: ${RUN_NAME}-cd-channel
EOF

create_k8s_object_and_record_uid CREATED_SVC_UID ATTEMPTED_SVC <<EOF
apiVersion: v1
kind: Service
metadata:
  name: ${RUN_NAME}-head
  labels:
    retool-run: ${RUN_NAME}
    retool-launch-id: ${LAUNCH_ID}
spec:
  clusterIP: None
  selector:
    retool-run: ${RUN_NAME}
    retool-role: head
  ports:
  - name: ray-gcs
    port: 6379
    targetPort: 6379
  - name: ray-dash
    port: 8265
    targetPort: 8265
EOF

for ((w = 0; w < NUM_WORKERS; w++)); do
    WORKER_POD_NAME="${WORKER_POD_NAMES[$w]}"
    WORKER_LABEL="worker-${w}"
    create_k8s_object_and_record_uid "CREATED_WORKER_POD_UIDS[$w]" "ATTEMPTED_WORKER_PODS[$w]" <<EOF
apiVersion: v1
kind: Pod
metadata:
  name: ${WORKER_POD_NAME}
  labels:
    retool-run: ${RUN_NAME}
    retool-role: worker
    retool-launch-id: ${LAUNCH_ID}
  annotations:
    gke-gcsfuse/volumes: "true"
spec:
  restartPolicy: Never
  serviceAccountName: ${K8S_SA}
  resourceClaims:
  - name: compute-domain-channel
    resourceClaimTemplateName: ${RUN_NAME}-cd-channel
  nodeSelector:
    cloud.google.com/gke-nodepool: ${NODEPOOL}
  affinity:
    podAntiAffinity:
      requiredDuringSchedulingIgnoredDuringExecution:
      - labelSelector:
          matchExpressions:
          - key: retool-run
            operator: In
            values:
            - ${RUN_NAME}
        topologyKey: kubernetes.io/hostname
  tolerations:
  - key: nvidia.com/gpu
    operator: Exists
    effect: NoSchedule
  - key: kubernetes.io/arch
    operator: Equal
    value: arm64
    effect: NoSchedule
  volumes:
  - name: dshm
    emptyDir:
      medium: Memory
      sizeLimit: 350Gi
  - name: data
    csi:
      driver: gcsfuse.csi.storage.gke.io
      volumeAttributes:
        bucketName: ${GCS_BUCKET}
        mountOptions: "implicit-dirs,file-mode=0777,dir-mode=0777"
  containers:
  - name: ray-worker
    image: ${IMAGE}
    workingDir: /workspace/verl-pin
    securityContext:
      privileged: true
    env:
    - name: RUN_NAME
      value: "${RUN_NAME}"
    - name: LAUNCH_ID
      value: "${LAUNCH_ID}"
    - name: RUN_DIR
      value: "${RUN_DIR}"
    - name: DATA_ROOT
      value: "${INPUTS_DIR}"
    - name: SANDBOX_FUSION_HOST
      value: "${SANDBOX_FUSION_HOST}"
    - name: SANDBOX_FUSION_PORT
      value: "${SANDBOX_FUSION_PORT}"
    - name: SANDBOX_FUSION_URL
      value: "${SANDBOX_FUSION_URL}"
    - name: TENSORBOARD_DIR
      value: "${TENSORBOARD_DIR}"
    - name: EXPECT_CHECKPOINT_SHA256
      value: "${EXPECT_CHECKPOINT_SHA256}"
    - name: EXPECT_GIT_SHA
      value: "${EXPECT_GIT_SHA}"
    - name: GLOO_SOCKET_IFNAME
      value: "eth0"
    - name: NCCL_SOCKET_IFNAME
      value: "eth0"
    - name: NCCL_SOCKET_FAMILY
      value: "AF_INET"
${WANDB_EXTRA_ENV}${JOBENV_YAML}    resources:
      claims:
      - name: compute-domain-channel
      requests:
        cpu: "64"
        memory: 500Gi
        nvidia.com/gpu: "${GPUS_PER_NODE}"
      limits:
        cpu: "120"
        memory: 750Gi
        nvidia.com/gpu: "${GPUS_PER_NODE}"
    volumeMounts:
    - name: dshm
      mountPath: /dev/shm
    - name: data
      mountPath: /data
    command: ["/bin/bash", "-lc"]
    args:
    - |
      set -xeuo pipefail
      HOST_TAG="\$(hostname)"
      python3 -c 'import socket; from pathlib import Path; [(Path(p).write_text("1\n", encoding="utf-8") if Path(p).exists() else None) for p in ("/proc/sys/net/ipv6/conf/eth0/disable_ipv6", "/proc/sys/net/ipv6/conf/default/disable_ipv6")]; gai = Path("/etc/gai.conf"); txt = gai.read_text(encoding="utf-8") if gai.exists() else ""; gai.write_text(txt + "\nprecedence ::ffff:0:0/96  100\n", encoding="utf-8") if "precedence ::ffff:0:0/96  100" not in txt else None; s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM); s.connect(("10.255.255.255", 1)); ipv4 = s.getsockname()[0]; s.close(); host = socket.gethostname(); hp = Path("/etc/hosts"); lines = [ln for ln in hp.read_text(encoding="utf-8").splitlines() if not (":" in (ln.split("#", 1)[0].split()[0] if ln.split("#", 1)[0].split() else "") and host in ln.split())]; lines = lines + ([f"{ipv4}\t{host}"] if not any(ln.split()[:2] == [ipv4, host] for ln in lines if ln.split()) else []); hp.write_text("\n".join(lines) + "\n", encoding="utf-8"); print(f"[ipv4-setup] host={host} ipv4={ipv4} addrinfo={socket.getaddrinfo(host, None, 0, socket.SOCK_STREAM)[0][4]}", flush=True)' || true
      mkdir -p "${RUN_DIR}/env"
      rm -f "${RUN_DIR}/exit_code.txt"
      if [[ -n "\${EXPECT_GIT_SHA:-}" && "\$(git rev-parse HEAD)" != "\${EXPECT_GIT_SHA}" ]]; then
        git fetch --depth=1 https://github.com/syangx39/verl-upstream.git "\${EXPECT_GIT_SHA}"
        git checkout -f "\${EXPECT_GIT_SHA}"
      fi
      nvidia-smi -L > "${RUN_DIR}/env/\${HOST_TAG}-nvidia-smi-L.txt"
      ACTUAL_GPUS="\$(grep -c '^GPU ' "${RUN_DIR}/env/\${HOST_TAG}-nvidia-smi-L.txt" || true)"
      if [[ "\${ACTUAL_GPUS}" -ne ${GPUS_PER_NODE} ]]; then
        echo "ERROR: expected GPUS_PER_NODE=${GPUS_PER_NODE} visible GPUs on \${HOST_TAG}, but nvidia-smi reported \${ACTUAL_GPUS}" >&2
        exit 1
      fi
      nvidia-smi topo -m > "${RUN_DIR}/env/\${HOST_TAG}-topo.txt" || true
      python3 examples/tpu/retool/gb200/verify_venv_lock.py \
        /workspace/verl-pin \
        --check-all-locked \
        --expect-git-sha "\${EXPECT_GIT_SHA:-\$(git -C /workspace/verl-pin rev-parse HEAD)}" \
        --expect-lock-sha256 df508db0b957fa181c229fb75885cd3863cb651fd5233d0fa5af95e990bda0be \
        --require-async \
        --require-torchtitan \
        --require-rl-moe-helper \
        --require-gpu \
        --require-gb200 \
        --output-json "${RUN_DIR}/env/${LAUNCH_ID}-\${HOST_TAG}-precheck.json" \
        > "${RUN_DIR}/env/${LAUNCH_ID}-\${HOST_TAG}-precheck.log"
      cp -f "${RUN_DIR}/env/${LAUNCH_ID}-\${HOST_TAG}-precheck.json" "${RUN_DIR}/env/\${HOST_TAG}-precheck.json"
      cp -f "${RUN_DIR}/env/${LAUNCH_ID}-\${HOST_TAG}-precheck.log" "${RUN_DIR}/env/\${HOST_TAG}-precheck.log"

      MODEL_SRC="\${MODEL_SRC_PATH:-/data/runs/p2-retool-sft-01/checkpoint/global_step_372/huggingface}"
      MODEL_DST="\${MODEL_PATH:-/dev/shm/models/multiturn-sft-qwen3-30b-a3b-tpu7x}"
      G0_EXPECT_ARGS=()
      if [[ -n "\${EXPECT_CHECKPOINT_SHA256:-}" ]]; then
        G0_EXPECT_ARGS=(--expect-composite-sha256 "\${EXPECT_CHECKPOINT_SHA256}")
      fi
      python3 examples/tpu/retool/gb200/verify_g0_gb200.py \
        --stage-from "\${MODEL_SRC}" \
        --model-dir "\${MODEL_DST}" \
        --ref-dir "\${MODEL_SRC}" \
        "\${G0_EXPECT_ARGS[@]}" \
        --output-json "${RUN_DIR}/env/${LAUNCH_ID}-\${HOST_TAG}-g0-verify.json"
      cp -f "${RUN_DIR}/env/${LAUNCH_ID}-\${HOST_TAG}-g0-verify.json" "${RUN_DIR}/env/\${HOST_TAG}-g0-verify.json"

      CACHE_DIR="/data/\$(dirname "${RUN_ROOT}")/cache"
      mkdir -p /root/.cache
      if [[ -f "\${CACHE_DIR}/flashinfer_cache_sm100a.tar.gz" ]]; then
        tar -xzf "\${CACHE_DIR}/flashinfer_cache_sm100a.tar.gz" -C /root/.cache || true
        find /root/.cache/flashinfer -name "*.o" -exec touch -c {} + 2>/dev/null || true
        sleep 1
        find /root/.cache/flashinfer -name "*.so" -exec touch -c {} + 2>/dev/null || true
      fi
      if [[ -f "\${CACHE_DIR}/vllm_cache_sm100a.tar.gz" ]]; then
        tar -xzf "\${CACHE_DIR}/vllm_cache_sm100a.tar.gz" -C /root/.cache || true
      fi

      SANDBOX_EFFECTIVE_HOST="\$(python3 -c 'import sys, urllib.parse; u = urllib.parse.urlparse(sys.argv[1]); print(u.hostname or sys.argv[2])' "\${SANDBOX_FUSION_URL}" "\${SANDBOX_FUSION_HOST}")"
      if [[ "\${SANDBOX_EFFECTIVE_HOST}" == "127.0.0.1" || "\${SANDBOX_EFFECTIVE_HOST}" == "localhost" || "\${SANDBOX_EFFECTIVE_HOST}" == "::1" ]]; then
        if [[ "\${SANDBOX_DISABLE_AUTO_START:-0}" == "1" ]]; then
          if ! python3 examples/tpu/retool/sandbox_fusion_server.py --check --url "\${SANDBOX_FUSION_URL}" >/dev/null 2>&1; then
            echo "ERROR: ${WORKER_LABEL} loopback sandbox_fusion_server is not running at \${SANDBOX_FUSION_URL} and SANDBOX_DISABLE_AUTO_START=1" >&2
            exit 1
          fi
        else
          nohup python3 examples/tpu/retool/sandbox_fusion_server.py \
            --host "\${SANDBOX_EFFECTIVE_HOST}" \
            --port "\${SANDBOX_FUSION_PORT}" \
            --workers 4 > "/tmp/sandbox_fusion_server.log" 2>&1 &
          SANDBOX_READY=0
          for _ in \$(seq 1 20); do
            if python3 examples/tpu/retool/sandbox_fusion_server.py --check --url "\${SANDBOX_FUSION_URL}" >/dev/null 2>&1; then
              SANDBOX_READY=1
              break
            fi
            sleep 0.5
          done
          if [[ "\${SANDBOX_READY}" != "1" ]]; then
            echo "ERROR: ${WORKER_LABEL} local sandbox_fusion_server failed healthcheck on \${SANDBOX_FUSION_URL}" >&2
            exit 1
          fi
        fi
      else
        export SANDBOX_DISABLE_AUTO_START=1
        if ! python3 examples/tpu/retool/sandbox_fusion_server.py --check --url "\${SANDBOX_FUSION_URL}" >/dev/null 2>&1; then
          echo "ERROR: ${WORKER_LABEL} external SandboxFusion endpoint \${SANDBOX_FUSION_URL} is unreachable (local server auto-start disabled)" >&2
          exit 1
        fi
      fi

      nvidia-smi --query-gpu=timestamp,index,name,memory.used,memory.reserved,memory.total,utilization.gpu,power.draw \
        --format=csv -l 2 > "${RUN_DIR}/env/\${HOST_TAG}-gpu-mem.csv" 2>&1 &

      HEAD_DNS_READY=0
      for _ in \$(seq 1 120); do
        if getent hosts "${RUN_NAME}-head" >/dev/null 2>&1; then
          HEAD_DNS_READY=1
          break
        fi
        sleep 1
      done
      if [[ "\${HEAD_DNS_READY}" != "1" ]]; then
        echo "ERROR: ${WORKER_LABEL} timed out waiting for DNS resolution of ${RUN_NAME}-head" >&2
        exit 1
      fi

      RAY_JOINED=0
      for _ in \$(seq 1 120); do
        if ray start --address="${RUN_NAME}-head:6379" --num-gpus=${GPUS_PER_NODE}; then
          RAY_JOINED=1
          break
        fi
        sleep 2
      done
      if [[ "\${RAY_JOINED}" != "1" ]]; then
        echo "ERROR: ${WORKER_LABEL} timed out trying to join Ray head at ${RUN_NAME}-head:6379" >&2
        exit 1
      fi
      while [[ ! -f "${RUN_DIR}/exit_code.txt" ]]; do
        sleep 3
      done
      ray stop || true
EOF
done

create_k8s_object_and_record_uid CREATED_HEAD_POD_UID ATTEMPTED_HEAD_POD <<EOF
apiVersion: v1
kind: Pod
metadata:
  name: ${RUN_NAME}-head
  labels:
    retool-run: ${RUN_NAME}
    retool-role: head
    retool-launch-id: ${LAUNCH_ID}
  annotations:
    gke-gcsfuse/volumes: "true"
spec:
  restartPolicy: Never
  serviceAccountName: ${K8S_SA}
  resourceClaims:
  - name: compute-domain-channel
    resourceClaimTemplateName: ${RUN_NAME}-cd-channel
  nodeSelector:
    cloud.google.com/gke-nodepool: ${NODEPOOL}
  affinity:
    podAntiAffinity:
      requiredDuringSchedulingIgnoredDuringExecution:
      - labelSelector:
          matchExpressions:
          - key: retool-run
            operator: In
            values:
            - ${RUN_NAME}
        topologyKey: kubernetes.io/hostname
  tolerations:
  - key: nvidia.com/gpu
    operator: Exists
    effect: NoSchedule
  - key: kubernetes.io/arch
    operator: Equal
    value: arm64
    effect: NoSchedule
  volumes:
  - name: dshm
    emptyDir:
      medium: Memory
      sizeLimit: 350Gi
  - name: data
    csi:
      driver: gcsfuse.csi.storage.gke.io
      volumeAttributes:
        bucketName: ${GCS_BUCKET}
        mountOptions: "implicit-dirs,file-mode=0777,dir-mode=0777"
  containers:
  - name: ray-head
    image: ${IMAGE}
    workingDir: /workspace/verl-pin
    securityContext:
      privileged: true
    env:
    - name: RUN_NAME
      value: "${RUN_NAME}"
    - name: LAUNCH_ID
      value: "${LAUNCH_ID}"
    - name: RUN_DIR
      value: "${RUN_DIR}"
    - name: DATA_ROOT
      value: "${INPUTS_DIR}"
    - name: SANDBOX_FUSION_HOST
      value: "${SANDBOX_FUSION_HOST}"
    - name: SANDBOX_FUSION_PORT
      value: "${SANDBOX_FUSION_PORT}"
    - name: SANDBOX_FUSION_URL
      value: "${SANDBOX_FUSION_URL}"
    - name: TENSORBOARD_DIR
      value: "${TENSORBOARD_DIR}"
    - name: EXPECT_CHECKPOINT_SHA256
      value: "${EXPECT_CHECKPOINT_SHA256}"
    - name: EXPECT_GIT_SHA
      value: "${EXPECT_GIT_SHA}"
    - name: GLOO_SOCKET_IFNAME
      value: "eth0"
    - name: NCCL_SOCKET_IFNAME
      value: "eth0"
    - name: NCCL_SOCKET_FAMILY
      value: "AF_INET"
${WANDB_EXTRA_ENV}${JOBENV_YAML}    resources:
      claims:
      - name: compute-domain-channel
      requests:
        cpu: "64"
        memory: 500Gi
        nvidia.com/gpu: "${GPUS_PER_NODE}"
      limits:
        cpu: "120"
        memory: 750Gi
        nvidia.com/gpu: "${GPUS_PER_NODE}"
    volumeMounts:
    - name: dshm
      mountPath: /dev/shm
    - name: data
      mountPath: /data
    command: ["/bin/bash", "-lc"]
    args:
    - |
      set -xeuo pipefail
      HOST_TAG="\$(hostname)"
      python3 -c 'import socket; from pathlib import Path; [(Path(p).write_text("1\n", encoding="utf-8") if Path(p).exists() else None) for p in ("/proc/sys/net/ipv6/conf/eth0/disable_ipv6", "/proc/sys/net/ipv6/conf/default/disable_ipv6")]; gai = Path("/etc/gai.conf"); txt = gai.read_text(encoding="utf-8") if gai.exists() else ""; gai.write_text(txt + "\nprecedence ::ffff:0:0/96  100\n", encoding="utf-8") if "precedence ::ffff:0:0/96  100" not in txt else None; s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM); s.connect(("10.255.255.255", 1)); ipv4 = s.getsockname()[0]; s.close(); host = socket.gethostname(); hp = Path("/etc/hosts"); lines = [ln for ln in hp.read_text(encoding="utf-8").splitlines() if not (":" in (ln.split("#", 1)[0].split()[0] if ln.split("#", 1)[0].split() else "") and host in ln.split())]; lines = lines + ([f"{ipv4}\t{host}"] if not any(ln.split()[:2] == [ipv4, host] for ln in lines if ln.split()) else []); hp.write_text("\n".join(lines) + "\n", encoding="utf-8"); print(f"[ipv4-setup] host={host} ipv4={ipv4} addrinfo={socket.getaddrinfo(host, None, 0, socket.SOCK_STREAM)[0][4]}", flush=True)' || true
      mkdir -p "${RUN_DIR}/env" "${RUN_DIR}/rollout" "${RUN_DIR}/validation" "${RUN_DIR}/rollout_audit" "${TENSORBOARD_DIR}"
      rm -f "${RUN_DIR}/exit_code.txt"
      if [[ -n "\${EXPECT_GIT_SHA:-}" && "\$(git rev-parse HEAD)" != "\${EXPECT_GIT_SHA}" ]]; then
        git fetch --depth=1 https://github.com/syangx39/verl-upstream.git "\${EXPECT_GIT_SHA}"
        git checkout -f "\${EXPECT_GIT_SHA}"
      fi
      nvidia-smi -L > "${RUN_DIR}/env/\${HOST_TAG}-nvidia-smi-L.txt"
      ACTUAL_GPUS="\$(grep -c '^GPU ' "${RUN_DIR}/env/\${HOST_TAG}-nvidia-smi-L.txt" || true)"
      if [[ "\${ACTUAL_GPUS}" -ne ${GPUS_PER_NODE} ]]; then
        echo "ERROR: expected GPUS_PER_NODE=${GPUS_PER_NODE} visible GPUs on \${HOST_TAG}, but nvidia-smi reported \${ACTUAL_GPUS}" >&2
        exit 1
      fi
      nvidia-smi topo -m > "${RUN_DIR}/env/\${HOST_TAG}-topo.txt" || true
      python3 examples/tpu/retool/gb200/verify_venv_lock.py \
        /workspace/verl-pin \
        --check-all-locked \
        --expect-git-sha "\${EXPECT_GIT_SHA:-\$(git -C /workspace/verl-pin rev-parse HEAD)}" \
        --expect-lock-sha256 df508db0b957fa181c229fb75885cd3863cb651fd5233d0fa5af95e990bda0be \
        --require-async \
        --require-torchtitan \
        --require-rl-moe-helper \
        --require-gpu \
        --require-gb200 \
        --output-json "${RUN_DIR}/env/${LAUNCH_ID}-\${HOST_TAG}-precheck.json" \
        > "${RUN_DIR}/env/${LAUNCH_ID}-\${HOST_TAG}-precheck.log"
      cp -f "${RUN_DIR}/env/${LAUNCH_ID}-\${HOST_TAG}-precheck.json" "${RUN_DIR}/env/\${HOST_TAG}-precheck.json"
      cp -f "${RUN_DIR}/env/${LAUNCH_ID}-\${HOST_TAG}-precheck.log" "${RUN_DIR}/env/\${HOST_TAG}-precheck.log"

      ray start --head --port=6379 --dashboard-port=8265 --num-gpus=${GPUS_PER_NODE}

      MODEL_SRC="\${MODEL_SRC_PATH:-/data/runs/p2-retool-sft-01/checkpoint/global_step_372/huggingface}"
      MODEL_DST="\${MODEL_PATH:-/dev/shm/models/multiturn-sft-qwen3-30b-a3b-tpu7x}"
      G0_EXPECT_ARGS=()
      if [[ -n "\${EXPECT_CHECKPOINT_SHA256:-}" ]]; then
        G0_EXPECT_ARGS=(--expect-composite-sha256 "\${EXPECT_CHECKPOINT_SHA256}")
      fi
      python3 examples/tpu/retool/gb200/verify_g0_gb200.py \
        --stage-from "\${MODEL_SRC}" \
        --model-dir "\${MODEL_DST}" \
        --ref-dir "\${MODEL_SRC}" \
        "\${G0_EXPECT_ARGS[@]}" \
        --output-json "${RUN_DIR}/env/${LAUNCH_ID}-\${HOST_TAG}-g0-verify.json"
      cp -f "${RUN_DIR}/env/${LAUNCH_ID}-\${HOST_TAG}-g0-verify.json" "${RUN_DIR}/env/\${HOST_TAG}-g0-verify.json"

      CACHE_DIR="/data/\$(dirname "${RUN_ROOT}")/cache"
      mkdir -p /root/.cache
      if [[ -f "\${CACHE_DIR}/flashinfer_cache_sm100a.tar.gz" ]]; then
        tar -xzf "\${CACHE_DIR}/flashinfer_cache_sm100a.tar.gz" -C /root/.cache || true
        find /root/.cache/flashinfer -name "*.o" -exec touch -c {} + 2>/dev/null || true
        sleep 1
        find /root/.cache/flashinfer -name "*.so" -exec touch -c {} + 2>/dev/null || true
      fi
      if [[ -f "\${CACHE_DIR}/vllm_cache_sm100a.tar.gz" ]]; then
        tar -xzf "\${CACHE_DIR}/vllm_cache_sm100a.tar.gz" -C /root/.cache || true
      fi

      # Wait for ${NUM_NODES} nodes (${TOTAL_GPUS} GPUs) and all nodes' precheck + G0 manifest files matching this LAUNCH_ID; fail explicitly on timeout
      CLUSTER_G0_READY=0
      for _ in \$(seq 1 400); do
        if ray status 2>/dev/null | grep -q "${TOTAL_GPUS}.0 GPU" && [[ \$(ls -1 "${RUN_DIR}/env/${LAUNCH_ID}-"*-g0-verify.json 2>/dev/null | wc -l) -ge ${NUM_NODES} ]] && [[ \$(ls -1 "${RUN_DIR}/env/${LAUNCH_ID}-"*-precheck.json 2>/dev/null | wc -l) -ge ${NUM_NODES} ]]; then
          python3 -c 'import glob, json, sys; env_dir, lid, expected, expect_sha = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4]; g0_files = sorted(glob.glob(f"{env_dir}/{lid}-*-g0-verify.json")); pre_files = sorted(glob.glob(f"{env_dir}/{lid}-*-precheck.json")); digests = {f: json.load(open(f))["g0_checkpoint_manifest"]["composite_sha256"] for f in g0_files}; pre_records = {f: json.load(open(f)) for f in pre_files}; shas = {f: rec.get("git_sha") for f, rec in pre_records.items()}; statuses = {f: rec.get("status") for f, rec in pre_records.items()}; print("[g0-cross-node] digests:", digests, "git_shas:", shas, flush=True); assert len(g0_files) >= expected and len(set(digests.values())) == 1, f"Cross-node manifest mismatch (expected >= {expected}): {digests}"; assert len(pre_files) >= expected and set(statuses.values()) == {"PASS"} and None not in shas.values() and len(set(shas.values())) == 1, f"Cross-node precheck/git_sha mismatch (expected >= {expected}): statuses={statuses} shas={shas}"; assert not expect_sha or next(iter(shas.values())) == expect_sha, f"Cross-node git_sha {shas} != EXPECT_GIT_SHA {expect_sha}"' "${RUN_DIR}/env" "${LAUNCH_ID}" "${NUM_NODES}" "\${EXPECT_GIT_SHA:-}"
          CLUSTER_G0_READY=1
          break
        fi
        sleep 3
      done
      if [[ "\${CLUSTER_G0_READY}" != "1" ]]; then
        echo "ERROR: timed out waiting for ${NUM_NODES} nodes (${TOTAL_GPUS}.0 GPU) and all nodes' G0 manifest files (launch_id=${LAUNCH_ID}) in ${RUN_DIR}/env" >&2
        echo "1" > "${RUN_DIR}/exit_code.txt"
        sync
        exit 1
      fi

      nvidia-smi --query-gpu=timestamp,index,name,memory.used,memory.reserved,memory.total,utilization.gpu,power.draw \
        --format=csv -l 2 > "${RUN_DIR}/env/\${HOST_TAG}-gpu-mem.csv" 2>&1 &

      set +e
      STAGE_MODEL=0 RUN_G0_CHECK=0 SANDBOX_CHECK_JSON="${RUN_DIR}/env/\${HOST_TAG}-sandbox-4case.json" \
        bash ${JOB_SCRIPT}${EXTRA_ARGS_STR} 2>&1 | tee "${RUN_DIR}/head.log"
      rc=\${PIPESTATUS[0]}
      set -e
      echo "\${rc}" > "${RUN_DIR}/exit_code.txt"
      sync
      exit "\${rc}"
EOF

echo "[gb200-launcher $(date -u +%H:%M:%S)] waiting for ${NUM_NODES} pods (${RUN_NAME}-head uid=${CREATED_HEAD_POD_UID} + ${NUM_WORKERS} worker(s): ${WORKER_POD_NAMES[*]}) to be Ready..."
kubectl --context="${KUBE_CONTEXT}" -n "${K8S_NAMESPACE}" wait --for=condition=Ready pod "${RUN_NAME}-head" "${WORKER_POD_NAMES[@]}" --timeout=600s
kubectl --context="${KUBE_CONTEXT}" -n "${K8S_NAMESPACE}" get pods -l "retool-run=${RUN_NAME}" -o wide

kubectl --context="${KUBE_CONTEXT}" -n "${K8S_NAMESPACE}" logs -f "${RUN_NAME}-head"
rc="$(kubectl --context="${KUBE_CONTEXT}" -n "${K8S_NAMESPACE}" get pod "${RUN_NAME}-head" -o jsonpath='{.status.containerStatuses[0].state.terminated.exitCode}' 2>/dev/null || echo 1)"
exit "${rc:-1}"
