#!/usr/bin/env bash
# Two-node (2 x 4-GPU NVIDIA GB200) Ray cluster launcher on GKE with GCSFuse /data and /dev/shm model staging.
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
#   RUN_NAME       Unique run identifier (e.g., g1-retool-rl-smoke-01)
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

SANDBOX_FUSION_HOST="${SANDBOX_FUSION_HOST:-${JOBENV_SANDBOX_FUSION_HOST:-127.0.0.1}}"
SANDBOX_FUSION_PORT="${SANDBOX_FUSION_PORT:-${JOBENV_SANDBOX_FUSION_PORT:-18080}}"
SANDBOX_FUSION_URL="${SANDBOX_FUSION_URL:-${JOBENV_SANDBOX_FUSION_URL:-http://${SANDBOX_FUSION_HOST}:${SANDBOX_FUSION_PORT}/run_code}}"

RUN_DIR="/data/${RUN_ROOT}/${RUN_NAME}"
INPUTS_DIR="/data/${INPUTS_ROOT}"

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
        if [[ "${env_key}" == "SANDBOX_FUSION_HOST" || "${env_key}" == "SANDBOX_FUSION_PORT" || "${env_key}" == "SANDBOX_FUSION_URL" ]]; then
            continue
        fi
        escaped_val="${val//\"/\\\"}"
        JOBENV_YAML+="    - name: ${env_key}"$'\n'"      value: \"${escaped_val}\""$'\n'
    fi
done < <(env | sort)

CREATED_CD_UID=""
CREATED_SVC_UID=""
CREATED_WORKER_POD_UID=""
CREATED_HEAD_POD_UID=""

atomic_delete_by_uid() {
    local proxy_addr="$1"
    local api_path="$2"
    local expected_uid="$3"
    if [[ -z "${expected_uid}" ]]; then
        return 0
    fi
    local payload
    payload="$(printf '{"apiVersion":"meta.k8s.io/v1","kind":"DeleteOptions","propagationPolicy":"Background","preconditions":{"uid":"%s"}}' "${expected_uid}")"
    curl -sS -X DELETE \
        -H "Content-Type: application/json" \
        -d "${payload}" \
        "http://${proxy_addr}${api_path}" >/dev/null 2>&1 || true
}

cleanup() {
    if [[ "${KEEP_PODS}" != "1" ]]; then
        if [[ -n "${CREATED_HEAD_POD_UID}${CREATED_WORKER_POD_UID}${CREATED_SVC_UID}${CREATED_CD_UID}" ]]; then
            echo "[gb200-launcher $(date -u +%H:%M:%S)] atomically cleaning up resources by DeleteOptions.preconditions.uid for ${RUN_NAME}"
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
                atomic_delete_by_uid "${proxy_addr}" "/api/v1/namespaces/${K8S_NAMESPACE}/pods/${RUN_NAME}-head" "${CREATED_HEAD_POD_UID}"
                atomic_delete_by_uid "${proxy_addr}" "/api/v1/namespaces/${K8S_NAMESPACE}/pods/${RUN_NAME}-worker-0" "${CREATED_WORKER_POD_UID}"
                atomic_delete_by_uid "${proxy_addr}" "/api/v1/namespaces/${K8S_NAMESPACE}/services/${RUN_NAME}-head" "${CREATED_SVC_UID}"
                atomic_delete_by_uid "${proxy_addr}" "/apis/resource.nvidia.com/v1beta1/namespaces/${K8S_NAMESPACE}/computedomains/${RUN_NAME}-cd" "${CREATED_CD_UID}"
            else
                echo "WARNING: kubectl proxy failed to start during cleanup; skipping unverified delete to avoid clobbering foreign resources." >&2
            fi
            kill "${proxy_pid}" 2>/dev/null || true
            wait "${proxy_pid}" 2>/dev/null || true
            rm -f "${proxy_log}"
        fi
    fi
}
trap cleanup EXIT

# 1. Fail fast if any target K8s resource name is already in use so we never clobber an active run.
for res_spec in \
    "pod/${RUN_NAME}-head" \
    "pod/${RUN_NAME}-worker-0" \
    "svc/${RUN_NAME}-head" \
    "computedomain.resource.nvidia.com/${RUN_NAME}-cd"; do
    if kubectl --context="${KUBE_CONTEXT}" -n "${K8S_NAMESPACE}" get "${res_spec}" >/dev/null 2>&1; then
        echo "ERROR: ${res_spec} already exists in namespace ${K8S_NAMESPACE} for RUN_NAME=${RUN_NAME}; refusing to delete or overwrite existing workload." >&2
        exit 1
    fi
done

# 2. Refuse to reuse an existing GCS run directory or stale G0 precheck files unless ALLOW_OVERWRITE_RUN_DIR=1.
if gsutil -q stat "gs://${GCS_BUCKET}/${RUN_ROOT}/${RUN_NAME}/**" 2>/dev/null; then
    if [[ "${ALLOW_OVERWRITE_RUN_DIR}" != "1" ]]; then
        echo "ERROR: gs://${GCS_BUCKET}/${RUN_ROOT}/${RUN_NAME}/ already contains artifacts from a previous run; refusing to reuse stale RUN_DIR or G0 precheck files. Choose a fresh RUN_NAME or set ALLOW_OVERWRITE_RUN_DIR=1." >&2
        exit 1
    fi
    gsutil -m rm -f \
        "gs://${GCS_BUCKET}/${RUN_ROOT}/${RUN_NAME}/exit_code.txt" \
        "gs://${GCS_BUCKET}/${RUN_ROOT}/${RUN_NAME}/env/*-g0-verify.json" >/dev/null 2>&1 || true
fi

EXTRA_ARGS_STR=""
for a in "${EXTRA_ARGS[@]}"; do
    EXTRA_ARGS_STR+=" $(printf '%q' "${a}")"
done

create_k8s_object_and_record_uid() {
    local var_name="$1"
    local uid
    uid="$(kubectl --context="${KUBE_CONTEXT}" -n "${K8S_NAMESPACE}" create -f - -o jsonpath='{.metadata.uid}')"
    if [[ -z "${uid}" ]]; then
        echo "ERROR: failed to capture metadata.uid from created Kubernetes object (${var_name})" >&2
        exit 1
    fi
    printf -v "${var_name}" '%s' "${uid}"
}

create_k8s_object_and_record_uid CREATED_CD_UID <<EOF
apiVersion: resource.nvidia.com/v1beta1
kind: ComputeDomain
metadata:
  name: ${RUN_NAME}-cd
spec:
  numNodes: 2
  channel:
    allocationMode: Single
    resourceClaimTemplate:
      name: ${RUN_NAME}-cd-channel
EOF

create_k8s_object_and_record_uid CREATED_SVC_UID <<EOF
apiVersion: v1
kind: Service
metadata:
  name: ${RUN_NAME}-head
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

create_k8s_object_and_record_uid CREATED_WORKER_POD_UID <<EOF
apiVersion: v1
kind: Pod
metadata:
  name: ${RUN_NAME}-worker-0
  labels:
    retool-run: ${RUN_NAME}
    retool-role: worker
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
${JOBENV_YAML}    resources:
      claims:
      - name: compute-domain-channel
      requests:
        cpu: "64"
        memory: 500Gi
        nvidia.com/gpu: "4"
      limits:
        cpu: "120"
        memory: 750Gi
        nvidia.com/gpu: "4"
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
      mkdir -p "${RUN_DIR}/env"
      rm -f "${RUN_DIR}/exit_code.txt"
      if [[ -n "\${EXPECT_GIT_SHA:-}" && "\$(git rev-parse HEAD)" != "\${EXPECT_GIT_SHA}" ]]; then
        git fetch --depth=1 https://github.com/syangx39/verl-upstream.git "\${EXPECT_GIT_SHA}"
        git checkout -f "\${EXPECT_GIT_SHA}"
      fi
      nvidia-smi -L > "${RUN_DIR}/env/\${HOST_TAG}-nvidia-smi-L.txt"
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
        --require-gb200 > "${RUN_DIR}/env/\${HOST_TAG}-precheck.json"

      MODEL_SRC="\${MODEL_SRC_PATH:-/data/runs/p2-retool-sft-01/checkpoint/global_step_372/huggingface}"
      MODEL_DST="\${MODEL_PATH:-/dev/shm/models/multiturn-sft-qwen3-30b-a3b-tpu7x}"
      python3 examples/tpu/retool/gb200/verify_g0_gb200.py \
        --stage-from "\${MODEL_SRC}" \
        --model-dir "\${MODEL_DST}" \
        --ref-dir "\${MODEL_SRC}" \
        --output-json "${RUN_DIR}/env/${LAUNCH_ID}-\${HOST_TAG}-g0-verify.json"
      cp -f "${RUN_DIR}/env/${LAUNCH_ID}-\${HOST_TAG}-g0-verify.json" "${RUN_DIR}/env/\${HOST_TAG}-g0-verify.json"

      CACHE_DIR="/data/\$(dirname "${RUN_ROOT}")/cache"
      mkdir -p /root/.cache
      if [[ -f "\${CACHE_DIR}/flashinfer_cache_sm100a.tar.gz" ]]; then
        tar -xzf "\${CACHE_DIR}/flashinfer_cache_sm100a.tar.gz" -C /root/.cache || true
      fi
      if [[ -f "\${CACHE_DIR}/vllm_cache_sm100a.tar.gz" ]]; then
        tar -xzf "\${CACHE_DIR}/vllm_cache_sm100a.tar.gz" -C /root/.cache || true
      fi

      SANDBOX_EFFECTIVE_HOST="\$(python3 -c 'import sys, urllib.parse; u = urllib.parse.urlparse(sys.argv[1]); print(u.hostname or sys.argv[2])' "\${SANDBOX_FUSION_URL}" "\${SANDBOX_FUSION_HOST}")"
      if [[ "\${SANDBOX_EFFECTIVE_HOST}" == "127.0.0.1" || "\${SANDBOX_EFFECTIVE_HOST}" == "localhost" || "\${SANDBOX_EFFECTIVE_HOST}" == "::1" ]]; then
        if [[ "\${SANDBOX_DISABLE_AUTO_START:-0}" == "1" ]]; then
          if ! python3 examples/tpu/retool/sandbox_fusion_server.py --check --url "\${SANDBOX_FUSION_URL}" >/dev/null 2>&1; then
            echo "ERROR: worker-0 loopback sandbox_fusion_server is not running at \${SANDBOX_FUSION_URL} and SANDBOX_DISABLE_AUTO_START=1" >&2
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
            echo "ERROR: worker-0 local sandbox_fusion_server failed healthcheck on \${SANDBOX_FUSION_URL}" >&2
            exit 1
          fi
        fi
      else
        export SANDBOX_DISABLE_AUTO_START=1
        if ! python3 examples/tpu/retool/sandbox_fusion_server.py --check --url "\${SANDBOX_FUSION_URL}" >/dev/null 2>&1; then
          echo "ERROR: worker-0 external SandboxFusion endpoint \${SANDBOX_FUSION_URL} is unreachable (local server auto-start disabled)" >&2
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
        echo "ERROR: worker-0 timed out waiting for DNS resolution of ${RUN_NAME}-head" >&2
        exit 1
      fi

      RAY_JOINED=0
      for _ in \$(seq 1 120); do
        if ray start --address="${RUN_NAME}-head:6379" --num-gpus=4; then
          RAY_JOINED=1
          break
        fi
        sleep 2
      done
      if [[ "\${RAY_JOINED}" != "1" ]]; then
        echo "ERROR: worker-0 timed out trying to join Ray head at ${RUN_NAME}-head:6379" >&2
        exit 1
      fi
      while [[ ! -f "${RUN_DIR}/exit_code.txt" ]]; do
        sleep 3
      done
      ray stop || true
EOF

create_k8s_object_and_record_uid CREATED_HEAD_POD_UID <<EOF
apiVersion: v1
kind: Pod
metadata:
  name: ${RUN_NAME}-head
  labels:
    retool-run: ${RUN_NAME}
    retool-role: head
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
${JOBENV_YAML}    resources:
      claims:
      - name: compute-domain-channel
      requests:
        cpu: "64"
        memory: 500Gi
        nvidia.com/gpu: "4"
      limits:
        cpu: "120"
        memory: 750Gi
        nvidia.com/gpu: "4"
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
      mkdir -p "${RUN_DIR}/env" "${RUN_DIR}/rollout" "${RUN_DIR}/validation" "${RUN_DIR}/rollout_audit"
      rm -f "${RUN_DIR}/exit_code.txt"
      if [[ -n "\${EXPECT_GIT_SHA:-}" && "\$(git rev-parse HEAD)" != "\${EXPECT_GIT_SHA}" ]]; then
        git fetch --depth=1 https://github.com/syangx39/verl-upstream.git "\${EXPECT_GIT_SHA}"
        git checkout -f "\${EXPECT_GIT_SHA}"
      fi
      nvidia-smi -L > "${RUN_DIR}/env/\${HOST_TAG}-nvidia-smi-L.txt"
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
        --require-gb200 > "${RUN_DIR}/env/\${HOST_TAG}-precheck.json"

      ray start --head --port=6379 --dashboard-port=8265 --num-gpus=4

      MODEL_SRC="\${MODEL_SRC_PATH:-/data/runs/p2-retool-sft-01/checkpoint/global_step_372/huggingface}"
      MODEL_DST="\${MODEL_PATH:-/dev/shm/models/multiturn-sft-qwen3-30b-a3b-tpu7x}"
      python3 examples/tpu/retool/gb200/verify_g0_gb200.py \
        --stage-from "\${MODEL_SRC}" \
        --model-dir "\${MODEL_DST}" \
        --ref-dir "\${MODEL_SRC}" \
        --output-json "${RUN_DIR}/env/${LAUNCH_ID}-\${HOST_TAG}-g0-verify.json"
      cp -f "${RUN_DIR}/env/${LAUNCH_ID}-\${HOST_TAG}-g0-verify.json" "${RUN_DIR}/env/\${HOST_TAG}-g0-verify.json"

      CACHE_DIR="/data/\$(dirname "${RUN_ROOT}")/cache"
      mkdir -p /root/.cache
      if [[ -f "\${CACHE_DIR}/flashinfer_cache_sm100a.tar.gz" ]]; then
        tar -xzf "\${CACHE_DIR}/flashinfer_cache_sm100a.tar.gz" -C /root/.cache || true
      fi
      if [[ -f "\${CACHE_DIR}/vllm_cache_sm100a.tar.gz" ]]; then
        tar -xzf "\${CACHE_DIR}/vllm_cache_sm100a.tar.gz" -C /root/.cache || true
      fi

      # Wait for 2 nodes (8 GPUs) and both nodes' G0 manifest files matching this LAUNCH_ID; fail explicitly on timeout
      CLUSTER_G0_READY=0
      for _ in \$(seq 1 200); do
        if ray status 2>/dev/null | grep -q "8.0 GPU" && [[ \$(ls -1 "${RUN_DIR}/env/${LAUNCH_ID}-"*-g0-verify.json 2>/dev/null | wc -l) -ge 2 ]]; then
          python3 -c 'import glob, json, sys; files = sorted(glob.glob(sys.argv[1] + "/" + sys.argv[2] + "-*-g0-verify.json")); digests = {f: json.load(open(f))["g0_checkpoint_manifest"]["composite_sha256"] for f in files}; print("[g0-cross-node] digests:", digests, flush=True); assert len(files) >= 2 and len(set(digests.values())) == 1, f"Cross-node manifest mismatch: {digests}"' "${RUN_DIR}/env" "${LAUNCH_ID}"
          CLUSTER_G0_READY=1
          break
        fi
        sleep 3
      done
      if [[ "\${CLUSTER_G0_READY}" != "1" ]]; then
        echo "ERROR: timed out waiting for 2 nodes (8.0 GPU) and both nodes' G0 manifest files (launch_id=${LAUNCH_ID}) in ${RUN_DIR}/env" >&2
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

echo "[gb200-launcher $(date -u +%H:%M:%S)] waiting for pods ${RUN_NAME}-head (uid=${CREATED_HEAD_POD_UID}) and ${RUN_NAME}-worker-0 (uid=${CREATED_WORKER_POD_UID}) to be Ready..."
kubectl --context="${KUBE_CONTEXT}" -n "${K8S_NAMESPACE}" wait --for=condition=Ready pod "${RUN_NAME}-head" "${RUN_NAME}-worker-0" --timeout=600s
kubectl --context="${KUBE_CONTEXT}" -n "${K8S_NAMESPACE}" get pods -l "retool-run=${RUN_NAME}" -o wide

kubectl --context="${KUBE_CONTEXT}" -n "${K8S_NAMESPACE}" logs -f "${RUN_NAME}-head"
rc="$(kubectl --context="${KUBE_CONTEXT}" -n "${K8S_NAMESPACE}" get pod "${RUN_NAME}-head" -o jsonpath='{.status.containerStatuses[0].state.terminated.exitCode}' 2>/dev/null || echo 1)"
exit "${rc:-1}"
