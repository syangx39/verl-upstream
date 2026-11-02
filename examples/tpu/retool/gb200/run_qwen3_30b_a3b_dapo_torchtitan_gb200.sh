#!/usr/bin/env bash
# ReTool Stage 2 (Multi-Turn DAPO RL with Code Interpreter) | Qwen3-30B-A3B (MoE)
# TorchTitan 0.3.0 (FSDP2=4, CP=1, TP=1, EP=1, flex_flash) & vLLM V1 Async Rollout (TP=4, NCCL weight sync)
# Target hardware: 2 x 4-GPU NVIDIA GB200 nodes (8 GPUs total: 4 trainer + 4 rollout)
#
# Usage:
#   Stage G1 (2-step multi-turn RL smoke):
#     TRAIN_BATCH_SIZE=8 PPO_MINI_BATCH_SIZE=8 N_RESP_PER_PROMPT=4 TOTAL_TRAINING_STEPS=2 \
#     VAL_BEFORE_TRAIN=False TEST_FREQ=-1 SAVE_FREQ=-1 \
#     bash examples/tpu/retool/gb200/run_qwen3_30b_a3b_dapo_torchtitan_gb200.sh
#
#   Stage G2 (25-step multi-turn RL pilot with Step 0 & Step 25 AIME 2024/2025 evals):
#     TRAIN_BATCH_SIZE=64 PPO_MINI_BATCH_SIZE=64 N_RESP_PER_PROMPT=8 TOTAL_TRAINING_STEPS=25 \
#     VAL_BEFORE_TRAIN=True TEST_FREQ=25 SAVE_FREQ=-1 \
#     bash examples/tpu/retool/gb200/run_qwen3_30b_a3b_dapo_torchtitan_gb200.sh

set -xeuo pipefail

export VERL_PLATFORM=cuda
export VLLM_USE_V1="${VLLM_USE_V1:-1}"
export RAY_OVERRIDE_JOB_RUNTIME_ENV=1
export PYTHONUNBUFFERED=1
export RAY_memory_monitor_refresh_ms=0
export RAY_memory_usage_threshold=0.99
export VERL_LOGGING_LEVEL="${VERL_LOGGING_LEVEL:-INFO}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RETOOL_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
VERL_ROOT="$(cd "${RETOOL_DIR}/../../.." && pwd)"
export PYTHONPATH=".:${VERL_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

# 1. Start loopback SandboxFusion FastAPI code execution server on 127.0.0.1:18080
SANDBOX_FUSION_HOST="${SANDBOX_FUSION_HOST:-127.0.0.1}"
SANDBOX_FUSION_PORT="${SANDBOX_FUSION_PORT:-18080}"
SANDBOX_FUSION_WORKERS="${SANDBOX_FUSION_WORKERS:-4}"
if ! python3 "${RETOOL_DIR}/sandbox_fusion_server.py" --host "${SANDBOX_FUSION_HOST}" --port "${SANDBOX_FUSION_PORT}" --check >/dev/null 2>&1; then
    nohup python3 "${RETOOL_DIR}/sandbox_fusion_server.py" \
        --host "${SANDBOX_FUSION_HOST}" \
        --port "${SANDBOX_FUSION_PORT}" \
        --workers "${SANDBOX_FUSION_WORKERS}" > /tmp/sandbox_fusion_server.log 2>&1 &
    for _ in $(seq 1 20); do
        if python3 "${RETOOL_DIR}/sandbox_fusion_server.py" --host "${SANDBOX_FUSION_HOST}" --port "${SANDBOX_FUSION_PORT}" --check >/dev/null 2>&1; then
            break
        fi
        sleep 0.5
    done
fi

# 2. Run 4-case SandboxFusion functional verification (arithmetic, stderr, timeout+cleanup, concurrency)
SANDBOX_CHECK_JSON="${SANDBOX_CHECK_JSON:-/tmp/sandbox_4case_check.json}"
python3 - <<'PY' "${SANDBOX_FUSION_HOST}" "${SANDBOX_FUSION_PORT}" "${SANDBOX_CHECK_JSON}"
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import sys
import time
import urllib.request

host, port, out_path = sys.argv[1], int(sys.argv[2]), Path(sys.argv[3])
url = f"http://{host}:{port}/run_code"

def call_run(code: str, run_timeout: float = 5.0):
    t0 = time.time()
    payload = json.dumps({"code": code, "language": "python", "run_timeout": run_timeout}).encode("utf-8")
    req = urllib.request.Request(url, data=payload, headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=run_timeout + 10.0) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    data["elapsed_s"] = round(time.time() - t0, 4)
    return data

# Case 1: Normal arithmetic
c1 = call_run("print(sum(i * i for i in range(1, 101)))")
assert c1["status"] == "Success" and c1["run_result"]["stdout"].strip() == "338350", f"Case 1 failed: {c1}"

# Case 2: Python exception / stderr
c2 = call_run("print(1 / 0)")
assert c2["status"] == "Failed" and "ZeroDivisionError" in c2["run_result"]["stderr"], f"Case 2 failed: {c2}"

# Case 3: Timeout & process cleanup
c3 = call_run("import time\nwhile True:\n    time.sleep(0.1)", run_timeout=1.5)
assert c3["status"] in ("Failed", "TimeLimitExceeded") and c3["run_result"]["status"] == "TimeLimitExceeded", f"Case 3 failed: {c3}"

# Case 4: 8 concurrent requests
with ThreadPoolExecutor(max_workers=8) as ex:
    c4_res = list(ex.map(lambda k: call_run(f"print({k} * {k})"), range(1, 9)))
assert all(r["status"] == "Success" for r in c4_res), f"Case 4 failed: {c4_res}"
assert [r["run_result"]["stdout"].strip() for r in c4_res] == [str(k * k) for k in range(1, 9)], f"Case 4 outputs mismatch: {c4_res}"

report = {
    "status": "PASS",
    "endpoint": url,
    "case1_arithmetic": c1,
    "case2_exception_stderr": c2,
    "case3_timeout_cleanup": c3,
    "case4_concurrent_8": {
        "all_succeeded": True,
        "outputs": [r["run_result"]["stdout"].strip() for r in c4_res],
        "max_elapsed_s": max(r["elapsed_s"] for r in c4_res),
    },
}
out_path.parent.mkdir(parents=True, exist_ok=True)
out_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
print(f"[sandbox-4case] PASS -> {out_path}")
PY

# ================= data/model/tool =================
DATA_ROOT="${DATA_ROOT:-/data/inputs}"
dapo_math_17k="${DAPO_MATH_17K:-${DATA_ROOT}/dataset/DAPO-Math-17k-Processed}"
aime_2024="${AIME_2024:-${DATA_ROOT}/dataset/aime-2024}"
aime_2025="${AIME_2025:-${DATA_ROOT}/dataset/aime-2025}"

MODEL_SRC_PATH="${MODEL_SRC_PATH:-/data/runs/p2-retool-sft-01/checkpoint/global_step_372/huggingface}"
model_path="${MODEL_PATH:-/dev/shm/models/multiturn-sft-qwen3-30b-a3b-tpu7x}"
STAGE_MODEL="${STAGE_MODEL:-1}"

if [[ "${RUN_G0_CHECK:-1}" == "1" ]]; then
    G0_REPORT_JSON="${G0_REPORT_JSON:-/tmp/g0_verify_report.json}"
    STAGE_ARGS=()
    if [[ "${STAGE_MODEL}" == "1" ]]; then
        STAGE_ARGS=(--stage-from "${MODEL_SRC_PATH}")
    fi
    python3 "${SCRIPT_DIR}/verify_g0_gb200.py" \
        "${STAGE_ARGS[@]}" \
        --model-dir "${model_path}" \
        --ref-dir "${MODEL_SRC_PATH}" \
        --output-json "${G0_REPORT_JSON}"
fi

train_files="['${dapo_math_17k}']"
test_files="['${aime_2025}', '${aime_2024}']"

tool_config_path="examples/tpu/retool/sandbox_fusion_tool_config_tpu.yaml"
retool_py_path="pkg://examples.tpu.retool.retool_tpu"

project_name="${PROJECT_NAME:-verl_gb200_retool}"
experiment_name="${EXPERIMENT_NAME:-qwen3-30b-a3b_retool_dapo_gb200}"
default_local_dir="${CKPTS_DIR:-/tmp/checkpoint/${experiment_name}}"

# ================= algorithm =================
adv_estimator=grpo
use_kl_in_reward=False
kl_coef=0.0
use_kl_loss="${USE_KL_LOSS:-False}"
kl_loss_coef=0.0

clip_ratio_low=0.2
clip_ratio_high=0.28

max_turns="${MAX_TURNS:-8}"
max_prompt_length="${MAX_PROMPT_LENGTH:-2048}"
max_response_length="${MAX_RESPONSE_LENGTH:-8192}"
max_model_len=$((max_prompt_length + max_response_length))
actor_lr="${ACTOR_LR:-1.4e-6}"
min_lr_factor="${MIN_LR_FACTOR:-1.0}"
clip_grad="${CLIP_GRAD:-0.088}"

train_batch_size="${TRAIN_BATCH_SIZE:-64}"
ppo_mini_batch_size="${PPO_MINI_BATCH_SIZE:-64}"
n_resp_per_prompt="${N_RESP_PER_PROMPT:-8}"
n_resp_per_prompt_val="${N_RESP_PER_PROMPT_VAL:-30}"

# ================= 2 x 4-GPU GB200 topology (4 trainer + 4 rollout) =================
export NNODES_TRAINER="${NNODES_TRAINER:-1}"
export N_GPUS_TRAINER="${N_GPUS_TRAINER:-4}"
export NNODES_ROLLOUT="${NNODES_ROLLOUT:-1}"
export N_GPUS_ROLLOUT="${N_GPUS_ROLLOUT:-4}"

TOTAL_TRAINER_GPUS=$((NNODES_TRAINER * N_GPUS_TRAINER))
TOTAL_ROLLOUT_GPUS=$((NNODES_ROLLOUT * N_GPUS_ROLLOUT))

ROLLOUT_TP="${ROLLOUT_TP:-${TOTAL_ROLLOUT_GPUS}}"
ROLLOUT_EP="${ROLLOUT_EP:-1}"
ROLLOUT_ENABLE_PREFIX_CACHING="${ROLLOUT_ENABLE_PREFIX_CACHING:-True}"
PARAMETER_SYNC_STEP="${PARAMETER_SYNC_STEP:-$((train_batch_size / ppo_mini_batch_size))}"
NUM_WARMUP_BATCHES="${NUM_WARMUP_BATCHES:-2}"
MAX_OFF_POLICY_THRESHOLD="${MAX_OFF_POLICY_THRESHOLD:-4}"
AGENT_NUM_WORKERS="${AGENT_NUM_WORKERS:-16}"

TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-1}"
TRAINER_EP="${TRAINER_EP:-1}"
CONTEXT_PARALLEL_SIZE="${CONTEXT_PARALLEL_SIZE:-1}"
DEFAULT_DP_SHARD=$((TOTAL_TRAINER_GPUS / (TENSOR_PARALLEL_SIZE * CONTEXT_PARALLEL_SIZE)))
DATA_PARALLEL_SHARD_SIZE="${DATA_PARALLEL_SHARD_SIZE:-${DEFAULT_DP_SHARD}}"

USE_DYNAMIC_BSZ="${USE_DYNAMIC_BSZ:-True}"
ACTOR_MAX_TOKEN_LEN_PER_GPU="${ACTOR_MAX_TOKEN_LEN_PER_GPU:-16384}"
LOG_PROB_MAX_TOKEN_LEN_PER_GPU="${LOG_PROB_MAX_TOKEN_LEN_PER_GPU:-16384}"
torchtitan_max_seq_len="${TORCHTITAN_MAX_SEQ_LEN:-16384}"

ATTN_TYPE="${ATTN_TYPE:-flex}"
AC_MODE="${AC_MODE:-selective}"
SPMD_BACKEND="${SPMD_BACKEND:-partial_dtensor}"
USE_TORCH_COMPILE="${USE_TORCH_COMPILE:-False}"
PAD_TO_LENGTH="${PAD_TO_LENGTH:-True}"
PAD_TO_LENGTH_BUCKET="${PAD_TO_LENGTH_BUCKET:-256}"

CHECKPOINT_BACKEND="${CHECKPOINT_BACKEND:-nccl}"
UPDATE_WEIGHTS_BUCKET_MB="${UPDATE_WEIGHTS_BUCKET_MB:-2048}"
ROLLOUT_GPU_MEMORY_UTILIZATION="${ROLLOUT_GPU_MEMORY_UTILIZATION:-0.75}"
ROLLOUT_MAX_NUM_SEQS="${ROLLOUT_MAX_NUM_SEQS:-256}"
ROLLOUT_MAX_NUM_BATCHED_TOKENS="${ROLLOUT_MAX_NUM_BATCHED_TOKENS:-8192}"
ROLLOUT_ENFORCE_EAGER="${ROLLOUT_ENFORCE_EAGER:-False}"

ROLLOUT_IS="${ROLLOUT_IS:-token}"
ROLLOUT_IS_THRESHOLD="${ROLLOUT_IS_THRESHOLD:-2.0}"
BYPASS_MODE="${BYPASS_MODE:-True}"
LOSS_ARGS=("algorithm.rollout_correction.bypass_mode=${BYPASS_MODE}")
if [[ "${BYPASS_MODE}" == "True" || "${BYPASS_MODE}" == "true" ]]; then
    LOSS_ARGS+=("actor_rollout_ref.actor.policy_loss.loss_mode=bypass_mode")
fi

VAL_BEFORE_TRAIN="${VAL_BEFORE_TRAIN:-True}"
SAVE_FREQ="${SAVE_FREQ:--1}"
MAX_CKPT_TO_KEEP="${MAX_CKPT_TO_KEEP:-2}"
TEST_FREQ="${TEST_FREQ:-25}"
TOTAL_EPOCHS="${TOTAL_EPOCHS:-2}"
TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS:-25}"
TRUNCATION="${TRUNCATION:-left}"
LOGGER_LIST="${LOGGER_LIST:-['console','tensorboard']}"

python3 -m verl.trainer.main_ppo \
    "ray_kwargs.ray_init.runtime_env.py_executable=null" \
    "+ray_kwargs.ray_init.runtime_env.env_vars.PYTHONPATH=${PYTHONPATH}" \
    "+ray_kwargs.ray_init.runtime_env.env_vars.SANDBOX_FUSION_PORT='${SANDBOX_FUSION_PORT}'" \
    "+ray_kwargs.ray_init.runtime_env.env_vars.VLLM_USE_V1='${VLLM_USE_V1}'" \
    trainer.use_v1=True \
    trainer.v1.trainer_mode=separate_async \
    trainer.v1.separate_async.num_warmup_batches="${NUM_WARMUP_BATCHES}" \
    trainer.v1.separate_async.parameter_sync_step="${PARAMETER_SYNC_STEP}" \
    trainer.v1.sampler.max_off_policy_threshold="${MAX_OFF_POLICY_THRESHOLD}" \
    transfer_queue.enable=True \
    model_engine=torchtitan \
    algorithm.adv_estimator="${adv_estimator}" \
    algorithm.use_kl_in_reward="${use_kl_in_reward}" \
    algorithm.kl_ctrl.kl_coef="${kl_coef}" \
    algorithm.rollout_correction.rollout_is="${ROLLOUT_IS}" \
    algorithm.rollout_correction.rollout_is_threshold="${ROLLOUT_IS_THRESHOLD}" \
    "${LOSS_ARGS[@]}" \
    data.train_files="${train_files}" \
    data.val_files="${test_files}" \
    data.return_raw_chat=True \
    data.train_batch_size="${train_batch_size}" \
    data.max_prompt_length="${max_prompt_length}" \
    data.max_response_length="${max_response_length}" \
    +data.max_length="${max_model_len}" \
    +data.max_token_len_per_gpu="${ACTOR_MAX_TOKEN_LEN_PER_GPU}" \
    data.filter_overlong_prompts=False \
    data.truncation="${TRUNCATION}" \
    +data.apply_chat_template_kwargs.enable_thinking=False \
    +data.pad_mode=no_padding \
    data.custom_cls.path="${retool_py_path}" \
    data.custom_cls.name=CustomRLHFDataset \
    custom_reward_function.path="${retool_py_path}" \
    custom_reward_function.name=compute_score \
    actor_rollout_ref.actor.strategy=torchtitan \
    "actor_rollout_ref.actor.checkpoint.save_contents=['hf_model']" \
    actor_rollout_ref.model.path="${model_path}" \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.use_kl_loss="${use_kl_loss}" \
    actor_rollout_ref.actor.kl_loss_coef="${kl_loss_coef}" \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.clip_ratio_low="${clip_ratio_low}" \
    actor_rollout_ref.actor.clip_ratio_high="${clip_ratio_high}" \
    actor_rollout_ref.actor.clip_ratio_c=10.0 \
    actor_rollout_ref.actor.optim.lr="${actor_lr}" \
    actor_rollout_ref.actor.optim.min_lr_factor="${min_lr_factor}" \
    actor_rollout_ref.actor.optim.clip_grad="${clip_grad}" \
    actor_rollout_ref.actor.use_dynamic_bsz="${USE_DYNAMIC_BSZ}" \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz="${USE_DYNAMIC_BSZ}" \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz="${USE_DYNAMIC_BSZ}" \
    actor_rollout_ref.actor.ppo_mini_batch_size="${ppo_mini_batch_size}" \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu="${ACTOR_MAX_TOKEN_LEN_PER_GPU}" \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu="${LOG_PROB_MAX_TOKEN_LEN_PER_GPU}" \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu="${LOG_PROB_MAX_TOKEN_LEN_PER_GPU}" \
    actor_rollout_ref.actor.use_torch_compile="${USE_TORCH_COMPILE}" \
    actor_rollout_ref.actor.torchtitan.use_torch_compile="${USE_TORCH_COMPILE}" \
    actor_rollout_ref.actor.torchtitan.tensor_parallel_size="${TENSOR_PARALLEL_SIZE}" \
    actor_rollout_ref.actor.torchtitan.expert_parallel_size="${TRAINER_EP}" \
    actor_rollout_ref.actor.torchtitan.context_parallel_size="${CONTEXT_PARALLEL_SIZE}" \
    actor_rollout_ref.actor.torchtitan.data_parallel_shard_size="${DATA_PARALLEL_SHARD_SIZE}" \
    actor_rollout_ref.actor.torchtitan.pipeline_parallel_size=1 \
    actor_rollout_ref.actor.torchtitan.attn_type="${ATTN_TYPE}" \
    actor_rollout_ref.actor.torchtitan.activation_checkpoint="${AC_MODE}" \
    actor_rollout_ref.actor.torchtitan.spmd_backend="${SPMD_BACKEND}" \
    actor_rollout_ref.actor.torchtitan.pad_to_length="${PAD_TO_LENGTH}" \
    actor_rollout_ref.actor.torchtitan.pad_to_length_bucket="${PAD_TO_LENGTH_BUCKET}" \
    actor_rollout_ref.actor.torchtitan.max_seq_len="${torchtitan_max_seq_len}" \
    actor_rollout_ref.actor.torchtitan.param_offload=False \
    actor_rollout_ref.actor.torchtitan.optimizer_offload=False \
    actor_rollout_ref.hybrid_engine=False \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.mode=async \
    actor_rollout_ref.rollout.calculate_log_probs=True \
    actor_rollout_ref.rollout.agent.num_workers="${AGENT_NUM_WORKERS}" \
    actor_rollout_ref.rollout.enable_prefix_caching="${ROLLOUT_ENABLE_PREFIX_CACHING}" \
    actor_rollout_ref.rollout.tensor_model_parallel_size="${ROLLOUT_TP}" \
    actor_rollout_ref.rollout.expert_parallel_size="${ROLLOUT_EP}" \
    actor_rollout_ref.rollout.enable_chunked_prefill=True \
    actor_rollout_ref.rollout.max_model_len="${max_model_len}" \
    actor_rollout_ref.rollout.max_num_batched_tokens="${ROLLOUT_MAX_NUM_BATCHED_TOKENS}" \
    actor_rollout_ref.rollout.max_num_seqs="${ROLLOUT_MAX_NUM_SEQS}" \
    actor_rollout_ref.rollout.load_format=safetensors \
    actor_rollout_ref.rollout.dtype=bfloat16 \
    actor_rollout_ref.rollout.enforce_eager="${ROLLOUT_ENFORCE_EAGER}" \
    actor_rollout_ref.rollout.checkpoint_engine.backend="${CHECKPOINT_BACKEND}" \
    actor_rollout_ref.rollout.checkpoint_engine.update_weights_bucket_megabytes="${UPDATE_WEIGHTS_BUCKET_MB}" \
    actor_rollout_ref.rollout.multi_turn.enable=True \
    actor_rollout_ref.rollout.multi_turn.max_user_turns="${max_turns}" \
    actor_rollout_ref.rollout.multi_turn.max_assistant_turns="${max_turns}" \
    actor_rollout_ref.rollout.multi_turn.tool_config_path="${tool_config_path}" \
    actor_rollout_ref.rollout.multi_turn.format=hermes \
    actor_rollout_ref.rollout.gpu_memory_utilization="${ROLLOUT_GPU_MEMORY_UTILIZATION}" \
    actor_rollout_ref.rollout.n="${n_resp_per_prompt}" \
    actor_rollout_ref.rollout.val_kwargs.top_p=0.6 \
    actor_rollout_ref.rollout.val_kwargs.temperature=1.0 \
    actor_rollout_ref.rollout.val_kwargs.n="${n_resp_per_prompt_val}" \
    trainer.logger="${LOGGER_LIST}" \
    trainer.project_name="${project_name}" \
    trainer.experiment_name="${experiment_name}" \
    trainer.n_gpus_per_node="${N_GPUS_TRAINER}" \
    trainer.nnodes="${NNODES_TRAINER}" \
    actor_rollout_ref.rollout.nnodes="${NNODES_ROLLOUT}" \
    actor_rollout_ref.rollout.n_gpus_per_node="${N_GPUS_ROLLOUT}" \
    +rollout.nnodes="${NNODES_ROLLOUT}" \
    +rollout.n_gpus_per_node="${N_GPUS_ROLLOUT}" \
    trainer.val_before_train="${VAL_BEFORE_TRAIN}" \
    trainer.log_val_generations=20 \
    trainer.save_freq="${SAVE_FREQ}" \
    trainer.max_actor_ckpt_to_keep="${MAX_CKPT_TO_KEEP}" \
    trainer.resume_mode=disable \
    trainer.default_local_dir="${default_local_dir}" \
    trainer.test_freq="${TEST_FREQ}" \
    trainer.total_epochs="${TOTAL_EPOCHS}" \
    trainer.total_training_steps="${TOTAL_TRAINING_STEPS}" \
    "$@"
