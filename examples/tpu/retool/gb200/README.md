# GB200 Repro: Qwen3-30B-A3B ReTool DAPO RL on GKE

Concise reproduction guide for multi-turn code-interpreter **ReTool** ([arXiv:2504.11536](https://arxiv.org/abs/2504.11536)) reinforcement learning on **`Qwen/Qwen3-30B-A3B`** (`30.5B` total parameters, `3.3B` active per token, `48` layers, `128` routed experts, `top_k=8`) on **NVIDIA GB200 (`aarch64`, SM100, 184.31 GiB HBM3e/GPU)** on GKE.

- **Detailed bilingual execution & audit log (PR #9)**: [`../REPRO_RESULTS_GB200_ZH.md`](../REPRO_RESULTS_GB200_ZH.md)
- **Archived raw evidence & deterministic metric recomputation (`G2`)**: [`./evidence/g2-retool-rl-25step-01/`](./evidence/g2-retool-rl-25step-01/)
- **Iso-accelerator-count TPU v7x vs. NVIDIA GB200 benchmark plan (PR #11)**: [`../ISO_CHIP_PERF_BENCHMARK_PLAN.md`](../ISO_CHIP_PERF_BENCHMARK_PLAN.md)

---

## 0. Overview

This workflow reproduces the Stage-2 multi-turn **DAPO / GRPO** reinforcement learning recipe for `Qwen3-30B-A3B` on **2 × 4-GPU NVIDIA GB200 nodes (8 GPUs total)** using `verl`'s disaggregated **`separate_async`** architecture (`hybrid_engine=False`):

- **Trainer Node (`head`, 4 × GB200)**: Runs the **TorchTitan `0.3.0`** SPMD engine (`FSDP2=4, TP=1, EP=1, CP=1`, `attn_type=flex`, `spmd_backend=partial_dtensor`, `activation_checkpoint=selective`) with `configure_qwen3_moe_for_rl()` disabling `MoE.Config.load_balance_coeff` (`0.001 -> None`) across all 48 MoE layers so unexported `expert_bias_E` buffers cannot desynchronize Trainer and Rollout routing.
- **Rollout Node (`worker-0`, 4 × GB200)**: Runs the **vLLM V1 (`0.32.0`, `VLLM_USE_V1=1`)** async multi-turn rollout server (`TP=4, EP=1`, `gpu_memory_utilization=0.75`) driving up to `8` assistant/tool turns per trajectory against **SandboxFusion**.
- **Cross-Node Weight Synchronization**: Streams updated Actor weights from TorchTitan `FSDP2=4` to vLLM `TP=4` after every training step (`parameter_sync_step=1`) over **NCCL (`2.29.7`)** (`checkpoint_engine.backend=nccl`, `2048 MB` buckets) in **`3.87 s – 5.37 s`** (mean **`4.35 s`**), maintaining a tight Trainer-to-Rollout perplexity ratio of **`1.0080 – 1.0142`** across all 25 steps.

All algorithmic and dataset invariants are frozen to match the TPU recipe ([`../run_qwen3_30b_a3b_dapo_torchtitan.sh`](../run_qwen3_30b_a3b_dapo_torchtitan.sh)).

---

## 1. Setup

### 1.1 Image

- **Dockerfile**: [`examples/tpu/retool/gb200/Dockerfile.gb200.torchtitan`](./Dockerfile.gb200.torchtitan)
- **Base Image (`linux/arm64`)**: `verlai/verl:uv.cu130.dev5@sha256:6d7c328b1e7aedd2dece2f939a827704295b13f4bb75f2f17a962a91a19376b9`
- **Artifact Registry (`supercomputer-testing` / `us-central1` / `pirillo-gcr`)**:
  - **Production ReTool Runtime Image (used in `G1` & `G2`)**:
    ```text
    us-central1-docker.pkg.dev/supercomputer-testing/pirillo-gcr/verl-torchtitan-cu130-arm64@sha256:a139e2b491ac23f06f790fe4277afb9bd543a94bc740c92173e5fdb13ddaba40
    ```
  - **Base `uv.lock` Verification Image (`Stage 0.1`)**:
    ```text
    us-central1-docker.pkg.dev/supercomputer-testing/pirillo-gcr/verl-torchtitan-cu130-arm64@sha256:ec7c927495a06279b128b47424ecd7f84eb70434848e82e46ba26747b5b805f1
    ```

To rebuild and push on a native `linux/arm64` builder:

```bash
IMAGE_URI="us-central1-docker.pkg.dev/supercomputer-testing/pirillo-gcr/verl-torchtitan-cu130-arm64:retool-$(git rev-parse --short HEAD)"
docker build --platform linux/arm64 \
  -f examples/tpu/retool/gb200/Dockerfile.gb200.torchtitan \
  -t "${IMAGE_URI}" .
docker push "${IMAGE_URI}"
```

| Component | Version (`linux/arm64`) | Notes |
| --- | --- | --- |
| **Base Image** | `verlai/verl:uv.cu130.dev5` | `sha256:6d7c328b1e7aedd2dece2f939a827704295b13f4bb75f2f17a962a91a19376b9` |
| **Python** | `3.12.3` (`aarch64`) | Virtualenv at `/workspace/verl-pin/.venv` (`264` locked packages, `0` mismatches vs. `uv.lock`) |
| **PyTorch / CUDA / NCCL** | `2.13.0+cu130` / CUDA `13.0` / NCCL `2.29.7` | Native Blackwell `sm_100a` (compute capability `10.0`) |
| **TorchTitan** | `0.3.0` | Slotted `MoE.Config`; `configure_qwen3_moe_for_rl()` sets `load_balance_coeff = None` across all 48 layers |
| **vLLM** | `0.32.0` (`VLLM_USE_V1=1`) | FlashInfer `0.7.0.post1` SM100 CUTLASS MoE & FMHA kernels |
| **FlashAttention** | `2.8.3` | `import flash_attn` verified in precheck |
| **Ray / TransferQueue** | `2.55.1` / `0.1.11` | Used by `PPOTrainerSeparateAsync` and `AgentLoopManager` |
| **Transformers** | `5.16.1` | `Qwen3MoeForCausalLM` tokenizer & config loader |
| **verl** | `64ef634b` (on upstream `3fdf2b548`) | Adds GB200 ReTool recipe, TorchTitan `0.3.0` MoE RL helper, and token-level `rollout_audit` |

> **GB200 NVLink5 / `ComputeDomain` Requirement**: On GB200 GKE nodes (`NV18` full 18-lane NVLink5 mesh), cross-GPU P2P requires the NVIDIA DRA `ComputeDomain` (`resource.nvidia.com/v1beta1`) IMEX channel (`/dev/nvidia-caps-imex-channels`) and `securityContext.privileged: true`. [`run_ray_jobset_gb200.sh`](./run_ray_jobset_gb200.sh) automatically provisions the 2-node `ComputeDomain` and [`verify_venv_lock.py`](./verify_venv_lock.py) (`--require-gb200`) asserts `torch.cuda.can_device_access_peer(i, j)` across all $4 \times 3 = 12$ intra-node GPU pairs before Ray starts.

---

### 1.2 Model, Dataset & Preprocessing

- **Model — `Qwen3-30B-A3B` Stage-1 Multi-Turn SFT Checkpoint (`global_step_372`)**:
  - **Source Path (GCSFuse `/data`)**: `/data/${INPUTS_OR_RUN_ROOT}/p2-retool-sft-01/checkpoint/global_step_372/huggingface` (`25` files, `61,084,185,820` bytes / `56.89 GiB`, exported in HuggingFace `.safetensors` format after 6 epochs / 372 steps of Stage-1 SFT on `Russwest0/ReTool-SFT` starting from `Qwen/Qwen3-30B-A3B`).
  - **Node-Local `/dev/shm` Staging & SHA-256 Manifest Verification**: Before starting Ray, [`verify_g0_gb200.py`](./verify_g0_gb200.py) copies all 25 files from `/data` to `/dev/shm/models/multiturn-sft-qwen3-30b-a3b-gb200` on both Head and Worker-0 nodes (~`49–62 s` at `1.0–1.2 GB/s`) and verifies the composite SHA-256 across the **20 core files** (`16` `.safetensors` shards + `config.json`, `model.safetensors.index.json`, `tokenizer.json`, `tokenizer_config.json`, totaling `61,079,708,754` bytes):
    ```text
    composite_sha256 = 59247a00c13f2364a12aab5ab6b88b6d971d59ff643c57aad8ac4e55a71231b0
    ```
- **Dataset — Training (`BytedTsinghua-SIA/DAPO-Math-17k`)**:
  - **Path**: `/data/${INPUTS_ROOT}/dataset/BytedTsinghua-SIA/DAPO-Math-17k` (`parquet`, `data_source="math_dapo"`).
- **Dataset — Validation (`Maxwell-Jia/AIME_2024` & `yentinglin/aime_2025`)**:
  - **Paths**:
    - `/data/${INPUTS_ROOT}/dataset/Maxwell-Jia/AIME_2024` (`30` problems $\times$ `val_kwargs.n=30` = `900` evaluation trajectories per checkpoint).
    - `/data/${INPUTS_ROOT}/dataset/yentinglin/aime_2025` (`30` problems $\times$ `val_kwargs.n=30` = `900` evaluation trajectories per checkpoint).
- **Preprocessing (`CustomRLHFDataset` & Tool Schema)**:
  - No offline dataset preprocessing is required. At runtime, [`examples/tpu/retool/retool_tpu.py`](../retool_tpu.py) (`CustomRLHFDataset`) wraps each math problem with the ReTool step-by-step code-interpreter prompt template, attaches the `code_interpreter` tool schema from [`examples/tpu/retool/sandbox_fusion_tool_config_tpu.yaml`](../sandbox_fusion_tool_config_tpu.yaml), sets `apply_chat_template_kwargs.enable_thinking=False`, and applies left prompt truncation (`max_prompt_length=2048`, `max_response_length=8192`).

---

### 1.3 Reward Function & Code Sandbox

| Component | Configured Path / Entrypoint | Runtime Behavior in `G1` & `G2` (`trainer.use_v1=True`) |
| --- | --- | --- |
| **Reward Scorer** | CLI passes top-level `custom_reward_function.path=pkg://examples.tpu.retool.retool_tpu` (`name=compute_score`) | Under `trainer.use_v1=True` (`TaskRunnerV1`), `RewardLoopWorker` reads `config.reward.custom_reward_function.path` (`None` unless nested `reward.custom_reward_function.path` is also passed) and falls back to `verl.utils.reward_score.default_compute_score` $\rightarrow$ `math_dapo.compute_score(..., strict_box_verify=False)` (pure binary outcome reward $\{-1.0, +1.0\}$, $\text{reward} = 2 \times \text{acc} - 1$). |
| **Strict-Box vs. Runtime Scorer Verification** | [`recompute_g2_metrics.py`](./evidence/g2-retool-rl-25step-01/recompute_g2_metrics.py) | Verifies all `3,600` validation samples under both `strict_box_verify=False` (runtime scorer, `0/3600` diff vs. `head.log`) and `strict_box_verify=True` (ReTool strict `\boxed{}` scorer). Both show consistent gains: AIME 2024 **`+5.00 pp`** (`strict=False`) / **`+5.44 pp`** (`strict=True`), AIME 2025 **`+0.89 pp`** (both). |
| **Code Sandbox (`SandboxFusion`)** | [`sandbox_fusion_server.py`](../sandbox_fusion_server.py) + [`sandbox_fusion_tools.py`](../sandbox_fusion_tools.py) | Default single-tenant mode auto-starts a 4-worker loopback FastAPI server (`http://127.0.0.1:18080/run_code`) that drops privileges fail-closed to `nobody:nogroup` (`65534:65534`), sets `CUDA_VISIBLE_DEVICES=""`, enforces `1 GiB` `RLIMIT_AS` + process-group `SIGKILL` on timeout, and passes a 4-case functional check before training. Setting `SANDBOX_FUSION_URL` or `SANDBOX_FUSION_HOST` to an external sandboxed endpoint disables local server auto-start (`SANDBOX_DISABLE_AUTO_START=1`) and routes all tool calls to the external service. |

---

## 2. Launching Runs

| File | Role |
| --- | --- |
| [`examples/tpu/retool/gb200/run_qwen3_30b_a3b_dapo_torchtitan_gb200.sh`](./run_qwen3_30b_a3b_dapo_torchtitan_gb200.sh) | Final GB200 multi-turn DAPO RL recipe script (`verl.trainer.main_ppo` with TorchTitan + vLLM V1) |
| [`examples/tpu/retool/gb200/run_ray_jobset_gb200.sh`](./run_ray_jobset_gb200.sh) | 2-node (`head + worker-0`) GKE K8s + `ComputeDomain` launcher with atomic UID-preconditioned cleanup |
| [`examples/tpu/retool/gb200/verify_venv_lock.py`](./verify_venv_lock.py) | Node precheck: 264-package `uv.lock`, GB200 SM100, 12-pair NVLink P2P `can_device_access_peer`, and `configure_qwen3_moe_for_rl` |
| [`examples/tpu/retool/gb200/verify_g0_gb200.py`](./verify_g0_gb200.py) | `/dev/shm` checkpoint staging, 20-file SHA-256 manifest check, and 1-step `debugmodel_moe` `to_hf()->from_hf()` routing check |
| [`examples/tpu/retool/retool_tpu.py`](../retool_tpu.py) | `CustomRLHFDataset` prompt/tool formatting and ReTool `compute_score` |
| [`examples/tpu/retool/sandbox_fusion_server.py`](../sandbox_fusion_server.py) | FastAPI `/run_code` Python execution server with fail-closed privilege drop |
| [`examples/tpu/retool/sandbox_fusion_tools.py`](../sandbox_fusion_tools.py) | Ray `ExecutionWorker` client with sliding-window rate limiting and external URL support |
| [`examples/tpu/retool/sandbox_fusion_tool_config_tpu.yaml`](../sandbox_fusion_tool_config_tpu.yaml) | `code_interpreter` tool schema and `SandboxFusionTool` configuration |

### 2.1 The Launch Script

#### Launch Stage G1 (3-Step Multi-Turn RL Smoke + Token-Level Mask Audit)

```bash
RUN_NAME="g1-retool-rl-smoke-03" \
KUBE_CONTEXT="<YOUR_GB200_KUBE_CONTEXT>" \
NODEPOOL="<YOUR_4xGB200_NV18_NODEPOOL>" \
GCS_BUCKET="<YOUR_GCS_BUCKET>" \
RUN_ROOT="<YOUR_USER>/retool-gb200/runs" \
INPUTS_ROOT="<YOUR_USER>/retool-v7x/inputs" \
IMAGE="us-central1-docker.pkg.dev/supercomputer-testing/pirillo-gcr/verl-torchtitan-cu130-arm64@sha256:a139e2b491ac23f06f790fe4277afb9bd543a94bc740c92173e5fdb13ddaba40" \
JOBENV_MODEL_SRC_PATH="/data/<YOUR_USER>/retool-v7x/runs/p2-retool-sft-01/checkpoint/global_step_372/huggingface" \
JOBENV_TRAIN_BATCH_SIZE="8" \
JOBENV_PPO_MINI_BATCH_SIZE="8" \
JOBENV_N_RESP_PER_PROMPT="4" \
JOBENV_NUM_WARMUP_BATCHES="1" \
JOBENV_TOTAL_TRAINING_STEPS="3" \
JOBENV_VAL_BEFORE_TRAIN="False" \
JOBENV_TEST_FREQ="-1" \
JOBENV_SAVE_FREQ="-1" \
bash examples/tpu/retool/gb200/run_ray_jobset_gb200.sh
```

#### Launch Stage G2 (25-Step Pilot + Step 0 & Step 25 AIME 2024/2025 `n=30` Evaluation)

```bash
RUN_NAME="g2-retool-rl-25step-01" \
KUBE_CONTEXT="<YOUR_GB200_KUBE_CONTEXT>" \
NODEPOOL="<YOUR_4xGB200_NV18_NODEPOOL>" \
GCS_BUCKET="<YOUR_GCS_BUCKET>" \
RUN_ROOT="<YOUR_USER>/retool-gb200/runs" \
INPUTS_ROOT="<YOUR_USER>/retool-v7x/inputs" \
IMAGE="us-central1-docker.pkg.dev/supercomputer-testing/pirillo-gcr/verl-torchtitan-cu130-arm64@sha256:a139e2b491ac23f06f790fe4277afb9bd543a94bc740c92173e5fdb13ddaba40" \
JOBENV_MODEL_SRC_PATH="/data/<YOUR_USER>/retool-v7x/runs/p2-retool-sft-01/checkpoint/global_step_372/huggingface" \
JOBENV_TRAIN_BATCH_SIZE="64" \
JOBENV_PPO_MINI_BATCH_SIZE="64" \
JOBENV_N_RESP_PER_PROMPT="8" \
JOBENV_N_RESP_PER_PROMPT_VAL="30" \
JOBENV_NUM_WARMUP_BATCHES="2" \
JOBENV_TOTAL_TRAINING_STEPS="25" \
JOBENV_VAL_BEFORE_TRAIN="True" \
JOBENV_TEST_FREQ="25" \
JOBENV_SAVE_FREQ="-1" \
bash examples/tpu/retool/gb200/run_ray_jobset_gb200.sh
```

#### Parameter Reference ([`run_qwen3_30b_a3b_dapo_torchtitan_gb200.sh`](./run_qwen3_30b_a3b_dapo_torchtitan_gb200.sh) & [`head.log`](./evidence/g2-retool-rl-25step-01/head.log))

- **Frozen invariants (aligned with TPU recipe [`../run_qwen3_30b_a3b_dapo_torchtitan.sh`](../run_qwen3_30b_a3b_dapo_torchtitan.sh))**:
  - `data.train_batch_size=64`, `actor.ppo_mini_batch_size=64`, `actor.ppo_epochs=1` $\rightarrow$ **ONE optimizer update per rollout step** ($\mu = 1$), `trainer.v1.separate_async.parameter_sync_step=1`
  - `rollout.n=8` (`64 × 8 = 512` multi-turn trajectories per step; `12,800` trajectories across 25 steps)
  - $\beta = 0$ — `actor.use_kl_loss=False`, `actor.kl_loss_coef=0.0`, `algorithm.use_kl_in_reward=False`, `algorithm.kl_ctrl.kl_coef=0.0`
  - DAPO asymmetric clipping & decoupled bypass IS — `algorithm.adv_estimator=grpo`, `algorithm.norm_adv_by_std_in_grpo=True`, `actor.clip_ratio_low=0.2`, `actor.clip_ratio_high=0.28`, `actor.clip_ratio_c=10.0`, `actor.loss_agg_mode=token-mean`, `actor.policy_loss.loss_mode=bypass_mode`, `algorithm.rollout_correction.bypass_mode=True`, `algorithm.rollout_correction.rollout_is=token`, `algorithm.rollout_correction.rollout_is_threshold=2.0`
  - Sequence & multi-turn caps — `data.max_prompt_length=2048`, `data.max_response_length=8192` (`rollout.max_model_len=10240`), `max_turns=8` (`multi_turn.max_user_turns=8`, `multi_turn.max_assistant_turns=8` $\rightarrow$ up to `16` total messages per trajectory)
  - Sampling & Optimizer — Train rollout: `temperature=1.0`, `top_p=1.0`, `top_k=-1`; Validation rollout (`val_kwargs`): `n=30`, `temperature=1.0`, `top_p=0.6`, `top_k=-1`, `do_sample=True`; Optimizer: `actor.optim.lr=1.4e-6`, `lr_scheduler_type=constant` (`min_lr_factor=1.0`), `weight_decay=0.1`, `betas=[0.9, 0.999]`, TorchTitan `training.max_norm=1.0` (note: `actor.optim.clip_grad=0.088` is passed on CLI, while TorchTitan clips at `training.max_norm=1.0`)
  - Async trainer & evaluation schedule — `trainer.use_v1=True`, `trainer.v1.trainer_mode=separate_async`, `trainer.v1.separate_async.num_warmup_batches=2`, `trainer.v1.sampler.max_off_policy_threshold=4`, `trainer.val_before_train=True`, `trainer.test_freq=25`, `trainer.save_freq=-1`
- **System adaptations (`2 nodes × 4 GB200 = 8 GPUs`, `hybrid_engine=False`)**:
  - **Node 0 (`head` — TorchTitan Trainer, `NNODES_TRAINER=1, N_GPUS_TRAINER=4`)**: `torchtitan.data_parallel_shard_degree=4` (`FSDP2=4`), `tensor_parallel_size=1`, `expert_parallel_size=1`, `context_parallel_size=1`, `attn_type=flex`, `use_remove_padding=True`, `activation_checkpoint.mode=selective`, `spmd_backend=partial_dtensor`, `use_dynamic_bsz=True`, `ppo_max_token_len_per_gpu=16384`, `log_prob_max_token_len_per_gpu=16384`, `torchtitan.max_seq_len=16384`
  - **Node 1 (`worker-0` — vLLM V1 Async Rollout, `NNODES_ROLLOUT=1, N_GPUS_ROLLOUT=4`)**: `VLLM_USE_V1=1`, `tensor_model_parallel_size=4` (`TP=4`), `expert_parallel_size=1` (`EP=1`), `gpu_memory_utilization=0.75`, `max_num_seqs=256`, `max_num_batched_tokens=8192`, `enable_prefix_caching=True`, `enforce_eager=False`, `load_format=safetensors`, `agent.num_workers=16`
  - **Weight Sync**: `checkpoint_engine.backend=nccl`, `checkpoint_engine.update_weights_bucket_megabytes=2048`
- **Scale / Stage knobs**:
  - **Stage G1 (3-Step Smoke)**: `TRAIN_BATCH_SIZE=8`, `PPO_MINI_BATCH_SIZE=8`, `N_RESP_PER_PROMPT=4` (`32` trajectories/step), `NUM_WARMUP_BATCHES=1`, `TOTAL_TRAINING_STEPS=3`, `VAL_BEFORE_TRAIN=False`, `TEST_FREQ=-1`
  - **Stage G2 (25-Step Pilot)**: `TRAIN_BATCH_SIZE=64`, `PPO_MINI_BATCH_SIZE=64`, `N_RESP_PER_PROMPT=8` (`512` trajectories/step), `NUM_WARMUP_BATCHES=2`, `TOTAL_TRAINING_STEPS=25`, `VAL_BEFORE_TRAIN=True`, `TEST_FREQ=25`

---

### 2.2 Collecting Results

- **Loggers (`console` + `tensorboard`; no WandB required)**:
  - `run_qwen3_30b_a3b_dapo_torchtitan_gb200.sh` defaults to `LOGGER_LIST="['console','tensorboard']"` (`WANDB_API_KEY` is not required).
  - TensorBoard event files are persisted directly to the shared GCSFuse run directory at:
    ```text
    /data/${RUN_ROOT}/${RUN_NAME}/tensorboard
    ```
- **Rollout, Validation & Token-Level Mask Audit Dumps (enabled by default in [`run_ray_jobset_gb200.sh`](./run_ray_jobset_gb200.sh))**:
  - `${RUN_DIR}/head.log`: Full `verl.trainer.main_ppo` console log including launch command, Hydra resolved config, `step:0` / `step:25` AIME evaluations, and per-step metrics.
  - `${RUN_DIR}/rollout/{1..25}.jsonl`: All `512` generated multi-turn trajectories per training step (`12,800` total in `G2`).
  - `${RUN_DIR}/validation/{0,25}.jsonl`: All `1,800` evaluation trajectories (`60` AIME problems $\times$ `n=30`) at Step 0 and Step 25.
  - `${RUN_DIR}/rollout_audit/{1..25}.jsonl`: Token-by-token `response_mask` audit (`8` multi-turn sessions/step, `200` sessions in `G2`) verifying `mask_length_matches=True` and asserting `0` `<tool_response>` tokens leak into `response_mask == 1`.
  - `${RUN_DIR}/env/`: Node precheck outputs (`*-precheck.json`, `*-g0-verify.json`, `*-nvidia-smi-L.txt`, `*-topo.txt`, `*-sandbox-4case.json`).
- **Archived Evidence & One-Command Independent Recomputation (`G2`)**:
  All raw `G2` logs, per-step metrics, and `3,600` validation sample records (`output_tail_300` + `output_sha256`) are checked into [`examples/tpu/retool/gb200/evidence/g2-retool-rl-25step-01/`](./evidence/g2-retool-rl-25step-01/). Run:
  ```bash
  python3 examples/tpu/retool/gb200/evidence/g2-retool-rl-25step-01/recompute_g2_metrics.py
  ```
- **How Step Time (`timing_s/step`) Decomposes**:
  Under `separate_async`, each training step logs `timing_s/step` along with:
  - `timing_s/gen`: Time the trainer spent waiting for the async rollout queue to yield a full batch of `64 × 8 = 512` multi-turn trajectories (`~0.04 s – 0.13 s` when rollout generation is fully overlapped with the previous step's `update_actor`; `18 s – 86 s` when multi-turn tool trajectories exceed actor update time).
  - `timing_s/old_log_prob` + `timing_s/adv`: Reference/old log-prob and GRPO advantage computation (`~0.11 s` + `~0.19 s`).
  - `timing_s/update_actor`: TorchTitan `FSDP2=4` forward/backward + AdamW step (`46.32 s` on Step 1 and `64.28 s` on Step 2 due to initial dynamic-shape `flex_attention` Triton autotuning; **`12.89 s – 18.95 s`**, mean **`15.20 s`** in steady state Steps 3–25).
  - `timing_s/update_weights`: Cross-node `actor -> NCCL -> vLLM` 30B weight broadcast + vLLM reload (**`3.87 s – 5.37 s`**, mean **`4.35 s`**).

---

### 2.3 Scaling & Results Summary Table

| Run / Stage | Hardware Scale | Trainer / Rollout Split | Batch (`BS × n`) | Mean `timing_s/step` | Steady-State Throughput (`tok/s` / `tok/s/accel`) | AIME 2024 `mean@30` (`Step 0 -> Step 25`) | AIME 2025 `mean@30` (`Step 0 -> Step 25`) | Status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| **GB200 `G1` (`g1-retool-rl-smoke-03`)** | **2 nodes × 4 GB200 (`8 GPUs`)** | `4 GPU (FSDP2=4)` + `4 GPU (TP=4)` | `8 × 4 = 32` (`3` steps) | `35.94 s` (`6.42 s` on warm Step 3) | `4,304 tok/s` (`538.0 tok/s/GPU` on Step 3) | Smoke only (`VAL_BEFORE_TRAIN=False`) | Smoke only (`VAL_BEFORE_TRAIN=False`) | **PASS** |
| **GB200 `G2` (`g2-retool-rl-25step-01`)** | **2 nodes × 4 GB200 (`8 GPUs`)** | `4 GPU (FSDP2=4)` + `4 GPU (TP=4)` | `64 × 8 = 512` (`25` steps) | **`39.84 s`** (Steps 2–25) / **`38.59 s`** (Steps 3–25) / **`18.55–24.63 s`** (overlapped) | **`14,362.0 tok/s`** (`1,795.2 tok/s/GPU`, Steps 3–25); peak **`28,659.7 tok/s`** (`3,582.5 tok/s/GPU`, Step 25) | **`23.78% -> 28.78%` (`+5.00 pp`)** (`strict=True`: `24.22% -> 29.67%`, `+5.44 pp`) | **`17.00% -> 17.89%` (`+0.89 pp`)** (`strict=True`: `17.11% -> 18.00%`, `+0.89 pp`) | **PASS** |
| **TPU v7x Tier A (Iso-8-Chip, Planned)** | **2 × `2×2×1` (`8 v7x chips` / 16 cores)** | `4 chips (FSDP2=8, CP=1)` + `4 chips (TP=8, EP=8)` | `64 × 8 = 512` (`25` steps) | TBD ([`ISO_CHIP_PERF_BENCHMARK_PLAN.md`](../ISO_CHIP_PERF_BENCHMARK_PLAN.md)) | TBD | TBD | TBD | Planned |
| **GB200 Tier B (Iso-32-GPU, Planned)** | **8 nodes × 4 GB200 (`32 GPUs`)** | `16 GPU (FSDP2=16)` + `16 GPU (4 × TP=4)` | `64 × 8 = 512` (`25` steps) | TBD ([`ISO_CHIP_PERF_BENCHMARK_PLAN.md`](../ISO_CHIP_PERF_BENCHMARK_PLAN.md)) | TBD | TBD | TBD | Planned |
| **TPU v7x `P3` / Tier B (Iso-32-Chip)** | **2 × `2×2×4` (`32 v7x chips` / 64 cores)** | `16 chips (FSDP2=8, CP=4)` + `16 chips (TP=32, EP=32)` | `64 × 8 = 512` (`25` steps) | TBD (pending `P1c` gate) | TBD | TBD | TBD | In Progress |
| **Askhat TPU v6e Baseline ([`#108`](https://github.com/jialei777/verl-upstream/pull/108))** | **2 × `v6e-32` (`64 v6e chips`)** | `32 chips (FSDP2=8, CP=4)` + `32 chips (TP=32, EP=32)` | `64 × 8 = 512` (`25+` steps) | `108.1 s` | - | `22.44% -> 30.00%` (`+7.56 pp`) | - | Reference |

---

### 2.4 Step-Time Breakdown & AIME 2024 / 2025 Validation (`G2`)

#### Step-Time & Wall-Clock Breakdown (`g2-retool-rl-25step-01`, `8 × GB200 GPUs`)

| Window / Subset | `timing_s/gen` | `old_log_prob` + `adv` | `timing_s/update_actor` | `timing_s/update_weights` | **`timing_s/step`** | Total Tokens / Wall-Clock | Throughput (`tok/s` / `tok/s/GPU`) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| **Step 1 (Cold `flex_attention` compile + warmup)** | `86.14 s` | `0.23 s` | `46.32 s` | `3.99 s` | **`136.92 s`** | `473,629 tok` / `136.92 s` | `3,459.2 tok/s` (`432.4 tok/s/GPU`) |
| **Step 2 (2nd dynamic-shape `flex_attention` autotune)** | `0.04 s` | `0.27 s` | `64.28 s` | `3.87 s` | **`68.58 s`** | `604,398 tok` / `68.58 s` | `8,813.1 tok/s` (`1,101.6 tok/s/GPU`) |
| **Steps 2–25 Mean (24 steps)** | `17.78 s` | `0.31 s` | `17.25 s` | `4.36 s` | **`39.84 s`** | `13,351,790 tok` / `956.17 s` | `13,963.8 tok/s` (`1,745.5 tok/s/GPU`) |
| **Steps 3–25 Steady-State Mean (`W2`, 23 steps)** | **`18.55 s`** | **`0.31 s`** | **`15.20 s`** (`12.89–18.95 s`) | **`4.38 s`** (`3.93–5.37 s`) | **`38.59 s`** | `12,747,392 tok` / `887.58 s` | **`14,362.0 tok/s` (`1,795.2 tok/s/GPU`)** |
| **Overlapped Async Steps (Steps 3, 4, 6, 10, 12, 18, 20, 23, 25)** | **`0.04 – 4.11 s`** | `0.30 s` | `13.75 – 18.95 s` | `4.23 – 5.37 s` | **`18.55 – 24.63 s`** | Peak Step 25 (`629,878 tok` / `21.98 s`) | **`28,659.7 tok/s` (`3,582.5 tok/s/GPU`)** |
| **All Steps 1–25 Training Wall-Clock (`W1`, excl. Step 0/25 evals)** | `20.51 s` (mean) | `0.30 s` (mean) | `18.41 s` (mean) | `4.35 s` (mean) | **`43.72 s`** (mean) | `13,825,419 tok` / **`1,104.5 s` (`18m 24s`)** | **`12,517.4 tok/s` (`1,564.7 tok/s/GPU`)** |

- **End-to-End Wall-Clock Milestones ([`head.log`](./evidence/g2-retool-rl-25step-01/head.log) UTC timestamps)**:
  - **Step 0 initial `update_weights`**: `4.13 s` (`23:50:01` $\rightarrow$ `23:50:05.530`)
  - **Step 0 AIME 2024/2025 (`n=30`, 1,800 trajectories) evaluation**: `180.1 s` (`3m 00s`, `23:50:05.950` $\rightarrow$ `23:53:06`)
  - **25-step RL training loop (`W1`)**: `1,104.5 s` (`18m 24s`, `23:53:06` $\rightarrow$ `00:11:30.538`)
  - **Step 25 AIME 2024/2025 (`n=30`, 1,800 trajectories) evaluation**: `190.4 s` (`3m 10s`, `00:11:30.538` $\rightarrow$ `00:14:42.324`)
  - **Total training + both Step 0 & Step 25 `n=30` evaluations**: **`1,476.4 s` (`24m 36s`)** (`tqdm` reports `21m 35s`, which covers 25 training steps + Step 25 evaluation; full `main_ppo` process lifetime including Ray init & 30B load is `27m 37s`).

#### AIME 2024 & AIME 2025 (`n=30` Samples/Problem) Validation Results (`Step 0 -> Step 25`)

| Benchmark & Metric (`n=30` samples / problem, `900` trajectories / dataset) | Step 0 (Stage-1 SFT `global_step_372`) | Step 25 (After 25 ReTool DAPO RL Steps) | Absolute Gain ($\Delta$) |
| --- | --- | --- | --- |
| **AIME 2024 `acc/mean@30` (Runtime `strict_box_verify=False`)** | `23.78%` (`214 / 900`) | **`28.78%` (`259 / 900`)** | **`+5.00 pp`** (`+45 / 900`) |
| **AIME 2024 `acc/mean@30` (ReTool `strict_box_verify=True`)** | `24.22%` (`218 / 900`) | **`29.67%` (`267 / 900`)** | **`+5.44 pp`** (`+49 / 900`) |
| **AIME 2024 `acc/maj@30/mean`** | `40.55%` | **`47.13%`** | **`+6.58 pp`** |
| **AIME 2024 `acc/best@30/mean`** | `64.29%` | **`69.05%`** | **`+4.76 pp`** |
| **AIME 2024 `acc/maj@8/mean` / `best@8/mean`** | `33.73%` / `53.13%` | **`41.57%`** / **`58.00%`** | **`+7.84 pp`** / **`+4.87 pp`** |
| **AIME 2025 `acc/mean@30` (Runtime `strict_box_verify=False`)** | `17.00%` (`153 / 900`) | **`17.89%` (`161 / 900`)** | **`+0.89 pp`** (`+8 / 900`) |
| **AIME 2025 `acc/mean@30` (ReTool `strict_box_verify=True`)** | `17.11%` (`154 / 900`) | **`18.00%` (`162 / 900`)** | **`+0.89 pp`** (`+8 / 900`) |
| **AIME 2025 `acc/maj@30/mean`** | `29.84%` | **`30.60%`** | **`+0.76 pp`** |
| **AIME 2025 `acc/best@30/mean`** | `45.52%` | **`57.34%`** | **`+11.82 pp`** |
| **AIME 2025 `acc/maj@8/mean` / `best@8/mean`** | `25.38%` / `36.78%` | **`25.41%`** / **`42.61%`** | **`+0.03 pp`** / **`+5.84 pp`** |

---

### 2.5 Workload Shape (`G2`, Steps 1–25)

| Metric | Step 1 | Step 25 | Steps 1–25 Mean / Total | Notes |
| --- | --- | --- | --- | --- |
| **Response Length (`response_length/mean`)** | `610.3` tokens | `898.6` tokens | **`760.4` tokens** (`593.5 – 948.6`) | `max_response_length=8192` |
| **Max Response Length (`response_length/max`)** | `2,203` tokens | `8,192` tokens | **`3,442.2` tokens** (mean of step maxima) | Reaches `8,192` cap on long multi-turn traces |
| **Response Clip Ratio (`response_length/clip_ratio`)** | `0.195%` | `0.195%` | **`0.23%`** (`0.0023`) | `< 0.3%` of responses hit the `8,192`-token cap |
| **Prompt Length (`prompt_length/mean` / `max`)** | `314.8` / `438` | `332.1` / `779` | **`319.7`** / **`528.5`** | `max_prompt_length=2048` |
| **Conversation Turns (`num_turns/mean` / `max`)** | `4.89` / `12` | `6.00` / `16` | **`5.77` turns** (`min=2`, `max=16`) | `8 user + 8 assistant = 16` max messages |
| **Sandbox Tool Invocations (`<tool_response>`)** | `740` (across `512` trajs) | `1,024` (across `512` trajs) | **`24,106` calls** (`11,933 / 12,800 = 93.23%` trajs) | `200 / 200` audited sessions have **`0` mask violations** |
| **Tokens per Step (`perf/total_num_tokens`)** | `473,629` | `629,878` | **`553,017` tok/step** (`13,825,419` total) | Balanced across 4 FSDP2 shards (`global_seqlen/mean=138,254/GPU`) |
| **Trainer HBM Peak (`max_memory_allocated` / `reserved`)** | `117.34 GB` / `125.76 GB` | `154.47 GB` / `172.57 GB` | **`143.86 GiB` allocated / `160.72 GiB` reserved** | Fits comfortably within `184.31 GiB` per GB200 GPU (`0` OOMs) |
