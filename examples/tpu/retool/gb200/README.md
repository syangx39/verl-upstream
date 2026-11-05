# GB200 Repro: Qwen3-30B-A3B ReTool DAPO RL on GKE

Concise reproduction guide for multi-turn code-interpreter **ReTool** ([arXiv:2504.11536](https://arxiv.org/abs/2504.11536)) reinforcement learning on **`Qwen/Qwen3-30B-A3B`** (`30.5B` total parameters, `3.3B` active per token, `48` layers, `128` routed experts, `top_k=8`) on **NVIDIA GB200 (`aarch64`, SM100, 184.31 GiB HBM3e/GPU)** on GKE.

- **Detailed bilingual execution & audit log (PR #9)**: [`exp/REPRO_RESULTS_GB200_ZH.md`](exp/REPRO_RESULTS_GB200_ZH.md)
- **Archived raw evidence & deterministic metric recomputation (`G2`)**: [`./logs/g2-retool-rl-25step-01/`](logs/g2-retool-rl-25step-01/)
- **Iso-accelerator-count TPU v7x vs. NVIDIA GB200 benchmark plan (PR #11)**: [`../ISO_CHIP_PERF_BENCHMARK_PLAN.md`](../ISO_CHIP_PERF_BENCHMARK_PLAN.md)

### Directory guide

- This directory contains the Dockerfile, launchers, verification tools, and this guide.
- [`exp/`](./exp/) holds the original GB200 plan and detailed results report.
- [`logs/`](./logs/) holds the historical G2 logs, environment checks, sample records, and metric recomputation script.
- The dataset, tool, and sandbox implementations in [`../`](../) are shared with TPU.

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
  - **Recommended Reproducible Runtime Image (`verl @ a2cbde47`, verified by `g1-retool-rl-smoke-04`)**:
    ```text
    us-central1-docker.pkg.dev/supercomputer-testing/pirillo-gcr/verl-torchtitan-cu130-arm64@sha256:087366ae58a31888c2af2c25992c33ea77471b1b2c58d221b370899f4721e689
    ```
    The following smoke result is reported by the run author; the G1-04 raw log, precheck and TensorBoard event file are not archived in this PR, so the image digest and runtime behavior cannot be independently verified from this repository alone. Built from branch `gb200-retool-runtime` at commit [`a2cbde47ba729d0cba7e881da1692df20d8e05e4`](https://github.com/syangx39/verl-upstream/commit/a2cbde47ba729d0cba7e881da1692df20d8e05e4) (`uv.lock` SHA-256 `df508db0b957fa181c229fb75885cd3863cb651fd5233d0fa5af95e990bda0be`). Contains the hardened launcher, `sandbox_fusion_server.py --check --url ...` + fail-closed `65534:65534` privilege drop, `TENSORBOARD_DIR=${RUN_DIR}/tensorboard` persistence, and `verify_g0_gb200.py --expect-composite-sha256 59247a00...`. Works out-of-the-box with [`run_ray_jobset_gb200.sh`](./run_ray_jobset_gb200.sh) without needing `JOBENV_EXPECT_GIT_SHA`.
  - **Historical `G2` (`g2-retool-rl-25step-01`) Image & Runtime Commit**:
    ```text
    us-central1-docker.pkg.dev/supercomputer-testing/pirillo-gcr/verl-torchtitan-cu130-arm64@sha256:a139e2b491ac23f06f790fe4277afb9bd543a94bc740c92173e5fdb13ddaba40
    ```
    Built at commit [`23113cb2defeaa84e5a23c0b8a7d1d7f13f19789`](https://github.com/syangx39/verl-upstream/commit/23113cb2defeaa84e5a23c0b8a7d1d7f13f19789); historical `G2` (`g2-retool-rl-25step-01`) and `G1` (`g1-retool-rl-smoke-03`) ran on this image with `JOBENV_EXPECT_GIT_SHA="64ef634b173c69ea31eff598caeee81d9a9caea1"`. Because `23113cb2` and `64ef634b` predate the PR #9 sandbox CLI flag `--url`, running the current [`run_ray_jobset_gb200.sh`](./run_ray_jobset_gb200.sh) against `@sha256:a139e2b4...` requires passing `JOBENV_EXPECT_GIT_SHA="a2cbde47ba729d0cba7e881da1692df20d8e05e4"` (or using the new recommended image above).
  - **Base `uv.lock` Verification Image (`Stage 0.1`, `verl @ 3fdf2b548`)**:
    ```text
    us-central1-docker.pkg.dev/supercomputer-testing/pirillo-gcr/verl-torchtitan-cu130-arm64@sha256:ec7c927495a06279b128b47424ecd7f84eb70434848e82e46ba26747b5b805f1
    ```

To rebuild and push on a native `linux/arm64` builder (note that the build context **must** be `examples/tpu/retool/gb200` so `COPY verify_venv_lock.py` resolves, and `VERL_PIN` must point to a commit whose `uv.lock` matches the base image cache `df508db0...`, such as `a2cbde47ba729d0cba7e881da1692df20d8e05e4`):

```bash
(
  set -euo pipefail
  cd examples/tpu/retool/gb200
  VERL_PIN="a2cbde47ba729d0cba7e881da1692df20d8e05e4"
  IMAGE_URI="us-central1-docker.pkg.dev/supercomputer-testing/pirillo-gcr/verl-torchtitan-cu130-arm64:${VERL_PIN:0:8}"
  docker buildx build --platform linux/arm64 --progress=plain \
    --build-arg VERL_REPO=https://github.com/syangx39/verl-upstream.git \
    --build-arg VERL_PIN="${VERL_PIN}" \
    --build-arg EXPECT_LOCK_SHA256=df508db0b957fa181c229fb75885cd3863cb651fd5233d0fa5af95e990bda0be \
    -f Dockerfile.gb200.torchtitan \
    -t "${IMAGE_URI}" --push .
)
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
| **verl** | `a2cbde47` (recommended) / `64ef634b` (historical `G2`) | Both branched from upstream `3fdf2b548` (`uv.lock` `df508db0...`); `a2cbde47` adds `--check --url`, fail-closed privilege drop, `TENSORBOARD_DIR` persistence, and `--expect-composite-sha256` |

> **GB200 NVLink5 / `ComputeDomain` Requirement**: On GB200 GKE nodes (`NV18` full 18-lane NVLink5 mesh), cross-GPU P2P requires the NVIDIA DRA `ComputeDomain` (`resource.nvidia.com/v1beta1`) IMEX channel (`/dev/nvidia-caps-imex-channels`) and `securityContext.privileged: true`. [`run_ray_jobset_gb200.sh`](./run_ray_jobset_gb200.sh) automatically provisions the 2-node `ComputeDomain` and [`verify_venv_lock.py`](./verify_venv_lock.py) (`--require-gb200`) asserts `torch.cuda.can_device_access_peer(i, j)` across all $4 \times 3 = 12$ intra-node GPU pairs before Ray starts.

---

### 1.2 Model, Dataset & Preprocessing

- **Model — `Qwen3-30B-A3B` Stage-1 Multi-Turn SFT Checkpoint (`global_step_372`)**:
  - **Source Path (GCSFuse `/data`)**: `/data/${INPUTS_OR_RUN_ROOT}/p2-retool-sft-01/checkpoint/global_step_372/huggingface` (`25` files, `61,084,185,820` bytes / `56.89 GiB`, exported in HuggingFace `.safetensors` format after 6 epochs / 372 steps of Stage-1 SFT on `Russwest0/ReTool-SFT` starting from `Qwen/Qwen3-30B-A3B` @ HF commit `ad44e777bcd18fa416d9da3bd8f70d33ebb85d39`).
  - **Node-Local `/dev/shm` Staging & Expected `composite_sha256` Assertion**: Before starting Ray, [`run_ray_jobset_gb200.sh`](./run_ray_jobset_gb200.sh) invokes [`verify_g0_gb200.py`](./verify_g0_gb200.py) on both `head` and `worker-0` to copy all 25 files from `/data` to `/dev/shm/models/multiturn-sft-qwen3-30b-a3b-tpu7x` (~`49–62 s` at `1.0–1.2 GB/s`), cross-check every file against `--ref-dir`, and assert `--expect-composite-sha256` (`EXPECT_CHECKPOINT_SHA256`) across the **20 core files** (`16` `.safetensors` shards + `config.json`, `model.safetensors.index.json`, `tokenizer.json`, `tokenizer_config.json`, totaling `61,079,708,754` bytes; per-file hashes archived in [`g2-retool-rl-25step-01-head-g0-verify.json`](logs/g2-retool-rl-25step-01/env/g2-retool-rl-25step-01-head-g0-verify.json)):
    ```text
    composite_sha256 = 59247a00c13f2364a12aab5ab6b88b6d971d59ff643c57aad8ac4e55a71231b0
    ```
- **Inputs prerequisite**: The Stage-1 checkpoint and the datasets in the manifest below must already be staged in the shared GCS bucket at the specified paths, with read access from both nodes. The Stage-1 checkpoint is produced by the TPU P2 run; its relative path is not a public download URL. Copy the checkpoint and datasets from an accessible source before launching, and compare the dataset SHA-256 digests and row counts against the manifest. The model manifest's 20-file composite SHA-256 is checked by G0.
- **Datasets — Staged Parquet Manifest & SHA-256 Digests**:

| Dataset Role | HuggingFace Source | Relative Path under `/data/${INPUTS_ROOT}` | Rows | Size (Bytes) | Parquet SHA-256 |
| --- | --- | --- | --- | --- | --- |
| **RL Training (`DAPO-Math-17k`)** | [`BytedTsinghua-SIA/DAPO-Math-17k`](https://huggingface.co/datasets/BytedTsinghua-SIA/DAPO-Math-17k) | `dataset/BytedTsinghua-SIA/DAPO-Math-17k/data/train-00000-of-00001.parquet` | `1,791,700` (`17,917` unique prompts $\times 100$) | `715,244,619` | `22e266f54502ce271ee08713feb8b30caf97a4ea948aca0e0506f58c4cdd0545` |
| **Validation (`AIME 2024`)** | [`Maxwell-Jia/AIME_2024`](https://huggingface.co/datasets/Maxwell-Jia/AIME_2024) | `dataset/Maxwell-Jia/AIME_2024/data/train-00000-of-00001.parquet` | `30` (`× val_kwargs.n=30 = 900` trajs/eval) | `29,259` | `1332803ea03085cce9222b6c0b295b4115507ab5984aee9d3e9cb3db3c2ee948` |
| **Validation (`AIME 2025`)** | [`yentinglin/aime_2025`](https://huggingface.co/datasets/yentinglin/aime_2025) | `dataset/yentinglin/aime_2025/data/train-00000-of-00001.parquet` | `30` (`× val_kwargs.n=30 = 900` trajs/eval) | `18,454` | `4f4432a5252cf673ddd696a3dba24f75bfd662c0ef1d54b6e0f442c24119f8ed` |
| **Stage-1 SFT Source (`ReTool-SFT`)** | [`Russwest0/ReTool-SFT`](https://huggingface.co/datasets/Russwest0/ReTool-SFT) | `dataset/ReTool-SFT/data/train-00000-of-00001.parquet` | `2,000` (multi-turn up to `17` turns) | `8,901,717` | `92a29cea03f72dc79066e1697f4d7423d91bd6a02bd1aac50f5c9dfd89b27faa` |

- **Preprocessing (`CustomRLHFDataset` & Tool Schema)**:
  - No offline dataset preprocessing is required for Stage-2 RL; `RLHFDataset` reads the raw HuggingFace parquet directories directly. At runtime, [`examples/tpu/retool/retool_tpu.py`](../retool_tpu.py) (`CustomRLHFDataset`) wraps each math problem with the ReTool step-by-step code-interpreter prompt template, attaches the `code_interpreter` tool schema from [`examples/tpu/retool/sandbox_fusion_tool_config_tpu.yaml`](../sandbox_fusion_tool_config_tpu.yaml), sets `apply_chat_template_kwargs.enable_thinking=False`, and applies left prompt truncation (`max_prompt_length=2048`, `max_response_length=8192`).

---

### 1.3 Reward Function & Code Sandbox

| Component | Configured Path / Entrypoint | Runtime Behavior (`trainer.use_v1=True`) |
| --- | --- | --- |
| **Reward Scorer** | CLI passes top-level `custom_reward_function.path=pkg://examples.tpu.retool.retool_tpu` (`name=compute_score`) | Under `trainer.use_v1=True` (`TaskRunnerV1`), `RewardLoopWorker` reads `config.reward.custom_reward_function.path` (`None` unless nested `reward.custom_reward_function.path` is also passed) and falls back to `verl.utils.reward_score.default_compute_score` $\rightarrow$ `math_dapo.compute_score(..., strict_box_verify=False)` (pure binary outcome reward $\{-1.0, +1.0\}$, $\text{reward} = 2 \times \text{acc} - 1$). |
| **Strict-Box vs. Runtime Scorer Verification** | [`recompute_g2_metrics.py`](logs/g2-retool-rl-25step-01/recompute_g2_metrics.py) | Verifies all `3,600` validation samples under both `strict_box_verify=False` (runtime scorer, `0/3600` diff vs. `head.log`) and `strict_box_verify=True` (ReTool strict `\boxed{}` scorer). Both show consistent gains: AIME 2024 **`+5.00 pp`** (`strict=False`) / **`+5.44 pp`** (`strict=True`), AIME 2025 **`+0.89 pp`** (both). |
| **Code Sandbox (`SandboxFusion`)** | [`sandbox_fusion_server.py`](../sandbox_fusion_server.py) + [`sandbox_fusion_tools.py`](../sandbox_fusion_tools.py) | **Current hardened server (`a2cbde47` / `tpu-main`)**: Default single-tenant mode auto-starts a 4-worker loopback FastAPI server (`http://127.0.0.1:18080/run_code`) that drops privileges fail-closed to `nobody:nogroup` (`65534:65534`), sets `CUDA_VISIBLE_DEVICES=""`, attempts to set `RLIMIT_AS` with a 4 GiB minimum (the `setrlimit` error is ignored) and uses process-group `SIGKILL` on timeout, and passes a 4-case functional check before training.<br>**Historical `G2` (`64ef634b`)**: Ran the in-Pod loopback server before `65534:65534` privilege drop and `--url` CLI support were added in PR #9 (`574f0b78`).<br>**Security boundary**: Even with `65534:65534` privilege drop, GB200 Pods run with `securityContext.privileged: true` (for NVIDIA DRA `ComputeDomain` IMEX channel injection) and mount writable GCSFuse `/data`, so local subprocesses share the container filesystem and kernel namespaces. In-Pod loopback mode must **only** be used for trusted single-tenant experiments; for untrusted prompts/models, set `SANDBOX_FUSION_HOST` or `SANDBOX_FUSION_URL` to an external sandboxed service (which automatically sets `SANDBOX_DISABLE_AUTO_START=1`). |

---

## 2. Launching Runs

| File | Role |
| --- | --- |
| [`examples/tpu/retool/gb200/run_qwen3_30b_a3b_dapo_torchtitan_gb200.sh`](./run_qwen3_30b_a3b_dapo_torchtitan_gb200.sh) | Final GB200 multi-turn DAPO RL recipe script (`verl.trainer.main_ppo` with TorchTitan + vLLM V1) |
| [`examples/tpu/retool/gb200/run_ray_jobset_gb200.sh`](./run_ray_jobset_gb200.sh) | Multi-node (`NUM_NODES` default `2`: `head + worker-0..worker-$((NUM_NODES-2))`) GKE K8s + `ComputeDomain` launcher with atomic UID-preconditioned cleanup |
| [`examples/tpu/retool/gb200/verify_venv_lock.py`](./verify_venv_lock.py) | Node precheck: 264-package `uv.lock`, GB200 SM100, 12-pair NVLink P2P `can_device_access_peer`, and `configure_qwen3_moe_for_rl` |
| [`examples/tpu/retool/gb200/verify_g0_gb200.py`](./verify_g0_gb200.py) | `/dev/shm` checkpoint staging, 20-file SHA-256 manifest check, and 1-step `debugmodel_moe` `to_hf()->from_hf()` routing check |
| [`examples/tpu/retool/retool_tpu.py`](../retool_tpu.py) | `CustomRLHFDataset` prompt/tool formatting and ReTool `compute_score` |
| [`examples/tpu/retool/sandbox_fusion_server.py`](../sandbox_fusion_server.py) | FastAPI `/run_code` Python execution server with fail-closed privilege drop |
| [`examples/tpu/retool/sandbox_fusion_tools.py`](../sandbox_fusion_tools.py) | Ray `ExecutionWorker` client with sliding-window rate limiting and external URL support |
| [`examples/tpu/retool/sandbox_fusion_tool_config_tpu.yaml`](../sandbox_fusion_tool_config_tpu.yaml) | `code_interpreter` tool schema and `SandboxFusionTool` configuration |

**Multi-node scaling (`NUM_NODES=2 / 8 / 16`):** [`run_ray_jobset_gb200.sh`](./run_ray_jobset_gb200.sh) defaults to `NUM_NODES=2` (`2 × 4 GPU = 8 GPUs`) and supports `NUM_NODES=8` (`32 GPUs`) or `NUM_NODES=16` (`64 GPUs`) by creating `worker-0 .. worker-$((NUM_NODES-2))`, waiting for `${TOTAL_GPUS}.0 GPU` and `${NUM_NODES}` G0 manifest files, and cleaning up every pod by `metadata.uid`. See [§2.6](#26-32--64-gb200-scaling-plan) for the candidate parallel configs (`G64-A..D`), Profile 1 vs. Profile 2 alignment with 64-chip TPU v7x (`p3c-retool-25s-01`), and cluster validation steps.

### 2.1 The Launch Script

> **Note on `RUN_NAME` Uniqueness**: [`run_ray_jobset_gb200.sh`](./run_ray_jobset_gb200.sh) fails closed if `gs://${GCS_BUCKET}/${RUN_ROOT}/${RUN_NAME}/` already contains files or `.launch_lock.json` (unless `ALLOW_OVERWRITE_RUN_DIR=1`). Always use a fresh `RUN_NAME` (such as the UTC-timestamped names below) rather than reusing archived run names like `g1-retool-rl-smoke-03` or `g2-retool-rl-25step-01`.

#### Launch Stage G1 (3-Step Multi-Turn RL Smoke + Token-Level Mask Audit)

```bash
RUN_NAME="g1-retool-rl-smoke-$(date -u +%Y%m%d-%H%M%S)" \
KUBE_CONTEXT="<YOUR_GB200_KUBE_CONTEXT>" \
NODEPOOL="<YOUR_4xGB200_NV18_NODEPOOL>" \
GCS_BUCKET="<YOUR_GCS_BUCKET>" \
RUN_ROOT="<YOUR_USER>/retool-gb200/runs" \
INPUTS_ROOT="<YOUR_USER>/retool-v7x/inputs" \
IMAGE="us-central1-docker.pkg.dev/supercomputer-testing/pirillo-gcr/verl-torchtitan-cu130-arm64@sha256:087366ae58a31888c2af2c25992c33ea77471b1b2c58d221b370899f4721e689" \
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
RUN_NAME="g2-retool-rl-25step-$(date -u +%Y%m%d-%H%M%S)" \
KUBE_CONTEXT="<YOUR_GB200_KUBE_CONTEXT>" \
NODEPOOL="<YOUR_4xGB200_NV18_NODEPOOL>" \
GCS_BUCKET="<YOUR_GCS_BUCKET>" \
RUN_ROOT="<YOUR_USER>/retool-gb200/runs" \
INPUTS_ROOT="<YOUR_USER>/retool-v7x/inputs" \
IMAGE="us-central1-docker.pkg.dev/supercomputer-testing/pirillo-gcr/verl-torchtitan-cu130-arm64@sha256:087366ae58a31888c2af2c25992c33ea77471b1b2c58d221b370899f4721e689" \
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

#### Parameter Reference ([`run_qwen3_30b_a3b_dapo_torchtitan_gb200.sh`](./run_qwen3_30b_a3b_dapo_torchtitan_gb200.sh) & [`head.log`](logs/g2-retool-rl-25step-01/head.log))

- **Frozen invariants (aligned with TPU recipe [`../run_qwen3_30b_a3b_dapo_torchtitan.sh`](../run_qwen3_30b_a3b_dapo_torchtitan.sh))**:
  - `data.train_batch_size=64`, `actor.ppo_mini_batch_size=64`, `actor.ppo_epochs=1` $\rightarrow$ **ONE optimizer update per rollout step** ($\mu = 1$), `trainer.v1.separate_async.parameter_sync_step=1`
  - `rollout.n=8` (`64 × 8 = 512` multi-turn trajectories per step; `12,800` trajectories across 25 steps)
  - $\beta = 0$ — `actor.use_kl_loss=False`, `actor.kl_loss_coef=0.0`, `algorithm.use_kl_in_reward=False`, `algorithm.kl_ctrl.kl_coef=0.0`
  - DAPO asymmetric clipping & decoupled bypass IS — `algorithm.adv_estimator=grpo`, `algorithm.norm_adv_by_std_in_grpo=True`, `actor.clip_ratio_low=0.2`, `actor.clip_ratio_high=0.28`, `actor.clip_ratio_c=10.0`, `actor.loss_agg_mode=token-mean`, `actor.policy_loss.loss_mode=bypass_mode`, `algorithm.rollout_correction.bypass_mode=True`, `algorithm.rollout_correction.rollout_is=token`, `algorithm.rollout_correction.rollout_is_threshold=2.0`
  - Sequence & multi-turn caps — `data.max_prompt_length=2048`, `data.max_response_length=8192` (`rollout.max_model_len=10240`), `max_turns=8` (`multi_turn.max_user_turns=8`, `multi_turn.max_assistant_turns=8` $\rightarrow$ up to `16` total messages per trajectory)
  - Sampling & Optimizer —
    - **Train rollout**: `temperature=1.0`, `top_p=1.0`, `top_k=-1`, `do_sample=True`
    - **Validation rollout (`rollout.val_kwargs`, [`head.log#L573-L577`](logs/g2-retool-rl-25step-01/head.log#L573-L577))**: `n=30`, `temperature=1.0`, `top_p=0.6`, `top_k=-1`, `do_sample=False` (schema default). In `verl` V1 (`trainer.use_v1=True`), `AgentLoopWorkerTQ.generate_sequences` ([`verl/trainer/ppo/v1/agent_loop_tq.py`](../../../../verl/trainer/ppo/v1/agent_loop_tq.py#L72-L76)) and `AgentLoopWorker.generate_sequences` ([`verl/experimental/agent_loop/agent_loop.py`](../../../../verl/experimental/agent_loop/agent_loop.py#L603-L607)) construct vLLM `sampling_params` directly from `val_kwargs.temperature` (`1.0`), `val_kwargs.top_p` (`0.6`), and `val_kwargs.top_k` (`-1`) and spawn `val_kwargs.n` (`30`) parallel sessions per prompt without reading `val_kwargs.do_sample` (which is only consumed by legacy `ray_trainer.py` / `hf_rollout.py`). Because `temperature=1.0 > 0` and `top_p=0.6`, vLLM V1 performs stochastic nucleus sampling across all 30 validation trajectories per problem (`std@30 = 0.2552` on AIME 2024).
    - **Optimizer ([`head.log#L168-L181`](logs/g2-retool-rl-25step-01/head.log#L168-L181))**: `actor.optim.lr=1.4e-6`, `lr_scheduler_type=constant` (`min_lr_factor=1.0`), `weight_decay=0.01`, `betas=[0.9, 0.999]`, TorchTitan `training.max_norm=1.0` (note: `actor.optim.clip_grad=0.088` is passed on CLI, while TorchTitan clips at `training.max_norm=1.0`)
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
  - **Current launcher (`a2cbde47` / `tpu-main`, verified in `g1-retool-rl-smoke-04`)**: Both [`run_ray_jobset_gb200.sh`](./run_ray_jobset_gb200.sh) and [`run_qwen3_30b_a3b_dapo_torchtitan_gb200.sh`](./run_qwen3_30b_a3b_dapo_torchtitan_gb200.sh) export `TENSORBOARD_DIR="${RUN_DIR}/tensorboard"` (and forward it via `+ray_kwargs.ray_init.runtime_env.env_vars.TENSORBOARD_DIR`), persisting TensorBoard `events.out.tfevents.*` files directly to the shared GCSFuse run directory at:
    ```text
    /data/${RUN_ROOT}/${RUN_NAME}/tensorboard
    ```
    (Verified in `g1-retool-rl-smoke-04`: `Saving tensorboard log to /data/xiaotongyang/retool-gb200/runs/g1-retool-rl-smoke-04/tensorboard.` $\rightarrow$ `tensorboard/events.out.tfevents.1793689701.g1-retool-rl-smoke-04-head.10162.0`, `12,762` bytes).
  - **Historical `G2` (`g2-retool-rl-25step-01` at `64ef634b`)**: Ran before `TENSORBOARD_DIR` was exported in the launcher, so `_TensorboardAdapter` ([`verl/utils/tracking.py`](../../../../verl/utils/tracking.py#L557-L560)) wrote to the container-local default relative path `tensorboard_log/verl_gb200_retool/qwen3-30b-a3b_retool_dapo_gb200` ([`head.log#L1940`](logs/g2-retool-rl-25step-01/head.log#L1940)). For `g2-retool-rl-25step-01`, all step-by-step scalar metrics and raw trajectories are persisted in `head.log`, `g2_25step_metrics.json`, `rollout/*.jsonl`, `validation/*.jsonl`, and `rollout_audit/*.jsonl`.
- **Rollout, Validation & Token-Level Mask Audit Dumps (enabled by default in [`run_ray_jobset_gb200.sh`](./run_ray_jobset_gb200.sh))**:
  - `${RUN_DIR}/head.log`: Full `verl.trainer.main_ppo` console log including launch command, Hydra resolved config, `step:0` / `step:25` AIME evaluations, and per-step metrics.
  - `${RUN_DIR}/rollout/{1..25}.jsonl`: All `512` generated multi-turn trajectories per training step (`12,800` total in `G2`).
  - `${RUN_DIR}/validation/{0,25}.jsonl`: All `1,800` evaluation trajectories (`60` AIME problems $\times$ `n=30`) at Step 0 and Step 25.
  - `${RUN_DIR}/rollout_audit/{1..25}.jsonl`: Token-by-token `response_mask` audit (`8` multi-turn sessions/step, `200` sessions in `G2`) verifying `mask_length_matches=True` and asserting `0` `<tool_response>` tokens leak into `response_mask == 1`.
  - `${RUN_DIR}/env/`: Node precheck outputs (`*-precheck.json`, `*-g0-verify.json`, `*-nvidia-smi-L.txt`, `*-topo.txt`, `*-sandbox-4case.json`).
- **Archived Evidence & One-Command Independent Recomputation (`G2`)**:
  All raw `G2` logs, per-step metrics, and `3,600` validation sample records (`output_tail_300` + `output_sha256`) are checked into [`examples/tpu/retool/gb200/logs/g2-retool-rl-25step-01/`](logs/g2-retool-rl-25step-01/). Run:
  ```bash
  python3 examples/tpu/retool/gb200/logs/g2-retool-rl-25step-01/recompute_g2_metrics.py
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
| **GB200 `G1` (`g1-retool-rl-smoke-03`, `@sha256:a139e2b4...` + `64ef634b`)** | **2 nodes × 4 GB200 (`8 GPUs`)** | `4 GPU (FSDP2=4)` + `4 GPU (TP=4)` | `8 × 4 = 32` (`3` steps) | `35.94 s` (`6.42 s` on warm Step 3) | `4,304 tok/s` (`538.0 tok/s/GPU` on Step 3) | Smoke only (`VAL_BEFORE_TRAIN=False`) | Smoke only (`VAL_BEFORE_TRAIN=False`) | **PASS** |
| **GB200 `G1` (`g1-retool-rl-smoke-04`, `@sha256:087366ae...` + `a2cbde47`)** | **2 nodes × 4 GB200 (`8 GPUs`)** | `4 GPU (FSDP2=4)` + `4 GPU (TP=4)` | `8 × 4 = 32` (`2` steps) | `49.94 s` (Steps 1–2 `flex_attention` autotune; `update_weights=3.85s / 4.02s`, `ppl_ratio=1.0155 / 1.0108`) | Smoke verification of new image (`--check --url`, `--expect-composite-sha256`, `TENSORBOARD_DIR`) | Smoke only (`VAL_BEFORE_TRAIN=False`) | Smoke only (`VAL_BEFORE_TRAIN=False`) | **PASS** |
| **GB200 `G2` (`g2-retool-rl-25step-01`)** | **2 nodes × 4 GB200 (`8 GPUs`)** | `4 GPU (FSDP2=4)` + `4 GPU (TP=4)` | `64 × 8 = 512` (`25` steps) | **`39.84 s`** (Steps 2–25) / **`38.59 s`** (Steps 3–25) / **`18.55–24.63 s`** (overlapped) | **`14,362.0 tok/s`** (`1,795.2 tok/s/GPU`, Steps 3–25); peak **`28,659.7 tok/s`** (`3,582.5 tok/s/GPU`, Step 25) | **`23.78% -> 28.78%` (`+5.00 pp`)** (`strict=True`: `24.22% -> 29.67%`, `+5.44 pp`) | **`17.00% -> 17.89%` (`+0.89 pp`)** (`strict=True`: `17.11% -> 18.00%`, `+0.89 pp`) | **PASS** |
| **TPU v7x Tier A (Iso-8-Chip, Planned)** | **2 × `2×2×1` (`8 v7x chips` / 16 cores)** | `4 chips (FSDP2=8, CP=1)` + `4 chips (TP=8, EP=8)` | `64 × 8 = 512` (`25` steps) | TBD ([`ISO_CHIP_PERF_BENCHMARK_PLAN.md`](../ISO_CHIP_PERF_BENCHMARK_PLAN.md)) | TBD | TBD | TBD | Planned |
| **GB200 Tier B (Iso-32-GPU, Planned)** | **8 nodes × 4 GB200 (`32 GPUs`)** | `16 GPU (FSDP2=16)` + `16 GPU (4 × TP=4)` | `64 × 8 = 512` (`25` steps) | TBD ([`ISO_CHIP_PERF_BENCHMARK_PLAN.md`](../ISO_CHIP_PERF_BENCHMARK_PLAN.md)) | TBD | TBD | TBD | Planned |
| **GB200 64-GPU Strong Scaling (`G64-A..D`, Planned)** | **16 nodes × 4 GB200 (`64 GPUs`)** | `G64-A`: `32 GPU (FSDP2=32)` + `32 GPU (8 × TP=4)`<br>`G64-D`: `16 GPU (FSDP2=16)` + `48 GPU (12 × TP=4)` | Profile 1: `64 × 8 = 512`<br>Profile 2: `64 × 16 = 1,024` (`25` steps) | TBD | TBD | TBD | TBD | Planned (`NUM_NODES=16` supported in launcher) |
| **TPU v7x `P3c` / Tier C (`p3c-retool-25s-01`, Iso-64-Chip)** | **4 × `2×2×4` (`64 v7x chips` / 128 cores)** | `16 chips (FSDP2=8, CP=4)` + `48 chips (3 × TP=32, EP=32)` | Profile 2: `64 × 16 = 1,024` (`25` steps, 16K thinking, shaped reward) | TBD (32-chip early partial smoke verified in PR #15; Steps 0–3 64-chip gate pending) | TBD | TBD | TBD | Scheduled on `r-b0f80` (`64` chips) |
| **TPU v7x `P3` / Tier B (Iso-32-Chip)** | **2 × `2×2×4` (`32 v7x chips` / 64 cores)** | `16 chips (FSDP2=8, CP=4)` + `16 chips (TP=32, EP=32)` | `64 × 8 = 512` (`25` steps) | See [`REPRO_RESULTS_TPU7X.md`](../REPRO_RESULTS_TPU7X.md) | See results | See results | See results | 25 steps completed; compare only after matching scorer and measurement windows |
| **Askhat TPU v6e Baseline ([`#108`](https://github.com/jialei777/verl-upstream/pull/108))** | **2 × `v6e-32` (`64 v6e chips`)** | `32 chips (FSDP2=8, CP=4)` + `32 chips (TP=32, EP=32)` | `64 × 8 = 512` (`yjowd03u` Steps 0–15 + `ci0k54pl` Steps 15–25) | `108.1 s` | - | PR #108 reports `22.44%` (Step 0) and `30.00%` (Step 25) from two separate runs (not a single continuous run; audited `mean@8`: `yjowd03u` `26.25% -> 29.58%` Steps 0–15, `ci0k54pl` `29.17% -> 33.75%` Steps 15–25) | PR #108 reports `14.17%` and `18.33%` from two separate runs (`mean@8`: `yjowd03u` `14.58% -> 20.42%`, `ci0k54pl` `20.00% -> 22.92%`) | Reference |

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

- **End-to-End Wall-Clock Milestones ([`head.log`](logs/g2-retool-rl-25step-01/head.log) UTC timestamps)**:
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

### 2.6 32 → 64 GB200 Scaling Plan & Tier C (`64-TPU v7x` vs. `64-GB200`) Alignment

**Status: Multi-node launcher (`NUM_NODES=8/16`) and unit tests implemented; 32/64-GPU cluster execution pending.** This plan covers two complementary evaluations:
1. **GB200 32 → 64 GPU Strong Scaling (Profile 1 / `G2` 8K Baseline Parity)**: Holds the global workload fixed (`BS=64, n=8, 8K response, 8 turns, enable_thinking=False, default_compute_score`) and measures speedup and parallel efficiency from 32 GPUs (`8` nodes) to 64 GPUs (`16` nodes). Comparing 64 GPUs with the existing 32-chip TPU `P3` run is an additional non-iso-resource reference, **not** an iso-accelerator comparison.
2. **Tier C (`64 × TPU v7x` vs. `64 × GB200` Iso-Accelerator & Iso-Split Comparison, Profile 2 / `P3c` 16K Thinking Parity)**: Compares 64 GB200 GPUs directly against our 64-chip TPU v7x run (`p3c-retool-25s-01`, `4 × 2×2×4` slices = `16 Trainer chips + 48 Rollout chips`) under the exact same **`16 Trainer : 48 Rollout` (`1:3`)** accelerator split (`G64-D`) and 16K thinking + shaped-reward (`retool_tpu.compute_score`) recipe. See [`../ISO_CHIP_PERF_BENCHMARK_PLAN.md`](../ISO_CHIP_PERF_BENCHMARK_PLAN.md) for the full matrix and result table.

#### Candidate Parallel Configurations (Verified Against `verl` & TorchTitan Constraints)

| Candidate | Scale (`4 GPU/node`) | Trainer / Rollout Split | Trainer Parallelism (`TorchTitan 0.3.0`) | Rollout Parallelism (`vLLM V1`) | Key Environment Overrides | Rationale & Code Constraints |
| --- | --- | --- | --- | --- | --- | --- |
| **`G32-Base`** | **32 GPUs** (`NUM_NODES=8`) | `4 Trainer (16 GPU)` + `4 Rollout (16 GPU)` (`1:1`) | `FSDP2 (dp_shard)=16, CP=1, TP=1, EP=1` | `4 × (TP=4, EP=1)` | `NUM_NODES=8 JOBENV_NNODES_TRAINER=4 JOBENV_N_GPUS_TRAINER=4 JOBENV_NNODES_ROLLOUT=4 JOBENV_N_GPUS_ROLLOUT=4 JOBENV_DATA_PARALLEL_SHARD_SIZE=16 JOBENV_ROLLOUT_TP=4 JOBENV_ROLLOUT_EP=1` | Tier B (`32 vs 32`) baseline and `1×` reference for 32→64 strong scaling |
| **`G64-A` (1:1 Default)** | **64 GPUs** (`NUM_NODES=16`) | `8 Trainer (32 GPU)` + `8 Rollout (32 GPU)` (`1:1`) | `FSDP2 (dp_shard)=32, CP=1, TP=1, EP=1` | `8 × (TP=4, EP=1)` | `NUM_NODES=16 JOBENV_NNODES_TRAINER=8 JOBENV_N_GPUS_TRAINER=4 JOBENV_NNODES_ROLLOUT=8 JOBENV_N_GPUS_ROLLOUT=4 JOBENV_DATA_PARALLEL_SHARD_SIZE=32 JOBENV_ROLLOUT_TP=4 JOBENV_ROLLOUT_EP=1` | Direct `2×` strong-scaling counterpart to `G32-Base`; each `TP=4` vLLM replica stays within one 4-GPU NVLink5 node |
| **`G64-B` (Rollout EP)** | **64 GPUs** (`NUM_NODES=16`) | `8 Trainer (32 GPU)` + `8 Rollout (32 GPU)` (`1:1`) | `FSDP2 (dp_shard)=32, CP=1, TP=1, EP=1` | `8 × (TP=4, EP=4)` (or `4 × (TP=4, DP=2, EP=8)`) | Single-node replica: `... JOBENV_ROLLOUT_TP=4 JOBENV_ROLLOUT_DP=1 JOBENV_ROLLOUT_EP=4`<br>2-node replica: `... JOBENV_ROLLOUT_TP=4 JOBENV_ROLLOUT_DP=2 JOBENV_ROLLOUT_EP=8` | [`verl/workers/config/rollout.py`](../../../../verl/workers/config/rollout.py#L298-L301) enforces `expert_parallel_size == tensor_model_parallel_size * data_parallel_size` (and `world_size = TP * DP * PP` in [`replica.py`](../../../../verl/workers/rollout/replica.py#L107-L111)), so single-node `TP=4, DP=1` requires `ROLLOUT_EP=4` (setting `ROLLOUT_EP=2` raises `AssertionError`) |
| **`G64-C` (Trainer EP)** | **64 GPUs** (`NUM_NODES=16`) | `8 Trainer (32 GPU)` + `8 Rollout (32 GPU)` (`1:1`) | Dense `FSDP2=32`; MoE Experts `EFSDP=8 × EP=4` | `8 × (TP=4, EP=1)` | `NUM_NODES=16 JOBENV_NNODES_TRAINER=8 JOBENV_N_GPUS_TRAINER=4 JOBENV_NNODES_ROLLOUT=8 JOBENV_N_GPUS_ROLLOUT=4 JOBENV_DATA_PARALLEL_SHARD_SIZE=32 JOBENV_TRAINER_EP=4 JOBENV_ROLLOUT_TP=4 JOBENV_ROLLOUT_EP=1` | In TorchTitan `0.3.0` `ParallelDims` ([`transformer_impl.py`](../../../../verl/workers/engine/torchtitan/transformer_impl.py#L452-L481)), `dp_replicate * dp_shard * cp * tp * pp` must equal `world_size (32)` while `ep=4` sub-divides `dp_shard=32` into `efsdp=8, ep=4` for experts; keep `DATA_PARALLEL_SHARD_SIZE=32` (setting `8` fails mesh validation) |
| **`G64-D` (1:3 Rollout-Heavy Candidate / Iso-Split with 64-Chip TPU v7x)** | **64 GPUs** (`NUM_NODES=16`) | **`4 Trainer (16 GPU)` + `12 Rollout (48 GPU)` (`1:3`)** | `FSDP2 (dp_shard)=16, CP=1, TP=1, EP=1` | **`12 × (TP=4, EP=1)`** (`+50%` rollout replicas vs. `G64-A`) | `NUM_NODES=16 JOBENV_NNODES_TRAINER=4 JOBENV_N_GPUS_TRAINER=4 JOBENV_NNODES_ROLLOUT=12 JOBENV_N_GPUS_ROLLOUT=4 JOBENV_DATA_PARALLEL_SHARD_SIZE=16 JOBENV_ROLLOUT_TP=4 JOBENV_ROLLOUT_EP=1` | Matches the exact `16 Trainer : 48 Rollout` (`1:3`) accelerator split of 64-chip TPU v7x (`1 Trainer Slice + 3 Rollout Slices`) for Tier C iso-split comparison, and tests whether adding rollout replicas improves throughput under 16K thinking (`Profile 2`). *(Note: in `G2` [`g2_25step_metrics.json`](./logs/g2-retool-rl-25step-01/g2_25step_metrics.json) over Steps 1–25, trainer queue-wait `timing_s/gen` was `512.74 s` and `update_actor + old_log_prob` was `463.13 s`; `timing_s/gen` under `separate_async` is queue wait time rather than full rollout wall-clock, so `1:3` vs. `1:1` on GB200 must be evaluated empirically rather than assumed from `G2`.)* |

> **Important Notes on `ROLLOUT_TP` and `EXPECT_GIT_SHA`**:
> 1. **`ROLLOUT_TP`**: In [`run_qwen3_30b_a3b_dapo_torchtitan_gb200.sh`](./run_qwen3_30b_a3b_dapo_torchtitan_gb200.sh), `ROLLOUT_TP` defaults to `TOTAL_ROLLOUT_GPUS`. For multi-node rollout (`16`, `32`, or `48` rollout GPUs), always pass `JOBENV_ROLLOUT_TP=4` explicitly so `verl` instantiates `4`, `8`, or `12` independent `TP=4` vLLM replicas.
> 2. **`EXPECT_GIT_SHA` (Must point to the `gb200-retool-runtime` lineage matching the image's `uv.lock`)**: The prebuilt image `@sha256:087366ae...` was built from commit `a2cbde47ba729d0cba7e881da1692df20d8e05e4` on the `gb200-retool-runtime` branch (whose `uv.lock` SHA-256 is `df508db0b957fa181c229fb75885cd3863cb651fd5233d0fa5af95e990bda0be` and matches all 264 locked packages in `/workspace/verl-pin/.venv`). Because `tpu-main` is based on an earlier upstream `uv.lock` (`d30f7e35...`), checking out a `tpu-main` commit inside the GB200 container would fail `verify_venv_lock.py --expect-lock-sha256 df508db0...`. Instead, pass `EXPECT_GIT_SHA="27c41116acfdd89435f78e0993455bf9338a79fd"` (on `gb200-retool-runtime`, directly on top of `a2cbde47`, with zero diff in `uv.lock` or `pyproject.toml`, containing the V1 validation `num_turns` dump fix in `verl/trainer/ppo/v1/trainer_base.py` for offline shaped-reward reconciliation, `EXCLUDE_NODES` support in `run_ray_jobset_gb200.sh`, and automatic `'wandb'` logger enablement in `run_qwen3_30b_a3b_dapo_torchtitan_gb200.sh` when `WANDB_API_KEY` is present) so every Pod checks out `27c41116...`, verifies both `uv.lock` (`df508db0...`) and `git_sha` via `verify_venv_lock.py --output-json "${RUN_DIR}/env/${LAUNCH_ID}-${HOST_TAG}-precheck.json"`, and passes the cross-node JSON readiness gate.

#### Example 64-GPU Launch Commands (`NUM_NODES=16`)

1. **Profile 1 (Track 1 — `G2` 8K Non-Thinking Strong Scaling on 64 GPUs, `G64-A` `32+32` or `G64-D` `16+48`)**:
   ```bash
   RUN_NAME="g64a-retool-rl-25step-$(date -u +%Y%m%d-%H%M%S)" \
   NUM_NODES="16" \
   GPUS_PER_NODE="4" \
   EXPECT_GIT_SHA="27c41116acfdd89435f78e0993455bf9338a79fd" \
   KUBE_CONTEXT="<YOUR_GB200_KUBE_CONTEXT>" \
   NODEPOOL="<YOUR_4xGB200_NV18_NODEPOOL>" \
   GCS_BUCKET="<YOUR_GCS_BUCKET>" \
   RUN_ROOT="<YOUR_USER>/retool-gb200/runs" \
   INPUTS_ROOT="<YOUR_USER>/retool-v7x/inputs" \
   IMAGE="us-central1-docker.pkg.dev/supercomputer-testing/pirillo-gcr/verl-torchtitan-cu130-arm64@sha256:087366ae58a31888c2af2c25992c33ea77471b1b2c58d221b370899f4721e689" \
   JOBENV_MODEL_SRC_PATH="/data/<YOUR_USER>/retool-v7x/runs/p2-retool-sft-01/checkpoint/global_step_372/huggingface" \
   JOBENV_NNODES_TRAINER="8" \
   JOBENV_N_GPUS_TRAINER="4" \
   JOBENV_NNODES_ROLLOUT="8" \
   JOBENV_N_GPUS_ROLLOUT="4" \
   JOBENV_DATA_PARALLEL_SHARD_SIZE="32" \
   JOBENV_ROLLOUT_TP="4" \
   JOBENV_ROLLOUT_EP="1" \
   JOBENV_ACTOR_LR="1.4e-6" \
   JOBENV_BYPASS_MODE="True" \
   JOBENV_ROLLOUT_IS="token" \
   JOBENV_ROLLOUT_IS_THRESHOLD="2.0" \
   JOBENV_ENABLE_THINKING="False" \
   JOBENV_USE_SHAPED_REWARD="False" \
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

2. **Profile 2 (Track 2 — `P3c` 16K Thinking + Shaped Reward Iso-64-Accelerator & Iso-1:3-Split Candidate vs. 64-Chip TPU v7x `p3c-retool-25s-01`, `G64-D` `16+48`)**:
   > **Parity lock requirement**: Before labeling a 64-GPU run as Profile 2 parity against 64-chip TPU v7x (`p3c-retool-25s-01`), verify that every resolved Hydra parameter (`actor.optim.lr=2e-6`, `bypass_mode=True`, `loss_mode=bypass_mode`, `rollout_is=token`, `rollout_is_threshold=2.0`, `num_warmup_batches=2`, `max_turns=16`, `max_response_length=16384`, `enable_thinking=True`, `reward.custom_reward_function.path=pkg://examples.tpu.retool.retool_tpu`) matches the resolved config of `p3c-retool-25s-01`.
   ```bash
   RUN_NAME="g64d-p3c-parity-25s-$(date -u +%Y%m%d-%H%M%S)" \
   NUM_NODES="16" \
   GPUS_PER_NODE="4" \
   EXPECT_GIT_SHA="27c41116acfdd89435f78e0993455bf9338a79fd" \
   KUBE_CONTEXT="<YOUR_GB200_KUBE_CONTEXT>" \
   NODEPOOL="<YOUR_4xGB200_NV18_NODEPOOL>" \
   GCS_BUCKET="<YOUR_GCS_BUCKET>" \
   RUN_ROOT="<YOUR_USER>/retool-gb200/runs" \
   INPUTS_ROOT="<YOUR_USER>/retool-v7x/inputs" \
   IMAGE="us-central1-docker.pkg.dev/supercomputer-testing/pirillo-gcr/verl-torchtitan-cu130-arm64@sha256:087366ae58a31888c2af2c25992c33ea77471b1b2c58d221b370899f4721e689" \
   WANDB_SECRET_NAME="<YOUR_WANDB_SECRET_NAME>" \
   WANDB_SECRET_KEY="WANDB_API_KEY" \
   JOBENV_LOGGER_LIST="['console','tensorboard','wandb']" \
   JOBENV_WANDB_PROJECT="verl-torchtpu-grpo-test" \
   JOBENV_WANDB_ENTITY="google-trellis" \
   JOBENV_WANDB_MODE="online" \
   JOBENV_MODEL_SRC_PATH="/data/<YOUR_USER>/retool-v7x/runs/p2-retool-sft-01/checkpoint/global_step_372/huggingface" \
   JOBENV_NNODES_TRAINER="4" \
   JOBENV_N_GPUS_TRAINER="4" \
   JOBENV_NNODES_ROLLOUT="12" \
   JOBENV_N_GPUS_ROLLOUT="4" \
   JOBENV_DATA_PARALLEL_SHARD_SIZE="16" \
   JOBENV_ROLLOUT_TP="4" \
   JOBENV_ROLLOUT_EP="1" \
   JOBENV_MAX_PROMPT_LENGTH="2048" \
   JOBENV_MAX_RESPONSE_LENGTH="16384" \
   JOBENV_ACTOR_MAX_TOKEN_LEN_PER_GPU="18432" \
   JOBENV_LOG_PROB_MAX_TOKEN_LEN_PER_GPU="18432" \
   JOBENV_TORCHTITAN_MAX_SEQ_LEN="18432" \
   JOBENV_MAX_TURNS="16" \
   JOBENV_N_RESP_PER_PROMPT="16" \
   JOBENV_N_RESP_PER_PROMPT_VAL="30" \
   JOBENV_ACTOR_LR="2e-6" \
   JOBENV_CLIP_GRAD="0.088" \
   JOBENV_ENABLE_THINKING="True" \
   JOBENV_BYPASS_MODE="True" \
   JOBENV_ROLLOUT_IS="token" \
   JOBENV_ROLLOUT_IS_THRESHOLD="2.0" \
   JOBENV_USE_SHAPED_REWARD="True" \
   JOBENV_TRAIN_BATCH_SIZE="64" \
   JOBENV_PPO_MINI_BATCH_SIZE="64" \
   JOBENV_NUM_WARMUP_BATCHES="2" \
   JOBENV_TOTAL_TRAINING_STEPS="25" \
   JOBENV_VAL_BEFORE_TRAIN="True" \
   JOBENV_TEST_FREQ="25" \
   JOBENV_SAVE_FREQ="5" \
   bash examples/tpu/retool/gb200/run_ray_jobset_gb200.sh
   ```

**Validation and execution sequence:**

1. Verify `ComputeDomain` / DRA domain limits on the target 8- or 16-node GKE nodepool: [`run_ray_jobset_gb200.sh`](./run_ray_jobset_gb200.sh) creates a single `ComputeDomain` (`numNodes: ${COMPUTE_DOMAIN_NUM_NODES}`, default `NUM_NODES`) whose `resourceClaimTemplate` is attached by all `${NUM_NODES}` Pods. Simply setting `COMPUTE_DOMAIN_NUM_NODES < NUM_NODES` on a single `ComputeDomain` while all `${NUM_NODES}` Pods still claim it is **not** a validated multi-domain solution; if 8 or 16 nodes span multiple NVLink domains that cannot join one `ComputeDomain`, per-domain `ComputeDomain` partitioning or non-IMEX cross-rack fallback must be validated on the target cluster first.
2. [`run_ray_jobset_gb200.sh`](./run_ray_jobset_gb200.sh) sets Pod `nvidia.com/gpu` requests/limits to `GPUS_PER_NODE` (`1..4`, default `4`), verifies that `nvidia-smi -L` reports exactly `GPUS_PER_NODE` GPUs on every node, runs [`verify_venv_lock.py`](./verify_venv_lock.py) (`${LAUNCH_ID}-${HOST_TAG}-precheck.json`) and [`verify_g0_gb200.py`](./verify_g0_gb200.py) (`${LAUNCH_ID}-${HOST_TAG}-g0-verify.json`) on **every** node (`head` + `worker-0 .. worker-$((NUM_NODES-2))`), and blocks training until all `${NUM_NODES}` nodes match `EXPECT_GIT_SHA`, `EXPECT_CHECKPOINT_SHA256`, and `ray status` (`${TOTAL_GPUS}.0 GPU`).
3. Run a smoke validation run at 64 GPUs first to verify resolved Hydra config parity, finite gradients, actual optimizer updates, `0` `<tool_response>` mask violations, multi-turn sandbox execution, no OOM, and NCCL weight broadcast to all `12` rollout replicas.
4. Run the 25-step benchmark (with Step 0 and Step 25 `n=30` AIME 2024/2025 evaluations) under Profile 1 (for 8→32→64 GPU strong scaling) and Profile 2 (for 64-GPU GB200 vs. 64-chip TPU v7x `p3c-retool-25s-01` iso-accelerator comparison), and archive all metrics and wall-clock windows (`W1`, `W2`, evaluation, and end-to-end).

---

### 64-GPU (`G64-D`, `16` Nodes = `4` Trainer + `12` Rollout) Results: 25-Step Run (`g64-retool-25s-01`) and 5-Step Smoke (`g64-retool-smoke-04`)

All tables below are generated by [`logs/gen_g64_report_tables.py`](./logs/gen_g64_report_tables.py) from each run's archived files ([`logs/g64-retool-25s-01/`](./logs/g64-retool-25s-01/), [`logs/g64-retool-smoke-04/`](./logs/g64-retool-smoke-04/)), which [`logs/build_g64_run_evidence.py`](./logs/build_g64_run_evidence.py) builds from the GCS run directories. Per-step metrics come from the `TaskRunnerV1` `step:N` lines in each `head.log` (`2969` / `1009` overlapping values agree with the runs' metrics JSON), wall clock per step and its split from `step_timing_breakdown.json` ([`step_timing_breakdown.py`](../step_timing_breakdown.py) on the archived TensorBoard event file), tool-call counts and the whole-session mask audit from `rollout_and_audit_summary.json` (the audit is recomputed from the archived raw sessions in `rollout_audit/`), evaluation results from the validation samples, and Askhat references from [`askhat_runs_summary.json`](../results/askhat-wandb-audit/askhat_runs_summary.json). [`test_launcher_and_sandbox.py`](./test_launcher_and_sandbox.py) checks that every `AUTO-GENERATED` block matches the generator output.

#### 1. Setup shared by both runs

- **Scheduling and preflight**: both runs used `a4x-highgpu-4g-a4x-pool-5` with `EXCLUDE_NODES=gke-a4x-baker-a4x-highgpu-4g-a4x-pool-aab7d37b-7329` (`GPU 00000009:01:00.0 is lost`) and one `ComputeDomain` (`numNodes: 16`). In each run all 16 `*-precheck.json` / `*-g0-verify.json` reports are `PASS` with `git_sha 27c41116` (the pods `git fetch` this commit from GitHub and check it out; pinned as tag [`run-code/g64-retool-smoke-04-25s-01`](https://github.com/syangx39/verl-upstream/tree/run-code/g64-retool-smoke-04-25s-01)) and one `uv_lock_sha256`, and the 16 `nvidia-smi -L` files list 64 GPUs ([`logs/recompute_g64_metrics.py --run <run>`](./logs/recompute_g64_metrics.py)).
- **Same configuration**: the resolved Hydra configs printed in the two `head.log` files differ only in the run name and output paths, `trainer.test_freq` (`5 → 25`) and `trainer.total_training_steps` (`5 → 25`) (`resolved_config_diff` in the generator). Key settings, at the same line numbers in both logs: `tensor_model_parallel_size: 4` (L583), `data_parallel_shard_size: 16` (L284), `lr: 2e-06` (L211), `loss_mode: bypass_mode` (L224), `rollout_is: token` (L618, trainer-side only; see §4), `max_user_turns: 16` (L524), `enable_thinking: True` (L777), `max_response_length: 16384` (L790), `num_warmup_batches: 2` (L1066), reward `pkg://examples.tpu.retool.retool_tpu` (L776–L780). Each log shows 12 distinct `vLLMHttpServer` processes (`TP=4`).
- **Checkpoint evidence boundary**: every node fully SHA-256-hashed its staged `/dev/shm` copy and matched the pinned composite `59247a00c13f2364a12aab5ab6b88b6d971d59ff643c57aad8ac4e55a71231b0`. The verifier version used in these runs then only size-checked `--ref-dir`, and its `ref_composite_sha256` field echoes the pinned value; it is not an independent hash of the reference copy. [`verify_g0_gb200.py`](./verify_g0_gb200.py) now reports `ref_check_mode="size_only"`, `ref_sha256_verified=false` and no reference hash in that mode, and `--full-ref-hash` hashes the reference files.
- **No checkpoint was saved**: both runs set `trainer.save_freq=-1`, `trainer.resume_mode=disable` and `default_local_dir=/tmp/checkpoint/...` (`head.log` command line), so neither the final Step 25 / Step 5 weights nor any intermediate checkpoint exist.
- **W&B**: 25-step run [`skzy1l0y`](https://wandb.ai/google-trellis/verl-torchtpu-grpo-test/runs/skzy1l0y), smoke [`hv1k9rwo`](https://wandb.ai/google-trellis/verl-torchtpu-grpo-test/runs/hv1k9rwo).

#### 2. 25-step run (`g64-retool-25s-01`)

**Step 0 vs. Step 25 AIME 2024 / 2025 (`n=30`)**

<!-- BEGIN AUTO-GENERATED g64-retool-25s-01/eval_table (gen_g64_report_tables.py) -->
| Metric (`n=30` samples per question, 30 questions) | Step 0 | Step 25 | Δ (Step 25 − Step 0) |
| --- | --- | --- | --- |
| AIME 2024 correct samples | `398 / 900` | `486 / 900` | `+88` |
| AIME 2024 `acc/mean@30` | `44.22%` | `54.00%` | `+9.78 pp` (paired bootstrap 95% CI `[+3.89 pp, +16.22 pp]`) |
| AIME 2024 `acc/maj@30/mean` | `61.97%` | `75.39%` | `+13.42 pp` |
| AIME 2024 `acc/best@30/mean` | `78.07%` | `89.19%` | `+11.13 pp` |
| AIME 2024 shaped `reward/mean@30` | `+0.0291` | `+0.1924` | `+0.1633` |
| AIME 2024 questions improved / worse / tied (per-question correct count) | – | – | `21 / 6 / 3` |
| AIME 2025 correct samples | `315 / 900` | `376 / 900` | `+61` |
| AIME 2025 `acc/mean@30` | `35.00%` | `41.78%` | `+6.78 pp` (paired bootstrap 95% CI `[+1.67 pp, +12.22 pp]`) |
| AIME 2025 `acc/maj@30/mean` | `55.90%` | `62.34%` | `+6.45 pp` |
| AIME 2025 `acc/best@30/mean` | `72.78%` | `79.40%` | `+6.62 pp` |
| AIME 2025 shaped `reward/mean@30` | `-0.1519` | `-0.0381` | `+0.1138` |
| AIME 2025 questions improved / worse / tied (per-question correct count) | – | – | `17 / 6 / 7` |
<!-- END AUTO-GENERATED g64-retool-25s-01/eval_table -->

- Both two-level paired bootstrap 95% intervals exclude zero (AIME 2024 `[+3.89 pp, +16.22 pp]`, AIME 2025 `[+1.67 pp, +12.22 pp]`), and most questions improved (AIME 2024 `21 / 6 / 3` up / down / tied, AIME 2025 `17 / 6 / 7`). The intervals cover evaluation sampling noise on this fixed question set only; this is one training run.
- The same SFT checkpoint scored `44.22% / 35.00%` at Step 0 here and `45.56% / 33.00%` in the smoke run, i.e. about ±2 pp of run-to-run evaluation spread at `n=30`.
- `recompute_g64_metrics.py --run g64-retool-25s-01` rescored all 3,600 Step 0/25 samples with the runtime `retool_tpu.compute_score` (verbatim source, archived `output_tail_300` + `num_turns`): `pred`/`acc`/`score` match `3,600 / 3,600`.

**Comparison with Askhat's v5p Regime 2 runs (16K + thinking + shaped reward, `max_user_turns=16`)**

<!-- BEGIN AUTO-GENERATED g64-retool-25s-01/askhat_table (gen_g64_report_tables.py) -->
| Run | Starts from | eval `n` per question | Step | AIME 2024 `acc` | AIME 2025 `acc` | train `critic/score/mean`, Steps 1–5 mean | Steps 21–25 mean |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `g64-retool-25s-01` (this run, 64× GB200) | SFT `p2-retool-sft-01` step 372 | `30` | 0 | `44.22%` | `35.00%` | – | – |
| `g64-retool-25s-01` (this run, 64× GB200) | SFT `p2-retool-sft-01` step 372 | `30` | 25 | `54.00%` | `41.78%` | `-0.202` | `+0.432` |
| `yjowd03u` (Askhat v5p) | Askhat's SFT checkpoint | `8` | 0 | `48.75%` | `30.83%` | – | – |
| `ci0k54pl` (Askhat v5p) | resumed from `fcze3xlx (step 20 checkpoint)` | `8` | 25 | `58.33%` | `38.75%` | – | `+0.479` |
| `rq1p980n` (Askhat v5p) | Askhat's SFT checkpoint | `8` | 25 | `56.67%` | `46.25%` | `-0.304` | `+0.483` |
| `yrcsp4z6` (Askhat v5p) | Askhat's SFT checkpoint | `8` | 25 | `60.00%` | `46.67%` | `-0.306` | `+0.482` |
<!-- END AUTO-GENERATED g64-retool-25s-01/askhat_table -->

- At Step 25 we are `6.0`–`2.7` pp below Askhat on AIME 2024 and `4.5`–`4.9` pp below the two from-scratch runs on AIME 2025 (`+3.0` pp vs. the resumed `ci0k54pl`). The training reward over Steps 21–25 is `+0.432` vs. `+0.479`–`+0.483`.
- These are not like-for-like: Askhat evaluates `n=8` per question (about 3 pp sampling standard error per number from that alone), Askhat's only Step 0 is from a different run (`yjowd03u`), the SFT checkpoints are not shown to be identical, and `ci0k54pl` resumed from Step 20. Within these limits there is no large gap; a difference of a few points cannot be resolved with these references.

**Steps 1–25 training metrics (generated from `head.log`)**

<!-- BEGIN AUTO-GENERATED g64-retool-25s-01/training_table (gen_g64_report_tables.py) -->
| Step (`head.log` line) | `timing_s/step` | `timing_s/gen` | `timing_s/update_actor` | `timing_s/update_weights` | `perf/throughput` (tok/s/GPU) | `critic/score/mean` | `actor/pg_loss` | `actor/grad_norm` | `response_length/mean` (`max`) | `response_length/clip_ratio` | `num_turns/mean` (`max`) | trajectories with `<tool_response>` (calls) | `score <= -2.0` | audit mask violations |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 (L2511) | `248.77s` | `34.11s` | `209.48s` | `4.01s` | `119.6` | `-0.1200` | `-0.0267` | `0.1767` | `1,557.6` (`4,413`) | `0.10%` | `8.79` (`32`) | `1022 / 1024` (`3,477`) | `0` | `0 / 8` |
| 2 (L2533) | `35.03s` | `0.03s` | `30.14s` | `4.28s` | `1406.4` | `-0.3013` | `-0.0823` | `0.1219` | `2,753.2` (`16,384`) | `0.10%` | `8.79` (`32`) | `1017 / 1024` (`3,475`) | `0` | `0 / 8` |
| 3 (L2545) | `34.47s` | `0.04s` | `24.91s` | `8.96s` | `1381.8` | `-0.2828` | `-0.0316` | `0.1229` | `2,667.1` (`16,384`) | `0.20%` | `8.64` (`32`) | `1016 / 1024` (`3,401`) | `0` | `0 / 8` |
| 4 (L2561) | `27.93s` | `0.04s` | `22.59s` | `4.73s` | `1585.0` | `-0.2601` | `-0.0527` | `0.1278` | `2,452.1` (`16,384`) | `0.10%` | `8.52` (`32`) | `1018 / 1024` (`3,340`) | `0` | `0 / 8` |
| 5 (L2577) | `26.65s` | `0.04s` | `21.37s` | `4.31s` | `1542.1` | `-0.0478` | `-0.0741` | `0.1259` | `2,252.4` (`16,384`) | `0.49%` | `7.95` (`32`) | `1019 / 1024` (`3,044`) | `0` | `0 / 8` |
| 6 (L2606) | `25.36s` | `0.04s` | `20.46s` | `4.30s` | `1591.6` | `-0.0373` | `-0.0909` | `0.1367` | `2,195.1` (`16,384`) | `0.20%` | `8.79` (`32`) | `1017 / 1024` (`3,476`) | `0` | `0 / 8` |
| 7 (L2621) | `27.65s` | `0.03s` | `22.76s` | `4.28s` | `1624.4` | `+0.1388` | `-0.0085` | `0.1236` | `2,496.0` (`16,384`) | `0.10%` | `8.38` (`32`) | `1020 / 1024` (`3,265`) | `0` | `0 / 8` |
| 8 (L2633) | `28.47s` | `0.04s` | `21.76s` | `5.74s` | `1498.3` | `+0.1380` | `-0.1245` | `0.1211` | `2,337.6` (`16,384`) | `0.10%` | `8.58` (`32`) | `1012 / 1024` (`3,371`) | `0` | `0 / 8` |
| 9 (L2652) | `28.89s` | `0.03s` | `22.41s` | `5.89s` | `1518.0` | `+0.2105` | `-0.0939` | `0.1238` | `2,419.6` (`16,384`) | `0.10%` | `8.86` (`32`) | `1015 / 1024` (`3,512`) | `0` | `0 / 8` |
| 10 (L2676) | `27.61s` | `0.04s` | `22.67s` | `4.33s` | `1617.5` | `+0.2861` | `-0.1354` | `0.1228` | `2,472.9` (`14,110`) | `0.10%` | `8.12` (`32`) | `1019 / 1024` (`3,133`) | `0` | `0 / 8` |
| 11 (L2691) | `29.26s` | `0.04s` | `22.98s` | `5.31s` | `1536.8` | `+0.2740` | `-0.0453` | `0.1187` | `2,493.8` (`16,384`) | `0.29%` | `8.22` (`32`) | `1010 / 1024` (`3,184`) | `0` | `0 / 8` |
| 12 (L2708) | `34.86s` | `0.04s` | `24.12s` | `10.12s` | `1348.7` | `+0.2293` | `-0.1551` | `0.1592` | `2,623.0` (`16,384`) | `0.20%` | `8.01` (`32`) | `1013 / 1024` (`3,079`) | `0` | `0 / 8` |
| 13 (L2722) | `26.34s` | `0.04s` | `20.90s` | `4.83s` | `1590.8` | `+0.4393` | `-0.0675` | `0.1550` | `2,309.0` (`16,384`) | `0.10%` | `8.62` (`32`) | `1018 / 1024` (`3,391`) | `0` | `0 / 8` |
| 14 (L2744) | `28.83s` | `0.03s` | `23.90s` | `4.31s` | `1615.7` | `+0.3871` | `-0.0991` | `0.1096` | `2,610.0` (`16,384`) | `0.10%` | `8.20` (`32`) | `1016 / 1024` (`3,176`) | `0` | `0 / 8` |
| 15 (L2762) | `30.58s` | `0.04s` | `25.11s` | `4.39s` | `1584.3` | `+0.4492` | `-0.0496` | `0.1046` | `2,709.0` (`16,384`) | `0.29%` | `8.44` (`32`) | `1017 / 1024` (`3,296`) | `0` | `0 / 8` |
| 16 (L2779) | `28.07s` | `0.04s` | `22.99s` | `4.31s` | `1651.1` | `+0.4584` | `-0.1208` | `0.1106` | `2,589.5` (`14,365`) | `0.10%` | `8.64` (`32`) | `1020 / 1024` (`3,402`) | `0` | `0 / 8` |
| 17 (L2793) | `30.88s` | `0.04s` | `25.89s` | `4.32s` | `1658.8` | `+0.3941` | `-0.1449` | `0.1080` | `2,866.4` (`14,817`) | `0.10%` | `9.01` (`32`) | `1017 / 1024` (`3,588`) | `0` | `0 / 8` |
| 18 (L2815) | `34.78s` | `0.04s` | `27.32s` | `6.41s` | `1492.9` | `+0.1824` | `-0.0906` | `0.1033` | `2,916.9` (`16,384`) | `0.39%` | `8.69` (`32`) | `1012 / 1024` (`3,424`) | `0` | `0 / 8` |
| 19 (L2844) | `32.42s` | `0.04s` | `26.67s` | `5.06s` | `1528.8` | `+0.3364` | `-0.0712` | `0.1070` | `2,780.4` (`16,384`) | `0.20%` | `9.42` (`32`) | `1019 / 1024` (`3,799`) | `0` | `0 / 8` |
| 20 (L2856) | `33.19s` | `0.04s` | `24.76s` | `7.75s` | `1434.6` | `+0.4260` | `-0.1119` | `0.1072` | `2,664.9` (`16,384`) | `0.10%` | `8.81` (`32`) | `1019 / 1024` (`3,489`) | `0` | `0 / 8` |
| 21 (L2871) | `35.04s` | `0.04s` | `26.00s` | `8.35s` | `1429.2` | `+0.3412` | `-0.0329` | `0.1069` | `2,826.8` (`16,384`) | `0.49%` | `8.81` (`32`) | `1011 / 1024` (`3,488`) | `0` | `0 / 8` |
| 22 (L2893) | `33.76s` | `0.04s` | `27.89s` | `5.18s` | `1562.3` | `+0.4788` | `-0.1389` | `0.0918` | `2,980.4` (`16,384`) | `0.10%` | `8.38` (`32`) | `1019 / 1024` (`3,269`) | `0` | `0 / 8` |
| 23 (L2913) | `34.12s` | `0.05s` | `28.99s` | `4.43s` | `1602.5` | `+0.4405` | `-0.0731` | `0.0957` | `3,103.8` (`16,384`) | `0.39%` | `8.19` (`32`) | `1017 / 1024` (`3,171`) | `0` | `0 / 8` |
| 24 (L2923) | `30.88s` | `0.04s` | `25.76s` | `4.33s` | `1620.0` | `+0.4258` | `-0.0470` | `0.1015` | `2,809.6` (`16,384`) | `0.49%` | `8.52` (`32`) | `1016 / 1024` (`3,336`) | `0` | `0 / 8` |
| 25 (L2968) | `33.71s` | `0.04s` | `28.28s` | `4.34s` | `1630.5` | `+0.4716` | `-0.0633` | `0.0911` | `3,113.8` (`16,384`) | `0.49%` | `8.02` (`32`) | `1016 / 1024` (`3,082`) | `0` | `0 / 8` |
| Steps 2–25 mean (24 steps) | `30.78s` (`25.36s`–`35.04s`) | `0.038s` | `24.61s` | `5.43s` | `1543.8` | – | – | – | – | – | – | total `25415 / 25600` (`83,668`) | total `0` | total `0 / 200` |
<!-- END AUTO-GENERATED g64-retool-25s-01/training_table -->

- Mean response length grew from `1,558` to `3,114` tokens with `response_length/clip_ratio <= 0.49%`; `0` trajectories scored `<= -2.0`. Peak trainer memory: `72.99` GB allocated / `112.31` GB reserved.
- **Whole-session token-mask audit** ([`rollout_mask_audit.py`](../rollout_mask_audit.py)), recomputed from the 200 archived raw sessions ([`logs/g64-retool-25s-01/rollout_audit/`](./logs/g64-retool-25s-01/rollout_audit/); sizes and MD5 equal the GCS objects, see `raw_manifest.json`): `0` violations. All `706` `<tool_response>` blocks (`22,963` tokens, tags included) are outside the loss, each of the `706` `mask=0` runs is exactly the injected tool turn (`\n<|im_start|>user\n<tool_response>…</tool_response><|im_end|>\n<|im_start|>assistant\n`), and the `710` complete `<tool_call>` blocks are in the loss. Two sessions contain malformed `<tool_call>` tags written by the model (a duplicated `</tool_call>`; a second `<tool_call>` before the first was closed), all inside the loss. The earlier per-segment substring search could miss a tool-response block whose content and closing tag fall into a `mask=1` segment; [`test_retool_evidence_tools_on_cpu.py`](../../../../tests/special_tpu/test_retool_evidence_tools_on_cpu.py) builds such a session, which only the new audit flags. Scope: 8 sessions per step, 200 of the 25,600 training trajectories (0.78%).

**Wall clock per step (TensorBoard timing)**

<!-- BEGIN AUTO-GENERATED g64-retool-25s-01/timing_table (gen_g64_report_tables.py) -->
| Step | wall clock (Δ of TensorBoard log starts) | `timing_s/step` | of which `gen` | of which `update_actor` | of which `update_weights` | `timing_s/testing` | TensorBoard logging of the previous step (scalars) | rest outside the timers | weight sync done → `logger.log` |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | `363.57s` | `248.77s` | `34.11s` | `209.48s` | `4.01s` | `0.00s` | `113.19s` (259) | `1.61s` | `1.25s` |
| 2 | `78.99s` | `35.03s` | `0.03s` | `30.14s` | `4.28s` | `0.00s` | `41.74s` (98) | `2.21s` | `2.37s` |
| 3 | `80.73s` | `34.47s` | `0.04s` | `24.91s` | `8.96s` | `0.00s` | `43.69s` (98) | `2.57s` | `2.66s` |
| 4 | `71.92s` | `27.93s` | `0.04s` | `22.59s` | `4.74s` | `0.00s` | `41.74s` (98) | `2.24s` | `2.30s` |
| 5 | `67.00s` | `26.65s` | `0.04s` | `21.37s` | `4.31s` | `0.00s` | `38.18s` (98) | `2.17s` | `2.29s` |
| 6 | `69.08s` | `25.36s` | `0.04s` | `20.46s` | `4.30s` | `0.00s` | `41.15s` (98) | `2.56s` | `2.63s` |
| 7 | `68.97s` | `27.65s` | `0.03s` | `22.76s` | `4.28s` | `0.00s` | `38.47s` (98) | `2.84s` | `2.34s` |
| 8 | `74.04s` | `28.47s` | `0.04s` | `21.76s` | `5.74s` | `0.00s` | `43.39s` (98) | `2.18s` | `2.68s` |
| 9 | `65.28s` | `28.89s` | `0.03s` | `22.41s` | `5.89s` | `0.00s` | `33.77s` (98) | `2.61s` | `2.72s` |
| 10 | `68.39s` | `27.61s` | `0.04s` | `22.67s` | `4.33s` | `0.00s` | `38.74s` (98) | `2.04s` | `2.19s` |
| 11 | `72.53s` | `29.26s` | `0.04s` | `22.98s` | `5.30s` | `0.00s` | `41.06s` (98) | `2.20s` | `2.84s` |
| 12 | `76.67s` | `34.86s` | `0.04s` | `24.12s` | `10.12s` | `0.00s` | `39.60s` (98) | `2.21s` | `2.35s` |
| 13 | `68.32s` | `26.34s` | `0.04s` | `20.90s` | `4.83s` | `0.00s` | `39.45s` (98) | `2.54s` | `2.68s` |
| 14 | `70.75s` | `28.83s` | `0.03s` | `23.90s` | `4.31s` | `0.00s` | `39.62s` (98) | `2.31s` | `2.36s` |
| 15 | `71.00s` | `30.58s` | `0.04s` | `25.11s` | `4.39s` | `0.00s` | `38.01s` (98) | `2.41s` | `2.60s` |
| 16 | `69.66s` | `28.07s` | `0.04s` | `22.99s` | `4.31s` | `0.00s` | `39.02s` (98) | `2.57s` | `2.65s` |
| 17 | `70.85s` | `30.88s` | `0.04s` | `25.89s` | `4.32s` | `0.00s` | `37.54s` (98) | `2.42s` | `2.58s` |
| 18 | `75.82s` | `34.78s` | `0.04s` | `27.32s` | `6.41s` | `0.00s` | `38.32s` (98) | `2.72s` | `2.55s` |
| 19 | `73.12s` | `32.42s` | `0.04s` | `26.68s` | `5.06s` | `0.00s` | `37.89s` (98) | `2.81s` | `2.53s` |
| 20 | `82.48s` | `33.19s` | `0.04s` | `24.76s` | `7.75s` | `0.00s` | `46.59s` (98) | `2.70s` | `3.18s` |
| 21 | `72.95s` | `35.04s` | `0.04s` | `26.00s` | `8.35s` | `0.00s` | `35.45s` (98) | `2.46s` | `2.56s` |
| 22 | `85.55s` | `33.76s` | `0.04s` | `27.89s` | `5.18s` | `0.00s` | `47.81s` (98) | `3.98s` | `2.58s` |
| 23 | `75.03s` | `34.12s` | `0.04s` | `29.00s` | `4.43s` | `0.00s` | `37.21s` (98) | `3.70s` | `2.88s` |
| 24 | `73.81s` | `30.88s` | `0.04s` | `25.77s` | `4.33s` | `0.00s` | `40.54s` (98) | `2.40s` | `2.51s` |
| 25 | `201.72s` | `33.71s` | `0.04s` | `28.28s` | `4.34s` | `126.53s` | `39.08s` (98) | `2.39s` | `128.98s` |
| Steps 2–24 mean (23 steps) | `73.17s` | `30.66s` | `0.038s` | `24.45s` | `5.47s` | `0.00s` | `39.96s` (98) | `2.56s` | `2.57s` |
<!-- END AUTO-GENERATED g64-retool-25s-01/timing_table -->

- `timing_s/step` averages `30.78s` over Steps 2–25 (`30.66s` over Steps 2–24). It covers only the `step` timer of `RayPPOTrainer.fit`: sampling the batch (waiting for rollouts, `timing_s/gen`), `old_log_prob`, `adv`, `update_actor` and `update_weights`. Validation, `_compute_metrics`, the rollout / audit dump preparation (TransferQueue fetch and decode; the file writes run on a background thread), `tq.kv_clear` and `logger.log` (W&B, then one TensorBoard `add_scalar` per metric, then the console line) run outside it ([`trainer_base.py` L474–L540 at the run commit `27c41116`](https://github.com/syangx39/verl-upstream/blob/27c41116acfdd89435f78e0993455bf9338a79fd/verl/trainer/ppo/v1/trainer_base.py#L474-L540)).
- TensorBoard records the wall time of every `add_scalar`, so the archived event file ([`logs/g64-retool-25s-01/tensorboard/`](./logs/g64-retool-25s-01/tensorboard/)) times each step independently of the trainer's timers; the first scalar of each step matches the W&B `_timestamp` to within 0.01 s (checked with the W&B API). Over Steps 2–24 the wall clock averages `73.17s` = `30.66s` `timing_s/step` + `39.96s` TensorBoard logging of the previous step (98 scalars, about `0.41s` each) + `2.56s` rest. The driver log gives `2.57s` from `update_weights done` to `logger.log`, which matches the rest, so the gap between wall clock and `timing_s/step` is the TensorBoard logging; it is not the rollout / audit GCS writes, which run on the dump thread.
- Progress-bar (tqdm) intervals are not per-step training times: tqdm updates only after each step's TensorBoard logging, and the Steps 2–25 progress-bar mean (`83.7s`) includes Step 25, whose interval (`328s`) contains the `126.53s` final validation and the `165s` TensorBoard logging of Step 25's 358 scalars. By the TensorBoard clock Step 25 took `201.72s` = `33.71s` step + `126.53s` validation + `39.08s` TensorBoard logging of Step 24 + `2.39s`.
- Likely cause of about `0.41s` per scalar (not tested separately): without TensorFlow installed, TensorBoard's `tensorflow_stub` `GFile` opens the event file in append mode, writes one record and closes it, for every record; `TENSORBOARD_DIR` is on GCS Fuse (`/data/...`), where each close uploads the whole file again. The TPU v7x run shows the same `0.41s` per scalar. To confirm, run 3 steps with `TENSORBOARD_DIR` on local disk (copied at the end) or without `tensorboard` in `LOGGER_LIST`.
- `timing_s/gen ≈ 0.04s` means the replay buffer always had a batch ready. With about 40 s of TensorBoard logging per step the rollout also had that much extra time, so this shows the rollout keeps up with a ~73 s step, not that it would keep up with a ~33 s step. The TPU v7x comparison of these numbers is in PR #18 (`RETOOL_REWARD_CLIP_RERUN_PLAN_ZH.md` §8.5.6).
- The 25,600 training trajectories are aggregated in `rollout_and_audit_summary.json`; they were not rescored per sample (the V1 training rollout dump has no `num_turns`).

#### 3. 5-step smoke run (`g64-retool-smoke-04`, preliminary)

<!-- BEGIN AUTO-GENERATED g64-retool-smoke-04/eval_table (gen_g64_report_tables.py) -->
| Metric (`n=30` samples per question, 30 questions) | Step 0 | Step 5 | Δ (Step 5 − Step 0) |
| --- | --- | --- | --- |
| AIME 2024 correct samples | `410 / 900` | `416 / 900` | `+6` |
| AIME 2024 `acc/mean@30` | `45.56%` | `46.22%` | `+0.67 pp` (paired bootstrap 95% CI `[-3.67 pp, +5.00 pp]`) |
| AIME 2024 `acc/maj@30/mean` | `64.81%` | `66.62%` | `+1.81 pp` |
| AIME 2024 `acc/best@30/mean` | `84.99%` | `84.61%` | `-0.38 pp` |
| AIME 2024 shaped `reward/mean@30` | `+0.0504` | `+0.0651` | `+0.0147` |
| AIME 2024 questions improved / worse / tied (per-question correct count) | – | – | `11 / 13 / 6` |
| AIME 2025 correct samples | `297 / 900` | `315 / 900` | `+18` |
| AIME 2025 `acc/mean@30` | `33.00%` | `35.00%` | `+2.00 pp` (paired bootstrap 95% CI `[-3.00 pp, +7.44 pp]`) |
| AIME 2025 `acc/maj@30/mean` | `54.37%` | `54.42%` | `+0.06 pp` |
| AIME 2025 `acc/best@30/mean` | `72.45%` | `71.40%` | `-1.05 pp` |
| AIME 2025 shaped `reward/mean@30` | `-0.1869` | `-0.1487` | `+0.0382` |
| AIME 2025 questions improved / worse / tied (per-question correct count) | – | – | `13 / 10 / 7` |
<!-- END AUTO-GENERATED g64-retool-smoke-04/eval_table -->

- A small increase in one run whose intervals include zero (AIME 2024 `[-3.67 pp, +5.00 pp]`, AIME 2025 `[-3.00 pp, +7.44 pp]`); the 25-step run above supersedes it.

<!-- BEGIN AUTO-GENERATED g64-retool-smoke-04/training_table (gen_g64_report_tables.py) -->
| Step (`head.log` line) | `timing_s/step` | `timing_s/gen` | `timing_s/update_actor` | `timing_s/update_weights` | `perf/throughput` (tok/s/GPU) | `critic/score/mean` | `actor/pg_loss` | `actor/grad_norm` | `response_length/mean` (`max`) | `response_length/clip_ratio` | `num_turns/mean` (`max`) | trajectories with `<tool_response>` (calls) | `score <= -2.0` | audit mask violations |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 (L2456) | `148.45s` | `34.11s` | `109.18s` | `3.99s` | `196.5` | `-0.2304` | `-0.0497` | `0.1716` | `1,512.5` (`3,782`) | `0.10%` | `8.37` (`32`) | `1020 / 1024` (`3,259`) | `0` | `0 / 8` |
| 2 (L2469) | `30.20s` | `0.03s` | `25.29s` | `4.25s` | `1579.9` | `-0.2464` | `-0.0947` | `0.1268` | `2,666.1` (`15,193`) | `0.10%` | `9.20` (`32`) | `1016 / 1024` (`3,684`) | `0` | `0 / 8` |
| 3 (L2498) | `32.15s` | `0.04s` | `24.49s` | `7.01s` | `1466.0` | `-0.2413` | `-0.0946` | `0.1225` | `2,633.4` (`16,384`) | `0.29%` | `8.33` (`32`) | `1019 / 1024` (`3,242`) | `0` | `0 / 8` |
| 4 (L2513) | `27.50s` | `0.04s` | `22.64s` | `4.23s` | `1610.4` | `-0.2223` | `-0.0779` | `0.1245` | `2,452.9` (`16,384`) | `0.29%` | `8.27` (`32`) | `1019 / 1024` (`3,208`) | `0` | `0 / 8` |
| 5 (L2544) | `25.64s` | `0.04s` | `20.43s` | `4.21s` | `1531.9` | `-0.0080` | `-0.1023` | `0.1441` | `2,138.6` (`16,384`) | `0.10%` | `7.98` (`32`) | `1014 / 1024` (`3,060`) | `0` | `0 / 8` |
| Steps 2–5 mean (4 steps) | `28.87s` (`25.64s`–`32.15s`) | `0.037s` | `23.21s` | `4.92s` | `1547.0` | – | – | – | – | – | – | total `5088 / 5120` (`16,453`) | total `0` | total `0 / 40` |
<!-- END AUTO-GENERATED g64-retool-smoke-04/training_table -->

<!-- BEGIN AUTO-GENERATED g64-retool-smoke-04/timing_table (gen_g64_report_tables.py) -->
| Step | wall clock (Δ of TensorBoard log starts) | `timing_s/step` | of which `gen` | of which `update_actor` | of which `update_weights` | `timing_s/testing` | TensorBoard logging of the previous step (scalars) | rest outside the timers | weight sync done → `logger.log` |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | `268.80s` | `148.45s` | `34.10s` | `109.18s` | `3.99s` | `0.00s` | `118.33s` (259) | `2.02s` | `1.22s` |
| 2 | `71.11s` | `30.20s` | `0.03s` | `25.29s` | `4.25s` | `0.00s` | `38.71s` (98) | `2.21s` | `2.35s` |
| 3 | `73.45s` | `32.15s` | `0.04s` | `24.49s` | `7.01s` | `0.00s` | `38.59s` (98) | `2.71s` | `2.85s` |
| 4 | `67.24s` | `27.50s` | `0.04s` | `22.64s` | `4.23s` | `0.00s` | `37.44s` (98) | `2.31s` | `2.43s` |
| 5 | `181.15s` | `25.64s` | `0.04s` | `20.43s` | `4.21s` | `115.89s` | `37.40s` (98) | `2.22s` | `118.20s` |
| Steps 2–4 mean (3 steps) | `70.60s` | `29.95s` | `0.037s` | `24.14s` | `5.16s` | `0.00s` | `38.25s` (98) | `2.41s` | `2.54s` |
<!-- END AUTO-GENERATED g64-retool-smoke-04/timing_table -->

- Steps 2–5 `timing_s/step` mean `28.87s`; wall clock over Steps 2–4 `70.60s` per step, of which `38.25s` is TensorBoard logging (Step 5 includes the validation). Only 4 steps, so not a steady-state measurement. The whole-session mask audit of the 40 archived raw sessions finds `0` violations.

#### 4. `actor/rollout_corr/rollout_is_*` diagnostics (same pattern on GPU and TPU)

- `rollout_is_ratio_fraction_low ≈ 0.98–0.99` and `rollout_is_mean ≈ 0.005–0.02` contradict `k3_kl ≈ 0.01` if read as token-level statistics (a token with `ρ < 0.5` has `k3 ≥ 0.193`). They are sequence-level: the V1 trainer bypass path ([`trainer_base.py` L1713–L1721](../../../../verl/trainer/ppo/v1/trainer_base.py#L1713-L1721)) does not call [`apply_bypass_mode`](../../../../verl/trainer/ppo/rollout_corr_helper.py#L1109-L1144), so the actor keeps the dataclass default `policy_loss.rollout_correction.rollout_is="sequence"` ([`algorithm.py` L179](../../../../verl/trainer/config/algorithm.py#L179)), and the configured `algorithm.rollout_correction.rollout_is=token` never reaches the actor. With `Σ log ρ ≈ −KL × L` over 1.5K–3K-token responses, sequence weights collapse toward `e^-20`. Running the repository's `compute_rollout_correction_and_rejection_mask` on synthetic data ([`logs/repro_rollout_is_level.py`](./logs/repro_rollout_is_level.py), CPU `torch`; token KL ≈ 0.0094) gives `rollout_is_mean = 1.0004`, `fraction_low = 0` at token level and `0.0112`, `0.969` (equal to the sequence fraction) at sequence level.
- With `loss_type="ppo_clip"` these weights are metrics only (`rollout_is_weights=None` in the loss). TPU 64-chip `p3c-retool-25s-01` Steps 1–9 show the same pattern (`rollout_is_mean 0.0022–0.0130`, `fraction_low 0.989–0.997`, `kl 0.0098–0.0126`). So these numbers are not evidence of a GPU-specific logprob misalignment, but they do not prove alignment either. Follow-ups: propagate `algorithm.rollout_correction` to the V1 actor (separate PR covering both platforms), and run a fixed-batch GPU/TPU logprob comparison on identical tokens and SFT weights.

<details><summary>Per-step <code>actor/rollout_corr/*</code> values, 25-step run</summary>

<!-- BEGIN AUTO-GENERATED g64-retool-25s-01/rollout_corr_table (gen_g64_report_tables.py) -->
| Step | `actor/rollout_corr/kl` | `actor/rollout_corr/k3_kl` | `actor/rollout_corr/ppl_ratio` | `actor/rollout_corr/chi2_token` | `actor/rollout_corr/rollout_is_mean` | `actor/rollout_corr/rollout_is_ratio_fraction_low` | `actor/rollout_corr/rollout_is_seq_fraction_low` | `actor/rollout_corr/rollout_is_max` | `actor/rollout_corr/rollout_is_min` |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | `0.0115` | `0.0118` | `1.0127` | `0.0691` | `0.0101` | `0.9903` | `0.9903` | `0.242` | `6.84e-11` |
| 2 | `0.0093` | `0.0093` | `1.0116` | `0.0331` | `0.0072` | `0.9953` | `0.9953` | `0.061` | `1.64e-04` |
| 3 | `0.0094` | `0.0093` | `1.0124` | `0.0350` | `0.0033` | `0.9974` | `0.9974` | `0.073` | `6.97e-12` |
| 4 | `0.0098` | `0.0098` | `1.0124` | `0.0386` | `0.0031` | `0.9966` | `0.9966` | `0.095` | `3.18e-11` |
| 5 | `0.0094` | `0.0096` | `1.0116` | `0.0646` | `0.0127` | `0.9863` | `0.9863` | `0.519` | `3.61e-11` |
| 6 | `0.0097` | `0.0100` | `1.0121` | `0.0716` | `0.0094` | `0.9918` | `0.9918` | `16.717` | `1.77e-05` |
| 7 | `0.0091` | `0.0094` | `1.0112` | `0.0880` | `0.0106` | `0.9884` | `0.9884` | `0.051` | `6.45e-03` |
| 8 | `0.0099` | `0.0098` | `1.0130` | `0.0459` | `0.0124` | `0.9895` | `0.9895` | `1.245` | `1.56e-06` |
| 9 | `0.0095` | `0.0093` | `1.0114` | `0.0420` | `0.0088` | `0.9883` | `0.9883` | `1.922` | `3.39e-10` |
| 10 | `0.0086` | `0.0086` | `1.0109` | `0.0309` | `0.0116` | `0.9903` | `0.9903` | `0.434` | `3.87e-11` |
| 11 | `0.0090` | `0.0090` | `1.0111` | `0.0318` | `0.0119` | `0.9856` | `0.9856` | `1.255` | `1.42e-09` |
| 12 | `0.0080` | `0.0090` | `1.0097` | `2.3181` | `0.0054` | `0.9907` | `0.9907` | `0.193` | `3.13e-10` |
| 13 | `0.0087` | `0.0089` | `1.0106` | `0.0332` | `0.0097` | `0.9909` | `0.9909` | `1.796` | `2.18e-10` |
| 14 | `0.0082` | `0.0084` | `1.0101` | `0.0475` | `0.0068` | `0.9947` | `0.9947` | `0.173` | `1.71e-10` |
| 15 | `0.0076` | `0.0076` | `1.0094` | `0.0290` | `0.0097` | `0.9900` | `0.9900` | `0.286` | `1.06e-08` |
| 16 | `0.0078` | `0.0078` | `1.0093` | `0.0371` | `0.0111` | `0.9870` | `0.9870` | `0.853` | `1.20e-09` |
| 17 | `0.0077` | `0.0077` | `1.0096` | `0.0230` | `0.0058` | `0.9888` | `0.9888` | `0.300` | `1.23e-09` |
| 18 | `0.0077` | `0.0077` | `1.0097` | `0.0267` | `0.0072` | `0.9889` | `0.9889` | `0.799` | `2.88e-10` |
| 19 | `0.0079` | `0.0078` | `1.0101` | `0.0226` | `0.0074` | `0.9877` | `0.9877` | `0.446` | `2.91e-10` |
| 20 | `0.0076` | `0.0079` | `1.0090` | `0.1237` | `0.0090` | `0.9914` | `0.9914` | `0.259` | `1.45e-10` |
| 21 | `0.0076` | `0.0075` | `1.0093` | `0.0240` | `0.0057` | `0.9935` | `0.9935` | `0.288` | `9.49e-10` |
| 22 | `0.0069` | `0.0068` | `1.0084` | `0.0230` | `0.0129` | `0.9854` | `0.9854` | `0.391` | `8.64e-09` |
| 23 | `0.0067` | `0.0066` | `1.0082` | `0.0178` | `0.0143` | `0.9864` | `0.9864` | `0.247` | `5.78e-09` |
| 24 | `0.0070` | `0.0070` | `1.0084` | `0.0231` | `0.0159` | `0.9802` | `0.9802` | `0.720` | `3.65e-08` |
| 25 | `0.0064` | `0.0064` | `1.0078` | `0.0200` | `0.0102` | `0.9857` | `0.9857` | `0.508` | `2.50e-08` |
<!-- END AUTO-GENERATED g64-retool-25s-01/rollout_corr_table -->

</details>

<details><summary>Per-step <code>actor/rollout_corr/*</code> values, 5-step smoke</summary>

<!-- BEGIN AUTO-GENERATED g64-retool-smoke-04/rollout_corr_table (gen_g64_report_tables.py) -->
| Step | `actor/rollout_corr/kl` | `actor/rollout_corr/k3_kl` | `actor/rollout_corr/ppl_ratio` | `actor/rollout_corr/chi2_token` | `actor/rollout_corr/rollout_is_mean` | `actor/rollout_corr/rollout_is_ratio_fraction_low` | `actor/rollout_corr/rollout_is_seq_fraction_low` | `actor/rollout_corr/rollout_is_max` | `actor/rollout_corr/rollout_is_min` |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | `0.0117` | `0.0116` | `1.0127` | `0.0639` | `0.0176` | `0.9844` | `0.9844` | `1.270` | `1.26e-10` |
| 2 | `0.0092` | `0.0090` | `1.0116` | `0.0319` | `0.0048` | `0.9916` | `0.9916` | `0.093` | `4.92e-11` |
| 3 | `0.0097` | `0.0097` | `1.0123` | `0.0348` | `0.0065` | `0.9931` | `0.9931` | `0.140` | `7.94e-09` |
| 4 | `0.0097` | `0.0095` | `1.0121` | `0.0275` | `0.0057` | `0.9924` | `0.9924` | `0.282` | `1.96e-11` |
| 5 | `0.0101` | `0.0100` | `1.0128` | `0.0327` | `0.0081` | `0.9897` | `0.9897` | `0.170` | `7.37e-11` |
<!-- END AUTO-GENERATED g64-retool-smoke-04/rollout_corr_table -->

</details>

#### 5. Shared-code change

- [`verl/utils/distributed.py`](../../../../verl/utils/distributed.py) keeps the `tpu-main` TPU branches (explicit `import torch_tpu` before `init_process_group`, TPU skip in `set_numa_affinity`); the new IPv4 Gloo/NCCL interface setup is a no-op on TPU.
