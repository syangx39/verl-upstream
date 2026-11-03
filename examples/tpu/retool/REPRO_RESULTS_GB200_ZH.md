# ReTool（Qwen3-30B-A3B）：NVIDIA GB200 复现进展、结果与 TPU v7x 对照记录

状态：**GB200 全部阶段已完成（Stage 0.1–0.3 / G0 / G1 / G2 25-Step 均 PASS，2026-11-03 更新）**
对应计划：[`REPRO_PLAN_GB200_ZH.md`](./REPRO_PLAN_GB200_ZH.md)（PR #8 已合入 `tpu-main` `10f16720`）
对照 TPU 基线：[`REPRO_RESULTS_TPU7X.md`](https://github.com/syangx39/verl-upstream/blob/retool-tpu7x-exp-results/examples/tpu/retool/REPRO_RESULTS_TPU7X.md)（PR #7）
负责人：@syangx39

---

## 1. 阶段执行总览（Execution Log）

| 阶段 | 任务 / Run 名称 | 代码 / 环境版本 | 硬件规模（GB200） | 核心目标 | 耗时 | 状态 |
| --- | --- | --- | --- | --- | --- | --- |
| **Stage 0.1** | 上游依赖环境验证镜像构建 | `verl@3fdf2b548` + `uv.lock` (`df508db0...`) | 原生 `linux/arm64` 构建节点 | 基于 `verlai/verl:uv.cu130.dev5` (`sha256:6d7c328b...`) 构建 `--extra fsdp --extra vllm --extra torchtitan` 锁定环境 | `uv sync` 12 s（热）/ 103 s（冷）；完整构建+快照 202 s | **DONE**（`sha256:ec7c9274...`） |
| **Stage 0.2** | 4×GB200 目标节点硬件与包锁预检 | `verify_venv_lock.py --require-gb200 --min-gpus 4` | 1 节点 × 4 张 NVIDIA GB200 (`aarch64`) | 校验 264 个锁包、CUDA 13.0、NCCL 2.29.7、SM 10.0、单卡 184.31 GiB HBM、NVLink 拓扑及 `torchtitan 0.3.0` slotted `MoE.Config` | 28 s | **PASS** |
| **Stage 0.3** | ReTool + TorchTitan `0.3.0` MoE 适配镜像构建 | `gb200-retool-runtime` (`23113cb2`, 基于 `3fdf2b548`) | 原生 `linux/arm64` 构建节点 | 适配 TorchTitan `0.3.0` `MoE.Config`（`load_balance_coeff = None`，消除 `expert_bias_E` 缓冲）、移植 `examples/tpu/retool/` 与 `rollout_audit`、构建正式运行镜像 | 80 s | **DONE**（`sha256:a139e2b4...`） |
| **Stage G0** | 共享 P2 Checkpoint 跨节点缓存 + 20 核心文件 SHA-256 清单 + `debugmodel_moe` 1 步 `to_hf()->from_hf()` 路由单测 + Sandbox 四用例 | [`verify_g0_gb200.py`](./gb200/verify_g0_gb200.py) (`bace5a04`) | 2 节点 × 4 张 GB200（`4 trainer + 4 rollout`） | 跨双节点 `/dev/shm` 拷贝全部 25 文件并对 20 个核心文件（16 分片 + 4 元数据）计算组合 SHA-256、校验 48 层 `30B-A3B` 配置并基于 8 层 `debugmodel_moe` 单测 1 步 `expert_bias_E` 与 `to_hf()->from_hf()` 路由一致性（注：**完整 30B 权重加载与真实 `actor -> NCCL -> vLLM` 同步由 `Stage G1` 验证**） | Head 59.2 s / Worker 72.2 s | **PASS（单元预检）** |
| **Stage G1 (Attempt 1)** | GB200 ReTool 多轮工具调用 2 步 RL Smoke（初次尝试） | `g1-retool-rl-smoke-01` (`bace5a04`) | 2 节点 × 4 张 GB200（`NV16/NV17` 降级链路节点，未挂载 `ComputeDomain`） | 进入 `actor_init_model()`（已打印 `disabled expert-bias load balancing on 48 layers`），在 `torch.distributed.broadcast` 报 `ncclUnhandledCudaError: P2P is disabled between NVLINK connected GPUs 1 and 0` | 110 s | **FAIL（缺 `ComputeDomain` IMEX 通道 + `NV16/NV17` 节点）** |
| **Stage G1 (Attempt 2)** | GB200 ReTool 多轮工具调用 3 步 RL Smoke（对照 TPU `P1c`） | `g1-retool-rl-smoke-02` (`bb704378`) | 2 节点 × 4 张 GB200（健康 `NV18` 全互联节点 + `ComputeDomain` IMEX 通道） | 挂载 `ComputeDomain` + `privileged: true` 并校验 `can_device_access_peer`，完成完整 30B 加载、真实 `actor -> NCCL -> vLLM` 同步（`4.42s / 3.07s / 3.02s / 3.89s`）及 3 步真实更新（`grad_norm: 0.637 -> 0.593 -> 0.864`，`score/mean: -0.625 -> -0.1875 -> +0.3125`，`num_turns/mean: 5.875 / 6.0 / 5.625`） | 冷启动含 SM100 JIT 44 m（其中 3 步训练仅 `108 s`） | **PASS（训练与多轮工具调用通过）** |
| **Stage G1 (Attempt 3)** | GB200 ReTool 多轮工具调用 3 步 RL Smoke + Token 级 `rollout_audit` 落盘 | `g1-retool-rl-smoke-03` (`64ef634b`) | 2 节点 × 4 张 GB200（复用热 SM100 FlashInfer/vLLM 缓存） | 默认开启 `trainer.rollout_data_dir` 与 `+trainer.rollout_audit_dir`，落盘 `rollout/{1,2,3}.jsonl`（共 172 次 `<tool_response>`）与 `rollout_audit/{1,2,3}.jsonl`（24 个审计会话，`mask_length_matches=True`，`<tool_response>` 位于 `mask==1` 违规数 **`0`**） | 7 m 15 s（其中 3 步训练 `107.8 s`） | **PASS（全项门槛通过）** |
| **Stage G2** | GB200 ReTool 25 步正式 RL 对照 + Step 0/25 AIME 2024/2025 (`n=30`) 评估（对照 TPU `P3`） | `g2-retool-rl-25step-01` (`64ef634b`) | 2 节点 × 4 张 GB200（`4 trainer FSDP2=4 + 4 rollout TP=4`） | 完成全部 **25/25 步**训练 + Step 0/25 AIME 2024/2025 (`n=30`) 评估（`rc=0`，`SAVE_FREQ=-1` 未保存 Step-25 权重 checkpoint，所有原始日志与评估样本已归档至 [`gb200/evidence/g2-retool-rl-25step-01/`](./gb200/evidence/g2-retool-rl-25step-01/)）：AIME 2024 `mean@30: 23.78% (214/900) -> 28.78% (259/900, +5.00 pp)`、`maj@30: 40.55% -> 47.13% (+6.58 pp)`、`best@30: 64.29% -> 69.05%`；AIME 2025 `mean@30: 17.00% (153/900) -> 17.89% (161/900)`、`best@30: 45.52% -> 57.34% (+11.82 pp)`；25 步共 12,800 条轨迹触发 `24,106` 次 `<tool_response>`，200 个审计会话掩码违规 **`0`**，Step 2–25 均值 **`39.84s/step`**（Step 3–25 稳态均值 **`38.59s/step`**，异步全重叠步 `18.55s–24.63s`） | 25 步训练 `18m 24s`；`tqdm`（25 步 + Step 25 评估）`21m 35s`；**含 Step 0 + Step 25 两次 `n=30` 评估共 `24m 36s`**（`main_ppo` 全程 `27m 37s`） | **PASS（25/25 完成）** |

---

## 2. Stage 0.1 & 0.2：GB200 派生镜像构建与 4×GB200 真机预检结果（PASS）

### 2.1 镜像构建与包锁核验记录

- **基础镜像（`linux/arm64`）**：`verlai/verl:uv.cu130.dev5@sha256:6d7c328b1e7aedd2dece2f939a827704295b13f4bb75f2f17a962a91a19376b9`
- **派生 Dockerfile**：[`examples/tpu/retool/gb200/Dockerfile.gb200.torchtitan`](./gb200/Dockerfile.gb200.torchtitan)
- **上游依赖验证镜像 Digest**：`verl-torchtitan-cu130-arm64@sha256:ec7c927495a06279b128b47424ecd7f84eb70434848e82e46ba26747b5b805f1`
- **关键依赖版本（全部 264 个 `uv.lock` 包 0 偏差通过）**：

| 组件 | 实测安装版本 (`aarch64`) | `uv.lock` 锁定版本 | 与 TPU v7x (`P0.2`) 对照 |
| --- | --- | --- | --- |
| `python` | `3.12.3` (`Linux/aarch64`) | `3.12.x` | `3.12` |
| `torch` | `2.13.0+cu130` (CUDA `13.0`, NCCL `2.29.7`) | `2.13.0+cu130` | `2.13.0+cpu` (`torch-tpu 0.1.1.dev20260912`) |
| `torchtitan` | `0.3.0` (`/workspace/verl-pin/.venv/.../torchtitan`) | `0.3.0` | `0.1.0 @ f880cc73` |
| `torch-checkpointing` | `0.1.0` | `0.1.0` | - |
| `vllm` | `0.32.0` (`VLLM_USE_V1=1`) | `0.32.0` | `0.29.0` (`VLLM_USE_V1=0`) |
| `flash-attn` | `2.8.3` (`import flash_attn` 通过) | `2.8.3` | Splash Attention / Pallas |
| `ray` | `2.55.1` | `2.55.1` | `2.56.0` |
| `transformers` | `5.16.1` | `5.16.1` | `5.16.1` |
| `transferqueue` | `0.1.11` | `0.1.11` | `0.1.8` |

### 2.2 4×GB200 节点硬件拓扑与 `--require-gb200` 实测输出

在单节点 4×GB200（`aarch64`，140 vCPU / 双 NUMA `0-69`、`70-139`，约 `905 GB` 主机内存）上运行宿主机预检：

```text
GPU 0: NVIDIA GB200
GPU 1: NVIDIA GB200
GPU 2: NVIDIA GB200
GPU 3: NVIDIA GB200
	GPU0	GPU1	GPU2	GPU3	NIC0	NIC1	NIC2	NIC3	CPU Affinity	NUMA Affinity
GPU0	 X 	NV17	NV17	NV17	NODE	NODE	SYS	SYS	0-69	0
GPU1	NV17	 X 	NV17	NV17	NODE	NODE	SYS	SYS	0-69	0
GPU2	NV16	NV16	 X 	NV16	SYS	SYS	NODE	NODE	70-139	1
GPU3	NV16	NV16	NV16	 X 	SYS	SYS	NODE	NODE	70-139	1
```

`verify_venv_lock.py --require-gb200 --min-gpus 4` 校验输出：

```text
platform=Linux/aarch64 python=3.12.3
uv.lock SHA256 verified: df508db0b957fa181c229fb75885cd3863cb651fd5233d0fa5af95e990bda0be
git HEAD SHA verified: 3fdf2b548882a8aeccf0842c2150d9c9f1790a55
full uv.lock check: verified 264 packages (mismatches=0)
verl from /workspace/verl-pin/verl/__init__.py
flash_attn 2.8.3 imports OK
PPOTrainerSeparateAsync imports OK
torchtitan 0.3.0 from /workspace/verl-pin/.venv/lib/python3.12/site-packages/torchtitan/__init__.py
torchtitan 0.3.0: resolved torchtitan.components.optimizer.LRSchedulersContainer
torchtitan 0.3.0: resolved torchtitan.models.qwen3.state_dict_adapter.Qwen3StateDictAdapter
torchtitan MoE.Config (torchtitan.models.common.moe): slotted=True required_fields=['routed_experts', 'router'] has_load_balance_coeff=True has_aux_loss=False default_load_balance_coeff=0.001
NOTE: verl.workers.engine.torchtitan.utils:missing-configure_qwen3_moe_for_rl (must disable moe.load_balance_coeff via 0.3-compatible helper before Qwen3-30B-A3B RL)
CUDA runtime: devices=4 nccl=(2, 29, 7)
  cuda:0 name='NVIDIA GB200' capability=10.0 total_memory=184.31 GiB (197.90 GB)
  cuda:1 name='NVIDIA GB200' capability=10.0 total_memory=184.31 GiB (197.90 GB)
  cuda:2 name='NVIDIA GB200' capability=10.0 total_memory=184.31 GiB (197.90 GB)
  cuda:3 name='NVIDIA GB200' capability=10.0 total_memory=184.31 GiB (197.90 GB)
GB200 TARGET-HOST PRECHECK OK (hardware/NCCL/venv verified; run G0 1-step MoE routing + weight-sync test next): arch=aarch64 ray 2.55.1 torch 2.13.0+cu130 (cuda 13.0) vllm 0.32.0 torchtitan 0.3.0 transformers 5.16.1
```

---

## 3. 节点规模需求与实验矩阵（`4 Trainer + 4 Rollout`）

- **单次运行所需节点数**：
  - **GPU 节点：2 个 `4 × GB200` 节点（共 8 张 GB200 GPU）**
    - **Node 0（Trainer 节点，4 × GB200）**：运行 TorchTitan Actor/Trainer（`FSDP=4, TP=1, EP=1, CP=1`），每卡静态 fp32 master + grads + Adam 状态约 `113.7 GiB`（占单卡实测 `184.31 GiB` 的 `61.7%`，留出约 `70.6 GiB / 卡` 给激活与通信 buffer）。
    - **Node 1（Rollout 节点，4 × GB200）**：运行 vLLM Async Rollout Server（`TP=4`，单节点内 `NV16/NV17` NVLink 全互联），每卡 bf16 权重约 `14.2 GiB`。
  - **CPU Head 节点（可选 1 个 CPU 节点，或直接复用 Trainer 节点 `140 vCPU / 905 GB RAM` 作为 Ray Head + SandboxFusion Server）**：每个 `4 × GB200` 节点自带 `140` 核 ARM64 CPU 与 `905 GB` 内存，足够承载 Ray Head 与本地 loopback SandboxFusion 服务。
- **优选同 Subblock 节点池**：优先选择同一 NVLink/RoCE 拓扑 subblock 且空闲节点充足的 `4 × GB200` 节点池（只需要 **2 个节点**）。

---

## 4. Stage 0.3 & Stage G0：共享 TPU P2 SFT Checkpoint 跨节点清单核验、`debugmodel_moe` 路由单测与 Sandbox 四用例实测

> **验证边界说明（针对代码 `bace5a04` 与原始 JSON 产物核实）**：
> [`verify_g0_gb200.py`](./gb200/verify_g0_gb200.py) 是启动 Ray 训练前的**节点级单卡/文件预检脚本**，其验证范围严格限定为以下三项，**不包含**完整 30B 权重的分布式加载或真实 `actor -> NCCL -> vLLM` 跨进程权重同步（后两者在 `Stage G1` 中验证）：
> 1. **`g0_stage_local`**：将共享 TPU `Stage P2` SFT 导出的全部 **25 个文件**（`61,084,185,820 bytes`）并行拷贝到各节点本地 `/dev/shm/models/multiturn-sft-qwen3-30b-a3b-gb200`；
> 2. **`g0_checkpoint_manifest`**：对 `all_names = required_meta + shards` 定义的 **20 个核心文件**（`16` 个 `.safetensors` 权重分片 + `config.json`、`model.safetensors.index.json`、`tokenizer.json`、`tokenizer_config.json`，合计 **`61,079,708,754 bytes`**；其余 5 个辅助文件 `added_tokens.json`、`chat_template.jinja`、`generation_config.json`、`merges.txt`、`special_tokens_map.json`、`vocab.json` 占 `4,477,066 bytes`）逐一计算 SHA-256 并生成组合摘要 `composite_sha256`；
> 3. **`g0_moe_torchtitan_0_3`**：
>    - 在 **48 层 `Qwen3-30B-A3B` 模型配置（`num_layers_30b_a3b = 48`）** 上验证 `configure_qwen3_moe_for_rl` 将全部 48 层 `MoE.Config.load_balance_coeff` 从 `0.001` 设为 `None`；
>    - 在单张 GPU (`cuda:0`) 上实例化 TorchTitan 内置的小型 **8 层 `debugmodel_moe`（`dim=256, n_layers=8, num_experts=8, top_k=2`）**，执行 1 步 `forward/backward/AdamW` 并通过 `Qwen3StateDictAdapter.to_hf() -> from_hf()` 在两个进程内 TorchTitan 模型之间模拟 HF state_dict 往返，验证 `expert_bias_E` 是否被丢弃及引起的路由/logits 偏差。

### 4.1 Stage 0.3：ReTool + TorchTitan `0.3.0` MoE 适配正式运行镜像

- **正式运行镜像 Digest**：`verl-torchtitan-cu130-arm64@sha256:a139e2b491ac23f06f790fe4277afb9bd543a94bc740c92173e5fdb13ddaba40`
- **运行代码 SHA**：`bace5a04`（`G1 Attempt 1`）/ `bb704378`（`G1 Attempt 2`，新增 `ComputeDomain` IMEX 通道与 `can_device_access_peer` 预检）
- **双节点启动时 `--require-rl-moe-helper --require-gb200` 校验输出**：

```text
platform=Linux/aarch64 python=3.12.3
uv.lock SHA256 verified: df508db0b957fa181c229fb75885cd3863cb651fd5233d0fa5af95e990bda0be
full uv.lock check: verified 264 packages (mismatches=0)
torchtitan MoE.Config (torchtitan.models.common.moe): slotted=True required_fields=['routed_experts', 'router'] has_load_balance_coeff=True has_aux_loss=False default_load_balance_coeff=0.001
verl.workers.engine.torchtitan.utils.configure_qwen3_moe_for_rl: present
```

### 4.2 Stage G0.1：`g0_stage_local`（25 文件拷贝）与 `g0_checkpoint_manifest`（20 核心文件 SHA-256）双节点实测数据

以下数据直接提取自双节点原始 JSON 产物（`g1-retool-rl-smoke-01-head-g0-verify.json` 与 `g1-retool-rl-smoke-01-worker-0-g0-verify.json`，代码 SHA `bace5a04`）：

| JSON 字段 | Trainer / Head 节点 (`4 × GB200`) | Rollout / Worker-0 节点 (`4 × GB200`) | TPU v7x `Stage P2` SFT 参照源 | 核验结论 |
| --- | --- | --- | --- | --- |
| `g0_stage_local.total_files` / `copied_files` | `25` / `25` | `25` / `25` | `25` | 完整目录共 25 个文件全部拷入 `/dev/shm` |
| `g0_stage_local.copied_bytes` | `61,084,185,820` (`56.89 GiB`) | `61,084,185,820` (`56.89 GiB`) | `61,084,185,820` (`56.89 GiB`) | 100% 一致 |
| `g0_stage_local.elapsed_s` (`throughput_gb_s`) | `49.091 s` (`1.244 GB/s`) | `62.158 s` (`0.983 GB/s`) | `55.66 s` (`1.097 GB/s`, 8 切片均值) | 正常 |
| `g0_checkpoint_manifest.num_files_hashed` / `num_safetensors_shards` | **`20` / `16`** | **`20` / `16`** | **`20` / `16`** | 校验 16 个 `.safetensors` 分片 + 4 个核心 JSON（`config.json`、`model.safetensors.index.json`、`tokenizer.json`、`tokenizer_config.json`） |
| `g0_checkpoint_manifest.total_bytes`（20 个被哈希文件） | **`61,079,708,754`** (`56.88 GiB`) | **`61,079,708,754`** (`56.88 GiB`) | **`61,079,708,754`** (`56.88 GiB`) | 100% 一致 |
| `g0_checkpoint_manifest.elapsed_s` | `10.099 s` | `10.047 s` | - | 正常 |
| `g0_checkpoint_manifest.composite_sha256` | `59247a00c13f2364a12aab5ab6b88b6d971d59ff643c57aad8ac4e55a71231b0` | `59247a00c13f2364a12aab5ab6b88b6d971d59ff643c57aad8ac4e55a71231b0` | `59247a00c13f2364a12aab5ab6b88b6d971d59ff643c57aad8ac4e55a71231b0` | **`ref_match = true`, `mismatches = []`** |

### 4.3 Stage G0.2：`g0_moe_torchtitan_0_3`（48 层 `30B-A3B` 配置检查 + 8 层小型 `debugmodel_moe` `to_hf()->from_hf()` 往返单测）

1. **48 层 `Qwen3-30B-A3B` 配置级检查（不加载 30B 权重）**：
   - `torchtitan_name = "qwen3"`, `torchtitan_flavor = "30B-A3B"`, `num_layers_30b_a3b = 48`
   - `moe_config_fields = ["param_init", "sharding_config", "num_experts", "routed_experts", "router", "load_balance_coeff", "shared_experts"]`
   - `before_load_balance_coeff_sample = [0.001, 0.001, 0.001]`（全 48 层均为 `0.001`）
   - 调用 `configure_qwen3_moe_for_rl(model_spec.model)` 后：`after_load_balance_coeff_sample = [null, null, null]`（全 48 层均为 `None`）
2. **8 层小型 `debugmodel_moe`（`dim=256, n_layers=8, num_experts=8, top_k=2`）单卡 1 步更新 + `Qwen3StateDictAdapter.to_hf() -> from_hf()` 模拟往返单测**：

| JSON 字段 (`g0_moe_torchtitan_0_3`) | `positive_rl_fix_check` (`apply_rl_fix=true`) | `negative_control_unfixed_check` (`apply_rl_fix=false`) | 物理含义与局限说明 |
| --- | --- | --- | --- |
| 测试模型规格 (`num_moe_layers`) | `8` 层 `debugmodel_moe` (`8` 专家, `top_k=2`) | `8` 层 `debugmodel_moe` (`8` 专家, `top_k=2`) | 进程内小型 MoE 单测；**非**完整 30B-A3B（48 层 128 专家）或真实 NCCL→vLLM 同步 |
| `has_expert_bias_buffer` | `false` | `true` (`layers.{0..7}.moe.expert_bias_E`) | `load_balance_coeff=0.001` 时注册 `expert_bias_E` 缓冲并在 `opt.step()` 后原地更新 |
| `expert_bias_abs_max_after_step1` | `0.0` | `0.0014531250344589353` | 1 步 AdamW 更新后 `expert_bias_E` 即偏离 `0` |
| `dropped_state_keys` (`to_hf() -> from_hf()`) | `[]` | `["layers.0.moe.expert_bias_E", ..., "layers.7.moe.expert_bias_E"]` | `Qwen3StateDictAdapter.to_hf()` 不导出 `expert_bias_E`，导致接收端丢失该 buffer |
| `router_topk_match_rate` | **`1.0`** (`100.0%`) | **`0.66229248046875`** (`66.23%`) | 接收端缺失 `expert_bias_E` 时，仅 1 步即有 `33.77%` top-k 专家选择不一致 |
| `logits_max_abs_diff` | **`0.0`** | **`0.056640625`** | 路由分叉导致前向 logits 不一致 |

### 4.4 Stage G0.3：SandboxFusion 本地 Loopback 服务四用例功能连通性核验与代码执行隔离边界说明

> **代码执行安全与隔离边界说明（重要）**：
> 1. **四用例测试仅验证 HTTP/子进程功能连通性，不构成安全隔离验证**：下表四项测试（正常算术、异常 `stderr`、超时 `killpg` 清理、8 路并发）仅用于验证 `/run_code` 接口契约与 `retool_tpu` 工具调用链路连通性。
> 2. **Pod 权限、失败即报错（Fail-Closed）降权与挂载边界**：在 GB200 多卡训练中，K8s 训练/推理 Pod 因挂载 NVIDIA DRA `ComputeDomain` IMEX 通道（`/dev/nvidia-caps-imex-channels`）必须开启 `securityContext.privileged: true`，且挂载了共享数据卷 `/data`。[`sandbox_fusion_server.py`](./sandbox_fusion_server.py) 采用本地 Python 子进程（`asyncio.create_subprocess_exec`）执行模型生成的代码；当服务以 `root` 运行时，已强制将执行目录 `chown/chmod` 并通过 `os.setgroups([])` / `os.setgid(65534)` / `os.setuid(65534)` 降权至 `nobody:nogroup`（**任何一步降权失败均直接报错拒绝执行，绝不回退为 `root`**），同时注入 `CUDA_VISIBLE_DEVICES=""` 屏蔽 GPU 并限制 `RLIMIT_AS` 与进程组超时强杀。但子进程仍与训练容器共享内核命名空间与文件系统挂载表。
> 3. **适用环境限制与外部沙箱端到端贯通**：[`run_ray_jobset_gb200.sh`](./gb200/run_ray_jobset_gb200.sh) + Pod 内置 loopback `sandbox_fusion_server.py` **严格限定在单租户、受信任基准数据集（DAPO-Math-17k / AIME）与受控复现实验环境中使用**。若用于非受信任提示词/模型或共享环境，可通过 `SANDBOX_FUSION_URL` 或 `SANDBOX_FUSION_HOST` + `SANDBOX_FUSION_PORT` 指向外部独立部署、非 `privileged`、不挂载 `/data` 且具备 `nsjail` / `gVisor` 隔离的沙箱服务——此时 [`run_ray_jobset_gb200.sh`](./gb200/run_ray_jobset_gb200.sh)（Head 与 Worker-0）、[`run_qwen3_30b_a3b_dapo_torchtitan_gb200.sh`](./gb200/run_qwen3_30b_a3b_dapo_torchtitan_gb200.sh) 及 [`sandbox_fusion_tools.py`](./sandbox_fusion_tools.py) 均会自动设置 `SANDBOX_DISABLE_AUTO_START=1`（`local_port=None`），**严禁自动启动任何 Pod 本地服务**，并将外部 URL 贯通至所有 Ray Rollout/Tool Worker。

在 Trainer Head 节点启动 `sandbox_fusion_server.py --host 127.0.0.1 --port 18080 --workers 4` 后，执行四项功能连通性实测（原始产物 `g1-retool-rl-smoke-01-head-sandbox-4case.json` / [`g2-retool-rl-25step-01-head-sandbox-4case.json`](./gb200/evidence/g2-retool-rl-25step-01/env/g2-retool-rl-25step-01-head-sandbox-4case.json)）均通过：

| 用例 | 输入代码摘要 | 期望行为 | 实测输出 (`status` / `stdout` / `stderr`) | 端到端耗时 | 结果 |
| --- | --- | --- | --- | --- | --- |
| **Case 1：正常算术** | `print(sum(i * i for i in range(1, 101)))` | `status=Success`, `stdout=338350` | `status="Success"`, `stdout="338350\n"`, `return_code=0` | `0.0149 s` | **PASS** |
| **Case 2：Python 异常与 `stderr`** | `print(1 / 0)` | `status=Failed`, `stderr` 含 `ZeroDivisionError` | `status="Failed"`, `stderr="...ZeroDivisionError: division by zero\n"` | `0.0155 s` | **PASS** |
| **Case 3：死循环超时与进程清理** | `while True: time.sleep(0.1)` (`run_timeout=1.5`) | 1.5 s 超时返回 `TimeLimitExceeded` 且无残留子进程 | `run_result.status="TimeLimitExceeded"`, `execution_time=1.5015 s` | `1.5051 s` | **PASS** |
| **Case 4：8 路并发请求** | `ThreadPoolExecutor(8)` 并发计算 `1²..8²` | 8 路全部 `Success` 且输出 `['1','4',...,'64']` | `all_succeeded=True`, `outputs=['1','4','9','16','25','36','49','64']` | 最大 `0.0256 s` | **PASS** |

---

## 5. Stage G1：GB200 ReTool 多轮工具调用 2 步 RL Smoke 执行记录

### 5.1 Attempt 1（`g1-retool-rl-smoke-01`，代码 SHA `bace5a04`）—— 失败根因与修复

- **运行进展**：
  1. 双节点（Head + Worker-0）完成 `verify_venv_lock.py`、`verify_g0_gb200.py` 与 `sandbox_fusion_server.py` 四用例检查，成功组建 `2 × 4 = 8 GPU` Ray 集群；
  2. `verl.trainer.main_ppo` 通过配置校验，启动 `WorkerDict`（4 个 Trainer GPU 进程，`FSDP2=4`），日志确认两项关键运行期行为：
     - `WARNING:2026-11-02 21:45:22,098:MoE: disabled expert-bias load balancing on 48 layers [repeated 3x across cluster]`
     - `WARNING:2026-11-02 21:45:22,099:TorchTitan grad clipping uses training.max_norm=1.0; optim.clip_grad=0.088 is not passed to TorchTitan [repeated 3x across cluster]`
  3. 随后在 `torchtitan/trainer.py:337 -> torchtitan/distributed/utils.py:262` 执行首次集合通信 `torch.distributed.broadcast(seed_tensor, src=0)` 时异常退出：
     ```text
     torch.distributed.DistBackendError: NCCL error in: /__w/pytorch/pytorch/torch/csrc/distributed/c10d/NCCLUtils.cpp:92, unhandled cuda error (run with NCCL_DEBUG=INFO for details), NCCL version 2.29.7
     ncclUnhandledCudaError: Call to CUDA function failed.
     Last error:
     P2P is disabled between NVLINK connected GPUs 1 and 0. This should not be the case given their connectivity, and is probably due to a hardware issue. If you still want to proceed, you can set NCCL_IGNORE_DISABLED_P2P=1.
     ```
- **根因定位**：
  1. **缺少 `ComputeDomain` DRA IMEX 通道**：在 GB200（NVLink5 / MNNVL 架构）K8s 节点上，跨 GPU NVLink P2P 访问依赖 `nvidia-dra-driver-gpu` 通过 `ComputeDomain`（`resource.nvidia.com/v1beta1`）向容器注入 `/dev/nvidia-caps-imex-channels` 设备及 `securityContext.privileged: true`；未声明 `ComputeDomain` resourceClaim 时，驱动层禁用 NVLink P2P；
  2. **初始节点池 NVLink5 链路降级**：Attempt 1 所在节点池的 `nvidia-smi topo -m` 显示为 `NV16` / `NV17`（部分 NVLink5 通道降级，而非满规格 18 通道 `NV18`）。
- **修复措施（已提交至 `bb704378`）**：
  1. [`run_ray_jobset_gb200.sh`](./gb200/run_ray_jobset_gb200.sh)：自动创建 `ComputeDomain`（`${RUN_NAME}-cd`，`numNodes: 2`），并为 Head 与 Worker-0 Pod 挂载 `resourceClaims: [{name: compute-domain-channel, resourceClaimTemplateName: ${RUN_NAME}-cd-channel}]` 与 `securityContext: {privileged: true}`；
  2. [`verify_venv_lock.py`](./gb200/verify_venv_lock.py)：在 `--require-gb200` 预检中新增全对（`4 × 3 = 12` 对）`torch.cuda.can_device_access_peer(i, j)` 强制检查，任何一对 GPU P2P 不可达立即在预检阶段 fail-fast 退出；
  3. 切换至全 18 通道健康 `NV18` 节点池启动 Attempt 2（`g1-retool-rl-smoke-02`）。

### 5.2 Attempt 2（`g1-retool-rl-smoke-02`，代码 SHA `bb704378`）—— 完整 30B 加载、真实 `actor -> NCCL -> vLLM` 同步与 3 步多轮 RL 全量实测指标

- **硬件与拓扑核验**：
  - 双节点 `nvidia-smi topo -m` 均为满规格 **`NV18`** 全互联；
  - `verify_venv_lock.py --require-gb200` 输出：`NVLink P2P peer access verified across all 4x3 GPU pairs`；
  - 双节点 `verify_g0_gb200.py` 跨节点清单比对：`composite_sha256 = 59247a00c13f2364a12aab5ab6b88b6d971d59ff643c57aad8ac4e55a71231b0`（100% 匹配）。
- **完整 30B-A3B（48 层 128 专家）模型加载与真实 `actor -> NCCL -> vLLM` 权重同步**：
  - **TorchTitan Actor & Ref（`FSDP2=4`, Head 节点）**：`MoE: disabled expert-bias load balancing on 48 layers`，`Actor use_remove_padding=True`，完成完整 30B-A3B 权重加载与进程组初始化；
  - **vLLM Rollout Server（`TP=4`, Worker-0 节点）**：`Loading safetensors checkpoint shards: 100% (16/16) [00:10<00:00, 1.54it/s]`，单卡静态权重占用 `14.32 GiB`，KV Cache 分配 `114.61 GiB`（`751,104 tokens`，最大并发 `110.02x`）；
  - **首次冷启动 SM100 (`sm_100a`) JIT 编译与 AutoTuner**：首次运行触发 FlashInfer `0.7.0.post1` SM100 MoE/FMHA CUTLASS 内核编译（约 31 分钟）及 `[Autotuner]: Tuning flashinfer::trtllm_bf16_moe (21/21)`（约 2 分钟），编译产物与调优配置已持久化至共享缓存（后续运行免除 34 分钟冷编译）；
  - **4 次真实 `actor -> NCCL -> vLLM` 跨节点权重同步耗时**：
    - Step 0 初始同步：`update_weights done, time cost: 4.42s`
    - Step 1 更新后同步：`update_weights done, time cost: 3.07s`（Trainer 端含汇总统计 `timing_s/update_weights: 3.909s`）
    - Step 2 更新后同步：`update_weights done, time cost: 3.02s`（Trainer 端 `timing_s/update_weights: 3.927s`）
    - Step 3 更新后同步：`update_weights done, time cost: 3.89s`（Trainer 端 `timing_s/update_weights: 4.785s`）
- **3 步多轮 RL 训练逐 Step 实测指标（`train_batch_size=8, ppo_mini_batch_size=8, n=4`，每步 32 条多轮轨迹，3 步训练总耗时 `108 s`）**：

| 指标 | Step 1 | Step 2 | Step 3 | 门槛要求与物理含义 |
| --- | --- | --- | --- | --- |
| `actor/grad_norm` | **`0.6371`** | **`0.5927`** | **`0.8636`** | 3 步均 `> 0`，证明真实反向传播与梯度更新生效 |
| `actor/pg_loss` | `0.014055` | `-0.054975` | `0.067018` | 正常无 NaN/Inf |
| `actor/ppo_kl` (`actor/rollout_corr/kl`) | `0.013654` | `0.006984` | `0.009124` | 正常 |
| `actor/rollout_corr/training_ppl` | `1.1546` | `1.1211` | `1.1366` | TorchTitan `FSDP2=4` 训练端困惑度正常（对比 TPU `p1c-retool-rl-03` 正常值 `1.14`） |
| `actor/rollout_corr/rollout_ppl` | **`1.1350`** | **`1.1107`** | **`1.1248`** | vLLM `TP=4` 推理端困惑度与训练端高度一致（无 TPU `p1c-retool-rl-03` Step-0 权重损坏的 `249.33` 异常） |
| `actor/rollout_corr/ppl_ratio` | **`1.0168`** | **`1.0093`** | **`1.0103`** | **训练/推理 PPL 比值全程稳定在 `1.009 ~ 1.017`**，证明 `configure_qwen3_moe_for_rl` + `NCCL` 30B 权重同步无路由/数值分叉 |
| `critic/score/mean` (`min` / `max`) | **`-0.6250`** (`-1.0` / `+1.0`) | **`-0.1875`** (`-1.0` / `+1.0`) | **`+0.3125`** (`-1.0` / `+1.0`) | 随训练步推进平均得分由 `-0.625` 稳步提升至 **`+0.3125`**（超半数样本答对得 `+1.0`） |
| `num_turns/mean` (`min` / `max`) | **`5.875`** (`2` / `16`) | **`6.000`** (`2` / `16`) | **`5.625`** (`4` / `14`) | 平均每条轨迹约 5.6 ~ 6.0 轮对话（对应每条轨迹约 2 ~ 7 次 SandboxFusion `code_interpreter` 工具调用） |
| `response_length/mean` (`min` / `max`) | `753.09` (`258` / `1502`) | `1083.31` (`166` / `8192`) | `563.00` (`166` / `971`) | `response/aborted_ratio = 0.0` |
| `actor/perf/max_memory_allocated_gb` / `reserved_gb` | `116.95 GB` / `124.12 GB` | `127.20 GB` / `161.86 GB` | `127.20 GB` / `161.86 GB` | 单卡峰值分配约 `118.5 GiB`、峰值预留约 `150.7 GiB`（均低于单卡实测 `184.31 GiB` HBM 上限） |
| `timing_s/gen` / `update_actor` / `update_weights` / `step` | `8.02s` / `28.42s` / `3.91s` / **`43.23s`** | `0.01s` / `45.61s` / `3.93s` / **`49.67s`** | `0.01s` / `1.50s` / `4.79s` / **`6.42s`** | Step 1/2 含首次动态序列长度 `flex_attention` Triton autotune（各约 25~40s）；Step 3 热核下 `update_actor` 仅 **`1.50 s`**，单步仅 **`6.42 s`**（吞吐 `538.0 tok/s/GPU`） |

### 5.3 Attempt 3（`g1-retool-rl-smoke-03`，代码 SHA `64ef634b`）—— 热 SM100 缓存复用与 `rollout_audit/{1,2,3}.jsonl` 逐 Token 掩码核验（PASS）

- **热缓存加速实测**：复用持久化至共享存储的 `flashinfer_cache_sm100a.tar.gz`（32 MB）与 `vllm_cache_sm100a.tar.gz`（31 MB），从 Pod 创建到 3 步训练全部完成并落盘仅耗时 **7 分 15 秒**（免除 33 分钟冷 JIT 编译，其中 3 步训练 `107.8 s`，`35.94 s/it`）。
- **逐 Step 工具调用与 Token 级 `response_mask` 审计汇总（提取自 `rollout/{1,2,3}.jsonl` 与 `rollout_audit/{1,2,3}.jsonl`）**：

| Step | 全批次轨迹数 (`rollout/*.jsonl`) | 全批次 `<tool_response>` 触发次数 | 抽样审计会话数 (`rollout_audit/*.jsonl`) | `mask_length_matches` | `num_loss_tokens` (`mask == 1`) | `num_non_loss_tokens` (`mask == 0`) | 审计会话 `<tool_call>` (`mask == 1`) | 审计会话 `<tool_response>` (`mask == 0`) | 掩码违规数（`<tool_response>` 落入 `mask == 1`） |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| **Step 1** | `32` | **`50`** | `8` | `True` (8/8) | `3,492` | `517` | `13` | `13` | **`0`** |
| **Step 2** | `32` | **`74`** | `8` | `True` (8/8) | `6,795` | `787` | `18` | `18` | **`0`** |
| **Step 3** | `32` | **`48`** | `8` | `True` (8/8) | `2,164` | `295` | `10` | `10` | **`0`** |
| **合计** | **`96`** | **`172`** | **`24`** | **100% `True`** | **`12,451`** | **`1,599`** | **`41`** | **`41`** | **`0`（PASS）** |

- **结论**：`Stage G1` 全部门槛（完整 30B SFT checkpoint 加载、真实 `actor -> NCCL -> vLLM` 同步、连续 3 次真实梯度更新、172 次多轮工具调用、Token 级 `response_mask` 0 违规）均已验证通过；随后完成 **`Stage G2`（`g2-retool-rl-25step-01`，25 步正式 RL 对照 + Step 0/25 AIME 2024/2025 `n=30` 评估）**。

---

## 6. Stage G2：GB200 ReTool 25 步正式 RL 对照与 Step 0 / Step 25 AIME 2024 & 2025 评估结果（PASS）

### 6.1 运行配置、起点对齐与可独立复算原始证据归档（`g2-retool-rl-25step-01`，代码 SHA `64ef634b`）

- **运行镜像与代码版本**：
  - 镜像 Digest：`verl-torchtitan-cu130-arm64@sha256:a139e2b491ac23f06f790fe4277afb9bd543a94bc740c92173e5fdb13ddaba40`
  - 运行代码 SHA：`64ef634b173c69ea31eff598caeee81d9a9caea1`（`uv.lock` SHA-256: `df508db0b957fa181c229fb75885cd3863cb651fd5233d0fa5af95e990bda0be`）
  - 实测软件栈（见 [`head-precheck.json`](./gb200/evidence/g2-retool-rl-25step-01/env/g2-retool-rl-25step-01-head-precheck.json)）：`python 3.12.3`、`torch 2.13.0+cu130`（CUDA `13.0`, NCCL `2.29.7`）、`torchtitan 0.3.0`、`vllm 0.32.0` (`VLLM_USE_V1=1`)、`flash-attn 2.8.3`、`ray 2.55.1`、`transformers 5.16.1`、`transferqueue 0.1.11`
- **SFT 起点（与 TPU 线完全共享）**：`p2-retool-sft-01/checkpoint/global_step_372/huggingface`（双节点 20 核心文件组合摘要 `59247a00c13f2364a12aab5ab6b88b6d971d59ff643c57aad8ac4e55a71231b0`，`ref_match = true`）
- **硬件拓扑与并行度**：**2 节点 × 4 张 NVIDIA GB200（共 8 张 GPU，`NV18` 满规格 NVLink5 + `ComputeDomain` IMEX 通道）**：
  - **Actor / Ref Trainer（Head 节点）**：4 张 GB200，`FSDP2=4, TP=1, EP=1, CP=1`（`MoE: disabled expert-bias load balancing on 48 layers`）
  - **Async vLLM Rollout（Worker-0 节点）**：4 张 GB200，`TP=4`（`VLLM_USE_V1=1`，FlashInfer SM100 热缓存复用）
- **核心训练与评估超参（经 [`head.log`](./gb200/evidence/g2-retool-rl-25step-01/head.log) 启动命令 `L130` 与 Hydra Resolved Config `L142–L1142` 逐项核实；与 TPU `Stage P3` 计划超参基本对齐，具体异同见下表 6.1.1）**：
  - `data.train_batch_size = 64`，`actor.ppo_mini_batch_size = 64`，`rollout.n = 8`（每步 `64 × 8 = 512` 条多轮轨迹；25 步合计 **`12,800` 条训练轨迹**，见 [`head.log#L130`](./gb200/evidence/g2-retool-rl-25step-01/head.log#L130)）
  - `rollout.val_kwargs.n = 30`（AIME 2024 共 30 题 + AIME 2025 共 30 题，每次评估 `60 × 30 = 1,800` 条多轮轨迹，见 [`head.log#L574-L580`](./gb200/evidence/g2-retool-rl-25step-01/head.log#L574-L580)）
  - `trainer.v1.separate_async.num_warmup_batches = 2`（见 [`head.log#L88`](./gb200/evidence/g2-retool-rl-25step-01/head.log#L88)、[`head.log#L130`](./gb200/evidence/g2-retool-rl-25step-01/head.log#L130)、[`head.log#L1069`](./gb200/evidence/g2-retool-rl-25step-01/head.log#L1069)），`trainer.v1.sampler.max_off_policy_threshold = 4`（注：早先报告正文曾误写为 `max_staleness = 3`，实际启动脚本 `L90` 与 Hydra 解析配置 [`head.log#L1064`](./gb200/evidence/g2-retool-rl-25step-01/head.log#L1064) 均为 `trainer.v1.sampler.max_off_policy_threshold = 4`），`trainer.total_training_steps = 25`，`trainer.val_before_train = True`，`trainer.test_freq = 25`
  - `max_prompt_length = 2048`，`max_response_length = 8192`，`multi_turn.max_user_turns = 8`、`multi_turn.max_assistant_turns = 8`（单条轨迹最多 `8 user + 8 assistant = 16` 轮总消息，对应日志 `val-aux/num_turns/max = 16`，见 [`head.log#L64`](./gb200/evidence/g2-retool-rl-25step-01/head.log#L64)、[`head.log#L498-L501`](./gb200/evidence/g2-retool-rl-25step-01/head.log#L498-L501)、[`head.log#L3028`](./gb200/evidence/g2-retool-rl-25step-01/head.log#L3028)），`actor_rollout_ref.actor.optim.lr = 1.4e-6`（见 [`head.log#L59`](./gb200/evidence/g2-retool-rl-25step-01/head.log#L59)、[`head.log#L130`](./gb200/evidence/g2-retool-rl-25step-01/head.log#L130)、[`head.log#L168-L176`](./gb200/evidence/g2-retool-rl-25step-01/head.log#L168-L176)，与 Askhat 的 [`run_qwen3_30b_a3b_dapo_torchtitan.sh`](https://github.com/jialei777/verl-upstream/blob/266e793a20c7baa6f849476ba72c8f8052da1477/examples/tpu/retool/run_qwen3_30b_a3b_dapo_torchtitan.sh#L105-L110) 默认值 `actor_lr=1.4e-6` 完全一致），`training.max_norm = 1.0`

#### 6.1.1 Askhat 脚本基线 vs G2 报告表述 vs G2 实际运行配置（`head.log`）逐项对照表

| 项目 | G2 早期报告表述 | G2 实际运行配置（[`head.log`](./gb200/evidence/g2-retool-rl-25step-01/head.log) 实证） | Askhat 30B-A3B 脚本默认值（[`run_qwen3_30b_a3b_dapo_torchtitan.sh`](https://github.com/jialei777/verl-upstream/blob/266e793a20c7baa6f849476ba72c8f8052da1477/examples/tpu/retool/run_qwen3_30b_a3b_dapo_torchtitan.sh)） | 状态与结论 |
| --- | --- | --- | --- | --- |
| **多轮上限** | 曾简写为 `max_turns=16` | `max_turns=8` $\rightarrow$ `multi_turn.max_user_turns=8`、`multi_turn.max_assistant_turns=8`（[`head.log#L64`](./gb200/evidence/g2-retool-rl-25step-01/head.log#L64), [`#L498-L501`](./gb200/evidence/g2-retool-rl-25step-01/head.log#L498-L501)）；单条轨迹最多 `8 user + 8 assistant = 16` 轮消息（`val-aux/num_turns/max = 16`，[`head.log#L2343`](./gb200/evidence/g2-retool-rl-25step-01/head.log#L2343), [`#L3028`](./gb200/evidence/g2-retool-rl-25step-01/head.log#L3028)） | `max_turns=8` $\rightarrow$ `multi_turn.max_user_turns=8`、`multi_turn.max_assistant_turns=8` | **已对齐**（报告先前写 `16` 是将分角色上限 `8+8` 的总对话轮数 `num_turns/max=16` 混写为配置名；实际配置与 Askhat 完全相同） |
| **Warmup 批次数** | 曾误写配置名为 `trainer.v1.sampler.num_warmup_steps=2` | `NUM_WARMUP_BATCHES=2` $\rightarrow$ `trainer.v1.separate_async.num_warmup_batches=2`（[`head.log#L88`](./gb200/evidence/g2-retool-rl-25step-01/head.log#L88), [`#L130`](./gb200/evidence/g2-retool-rl-25step-01/head.log#L130), [`#L1069`](./gb200/evidence/g2-retool-rl-25step-01/head.log#L1069)） | `NUM_WARMUP_BATCHES=2` $\rightarrow$ `trainer.v1.separate_async.num_warmup_batches=2` | **已对齐**（实际配置键名为 `trainer.v1.separate_async.num_warmup_batches=2`，与 Askhat 完全一致） |
| **Off-policy 阈值** | 曾误写为 `max_staleness=3` | `MAX_OFF_POLICY_THRESHOLD=4` $\rightarrow$ `trainer.v1.sampler.max_off_policy_threshold=4`（[`head.log#L90`](./gb200/evidence/g2-retool-rl-25step-01/head.log#L90), [`#L130`](./gb200/evidence/g2-retool-rl-25step-01/head.log#L130), [`#L1064`](./gb200/evidence/g2-retool-rl-25step-01/head.log#L1064)） | `MAX_OFF_POLICY_THRESHOLD=4` $\rightarrow$ `trainer.v1.sampler.max_off_policy_threshold=4` | **已对齐**（`max_staleness=3` 系早期文档笔误；G2 实际运行参数名称与数值均为 `trainer.v1.sampler.max_off_policy_threshold=4`，与 Askhat 完全一致） |
| **SFT 起点 Checkpoint** | 使用本地 TPU `Stage P2` Step-372 导出 checkpoint | `/dev/shm/models/multiturn-sft-qwen3-30b-a3b-tpu7x`（20 核心文件 `composite_sha256 = 59247a00...`，与本地 TPU `p2-retool-sft-01` Step-372 逐字节一致） | Askhat 自身训练导出的 SFT checkpoint（末 10 步 `train/loss=0.039429` vs 本地 `P2` `0.039265`） | **不同 / 未确认逐位相同**（已严格证明 GB200 `G2` 与本地 TPU `P2/P3` 起点逐位相同；但与 Askhat 的 SFT checkpoint 属于不同训练实例，尚未证明权重相同） |
| **评分函数（Reward Scorer）** | 启动脚本指定 `custom_reward_function.path=pkg://examples.tpu.retool.retool_tpu` | CLI 传入顶层 `custom_reward_function.path=pkg://examples.tpu.retool.retool_tpu`（[`head.log#L765-L766`](./gb200/evidence/g2-retool-rl-25step-01/head.log#L765-L766)），但 `trainer.use_v1=True` 下 `config.reward.custom_reward_function.path` 为 `None`（[`head.log#L899-L904`](./gb200/evidence/g2-retool-rl-25step-01/head.log#L899-L904)），运行期**已从日志与 3,600 条样本确认**回退加载 `default_compute_score` $\rightarrow$ `math_dapo.compute_score(..., strict_box_verify=False)` | 脚本同样传入顶层 `custom_reward_function.path=pkg://examples.tpu.retool.retool_tpu` 且 `trainer.use_v1=True`（未设 `reward.custom_reward_function.path`） | **G2 已确认回退至 `strict_box_verify=False`；Askhat 实际 run 是否同样回退目前是基于脚本与代码的推断，待其实际运行配置或日志确认**（详见 6.2.1 节双口径复算） |
| **Batch / Rollout / LR** | `BS=64, mini_bs=64, n=8, val_n=30, lr=1.4e-6` | `train_batch_size=64, ppo_mini_batch_size=64, rollout.n=8, val_kwargs.n=30, actor_rollout_ref.actor.optim.lr=1.4e-6, clip_ratio_low=0.2, clip_ratio_high=0.28`（[`head.log#L59`](./gb200/evidence/g2-retool-rl-25step-01/head.log#L59), [`#L130`](./gb200/evidence/g2-retool-rl-25step-01/head.log#L130), [`#L168-L176`](./gb200/evidence/g2-retool-rl-25step-01/head.log#L168-L176)） | `train_batch_size=64, ppo_mini_batch_size=64, rollout.n=8, val_kwargs.n=30, actor_lr=1.4e-6, clip_ratio_low=0.2, clip_ratio_high=0.28`（[`run_qwen3_30b_a3b_dapo_torchtitan.sh#L105-L110`](https://github.com/jialei777/verl-upstream/blob/266e793a20c7baa6f849476ba72c8f8052da1477/examples/tpu/retool/run_qwen3_30b_a3b_dapo_torchtitan.sh#L105-L110)） | **已对齐**（注：`32B` dense 脚本默认 `1e-6`，而 `30B-A3B` MoE 脚本与 G2 实际解析配置均为 **`1.4e-6`**） |
| **加速器类型与物理卡数** | 2 节点 × 4 张 GB200 = **8 张 GB200 GPU**（`4 trainer + 4 rollout`） | **8 张 NVIDIA GB200 GPU**（`4 trainer FSDP2=4 + 4 rollout TP=4`） | Askhat TPU v6e 配置（[`jialei777/verl-upstream#108`](https://github.com/jialei777/verl-upstream/pull/108)）为 **64 张 v6e 物理芯片**（`2 × v6e-32`，`32 trainer + 32 rollout`）；本仓库 TPU v7x 配置为 **32 张 v7x 物理芯片 / 64 逻辑核**（`2 × 2×2×4`，`16 trainer + 16 rollout`） | **不同**（跨平台性能需在相同物理加速器数量下进一步对照，见第 8 节与 PR #11） |
- **Step-25 模型权重 Checkpoint 保存状态说明**：
  - 本次 `g2-retool-rl-25step-01` 启动参数为默认 `SAVE_FREQ=-1`（即 `trainer.save_freq = -1`），因此 **未将 Step-25 模型权重保存为磁盘 checkpoint**；Step 25 的 AIME 2024 / 2025 (`n=30`) 评估是在 Step 25 完成 `actor -> NCCL -> vLLM` 权重同步后直接在内存中触发执行（`trainer.test_freq = 25`），并将全部评估样本、训练轨迹与掩码审计落盘保存。
- **可独立复算原始证据文件清单与验证范围说明（已归档至本 PR [`examples/tpu/retool/gb200/evidence/g2-retool-rl-25step-01/`](./gb200/evidence/g2-retool-rl-25step-01/)）**：
  1. [`head.log`](./gb200/evidence/g2-retool-rl-25step-01/head.log)（`810 KiB`，完整 3,037 行主节点控制台日志）：
     - 启动命令与环境变量：`L1–L130`；
     - **Hydra 解析后完整配置（Resolved Config）**：`L142–L1142`；
     - Step 0 初始 `update_weights` 与 AIME `n=30` 评估原始输出：`L1938–L2345`；
     - `step:1..25` 逐步训练指标与 Step 25 AIME `n=30` 评估原始输出：`L2393–L2711`；
  2. [`g2_25step_metrics.json`](./gb200/evidence/g2-retool-rl-25step-01/g2_25step_metrics.json)（从 `head.log` 提取的 `step:0` 与 `step:1..25` 全量键值指标 JSON）；
  3. [`validation_step0_samples.jsonl`](./gb200/evidence/g2-retool-rl-25step-01/validation_step0_samples.jsonl) 与 [`validation_step25_samples.jsonl`](./gb200/evidence/g2-retool-rl-25step-01/validation_step25_samples.jsonl)（Step 0 与 Step 25 各 `1,800` 条、合计 `3,600` 条 AIME 2024/2025 样本记录）：
     - **验证范围说明**：每条记录包含 `question_id`、`uid`、`gts`、`pred`、`acc`、`score`、`reward`、`tool_response_count`、全文 `output_sha256` 以及 **`output_tail_300`（模型原始输出的末尾 300 字符，即 `math_dapo.compute_score` 实际截取解析的原始文本窗口）**，不包含完整多轮中间长文本（原始 `validation/{0,25}.jsonl` 各约 `8.7 MB`）；
     - 从 `output_tail_300` 调用 `math_dapo.compute_score(..., strict_box_verify=False)` 可 100% 复现全部 `3,600 / 3,600` 条样本的 `pred` 与 `acc`（`0` 偏差），并确定性复算出 AIME 2024 `mean@30`（`214/900 = 23.78%` $\rightarrow$ `259/900 = 28.78%`）、AIME 2025 `mean@30`（`153/900 = 17.00%` $\rightarrow$ `161/900 = 17.89%`）及 `std@30`；
     - 报告中的 `maj@30/mean`、`best@30/mean`、`maj@8/mean`、`best@8/mean` 来自 `verl` 运行期 `process_validation_metrics(..., seed=42)`（见 `head.log` `L2345` 与 `L2707`），其中 `maj@k` 在票数并列时依赖运行期异步 rollout 完成顺序（落盘前未按 `uid` 排序）；
  4. [`rollout_and_audit_summary.json`](./gb200/evidence/g2-retool-rl-25step-01/rollout_and_audit_summary.json)（`rollout/1..25.jsonl` 共 `12,800` 条训练轨迹与 `rollout_audit/1..25.jsonl` 共 `200` 个审计会话的逐 Step 统计）；
  5. [`env/`](./gb200/evidence/g2-retool-rl-25step-01/env/)（双节点 `*-precheck.json`、`*-g0-verify.json`、`*-nvidia-smi-L.txt`、`*-topo.txt` 与 `*-sandbox-4case.json`）；
  6. [`recompute_g2_metrics.py`](./gb200/evidence/g2-retool-rl-25step-01/recompute_g2_metrics.py)（独立复算并断言校验下述全部训练均值、步耗时、墙钟计时边界与 AIME 评估指标的脚本：`python3 examples/tpu/retool/gb200/evidence/g2-retool-rl-25step-01/recompute_g2_metrics.py`）。
- **墙钟计时边界核实（基于 [`head.log`](./gb200/evidence/g2-retool-rl-25step-01/head.log) UTC 时间戳）**：
  - `23:47:07.289`（`L131`）：`verl.trainer.main_ppo` 启动（Ray 连接、30B-A3B 权重加载与进程组初始化）；
  - `23:50:05.530`（`L1938`）：Step 0 初始 `update_weights` 完成（耗时 `4.13s`）；
  - `23:50:05.950` $\rightarrow$ `23:53:06`（`L1939–L2345`）：**Step 0 初始 AIME 2024/2025 (`n=30`) 评估，耗时 `3m 00s`（`180.1 s`）**；评估结束后初始化 `tqdm` 进度条 `Training Progress: 0/25 [00:00]`（`L2022`）；
  - `23:53:06` $\rightarrow$ `00:11:30.538`（`L2022–L2686`）：**25 步多轮 RL 训练本体（不含首尾两次评估，包含训练过程中发生的动态编译与 autotune），墙钟耗时 `18m 24s`（`1,104.5 s`）**；
  - `00:11:30.538` $\rightarrow$ `00:14:42.324`（`L2686–L2709`）：**Step 25 末次 AIME 2024/2025 (`n=30`) 评估，耗时 `3m 10s`（`timing_s/testing = 190.36 s`）**；
  - 因此：
    - `tqdm` 进度条显示的 **`21m 35s`**（`L2706`，`00:14:42` 结束时为 `21m 36s`）包含 **25 步训练（`18m 24s`）+ Step 25 末次评估（`3m 10s`）**，**不含 Step 0 初始评估**；
    - **包含 Step 0 + Step 25 两次完整 `n=30` AIME 评估 + 25 步训练**的总耗时为 **`24m 36s`（`1,476.4 s`，`23:50:05.950` $\rightarrow$ `00:14:42.324`）**；
    - 从 `main_ppo` 进程启动到退出的完整端到端墙钟耗时为 **`27m 37s`（`1,657.1 s`，`23:47:07.289` $\rightarrow$ `00:14:44.392`）**。

### 6.2 Step 0（RL 前 SFT 基线）vs Step 25（25 步 RL 后）AIME 2024 & AIME 2025 (`n=30`) 评估对照

| 评估指标（`n=30` 采样 / 题） | Step 0（共享 P2 SFT 起点） | Step 25（25 步 ReTool RL 后） | 绝对增益（$\Delta$ Step 25 $-$ Step 0） | 对照 TPU v5p 历史基线（25 步区间） |
| --- | --- | --- | --- | --- |
| **AIME 2024 `acc/mean@30` (`pass@1`)** | `23.78%` (`0.2378`) | **`28.78%`** (`0.2878`) | **`+5.00 pp`**（相对 `+21.0%`） | Step 0 `~16.7%` $\rightarrow$ Step 20 `~26.7%` |
| **AIME 2024 `acc/maj@30/mean`** | `40.55%` (`0.4055`) | **`47.13%`** (`0.4713`) | **`+6.58 pp`** | - |
| **AIME 2024 `acc/best@30/mean`** | `64.29%` (`0.6429`) | **`69.05%`** (`0.6905`) | **`+4.76 pp`** | - |
| **AIME 2024 `acc/maj@8/mean`** | `33.73%` (`0.3373`) | **`41.57%`** (`0.4157`) | **`+7.84 pp`** | - |
| **AIME 2024 `acc/best@8/mean`** | `53.13%` (`0.5313`) | **`58.00%`** (`0.5800`) | **`+4.87 pp`** | - |
| **AIME 2024 `acc/std@30`** | `25.52%` (`0.2552`) | `26.78%` (`0.2678`) | `+1.26 pp` | - |
| **AIME 2025 `acc/mean@30` (`pass@1`)** | `17.00%` (`0.1700`) | **`17.89%`** (`0.1789`) | **`+0.89 pp`** | - |
| **AIME 2025 `acc/maj@30/mean`** | `29.84%` (`0.2984`) | **`30.60%`** (`0.3060`) | **`+0.76 pp`** | - |
| **AIME 2025 `acc/best@30/mean`** | `45.52%` (`0.4552`) | **`57.34%`** (`0.5734`) | **`+11.82 pp`**（相对 `+26.0%`） | - |
| **AIME 2025 `acc/maj@8/mean`** | `25.38%` (`0.2538`) | **`25.41%`** (`0.2541`) | **`+0.03 pp`** | - |
| **AIME 2025 `acc/best@8/mean`** | `36.78%` (`0.3678`) | **`42.61%`** (`0.4261`) | **`+5.84 pp`** | - |
| **AIME 2025 `acc/std@30`** | `17.79%` (`0.1779`) | `21.14%` (`0.2114`) | `+3.35 pp` | - |
| **评估集多轮对话轮数 `val-aux/num_turns/mean` (`min`–`max`)** | `5.9467` (`2`–`16`) | **`5.5878`** (`2`–`16`) | `-0.3589`（工具调用更精炼） | - |
| **评估轨迹含 `<tool_response>` 比例 (`validation/{0,25}.jsonl`)** | `1,634 / 1,800` (**`90.78%`**, 共 `3,552` 次调用) | `1,591 / 1,800` (**`88.39%`**, 共 `3,229` 次调用) | 全程保持 ~90% 工具调用率 | - |

#### 6.2.1 评分器（Scorer）加载路径核实与 `strict_box_verify=False` vs `strict_box_verify=True` 双口径全量复算

1. **运行期实际加载的评分器（从 [`head.log`](./gb200/evidence/g2-retool-rl-25step-01/head.log) 与 `verl` 源码确认）**：
   - 启动脚本 [`run_qwen3_30b_a3b_dapo_torchtitan_gb200.sh`](./gb200/run_qwen3_30b_a3b_dapo_torchtitan_gb200.sh)（沿用自 Askhat 的 [`run_qwen3_30b_a3b_dapo_torchtitan.sh`](https://github.com/jialei777/verl-upstream/blob/266e793a20c7baa6f849476ba72c8f8052da1477/examples/tpu/retool/run_qwen3_30b_a3b_dapo_torchtitan.sh)）在命令行传入了顶层参数 `custom_reward_function.path=pkg://examples.tpu.retool.retool_tpu` 与 `custom_reward_function.name=compute_score`（见 [`head.log#L53-L54`](./gb200/evidence/g2-retool-rl-25step-01/head.log#L53-L54)、[`head.log#L130`](./gb200/evidence/g2-retool-rl-25step-01/head.log#L130)）。
   - 但在 `trainer.use_v1=True`（`TaskRunnerV1`，见 [`verl/trainer/main_ppo.py`](../../../verl/trainer/main_ppo.py)）路径下，`migrate_legacy_reward_impl()` 未被调用，因此顶层 `config.custom_reward_function` 为 `{'name': 'compute_score', 'path': 'pkg://examples.tpu.retool.retool_tpu'}`（[`head.log#L765-L766`](./gb200/evidence/g2-retool-rl-25step-01/head.log#L765-L766)），而 V1 `RewardLoopWorker` 实际读取的嵌套字段 `config.reward.custom_reward_function` 保持默认值 `{'name': 'compute_score', 'path': None}`（[`head.log#L899-L904`](./gb200/evidence/g2-retool-rl-25step-01/head.log#L899-L904)）。
   - 因此，[`verl/trainer/ppo/reward.py:get_custom_reward_fn()`](../../../verl/trainer/ppo/reward.py) 因 `module_path is None` 返回 `None`，`load_reward_manager()` 自动回退至 [`verl/utils/reward_score/__init__.py:default_compute_score`](../../../verl/utils/reward_score/__init__.py)，对 `math_dapo` / `aime*` 数据源调用 `math_dapo.compute_score(solution_str, ground_truth, strict_box_verify=False)`（纯结果奖励 `{-1.0, +1.0}`）。（注：Askhat 的实际运行是否同样发生该回退，目前是基于其脚本与 `verl` V1 代码的推断，待其实际运行配置或日志确认。）
   - 这一点在 3,600 条样本记录及 `head.log` 中得到 100% 交叉印证：
     - 全部 `3,600 / 3,600` 条评估样本的 `pred`、`acc` 与 `score`（仅含 `-1.0` 与 `+1.0`）与 `math_dapo.compute_score(output_tail_300, gts, strict_box_verify=False)` **零偏差完全一致**；
     - `head.log` 打印的 `val-aux/aime_2024/reward/mean@30`（Step 0 `-0.524444`、Step 25 `-0.424444`）与 `val-core/aime_2024/acc/mean@30`（Step 0 `0.237778`、Step 25 `0.287778`）严格满足 $\text{reward} = 2 \times \text{acc} - 1$。
2. **为什么 `strict_box_verify=False` 与 `strict_box_verify=True` 在 Step 0 相差 5 条、Step 25 相差 9 条样本？**
   - 在 [`verl/utils/reward_score/math_dapo.py:verify()`](../../../verl/utils/reward_score/math_dapo.py) 中，当 `strict_box_verify=False` 时，会优先尝试 Minerva 正则（`(?i)Answer\s*:\s*([^\n]+)`），仅当其返回 `"[INVALID]"` 时才回退到 `\boxed{...}` 提取；
   - 当模型输出形如 `**Answer:** \boxed{45}`、`Final answer: \(\boxed{204}\)` 或 `**Final Answer:**\n\n\boxed{204}` 时，Minerva 正则匹配到了冒号前后的 Markdown 符号并经 `normalize_final_answer` 归一化为 `'**45'`、`'\\(204\\)'` 或 `'**'`（非 `"[INVALID]"`），导致 `strict_box_verify=False` 直接判定为错误（`acc=False`）而未触发 `\boxed{}` 回退；
   - 反之，`strict_box_verify=True`（即 `retool_tpu.compute_score` 指定的方式）直接提取最后一个 `\boxed{...}` 内的数字（`'45'`、`'204'`、`'211'`、`'16'`、`'77'`），将这 5 条（Step 0）与 9 条（Step 25）实际答对并放在 `\boxed{}` 内的样本全部正确判为 **`acc=True`**。
3. **双口径（运行期实际 `strict_box_verify=False` vs ReTool `strict_box_verify=True`）全量 3,600 条样本复算对照表**（由 [`recompute_g2_metrics.py`](./gb200/evidence/g2-retool-rl-25step-01/recompute_g2_metrics.py) 自动复算验证）：

| 评估数据集与指标 | 运行期实际评分器（`strict_box_verify=False`，与 `head.log` 一致） | ReTool 严格 Box 评分器（`strict_box_verify=True`） | 两种评分方式差异说明 |
| --- | --- | --- | --- |
| **AIME 2024 Step 0 正确样本数 / `mean@30` (`std@30`)** | **`214 / 900` (`23.78%`**, `0.237778`, `std=0.255242`) | **`218 / 900` (`24.22%`**, `0.242222`, `std=0.255773`) | Step 0 共 **4 条** AIME 2024 样本因 `**Answer:** \boxed{45}` / `Final answer: \(\boxed{204}\)` 在 `strict=True` 下恢复为正确（`+0.44 pp`） |
| **AIME 2024 Step 25 正确样本数 / `mean@30` (`std@30`)** | **`259 / 900` (`28.78%`**, `0.287778`, `std=0.267813`) | **`267 / 900` (`29.67%`**, `0.296667`, `std=0.268052`) | Step 25 共 **8 条** AIME 2024 样本（答案 `45`、`204`、`211`）在 `strict=True` 下恢复为正确（`+0.89 pp`） |
| **AIME 2024 25 步绝对增益（$\Delta$ Step 25 $-$ Step 0）** | **`+45 / 900` (`+5.00 pp`)** | **`+49 / 900` (`+5.44 pp`)** | 无论采用哪种口径，25 步 RL 在 AIME 2024 上均取得 **`+5.00 pp ~ +5.44 pp`** 的显著提升 |
| **AIME 2025 Step 0 正确样本数 / `mean@30` (`std@30`)** | **`153 / 900` (`17.00%`**, `0.170000`, `std=0.177899`) | **`154 / 900` (`17.11%`**, `0.171111`, `std=0.177257`) | Step 0 共 **1 条** AIME 2025 样本（`**Answer:** \boxed{16}`）在 `strict=True` 下恢复为正确（`+0.11 pp`） |
| **AIME 2025 Step 25 正确样本数 / `mean@30` (`std@30`)** | **`161 / 900` (`17.89%`**, `0.178889`, `std=0.211361`) | **`162 / 900` (`18.00%`**, `0.180000`, `std=0.212003`) | Step 25 共 **1 条** AIME 2025 样本（`**Final Answer:**\n\n\boxed{77}`）在 `strict=True` 下恢复为正确（`+0.11 pp`） |
| **AIME 2025 25 步绝对增益（$\Delta$ Step 25 $-$ Step 0）** | **`+8 / 900` (`+0.89 pp`)** | **`+8 / 900` (`+0.89 pp`)** | 两种口径下 AIME 2025 的 25 步净增益完全一致（均为 **`+8 / 900 = +0.89 pp`**） |

### 6.3 25 步多轮 RL 训练逐 Step 全量指标表（`step:1` – `step:25`）

| Step | `pg_loss` | `grad_norm` | `ppo_kl` | `training_ppl` / `rollout_ppl` | **`ppl_ratio`** | `score/mean` | `resp_len/mean` | `num_turns/mean` | `<tool_response>` 次数 (`/512` 轨迹) | `gen` / `update_actor` / `update_weights` | **`timing_s/step`** | `total_tokens` (`perf/throughput tok/s/GPU`) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| **1** | `-0.0173` | `0.2843` | `0.01205` | `1.1636` / `1.1476` | **`1.0137`** | `-0.2891` | `610.3` | `4.89` | `740` | `86.14s` / `46.32s` / `3.99s` | **`136.92s`** | `473,629` (`432`) |
| **2** | `+0.0511` | `0.2462` | `0.01121` | `1.1440` / `1.1279` | **`1.0141`** | `-0.3203` | `864.3` | `6.72` | `1,208` | `0.04s` / `64.28s` / `3.87s` | **`68.58s`** | `604,398` (`1,102`) |
| **3** | `+0.0650` | `0.1998` | `0.01129` | `1.1569` / `1.1408` | **`1.0142`** | `-0.4531` | `948.6` | `6.52` | `1,156` | `0.04s` / `18.95s` / `5.24s` | **`24.63s`** | `649,181` (`3,295`) |
| **4** | `+0.0288` | `0.2425` | `0.01208` | `1.1583` / `1.1422` | **`1.0140`** | `-0.2578` | `772.3` | `6.08` | `1,044` | `0.04s` / `15.34s` / `4.43s` | **`20.19s`** | `558,587` (`3,458`) |
| **5** | `-0.0203` | `0.3207` | `0.01192` | `1.1535` / `1.1384` | **`1.0132`** | `-0.0977` | `593.5` | `4.95` | `756` | `60.19s` / `12.89s` / `3.93s` | **`77.39s`** | `465,108` (`751`) |
| **6** | `+0.0135` | `0.2718` | `0.01157` | `1.1471` / `1.1316` | **`1.0135`** | `-0.3086` | `853.0` | `6.75` | `1,216` | `0.04s` / `16.67s` / `4.23s` | **`21.67s`** | `607,983` (`3,507`) |
| **7** | `+0.0205` | `0.2332` | `0.01108` | `1.1527` / `1.1386` | **`1.0123`** | `-0.2852` | `784.5` | `6.23` | `1,082` | `24.18s` / `15.28s` / `4.30s` | **`44.15s`** | `563,036` (`1,594`) |
| **8** | `-0.0230` | `0.2810` | `0.01272` | `1.1525` / `1.1363` | **`1.0141`** | `-0.1406` | `637.4` | `5.09` | `792` | `22.21s` / `13.45s` / `4.47s` | **`40.53s`** | `495,925` (`1,529`) |
| **9** | `-0.0292` | `0.2640` | `0.01104` | `1.1365` / `1.1220` | **`1.0127`** | `-0.2773` | `789.3` | `6.21` | `1,077` | `18.10s` / `15.46s` / `4.11s` | **`38.07s`** | `574,017` (`1,885`) |
| **10** | `-0.0284` | `0.2497` | `0.01192` | `1.1496` / `1.1342` | **`1.0135`** | `-0.2344` | `698.6` | `5.51` | `899` | `0.11s` / `13.75s` / `4.30s` | **`18.55s`** | `517,372` (`3,485`) |
| **11** | `+0.0181` | `0.2729` | `0.01007` | `1.1486` / `1.1355` | **`1.0113`** | `-0.2227` | `766.3` | `5.61` | `923` | `46.24s` / `15.15s` / `4.54s` | **`66.32s`** | `557,495` (`1,051`) |
| **12** | `-0.0168` | `0.2106` | `0.01002` | `1.1576` / `1.1442` | **`1.0114`** | `-0.4297` | `810.1` | `6.05` | `1,036` | `0.04s` / `15.42s` / `4.89s` | **`20.76s`** | `577,381` (`3,477`) |
| **13** | `+0.0293` | `0.2126` | `0.01088` | `1.1414` / `1.1275` | **`1.0121`** | `-0.1719` | `728.6` | `5.57` | `913` | `30.17s` / `14.48s` / `4.54s` | **`49.95s`** | `529,660` (`1,326`) |
| **14** | `+0.0290` | `0.2827` | `0.01009` | `1.1394` / `1.1268` | **`1.0110`** | `-0.0586` | `690.0` | `5.07` | `786` | `18.25s` / `13.74s` / `4.24s` | **`36.64s`** | `510,879` (`1,743`) |
| **15** | `-0.0148` | `0.2590` | `0.00951` | `1.1387` / `1.1267` | **`1.0106`** | `-0.1133` | `758.2` | `5.76` | `962` | `20.19s` / `15.06s` / `4.20s` | **`39.84s`** | `552,582` (`1,734`) |
| **16** | `+0.0086` | `0.2583` | `0.01066` | `1.1428` / `1.1294` | **`1.0118`** | `-0.1914` | `737.3` | `5.56` | `911` | `23.01s` / `14.95s` / `4.20s` | **`42.78s`** | `541,943` (`1,584`) |
| **17** | `-0.0111` | `0.2691` | `0.01082` | `1.1572` / `1.1434` | **`1.0119`** | `-0.4844` | `768.7` | `5.79` | `971` | `22.26s` / `15.52s` / `4.31s` | **`42.49s`** | `567,372` (`1,669`) |
| **18** | `+0.0293` | `0.2483` | `0.01028` | `1.1415` / `1.1284` | **`1.0116`** | `-0.3047` | `799.9` | `6.64` | `1,188` | `0.13s` / `15.44s` / `4.38s` | **`20.34s`** | `575,115` (`3,534`) |
| **19** | `-0.0203` | `0.2486` | `0.00954` | `1.1493` / `1.1371` | **`1.0106`** | `-0.2188` | `739.8` | `5.45` | `882` | `42.23s` / `15.21s` / `4.01s` | **`61.84s`** | `538,693` (`1,089`) |
| **20** | `-0.0328` | `0.2302` | `0.00913` | `1.1560` / `1.1439` | **`1.0105`** | `-0.3359` | `753.6` | `5.70` | `946` | `0.04s` / `15.09s` / `5.37s` | **`21.25s`** | `546,727` (`3,216`) |
| **21** | `+0.0017` | `0.2317` | `0.00833` | `1.1372` / `1.1266` | **`1.0094`** | `-0.1445` | `771.8` | `5.60` | `921` | `28.24s` / `15.28s` / `4.26s` | **`48.23s`** | `552,465` (`1,432`) |
| **22** | `-0.0268` | `0.2568` | `0.00811` | `1.1426` / `1.1323` | **`1.0090`** | `+0.0156` | `717.0` | `5.08` | `788` | `22.23s` / `14.79s` / `4.15s` | **`41.60s`** | `535,019` (`1,608`) |
| **23** | `+0.0136` | `0.2273` | `0.00790` | `1.1462` / `1.1358` | **`1.0091`** | `-0.2461` | `736.0` | `5.71` | `949` | `4.11s` / `15.29s` / `4.47s` | **`24.30s`** | `538,240` (`2,768`) |
| **24** | `+0.0109` | `0.1950` | `0.00829` | `1.1392` / `1.1286` | **`1.0094`** | `-0.2188` | `781.6` | `5.66` | `936` | `44.39s` / `15.32s` / `3.96s` | **`64.10s`** | `562,734` (`1,097`) |
| **25** | `+0.0076` | `0.1901` | `0.00733` | `1.1425` / `1.1334` | **`1.0080`** | `-0.4375` | `898.6` | `6.00` | `1,024` | `0.13s` / `17.17s` / `4.23s` | **`21.98s`**（不含末尾 190.4s 评估） | `629,878` (`3,582`) |
| **汇总 / 均值** | 均值 `+0.0034` | 均值 `0.2475` | `0.01205` $\rightarrow$ `0.00733` | `1.1478` / `1.1344` | **`1.0117` (`1.0080–1.0142`)** | 峰值 `+0.0156` | 均值 `760.4` | 均值 `5.77` | **合计 `24,106` 次**（`11,933/12,800` 轨迹 = **`93.23%`**） | Step 3–25 稳态 `update_actor=15.20s`, `update_weights=4.38s`（Step 1–25 均值 `4.35s`） | Step 2–25 均值 **`39.84s/step`**；Step 3–25 稳态均值 **`38.59s/step`**；Step 1–25 全程均值 **`43.72s/step`**（全重叠步 **`18.55s–24.63s`**） | 合计 **`13,825,419`** tokens；25 步训练墙钟（`1,104.5s`，不含首尾评估，包含训练中的动态编译与 autotune）总吞吐 **`12,517.4 tok/s`**（**`1,564.7 tok/s/GPU`**）；Step 3–25 稳态（`12,747,392 tok / 887.58s`）总吞吐 **`14,362.0 tok/s`**（**`1,795.2 tok/s/GPU`**）；Step 25 单步峰值 **`28,656.9 tok/s`**（**`3,582.1 tok/s/GPU`**） |

### 6.4 25 步全量 `rollout_audit/{1..25}.jsonl` 掩码与数据完整性审计

- **审计覆盖范围**：25 步 × 每步 8 个完整多轮会话 = **`200` 个审计会话**（同时对 `rollout/{1..25}.jsonl` 全部 **`12,800` 条训练轨迹**进行工具块扫描）：
  - 全量训练轨迹含 `<tool_response>` 比例：**`11,933 / 12,800` (`93.23%`)**，累计触发 **`24,106` 次** SandboxFusion 代码执行；
  - 200 个抽样审计会话中，`183 / 200` (`91.50%`) 含 `<tool_response>`，共包含 `391` 个 `<tool_call>` 助手生成片段（`mask == 1`）与 `386` 个 `<tool_response>` 工具返回片段（`mask == 0`）；
  - **`mask_length_matches == True`**：**`200 / 200` (`100%`)**；
  - **`<tool_response>` 落入 `mask == 1` 的违规片段数**：**`0 / 386` (`0` 违规)**；
  - **`warnings` / `error` 计数**：**`0` / `0`**。
- **显存水位（全 25 步恒定无泄漏）**：
  - `actor/perf/max_memory_allocated_gb`：**`154.47 GB` (`143.86 GiB`)**
  - `actor/perf/max_memory_reserved_gb`：**`172.57 GB` (`160.72 GiB`)**（低于单卡实测 `184.31 GiB` HBM 容量，留有 `23.6 GiB` 裕量，全 25 步 0 OOM）

---

## 7. NVIDIA GB200 与 TPU v7x / TPU v6e 跨平台阶段与性能对照总结

| 对照维度 | NVIDIA GB200（本 PR #9 实测） | TPU v7x Ironwood（PR #7 实测） | Askhat TPU v6e 参照基线（[`jialei777/verl-upstream#108`](https://github.com/jialei777/verl-upstream/pull/108) 8K/8-turn run） |
| --- | --- | --- | --- |
| **硬件规模** | **2 节点 × 4 张 GB200 = 8 张 GPU**（`4 trainer FSDP2=4 + 4 rollout TP=4`） | **2 Slice × `2×2×4` = 32 芯片 / 64 TensorCore**（`16 trainer + 16 rollout`，共 8 个 TPU Hosts） | **2 Slice × `v6e-32` = 64 张 v6e 物理芯片**（`32 trainer + 32 rollout`） |
| **SFT 起点** | 复用 TPU `p2-retool-sft-01/global_step_372/huggingface`（SHA256 `59247a00...`） | `p2-retool-sft-01`（372/372 步完成，末 10 步 `train/loss=0.039265`） | 372 步 SFT（末 10 步 `train/loss=0.039429`） |
| **多轮 RL Smoke（`G1` vs `P1c`）** | **PASS（`g1-retool-rl-smoke-02`/`03`，3/3 步完成）**：`ppl_ratio=1.009–1.017`，`172` 次工具调用，掩码违规 `0` | `p1c-retool-rl-03` Step 1（同步前）通过工具调用（`18` 次）、掩码校验（`0` 违规）与 Actor 更新（`grad_norm=0.7330`, `ppl_ratio=0.9923`）；`p1c-retool-rl-05`（`VERIFY_PARITY=all`）已将 Step-0 `10,414` 处 exact mismatch 定位至 Raiden 权重同步路径，具体根因待受控运行验证（见 PR #10） | - |
| **25 步正式 RL（`G2` vs `P3`）完成度** | **PASS（`g2-retool-rl-25step-01`，25/25 步 + Step 0/25 AIME `n=30` 全部完成）** | 待 `P1c` 完成受控验证后启动（`0/25` 步） | 完成 25+ 步（PR #108 正文报告 Step 0 $\rightarrow$ Step 25 结果） |
| **AIME 2024 `acc/mean@30` (Step 0 $\rightarrow$ Step 25)** | **`23.78%` $\rightarrow$ `28.78%` (`+5.00 pp`)**（`strict=True` 口径为 **`24.22%` $\rightarrow$ `29.67%`, `+5.44 pp`**），`maj@30`: **`40.55%` $\rightarrow$ `47.13%`**，`best@30`: **`64.29%` $\rightarrow$ `69.05%`** | 待 `Stage P3` 实测 | Step 0 **`22.44%`** $\rightarrow$ Step 25 **`30.00%`**（`+7.56 pp`，PR #108 8K/8-turn run） |
| **AIME 2025 `acc/mean@30` (Step 0 $\rightarrow$ Step 25)** | **`17.00%` $\rightarrow$ `17.89%` (`+0.89 pp`)**（`strict=True` 口径为 **`17.11%` $\rightarrow$ `18.00%`, `+0.89 pp`**），`maj@30`: **`29.84%` $\rightarrow$ `30.60%`**，`best@30`: **`45.52%` $\rightarrow$ `57.34%` (`+11.82 pp`)** | 待 `Stage P3` 实测 | - |
| **单次权重同步耗时 (`update_weights`)** | **`3.87 s – 5.37 s`**（Step 1–25 均值 **`4.35 s`**，Step 3–25 均值 **`4.38 s`**，`actor FSDP2=4 -> NCCL -> vLLM TP=4`） | `P1a` 实测约 `6–10 s`（Raiden DCN） | **`13.2 s`**（Raiden P2P，PR #108） |
| **单步 Actor 更新耗时 (`update_actor`, BS=64×8)** | 稳态（Step 3–25）**`12.89 s – 18.95 s`**（均值 **`15.20 s`**） | 待 `Stage P3` 实测 | **`94.2 s`**（PR #108） |
| **单步端到端耗时与吞吐 (`timing_s/step`, BS=64×8)** | Step 2–25 均值 **`39.84 s/step`**、Step 3–25 稳态均值 **`38.59 s/step`**、Step 1–25 全程均值 **`43.72 s/step`**（异步全重叠步 **`18.55 s – 24.63 s/step`**）；总吞吐：25 步训练墙钟（`1,104.5s`，不含首尾评估，包含训练中的动态编译与 autotune）**`12,517.4 tok/s`**（**`1,564.7 tok/s/GPU`**），Step 3–25 稳态（`12,747,392 tok / 887.58s`）**`14,362.0 tok/s`**（**`1,795.2 tok/s/GPU`**），Step 25 单步峰值 **`28,656.9 tok/s`**（**`3,582.1 tok/s/GPU`**）；墙钟耗时：25 步训练本体 **`18m 24s`**，`tqdm`（25 步 + Step 25 评估）**`21m 35s`**，**含 Step 0 + Step 25 两次完整 `n=30` AIME 评估共 `24m 36s`**（`main_ppo` 全程 `27m 37s`） | 待 `Stage P3` 实测 | **`108.1 s/step`**（PR #108） |

---

## 8. 待办事项（TODO）：同加速器数量（Iso-Accelerator-Count）TPU v7x vs NVIDIA GB200 性能基准对照

当前各线的硬件物理卡数尚未对齐，因此不能直接将上述单步耗时作为等规模硬件性能对比结论：

| 对照线 | 物理芯片 / GPU 总数 | 逻辑加速核总数 | Trainer / Rollout 切分 | 备注 |
| --- | --- | --- | --- | --- |
| **Askhat 原始基线（TPU v6e，PR #108）** | **64 张 v6e 物理芯片** | 64 核心（1 核/芯片） | `32 v6e trainer + 32 v6e rollout`（`2 × v6e-32`） | 单芯片 32 GiB HBM |
| **本仓库 TPU v7x 当前配置（PR #7）** | **32 张 v7x 物理芯片** | 64 逻辑 TensorCore（2 核/芯片） | `16 v7x 芯片 (32 核) trainer + 16 v7x 芯片 (32 核) rollout`（`2 × 2×2×4`，共 8 个 TPU Hosts） | 单芯片 192 GiB HBM（每逻辑核 96 GiB） |
| **本仓库 NVIDIA GB200 当前配置（本 PR #9）** | **8 张 GB200 GPU** | 8 张 GPU | `4 GB200 trainer + 4 GB200 rollout`（2 个 GPU 节点） | 单卡 184.31 GiB HBM |

- **后续同卡数（Iso-Accelerator-Count）性能对照 TODO（已在独立 PR #11 [`ISO_CHIP_PERF_BENCHMARK_PLAN.md`](https://github.com/syangx39/verl-upstream/pull/11) 中建立专门基准计划与占位记录）**：
  1. **先行完成 `Stage P1c` 受控验证与 `Stage P3`（32 芯片 `2 × 2×2×4` 基线）**；
  2. **Tier A：8 物理芯片主对照档（`8 × TPU v7x 物理芯片` vs 现有 `8 × NVIDIA GB200 GPU` `G2`）**：在 TPU v7x（`4 芯片 trainer + 4 芯片 rollout`，即双 `2×2×1` 拓扑、每侧 1 个 TPU Host、合计 2 个 TPU Hosts / 8 颗物理芯片 / 16 逻辑 TensorCore）上采用与 `G2` **完全一致的运行期奖励语义（`default_compute_score` / `strict_box_verify=False`）** 和 `BS=64, n=8, lr=1.4e-6, trainer.v1.separate_async.num_warmup_batches=2`，直接与现有 `G2` 对比稳态 `update_actor`、`update_weights`、`gen`、`timing_s/step` 及总吞吐（`tok/s` 与 `tok/s/accelerator`）；
  3. **Tier B：32 物理芯片扩展档（`32 × TPU v7x 物理芯片` vs `32 × NVIDIA GB200 GPU`）**：作为后续扩展，在 TPU v7x `Stage P3` 原规格（`2 × 2×2×4`，Trainer `FSDP2=8, CP=4, TP=1, EP=1` + Rollout `1 × TP=32, EP=32`）与 8 节点 `32 × GB200 GPU`（`16 GPU trainer + 16 GPU rollout`）之间进行 32 物理加速器规模扩展对照，同时完整记录拓扑、总 token 数与计时边界以解释跨架构性能差异。

