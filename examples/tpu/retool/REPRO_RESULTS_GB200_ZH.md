# ReTool（Qwen3-30B-A3B）：NVIDIA GB200 复现进展、结果与 TPU v7x 对照记录

状态：**执行中（In Progress，2026-11-02 启动）**
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
| **Stage G1 (Attempt 3)** | GB200 ReTool 多轮工具调用 3 步 RL Smoke + Token 级 `rollout_audit` 落盘 | `g1-retool-rl-smoke-03` (`64ef634b`) | 2 节点 × 4 张 GB200（复用热 SM100 FlashInfer/vLLM 缓存） | 默认开启 `trainer.rollout_data_dir` 与 `+trainer.rollout_audit_dir`，落盘 `rollout_audit/{1,2,3}.jsonl` 以执行逐 token `response_mask` 离线审计 | 运行中（预计 ~5 m） | **RUNNING** |
| **Stage G2** | GB200 ReTool 25 步 RL Pilot + Step 0/25 AIME 2024/2025 评估（对照 TPU `P3`） | `g2-retool-rl-pilot-01` (`64ef634b`) | 2 节点 × 4 张 GB200（`4 trainer + 4 rollout`） | **严格前置条件：需 `Stage G1` 落盘 `rollout_audit` 掩码核验与 TPU `Stage P1c` 均通过后方可启动** | 待 `G1` 审计与 `P1c` 通过后启动 | **BLOCKED on G1 audit & P1c** |

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

### 4.4 Stage G0.3：SandboxFusion 本地 Loopback 服务四用例功能核验

在 Trainer Head 节点启动 `sandbox_fusion_server.py --host 127.0.0.1 --port 18080 --workers 4` 后，执行四项实测（原始产物 `g1-retool-rl-smoke-01-head-sandbox-4case.json`）均通过：

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

- **补充落盘运行（`g1-retool-rl-smoke-03`，代码 SHA `64ef634b`）**：
  - `g1-retool-rl-smoke-02` 启动命令未显式传入 `trainer.rollout_data_dir` 与 `+trainer.rollout_audit_dir`；已在 `64ef634b` 中将上述落盘参数及 SM100 缓存自动恢复设为默认并启动 `g1-retool-rl-smoke-03`，用于归档 `rollout_audit/{1,2,3}.jsonl` 逐 token 掩码审计文件。



