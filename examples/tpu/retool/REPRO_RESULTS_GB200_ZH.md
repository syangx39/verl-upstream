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
| **Stage G0** | 共享 P2 Checkpoint 跨节点加载 + 1 步 MoE 路由正负对照 + Sandbox 四用例预检 | [`verify_g0_gb200.py`](./gb200/verify_g0_gb200.py) (`23113cb2`) | 2 节点 × 4 张 GB200（`4 trainer + 4 rollout`） | 跨双节点 `/dev/shm` 缓存并校验 25 文件 / 16 分片组合 SHA-256、正负对照 1 步 `expert_bias_E` 与 `Qwen3StateDictAdapter.to_hf()->from_hf()` 往返路由一致性、验证 Sandbox 四用例 | Head 61 s / Worker 81 s | **PASS** |
| **Stage G1** | GB200 ReTool 多轮工具调用 2 步 RL Smoke（对照 TPU `P1c`） | `g1-retool-rl-smoke-01` (`23113cb2`) | 2 节点 × 4 张 GB200（`4 trainer + 4 rollout`） | `train_batch_size=8, ppo_mini_batch_size=8, n=4, steps=2`，验证 Sandbox 四项测试、`rollout_audit` 掩码、NCCL 同步耗时与 HBM 峰值 | 运行中 | **RUNNING** |
| **Stage G2** | GB200 ReTool 25 步 RL Pilot + Step 0/25 AIME 2024/2025 评估（对照 TPU `P3`） | `g2-retool-rl-pilot-01` (`23113cb2`) | 2 节点 × 4 张 GB200（`4 trainer + 4 rollout`） | 从同一 TPU P2 `global_step_372/huggingface` 出发跑 25 步 RL（`64×8=512` rollouts/step），对比 Step 0/25 AIME `30×30` 准确率与单步耗时分解 | 待启动 | **QUEUED** |

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

## 4. Stage 0.3 & Stage G0：共享 TPU P2 SFT Checkpoint 跨节点核验、TorchTitan `0.3.0` MoE 路由正负对照与 Sandbox 四用例实测（PASS）

### 4.1 Stage 0.3：ReTool + TorchTitan `0.3.0` MoE 适配正式运行镜像

- **正式运行镜像 Digest**：`verl-torchtitan-cu130-arm64@sha256:a139e2b491ac23f06f790fe4277afb9bd543a94bc740c92173e5fdb13ddaba40`
- **双节点启动时 `--require-rl-moe-helper --require-gb200` 校验输出**：

```text
platform=Linux/aarch64 python=3.12.3
uv.lock SHA256 verified: df508db0b957fa181c229fb75885cd3863cb651fd5233d0fa5af95e990bda0be
full uv.lock check: verified 264 packages (mismatches=0)
torchtitan MoE.Config (torchtitan.models.common.moe): slotted=True required_fields=['routed_experts', 'router'] has_load_balance_coeff=True has_aux_loss=False default_load_balance_coeff=0.001
verl.workers.engine.torchtitan.utils:found-configure_qwen3_moe_for_rl
```

### 4.2 Stage G0.1：共享 TPU `Stage P2` SFT Checkpoint（`global_step_372/huggingface`）双节点 `/dev/shm` 缓存与逐文件 SHA-256 对齐

为严格控制变量（区分训练/推理栈与硬件差异，排除 SFT 起点差异），GB200 线直接从 TPU v7x `Stage P2` 产出的同一份 SFT checkpoint（`global_step_372/huggingface`，共 25 个文件 / 16 个 `.safetensors` 分片，`61,084,185,820 bytes`）启动，并在两个 `4 × GB200` 节点上分别拷入本地 `/dev/shm` 后执行逐文件 SHA-256 与参照源比对：

| 校验项 | Trainer / Head 节点 (`4 × GB200`) | Rollout / Worker-0 节点 (`4 × GB200`) | TPU v7x `Stage P2` SFT 参照源 | 对齐结论 |
| --- | --- | --- | --- | --- |
| 文件总数 / `.safetensors` 分片数 | `25` / `16` | `25` / `16` | `25` / `16` | **100% 一致** |
| 总字节数 (`total_bytes`) | `61,084,185,820` (`56.89 GiB`) | `61,084,185,820` (`56.89 GiB`) | `61,084,185,820` (`56.89 GiB`) | **100% 一致** |
| `/dev/shm` 缓存耗时与吞吐 | `46.16 s` (`1.323 GB/s`) | `66.27 s` (`0.922 GB/s`) | `55.66 s` (`1.097 GB/s`, 8 切片均值) | **正常** |
| 25 文件组合 SHA-256 (`composite_sha256`) | `59247a00c13f2364a12aab5ab6b88b6d971d59ff643c57aad8ac4e55a71231b0` | `59247a00c13f2364a12aab5ab6b88b6d971d59ff643c57aad8ac4e55a71231b0` | `59247a00c13f2364a12aab5ab6b88b6d971d59ff643c57aad8ac4e55a71231b0` | **逐字节 100% 匹配 (`matches_reference=true`)** |

### 4.3 Stage G0.2：TorchTitan `0.3.0` MoE `expert_bias_E` 1 步更新 + `Qwen3StateDictAdapter` 往返同步正负对照实验

在 GB200 (`cuda:0`, `bfloat16`) 上运行 [`verify_g0_gb200.py`](./gb200/verify_g0_gb200.py)，构建 8 层 Qwen3-30B-A3B MoE 模型（`128` 专家，`top_k=8`），执行 1 次真实 forward/backward + AdamW step，随后通过 `Qwen3StateDictAdapter.to_hf()` 与 `from_hf()` 模拟 Trainer → vLLM 的 HF 权重导出与加载（因为 HF `Qwen3MoeForCausalLM` 与 vLLM 均无 `expert_bias_E` 缓冲），对比 Actor 与同步后模型在同一批输入 token 上的每层 top-k 专家路由与输出 logits：

| 检查维度 | 正向实验（`apply_rl_fix=True`，调用 `configure_qwen3_moe_for_rl`） | 负向对照（`apply_rl_fix=False`，保持 TorchTitan `0.3.0` 默认） | 根因说明 |
| --- | --- | --- | --- |
| 各层 `moe.load_balance_coeff` | `[None, None, ..., None]`（全部 8 层为 `None`） | `[0.001, 0.001, ..., 0.001]` | TorchTitan `0.3.0` 默认开启 auxiliary-loss-free load balancing (`1e-3`) |
| 模型是否注册 `expert_bias_E` buffer | `False` | `True`（`layers.{0..7}.moe.expert_bias_E`） | 当 `load_balance_coeff is not None` 时，TorchTitan 在每次 step 中原地更新 `expert_bias_E` |
| 1 步训练后 `\|expert_bias_E\|_max` | `0.0` | `0.001453125` | 仅 1 步训练后，`expert_bias_E` 即发生非零偏移 |
| `Qwen3StateDictAdapter.to_hf()` 丢弃的键 (`dropped_state_keys`) | `[]`（**0 个键被丢弃**） | `['layers.0.moe.expert_bias_E', ..., 'layers.7.moe.expert_bias_E']`（**8 层 `expert_bias_E` 全部在同步时丢失**） | HF `Qwen3Moe` 架构不存在 `expert_bias_E`，导致同步到 vLLM 时被静默丢弃 |
| 1 步同步后 Actor vs Rollout **Top-8 专家路由一致率 (`router_topk_match_rate`)** | **`1.0000` (`100.00%`)** | **`0.6623` (`66.23%`，仅 1 步即出现 `33.77%` 专家路由分叉！)** | Actor 用含偏移的 `gate + expert_bias_E` 选专家，而 vLLM 用纯 `gate` 选专家 |
| 1 步同步后 Actor vs Rollout **输出 Logits 最大绝对误差 (`logits_max_abs_diff`)** | **`0.0`** | **`0.056640625`**（平均误差 `0.005310`） | 路由分叉直接导致 Actor 与 Rollout 的前向概率分布失配（破坏 PPO/GRPO importance ratio） |

### 4.4 Stage G0.3：SandboxFusion 本地 Loopback 服务四用例功能核验

在 Trainer Head 节点启动 `sandbox_fusion_server.py --host 127.0.0.1 --port 18080 --workers 4` 后，执行四项实测均 100% 通过：

| 用例 | 输入代码摘要 | 期望行为 | 实测输出 (`status` / `stdout` / `stderr`) | 端到端耗时 | 结果 |
| --- | --- | --- | --- | --- | --- |
| **Case 1：正常算术** | `print(sum(i * i for i in range(1, 101)))` | `status=Success`, `stdout=338350` | `status="Success"`, `stdout="338350\n"`, `return_code=0` | `0.0149 s` | **PASS** |
| **Case 2：Python 异常与 `stderr`** | `print(1 / 0)` | `status=Failed`, `stderr` 含 `ZeroDivisionError` | `status="Failed"`, `stderr="...ZeroDivisionError: division by zero\n"` | `0.0155 s` | **PASS** |
| **Case 3：死循环超时与进程清理** | `while True: time.sleep(0.1)` (`run_timeout=1.5`) | 1.5 s 超时返回 `TimeLimitExceeded` 且无残留子进程 | `run_result.status="TimeLimitExceeded"`, `execution_time=1.5015 s` | `1.5051 s` | **PASS** |
| **Case 4：8 路并发请求** | `ThreadPoolExecutor(8)` 并发计算 `1²..8²` | 8 路全部 `Success` 且输出 `['1','4',...,'64']` | `all_succeeded=True`, `outputs=['1','4','9','16','25','36','49','64']` | 最大 `0.0256 s` | **PASS** |

