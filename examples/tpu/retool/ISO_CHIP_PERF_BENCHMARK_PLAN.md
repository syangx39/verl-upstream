# ReTool（Qwen3-30B-A3B）：同物理加速器数量（Iso-Accelerator-Count）TPU v7x vs NVIDIA GB200 性能基准对照计划与占位记录

状态：**规划中**。TPU `P3` 与 8-GPU `G2` 已完成；32-GPU 同卡数对照和 64-GPU 扩展均待实现与运行。本文新增 32→64 GB200 强扩展（strong scaling）；64 GPU 对 32 颗 v7x 是不同资源数的补充比较，不属于 iso-accelerator 对照。
关联 PR：
- TPU v7x 实验记录与 `P1c/P3` 规格：PR #7（[`REPRO_RESULTS_TPU7X.md`](https://github.com/syangx39/verl-upstream/pull/7)）
- NVIDIA GB200 实验结果报告（`G2` 8-GPU 基线）：PR #9（[`REPRO_RESULTS_GB200_ZH.md`](https://github.com/syangx39/verl-upstream/pull/9)）
- TPU v7x `Stage P1c` Step-0 Raiden 权重同步现象、假设与受控验证计划：PR #10（[`TPU7X_RAIDEN_WEIGHT_SYNC_DIAGNOSIS.md`](https://github.com/syangx39/verl-upstream/pull/10)）

---

## 1. 背景与比较方法论：同物理加速器数量视角与多维归因要素

在现有 ReTool（`Qwen3-30B-A3B` 多轮代码调用 DAPO RL）复现实验中，三条基线所使用的**物理加速器芯片数量与拓扑**各不相同：

| 实验线 | 加速器型号 | 物理芯片 / GPU 总数 | 逻辑加速核与主机拓扑 | Trainer / Rollout 资源切分与并行策略 | 单物理芯片 / 单卡 HBM 容量 | 全集群总 HBM 容量 | 当前状态 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| **Askhat 上游参照（[`jialei777/verl-upstream#108`](https://github.com/jialei777/verl-upstream/pull/108)）** | Google Cloud TPU v6e (Trillium) | **64 颗物理芯片**（`2 × v6e-32`） | 64 核（1 核/芯片，16 个 TPU Hosts） | `32 芯片 trainer (FSDP2=8, CP=4) + 32 芯片 rollout (TP=32, EP=32)` | `32 GiB` | `2,048 GiB` | PR #108 历史基线（`108.1 s/step`） |
| **本仓库 TPU v7x 原规格（PR #7 `P1c/P3`）** | Google Cloud TPU v7x (Ironwood) | **32 颗物理芯片**（`2 × 2×2×4` slice） | 64 逻辑 TensorCore（2 核/芯片，4 芯片/Host，共 **8 个 TPU Hosts**） | `16 芯片 / 32 核 trainer (FSDP2=8, CP=4, TP=1, EP=1) + 16 芯片 / 32 核 rollout (1 × TP=32, EP=32)` | `192 GiB`（`96 GiB` / 逻辑核） | `6,144 GiB` | `P2` SFT、`P3` 25 步已完成；运行期为默认 scorer，详见 PR #7 / #10 |
| **本仓库 NVIDIA GB200 基线（PR #9 `G2`）** | NVIDIA GB200 (`sm_100a`) | **8 张 GPU**（2 节点 × 4 卡） | 8 张 GPU（2 个 GPU 节点） | `4 GPU trainer (FSDP2=4, TP=1, EP=1, CP=1) + 4 GPU rollout (1 × vLLM TP=4)` | `184.31 GiB` | `1,474.5 GiB` | **PASS**（`G2` 25/25 步完成，Step 3–25 稳态 `38.59 s/step`） |

> **核心比较原则**：
> 1. **同物理加速器数量（Iso-Accelerator-Count）视角**：TPU v7x 单颗物理芯片（封装 2 个逻辑 TensorCore，共 `192 GiB` HBM3e）与单张 NVIDIA GB200 GPU（`184.31 GiB` HBM3e）在单卡 HBM 容量与算力等级上处于同一量级。按相同物理芯片/GPU 数（**Tier A：8 vs 8**；**Tier B：32 vs 32**）对齐，可消除当前 PR #7（32 芯片）与 PR #9（8 GPU）之间 4 倍卡数差异。
> 2. **同时保留拓扑、实际 Token 数与计时边界以解释性能差距**：同卡数对照并非唯一维度——由于 TPU v7x 采用每颗芯片 2 个逻辑 TensorCore（例如 Tier A 中 4 颗芯片对应 8 个逻辑核 `FSDP2=8` / `TP=8`，而 4 张 GB200 对应 `FSDP2=4` / `TP=4`），且 RL 多轮生成轨迹长度随训练动态变化，所有对照表**必须同时记录并行拓扑（`FSDP/CP/TP/EP` 及 replica 数）、实际生成的总 Token 数（`sum(total_tokens)` 与 `resp_len/mean`）以及明确的墙钟计时边界（是否包含冷/热编译与 Step 0/25 评估）**，才能准确拆解计算、通信、权重同步与长尾生成对单步耗时的贡献。
> 3. **执行顺序（Phased Execution Order）**：
>    - **第一步（已完成）**：TPU v7x `Stage P1c` 受控验证（PR #10）与 `Stage P3`（32 颗物理芯片 `2 × 2×2×4` 原规格 25 步实验，PR #7）；
>    - **第二步（主线同卡数对照 Tier A）**：在 TPU v7x 上使用**与现有 GB200 `G2` 完全相同的运行期完整 Reward 语义**运行 **Tier A（`8 × TPU v7x 物理芯片` vs 现有 `8 × GB200 GPU` `G2`）**，直接复用 `G2` 结果完成首轮同卡数对照；
>    - **第三步（扩展 Tier B）**：扩展至 **Tier B（`32 × TPU v7x 物理芯片` vs `32 × GB200 GPU`）**；新增 **64 GB200** 以研究 GPU 32→64 强扩展，见 §2.5 和 §3.3。
>    - **独立配方**：自定义 `retool_tpu`（`strict_box_verify=True` + 奖励塑形）需要 GPU / TPU 双边按同一配置重跑，不能和 G2 的默认 scorer 轨道混用。

---

## 2. 奖励函数（Reward Semantics）、超参对齐与实验矩阵设计

### 2.1 奖励函数（Reward Scorer）语义对齐：为什么主线必须与现有 `G2` 保持相同运行期 Scorer

在 [`examples/tpu/retool/retool_tpu.py`](./retool_tpu.py) 中，自定义的 `retool_tpu.compute_score` 与 `verl` V1 默认回退的 [`math_dapo.compute_score(..., strict_box_verify=False)`](../../../verl/utils/reward_score/math_dapo.py) 存在**两处实质性的训练期差异**：
1. **答案提取规则不同**：`strict_box_verify=True` 仅提取 `\boxed{...}`；`strict_box_verify=False` 先匹配 Minerva `Answer:` 正则再回退 `\boxed{...}`；
2. **错误样本的工具调用奖励塑形（Tool-Use Reward Shaping）不同**：
   - `default_compute_score`（`strict_box_verify=False`）返回**纯二值结果奖励 `{-1.0, +1.0}`**；
   - `retool_tpu.compute_score` 对错误答案（`result["score"] < 0`）读取 `extra_info["num_turns"]`，将奖励设为 `min(-0.6, result["score"] + (num_turns - 2) / 2 * 0.1)`。实现直接使用对话轮数，而非读取工具调用次数；这一塑形会改变 GRPO 组内优势（Advantage）分布与后续策略梯度更新。

由于 PR #9 中已完成的 `G2`（`g2-retool-rl-25step-01`）在运行期实际使用的是 `default_compute_score`（`strict_box_verify=False`，纯 `{-1.0, +1.0}` 奖励），**事后对 `G2` 的评估输出重评分并不能改变 `G2` 在 25 步训练中已经发生的 RL 梯度更新**。因此，本计划严格区分两条对照轨道：

- **主线对照轨道（Track 1 — 与现有 `G2` 完全同配方，直接复用 `G2` 无需重跑 GPU）**：
  - **运行期训练与评估 Scorer**：TPU v7x 与 GB200 `G2` 均保持 `config.reward.custom_reward_function.path = null`（即运行期统一调用 `default_compute_score -> math_dapo.compute_score(..., strict_box_verify=False)`，纯 `{-1.0, +1.0}` 奖励）；
  - **评估双口径报告**：在完成训练后，对落盘的 Step 0 / Step 25 `validation/*.jsonl` 样本同时复算并报告 `strict_box_verify=False`（运行期口径）与 `strict_box_verify=True`（严格 `\boxed{}` 口径）两套 AIME 2024/2025 分数。
- **后续扩展轨道（Track 2 — `strict_box_verify=True` + 工具调用 Reward Shaping 双边重跑对照，可选）**：
  - 若需验证 `retool_tpu.compute_score`（`strict_box_verify=True` + 错误样本工具调用塑形奖励），必须通过 `reward.custom_reward_function.path=pkg://examples.tpu.retool.retool_tpu` 与 `reward.custom_reward_function.name=compute_score` **在 GPU 与 TPU 两边都从 SFT 起点重新运行 25 步训练**，不能将现有 `G2` 直接作为 Track 2 的同配方基线。

### 2.2 本项目 TPU/GPU 对照的训练与评估配置

以下以 `G2` [`head.log`](gb200/logs/g2-retool-rl-25step-01/head.log) 为对照；可核对的共同参数同时参考 Askhat 的 [`run_qwen3_30b_a3b_dapo_torchtitan.sh`](https://github.com/jialei777/verl-upstream/blob/266e793a20c7baa6f849476ba72c8f8052da1477/examples/tpu/retool/run_qwen3_30b_a3b_dapo_torchtitan.sh)。本项目 TPU/GPU 共享本地 P2 checkpoint，尚未证明其与 Askhat 的 checkpoint 一致；本对照运行 25 步，Askhat 的主运行是 400 步，具体运行的 scorer 需以其 resolved config 与证据为准。

- **SFT 起点**：共享 `p2-retool-sft-01/checkpoint/global_step_372/huggingface`（20 核心文件组合 SHA-256 `59247a00c13f2364a12aab5ab6b88b6d971d59ff643c57aad8ac4e55a71231b0`）
- **批次与采样**：`data.train_batch_size = 64`，`actor_rollout_ref.actor.ppo_mini_batch_size = 64`，`actor_rollout_ref.rollout.n = 8`（每步 `64 × 8 = 512` 条训练轨迹；25 步共 `12,800` 条轨迹）
- **学习率与裁剪**：`actor_rollout_ref.actor.optim.lr = 1.4e-6`，`clip_ratio_low = 0.2`，`clip_ratio_high = 0.28`，`clip_ratio_c = 10.0`，TorchTitan `training.max_norm = 1.0`
- **序列长度与多轮上限**：`max_prompt_length = 2048`，`max_response_length = 8192`，`multi_turn.max_user_turns = 8`，`multi_turn.max_assistant_turns = 8`（单条轨迹最多 `8 user + 8 assistant = 16` 轮消息）
- **V1 Separate Async 调度参数**：
  - `trainer.v1.trainer_mode = separate_async`
  - `trainer.v1.separate_async.num_warmup_batches = 2`
  - `trainer.v1.separate_async.parameter_sync_step = 1`
  - `trainer.v1.sampler.max_off_policy_threshold = 4`
- **评估配置**：`trainer.total_training_steps = 25`，`trainer.val_before_train = True`，`trainer.test_freq = 25`，`actor_rollout_ref.rollout.val_kwargs.n = 30`（AIME 2024 + AIME 2025，每次评估 `1,800` 条多轮轨迹）

### 2.3 Tier A（主线同卡数对照）：`8 × TPU v7x 物理芯片` vs `8 × NVIDIA GB200 GPU`

> **TPU v7x 拓扑说明（参考 [Google Cloud TPU v7x 文档](https://docs.cloud.google.com/tpu/docs/tpu7x)）**：每个 TPU v7x Host VM（`tpu7x-standard-4t`）搭载 **4 颗物理芯片（8 个逻辑 TensorCore 设备）**，对应 **`2×2×1`** 物理拓扑。因此 Tier A 每侧 4 颗物理芯片即为 **`2×2×1`（1 个 TPU Host，8 个逻辑 TensorCore）**，Trainer + Rollout 合计 **2 个 TPU Hosts（8 颗物理芯片 / 16 个逻辑 TensorCore）**。

| 配置项 | TPU v7x（8 物理芯片档，待运行） | NVIDIA GB200（8 GPU 档，复用 PR #9 `G2` 实测） |
| --- | --- | --- |
| **物理加速器数量与主机拓扑** | **8 颗 TPU v7x 物理芯片**（`2 × 2×2×1` slice：**Trainer 1 个 TPU Host + Rollout 1 个 TPU Host = 共 2 个 TPU Hosts**，16 个逻辑 TensorCore 设备） | **8 张 NVIDIA GB200 GPU**（**Trainer 1 个 4-GPU 节点 + Rollout 1 个 4-GPU 节点 = 共 2 个 GPU 节点**） |
| **集群总 HBM 容量** | `8 × 192 GiB = 1,536 GiB`（`96 GiB` / 逻辑 TensorCore） | `8 × 184.31 GiB = 1,474.5 GiB` |
| **Trainer 侧分配与并行策略** | **4 颗 v7x 物理芯片 / 1 个 TPU Host**（8 逻辑 TensorCore：主配置 `FSDP2 (dp_shard)=8, CP=1, TP=1, EP=1`；备选 `FSDP2=4, CP=2, TP=1, EP=1`） | **4 张 GB200 GPU / 1 个节点**（`FSDP2 (dp_shard)=4, CP=1, TP=1, EP=1`） |
| **Rollout 侧分配与并行策略** | **4 颗 v7x 物理芯片 / 1 个 TPU Host**（8 逻辑 TensorCore：单 replica `1 × vLLM TP=8, EP=8`） | **4 张 GB200 GPU / 1 个节点**（单 replica `1 × vLLM TP=4`） |
| **权重同步通道** | Raiden DCN P2P（1 Host / 8 TensorCore Trainer $\rightarrow$ 1 Host / 8 TensorCore Rollout） | NCCL P2P + NVLink5 `ComputeDomain`（4 GPU Trainer $\rightarrow$ 4 GPU Rollout） |
| **运行期 Reward 语义（Track 1 主线）** | `default_compute_score`（`strict_box_verify=False`，纯 `{-1.0, +1.0}` 奖励，与 `G2` 严格一致） | `default_compute_score`（`strict_box_verify=False`，纯 `{-1.0, +1.0}` 奖励，`G2` 已完成） |

### 2.4 Tier B（后续扩展同卡数对照）：`32 × TPU v7x 物理芯片` vs `32 × NVIDIA GB200 GPU`

> **Tier B TPU 并行配置约束说明**：
> 1. 在当前代码 [`verl/workers/engine/torchtitan/utils.py#L210-L218`](../../../verl/workers/engine/torchtitan/utils.py#L210-L218) 中，`configure_qwen3_moe_for_rl(..., use_pallas_grouped_mm=True)` 要求 `expert_parallel_size == 1`（当 `EP > 1` 时直接抛出 `ValueError: use_pallas_grouped_mm requires expert_parallel_size=1`），因此**当前 Qwen3 MoE TPU TorchTitan 训练端不支持 `EP > 1`**；
> 2. PR #7 中 `Stage P1c` 与 `Stage P3` 的原规格为：每侧 **`2×2×4` slice（16 颗物理芯片 / 4 个 TPU Hosts / 32 个逻辑 TensorCore 设备）**，其中 **Trainer 采用 `FSDP2 (dp_shard)=8, CP=4, TP=1, EP=1`**，**Rollout 采用单 replica `1 × vLLM TP=32, EP=32`**；
> 3. 因此 Tier B 的主配置严格采用 PR #7 的 `P1c/P3` 原规格；诸如 Rollout 多 replica（如 `4 × TP=8` 或 `2 × TP=16`）或 Trainer `EP > 1` 仅作为后续待验证的优化选项单独列出，不混入基线规格。

| 配置项 | TPU v7x（32 物理芯片档，PR #7 `Stage P3` 原规格） | NVIDIA GB200（32 GPU 档，后续扩展） |
| --- | --- | --- |
| **物理加速器数量与主机拓扑** | **32 颗 TPU v7x 物理芯片**（`2 × 2×2×4` slice：**Trainer 4 个 TPU Hosts + Rollout 4 个 TPU Hosts = 共 8 个 TPU Hosts**，64 个逻辑 TensorCore 设备） | **32 张 NVIDIA GB200 GPU**（**Trainer 4 个 4-GPU 节点 + Rollout 4 个 4-GPU 节点 = 共 8 个 GPU 节点**） |
| **集群总 HBM 容量** | `32 × 192 GiB = 6,144 GiB`（`96 GiB` / 逻辑 TensorCore） | `32 × 184.31 GiB = 5,897.9 GiB` |
| **Trainer 侧分配与主配置并行度** | **16 颗 v7x 物理芯片 / 4 个 TPU Hosts**（32 逻辑 TensorCore：**`FSDP2 (dp_shard)=8, CP=4, TP=1, EP=1`**，与 PR #7 `P1c/P3` 完全一致） | **16 张 GB200 GPU / 4 个节点**（对称配置 `FSDP2=4, CP=4, TP=1, EP=1` 或纯数据并行 `FSDP2=16, CP=1, TP=1, EP=1`） |
| **Rollout 侧分配与主配置并行度** | **16 颗 v7x 物理芯片 / 4 个 TPU Hosts**（32 逻辑 TensorCore：**单 replica `1 × vLLM TP=32, EP=32`**，与 PR #7 `P1c/P3` 完全一致） | **16 张 GB200 GPU / 4 个节点**（4 × vLLM replica `TP=4` 或 2 × vLLM replica `TP=8`） |
| **待验证优化选项（非基线）** | Rollout 多 replica（`2 × TP=16` 或 `4 × TP=8`）；Trainer `EP > 1`（需先解决 `TpuGroupedExperts` / `torch_tpu#2668` 限制） | Trainer `EP > 1`；Rollout 跨节点 `TP=16/32` 对照 |

---

### 2.5 64 × GB200 扩展：32→64 GPU 强扩展（待实现、待运行）

64 卡的主要问题是 **固定训练工作量时，GB200 从 32 卡扩到 64 卡能快多少、效率如何变化**。同时报告其与现有 32 颗 v7x P3 的实际端到端差距；后者是 **32 TPU 芯片 vs 64 GPU 的非等资源对照**，不能作为同卡数硬件效率结论。仅当另行运行 64 颗 v7x 且语义对齐后，才能报告 64 vs 64 的同卡数对照。

| 配置 | 32 × GB200：基准（待运行） | 64 × GB200：扩展（待运行） |
| --- | --- | --- |
| 资源假设 | 8 节点 × 4 GPU；16 trainer + 16 rollout | 16 节点 × 4 GPU；32 trainer + 32 rollout |
| Trainer 候选 | 4 节点、16 GPU；`dp_shard=16, CP=1, TP=1, EP=1` | 8 节点、32 GPU；`dp_shard=32, CP=1, TP=1, EP=1` |
| Rollout 候选 | 4 节点、16 GPU；4 个 vLLM replica × `TP=4, EP=1` | 8 节点、32 GPU；8 个 vLLM replica × `TP=4, EP=1` |
| 核心工作量 | 同 §2.2：每步 64 prompts × 8 responses、25 更新；共享 P2 checkpoint 与默认 scorer | 与 32-GPU 完全相同的全局工作量和评估；不随 GPU 数扩大 batch |
| 状态 | **计划，尚无多节点结果** | **计划，尚无多节点结果** |

上述节点规格、mesh、replica 数均为候选，需根据实际机器拓扑、GPU 可见性、Ray placement、TorchTitan / vLLM 支持和 NCCL 权重同步验证。`ROLLOUT_TP=4` 必须显式设置：现有训练脚本默认使用全部 rollout GPU 作为 TP，不能靠节点数自动得到上述 replicas。保留 `EP=1`、固定 image digest、代码 SHA、依赖与 sandbox 服务配置，并记录每个 replica 的实际放置。不要预设 16 个节点可归属同一 NVLink / ComputeDomain；记录实际 GPU 拓扑及 NVLink、跨节点网络和 ComputeDomain 的边界与带宽配置。

**实现与运行门槛**：当前 [`gb200/run_ray_jobset_gb200.sh`](gb200/run_ray_jobset_gb200.sh) 仅实现 2 个 4-GPU pod 和 8-GPU readiness，不能通过修改 `NNODES_*` 直接扩到 32 / 64 GPU。先实现并测试多节点资源创建、角色放置、实际 GPU/节点数与所有节点 readiness、checkpoint staging/hash、日志及 UID 安全清理；验证 ComputeDomain 与跨节点 NCCL 的实际要求。逐节点运行环境/模型检查，再验证全 rank NCCL、trainer→每个 rollout replica 的初始和更新后权重同步、sandbox 可达性与完整响应。先分别完成 32 / 64 GPU 的短 smoke，再运行各自的 25 步正式 benchmark；细化步骤见 [GB200 README](gb200/README.md#26-32--64-gb200-scaling-plan)。

**口径与解释**：

- 32 / 64 GPU 同一代码、image、checkpoint、scorer、全局 batch、采样参数和 trainer/rollout 分配比例；固定 `n=8`、8K / 8+8 turns、学习率、异步队列和评估配置，采集 resolved config 与参数差异。P3 使用默认 scorer；PR #15 的 16K / thinking / shaped-reward 配置是另一实验轨道，不能直接混入 G2/P3 对比。
- 以训练 W1 同边界墙钟计算 `speedup = T32 / T64`、`parallel efficiency = speedup / 2`；同时记录实际 prompt / response / 总 token、有效 session、输出长度分布、截断率、队列等待和 staleness。异步 rollout 与采样可能产生不同工作量：报告 token 吞吐和分布，不把单次结果当成严格相同轨迹的硬件性能结论。
- 报告训练 W1、Step 3–25 W2（称 warm window，不假设再无动态编译）、Step 0 / 25 评估、启动与完整端到端耗时；集群 `tok/s = 窗口总 token / 窗口墙钟`，每 GPU 再除以 32 或 64；同时报告 GPU-hours 与必要时的实际成本。重复运行并记录 seed、样本身份和波动；正式结果至少说明重复次数，单次结果仅为探索性证据。

---

## 3. 结果记录表（占位 / Placeholder）

### 3.1 Tier A（8 物理加速器主线对照：`8 × TPU v7x 物理芯片` vs `8 × NVIDIA GB200 GPU`，Track 1 相同 Reward 语义）

> **吞吐量口径与统计窗口定义（重要）**：
> - **`verl` 日志中的 `perf/throughput`**：在 `verl` 中定义为 $\text{total\_num\_tokens} / (N_{\text{accel}} \times \text{timing\_s/step})$，即**单步每加速器吞吐（`tok/s/accel`）**。例如 `G2` 在第 25 步（吞吐最高的异步重叠步，约 `21.98 s` 处理 `629,878 tokens`）记录的 `3,582.46` 是**单步峰值每 GPU 吞吐（`tok/s/GPU`）**，对应的 8 卡集群总吞吐为 `28,659.7 tok/s`，**不能与 25 步平均吞吐混用**。
> - 下表将吞吐量拆分为三个明确统计窗口，并同时报告**集群总吞吐（`Cluster Total tok/s`）**与**按物理芯片/GPU 数归一化的单加速器吞吐（`Per-Accelerator tok/s/accel`，Tier A 除以 `8`）**：
>   - **窗口 W1（25 步训练本体墙钟窗口）**：覆盖 Step 1 开始至 Step 25 权重同步完成（**不含容器启动和训练前编译，也不含 Step 0 与 Step 25 AIME 评估**；包含 Step 1–2 warmup 及训练窗口内的动态形状编译/autotune）；
>   - **窗口 W2（Step 3–25 稳态步耗时窗口）**：覆盖 Step 3 至 Step 25 共 23 个稳态训练步（**不含 Step 1–2 初始编译/autotune、不含 Step 0/25 评估**）；
>   - **窗口 W3（单步最高吞吐窗口）**：选择 `perf/throughput` 最高的训练步；它不一定是 `timing_s/step` 最短的步。G2 的最高吞吐为 Step 25，最短步为 Step 10（`18.55 s`）。

| 指标维度 | 统计窗口 / 包含边界 | NVIDIA GB200（`8 × GPU`，`g2-retool-rl-25step-01` 已完成） | TPU v7x（`8 × 物理芯片 / 16 逻辑核 / 2 Hosts`，TODO） | 比值 / 归因说明（TODO） |
| --- | --- | --- | --- | --- |
| **Run 名称 / 代码 SHA / 镜像** | - | `g2-retool-rl-25step-01` (`64ef634b`, `sha256:a139e2b4...`) | *TODO* | - |
| **并行拓扑（Trainer / Rollout）** | - | Trainer `FSDP2=4, CP=1, TP=1, EP=1` (4 GPU)<br>Rollout `1 × TP=4` (4 GPU) | Trainer `FSDP2=8, CP=1, TP=1, EP=1` (4 芯片 / 8 核)<br>Rollout `1 × TP=8, EP=8` (4 芯片 / 8 核) | 记录逻辑核切分差异 |
| **运行期实际 Reward Scorer** | 训练 + 评估全程 | `default_compute_score` (`strict_box_verify=False`, 纯 `{-1,+1}`) | `default_compute_score` (`strict_box_verify=False`, 纯 `{-1,+1}`) | **Track 1 完全对齐** |
| **25 步累计训练总 Token 数** | Steps 1–25 合计（不含评估） | **`13,825,419 tokens`**（均值 `553,017 tok/step`；`resp_len/mean = 760.4`） | *TODO* | 对比实际处理 token 量 |
| **Step 3–25 稳态累计训练 Token 数** | Steps 3–25 合计（23 步，不含评估） | **`12,747,392 tokens`**（均值 `554,234 tok/step`） | *TODO* | - |
| **单次权重同步 `timing_s/update_weights`** | Step 1–25 均值 / Step 3–25 稳态均值 | **`4.35 s`** / **`4.38 s`**（范围 `3.87 s – 5.37 s`） | *TODO* | NCCL P2P vs Raiden DCN |
| **单步 Actor 更新 `timing_s/update_actor`** | Step 1–2 (含 autotune) / Step 3–25 稳态均值 | Step 1–2: `46.32 s, 64.28 s`<br>**Step 3–25 稳态均值: `15.20 s`**（`12.89 s – 18.95 s`） | *TODO* | 对比反向传播+优化器步耗时 |
| **单步端到端耗时 `timing_s/step`** | Step 1–25 全程 / Step 2–25 / **Step 3–25 稳态** | Step 1–25: **`43.72 s`** \| Step 2–25: **`39.84 s`**<br>**Step 3–25 稳态: `38.59 s`**（全重叠步 `18.55 s – 24.63 s`） | *TODO* | - |
| **平均总吞吐 & 单卡吞吐（窗口 W1：25 步训练墙钟）** | 仅 25 步训练本体（`1,104.5 s`），**不含训练前编译与 Step 0/25 评估，包含训练期动态编译/autotune** | 集群总吞吐: **`12,517.4 tok/s`**<br>单物理加速器 (`÷ 8`): **`1,564.7 tok/s/GPU`** | *TODO* | `13,825,419 tok / 1,104.5 s` |
| **稳态总吞吐 & 单卡吞吐（窗口 W2：Step 3–25）** | Step 3–25 共 23 步（`887.58 s`），**不含 Step 1–2 编译与评估** | 集群总吞吐: **`14,362.0 tok/s`**<br>单物理加速器 (`÷ 8`): **`1,795.2 tok/s/GPU`** | *TODO* | `12,747,392 tok / 887.58 s` |
| **单步峰值总吞吐 & 单卡吞吐（窗口 W3：峰值步）** | Step 25 单步（`629,878 tok` / `21.98 s`），异步全重叠 | 集群总吞吐: **`28,659.7 tok/s`**<br>单物理加速器 (`perf/throughput`): **`3,582.46 tok/s/GPU`** | *TODO* | 单步峰值（非全程均值） |
| **墙钟耗时拆解（Wall-Clock Breakdown）** | 1. 启动+模型加载+Step 0 同步<br>2. Step 0 初始评估 (`n=30`)<br>3. **25 步训练本体**<br>4. Step 25 末次评估 (`n=30`)<br>5. **含首尾两次评估 + 25 步总计**<br>6. `main_ppo` 端到端全程 | 1. `2m 58s` (`178.2 s`，复用热编译缓存)<br>2. `3m 00s` (`180.1 s`)<br>3. **`18m 24s` (`1,104.5 s`)**<br>4. `3m 10s` (`190.4 s`)<br>5. **`24m 36s` (`1,476.4 s`)**<br>6. `27m 37s` (`1,657.1 s`) | *TODO* | 明确区分训练本体、评估与编译耗时 |
| **AIME 2024 `acc/mean@30` (Step 0 $\rightarrow$ Step 25)** | 运行期 `strict=False` / 事后复算 `strict=True` | `strict=False`: **`23.78%` $\rightarrow$ `28.78%` (`+5.00 pp`)**<br>`strict=True`: **`24.22%` $\rightarrow$ `29.67%` (`+5.44 pp`)** | *TODO* | 双口径同时报告 |
| **AIME 2025 `acc/mean@30` (Step 0 $\rightarrow$ Step 25)** | 运行期 `strict=False` / 事后复算 `strict=True` | `strict=False`: **`17.00%` $\rightarrow$ `17.89%` (`+0.89 pp`)**<br>`strict=True`: **`17.11%` $\rightarrow$ `18.00%` (`+0.89 pp`)** | *TODO* | 双口径同时报告 |

### 3.2 Tier B（32 物理加速器扩展对照：`32 × TPU v7x 物理芯片` vs `32 × NVIDIA GB200 GPU`）与 Askhat PR #108 (`64 × TPU v6e`) 对照表

> **Askhat 参考数据来源说明**：下表第一列严格引用 Askhat 在 [`jialei777/verl-upstream#108`](https://github.com/jialei777/verl-upstream/pull/108) 正文报告的 **`Qwen3-30B-A3B` 8K response length / 8-turn ReTool DAPO run（`2 × v6e-32` = 64 颗 TPU v6e 物理芯片）** 实测数字（`timing_s/step = 108.1 s`、`update_actor = 94.2 s`、`Raiden P2P update_weights = 13.2 s`、AIME 2024 `22.44% -> 30.00%` at Step 25），不混用早期 v5p 或 16K/16-turn 变体数据。

| 指标维度 | Askhat PR #108 基线（`64 × TPU v6e 物理芯片`，`2 × v6e-32`） | TPU v7x `Stage P3`（`32 × 物理芯片 / 64 逻辑核`，已完成；本表待填） | NVIDIA GB200（`32 × GPU`，`8 × 4-GPU 节点`，待运行） |
| --- | --- | --- | --- |
| **物理芯片 / GPU 数与主机数** | **64 颗 v6e 物理芯片**（`32 trainer + 32 rollout`，16 Hosts） | **32 颗 v7x 物理芯片**（`16 trainer + 16 rollout`，**8 个 TPU Hosts**，64 逻辑 TensorCore） | **32 张 GB200 GPU**（`16 trainer + 16 rollout`，**8 个 GPU 节点**） |
| **Trainer 并行配置** | `FSDP2=8, CP=4, TP=1, EP=1`（32 核） | **`FSDP2=8, CP=4, TP=1, EP=1`**（32 逻辑 TensorCore，PR #7 原规格） | `FSDP2=4, CP=4, TP=1, EP=1` 或 `FSDP2=16, CP=1`（16 GPU） |
| **Rollout 并行配置** | `1 × vLLM TP=32, EP=32`（32 核） | **`1 × vLLM TP=32, EP=32`**（32 逻辑 TensorCore，PR #7 原规格） | `4 × vLLM TP=4` 或 `2 × vLLM TP=8`（16 GPU） |
| **单次权重同步耗时 `timing_s/update_weights`** | **`13.2 s`**（Raiden P2P，[`jialei777/verl-upstream#108`](https://github.com/jialei777/verl-upstream/pull/108)） | *TODO* | *TODO* |
| **单步 Actor 更新耗时 `timing_s/update_actor`** | **`94.2 s`**（[`jialei777/verl-upstream#108`](https://github.com/jialei777/verl-upstream/pull/108)） | *TODO* | *TODO* |
| **单步端到端耗时 `timing_s/step`** | **`108.1 s/step`**（[`jialei777/verl-upstream#108`](https://github.com/jialei777/verl-upstream/pull/108)） | *TODO* | *TODO* |
| **25 步累计训练总 Token 数** | 未在 PR #108 正文单独列出 | *TODO* | *TODO* |
| **窗口 W1（25 步训练墙钟）总吞吐 `tok/s` / 单物理加速器 `tok/s/accel`** | - | *TODO*（总 `tok/s` 及 `÷ 32` 芯片 `tok/s/chip`） | *TODO*（总 `tok/s` 及 `÷ 32` 卡 `tok/s/GPU`） |
| **窗口 W2（Step 3–25 稳态）总吞吐 `tok/s` / 单物理加速器 `tok/s/accel`** | - | *TODO*（总 `tok/s` 及 `÷ 32` 芯片 `tok/s/chip`） | *TODO*（总 `tok/s` 及 `÷ 32` 卡 `tok/s/GPU`） |
| **Step 0 $\rightarrow$ Step 25 AIME 2024 `acc/mean@30`** | PR #108 报告 **`22.44%` $\rightarrow$ `30.00%`**；PR #15 的运行审计显示两个端点来自不同运行，不应解释为单次训练提升 | *TODO* | *TODO* |
| **Step 0 $\rightarrow$ Step 25 AIME 2025 `acc/mean@30`** | PR #108 报告 **`14.17%` $\rightarrow$ `18.33%`**；PR #15 的运行审计显示两个端点来自不同运行，不应解释为单次训练提升 | *TODO* | *TODO* |

---

### 3.3 GB200 32→64 强扩展与 TPU 参考（全部待测，P3 已完成）

| 指标 | 32 GB200 | 64 GB200 | 32 v7x P3（非等资源参考） |
| --- | --- | --- | --- |
| Run / 代码 / image digest / 实际节点拓扑 | TODO | TODO | 查 PR #7 原始记录 |
| Trainer / rollout、mesh、replica 放置 | 16 / 16，候选见 §2.5 | 32 / 32，候选见 §2.5 | 16 / 16 物理芯片，见 §2.4 |
| Resolved config / scorer / checkpoint / 实际 prompt、response tokens | TODO | TODO | TODO，验证是否同口径 |
| W1 训练墙钟 / 总 tok/s / 每卡 tok/s | TODO | TODO | TODO |
| W2 Step 3–25 耗时 / 总 tok/s / 每卡 tok/s | TODO | TODO | TODO |
| `timing_s/gen`、`update_actor`、`update_weights`，队列与 staleness | TODO | TODO | TODO |
| Step 0 / 25 评估、端到端墙钟、GPU-hours | TODO | TODO | TODO / TPU-chip-hours |
| Step 0→25 AIME 2024 / 2025、样本完整性 | TODO | TODO | 查 PR #7，核对评估口径 |
| 32→64 加速比 / 效率 | 基准 1× | TODO：`T32/T64` / `(T32/T64)/2` | 不适用 |

---

## 4. 分阶段执行路线图与前置条件清单（Checklist）

- [x] **完成 NVIDIA GB200 `8 × GPU`（`4 trainer + 4 rollout`）25 步基线实测与原始证据归档（PR #9）**：已归档完整 `head.log`、3,600 条评估样本、双口径（`strict_box_verify=False` vs `True`）复算脚本与墙钟计时拆解。
- [x] **完成 TPU v7x `Stage P1c` 同步问题诊断与新版路径验证（PR #10）**：`p1c-retool-rl-05` 曾出现大量 Step-0 `exact` mismatch；后续运行的初始逐元素检查和逐步 norm 检查范围不同，详细证据与归因边界见 PR #10。
- [x] **阶段 1：完成 `Stage P1c` 受控验证与 `Stage P3` 原规格 25 步**；具体证据与逐步同步检查边界见 PR #7 / #10。表 3.2 待按统一口径填数。
- [ ] **阶段 2（执行 Tier A 主线同物理卡数对照：`8 × TPU v7x 物理芯片` vs 现有 `8 × GB200 GPU` `G2`）**：
  - [ ] 在 2 个 TPU v7x Hosts（`2 × 2×2×1` = 8 颗物理芯片 / 16 逻辑 TensorCore）上使用**与 `G2` 完全相同的运行期 Reward 语义（Track 1：`default_compute_score` / `strict_box_verify=False`）**与超参（`BS=64, n=8, lr=1.4e-6, trainer.v1.separate_async.num_warmup_batches=2`）运行 25 步实验，填入表 3.1 并完成同卡数吞吐/耗时拆解分析。
- [ ] **阶段 3（后续扩展：Tier B `32 × GB200 GPU` 对照与可选 Track 2 Shaped-Reward 实验）**：
  - [ ] 视集群资源情况扩展运行 `32 × NVIDIA GB200 GPU`（8 节点）25 步实验，与 `Stage P3`（`32 × TPU v7x 物理芯片`）完成 Tier B 对照；
  - [ ] （可选 Track 2）若需评估 `retool_tpu.compute_score`（`strict_box_verify=True` + 错误样本工具调用奖励塑形），在 GPU 与 TPU 两侧均显式设置 `reward.custom_reward_function.path=pkg://examples.tpu.retool.retool_tpu` 重新运行 25 步对照。
- [ ] **阶段 4（64 GB200 强扩展）**：完成多节点 launcher 和通信/同步验证，分别执行 32 与 64 GPU smoke 和 25 步固定工作量 benchmark，收集 §3.3 的完整证据与资源开销；64 GPU vs 32 v7x 明确标为非等资源对照。

