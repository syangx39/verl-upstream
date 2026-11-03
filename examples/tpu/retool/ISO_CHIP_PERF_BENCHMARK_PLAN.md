# ReTool（Qwen3-30B-A3B）：同物理加速器数量（Iso-Accelerator-Count）TPU v7x vs NVIDIA GB200 性能基准对照计划与占位记录

状态：**TODO / 待执行（Placeholder PR，待 TPU v7x `Stage P1c` Raiden P2P 同步修复后执行）**
关联 PR：
- TPU v7x 实验记录：PR #7（[`REPRO_RESULTS_TPU7X.md`](https://github.com/syangx39/verl-upstream/pull/7)）
- NVIDIA GB200 实验记录：PR #9（[`REPRO_RESULTS_GB200_ZH.md`](https://github.com/syangx39/verl-upstream/pull/9)）
- TPU v7x Step-0 Raiden P2P 权重同步截断根因诊断：PR #10（[`TPU7X_RAIDEN_WEIGHT_SYNC_DIAGNOSIS.md`](https://github.com/syangx39/verl-upstream/pull/10)）

---

## 1. 背景与问题定义：当前各线物理芯片 / GPU 数量差异

在现有 ReTool（`Qwen3-30B-A3B` 多轮代码调用 DAPO RL）复现实验中，三条基线所使用的**物理加速器芯片数量**并不相同：

| 实验线 | 加速器型号 | 物理芯片 / GPU 总数 | 逻辑加速核总数 | Trainer / Rollout 资源切分 | 单物理芯片 / 单卡 HBM 容量 | 全集群总 HBM 容量 | 当前状态 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| **Askhat 原始基线** | Google Cloud TPU v6e (Trillium) | **64 张物理芯片**（2 × `v6e-32`） | 64 核（1 核/芯片） | `32 芯片 trainer + 32 芯片 rollout` | `32 GiB` | `2,048 GiB` | 历史基线 |
| **本仓库 TPU v7x 配置（PR #7）** | Google Cloud TPU v7x (Ironwood) | **32 张物理芯片**（2 × `4x4` slice） | 64 逻辑 TensorCore（2 核/芯片） | `16 物理芯片 (32 核) trainer + 16 物理芯片 (32 核) rollout` | `192 GiB`（`96 GiB` / 逻辑核） | `6,144 GiB` | `P2` SFT 完成；`P1c` 定位到 Raiden P2P 同步 bug（PR #10） |
| **本仓库 NVIDIA GB200 配置（PR #9）** | NVIDIA GB200 (`sm_100a`) | **8 张 GPU**（2 节点 × 4 卡） | 8 张 GPU | `4 GPU trainer (FSDP2=4) + 4 GPU rollout (vLLM TP=4)` | `184.31 GiB` | `1,474.5 GiB` | **PASS**（G2 25/25 步完成，稳态 `38.59 s/step`） |

> **关键结论**：
> - TPU v7x 单颗物理芯片封装了 2 个逻辑 TensorCore（总计 `192 GiB` HBM3e），与单张 NVIDIA GB200 GPU（`184.31 GiB` HBM3e）在**单物理芯片 / 单加速卡 HBM 容量（`192 GiB` vs `184.31 GiB`）与算力等级**上处于同一量级；
> - 但当前 PR #7 配置使用的是 **32 颗 TPU v7x 物理芯片**，而 PR #9 使用的是 **8 张 NVIDIA GB200 GPU**（相差 4 倍物理芯片数），Askhat 基线使用的是 **64 颗 TPU v6e 物理芯片**；
> - 因此，为了进行公平、严谨的跨平台性能（吞吐、单步耗时、权重同步延迟、有效 MFU）比较，必须在**相同物理加速器数量（Iso-Accelerator-Count：相同数量的 TPU v7x 物理芯片 vs 相同数量的 NVIDIA GB200 GPU）**下开展对照测试。

---

## 2. 同物理加速器数量（Iso-Accelerator-Count）实验矩阵设计

我们规划两个等物理卡数对照档位（**Tier A：8 物理芯片 vs 8 GPU** 与 **Tier B：32 物理芯片 vs 32 GPU**），固定相同的训练起点（共享 `p2-retool-sft-01` Step-372 SFT checkpoint，`SHA256=59247a00...`）与相同的算法超参：

- **统一超参**：
  - `data.train_batch_size = 64`，`actor.ppo_mini_batch_size = 64`，`actor_rollout_ref.rollout.n = 8`（每步 `512` 条多轮轨迹）
  - `max_prompt_length = 2048`，`max_response_length = 8192`，`multi_turn.max_user_turns = 8`，`multi_turn.max_assistant_turns = 8`
  - `trainer.v1.sampler.num_warmup_steps = 2`，`trainer.v1.sampler.max_off_policy_threshold = 4`
  - 统一评分器：显式传入 `reward.custom_reward_function.path=pkg://examples.tpu.retool.retool_tpu` 与 `reward.custom_reward_function.name=compute_score`（确保 V1 `RewardLoopWorker` 激活 `strict_box_verify=True`）

### 2.1 Tier A：`8 × TPU v7x 物理芯片` vs `8 × NVIDIA GB200 GPU`（8-Accelerator Iso-Count）

| 配置项 | TPU v7x（8 物理芯片档） | NVIDIA GB200（8 GPU 档，已在 PR #9 G2 实测） |
| --- | --- | --- |
| **物理加速器数量** | **8 颗 TPU v7x 物理芯片**（2 × `2x2` slice，共 16 逻辑 TensorCore / 4 Host） | **8 张 NVIDIA GB200 GPU**（2 × 4-GPU 节点） |
| **集群总 HBM 容量** | `8 × 192 GiB = 1,536 GiB` | `8 × 184.31 GiB = 1,474.5 GiB` |
| **Trainer 侧分配** | **4 颗 v7x 物理芯片**（8 逻辑 TensorCore，`FSDP2=8, TP=1, EP=1`） | **4 张 GB200 GPU**（`FSDP2=4, TP=1, EP=1`） |
| **Rollout 侧分配** | **4 颗 v7x 物理芯片**（8 逻辑 TensorCore，vLLM TPU `TP=8`） | **4 张 GB200 GPU**（vLLM CUDA `TP=4`） |
| **权重同步通道** | Raiden DCN P2P（4 芯片 Trainer $\rightarrow$ 4 芯片 Rollout） | NCCL P2P + NVLink5 `ComputeDomain`（4 GPU Trainer $\rightarrow$ 4 GPU Rollout） |

### 2.2 Tier B：`32 × TPU v7x 物理芯片` vs `32 × NVIDIA GB200 GPU`（32-Accelerator Iso-Count）

| 配置项 | TPU v7x（32 物理芯片档，PR #7 P3 规格） | NVIDIA GB200（32 GPU 档） |
| --- | --- | --- |
| **物理加速器数量** | **32 颗 TPU v7x 物理芯片**（2 × `4x4` slice，共 64 逻辑 TensorCore / 8 Host） | **32 张 NVIDIA GB200 GPU**（8 × 4-GPU 节点） |
| **集群总 HBM 容量** | `32 × 192 GiB = 6,144 GiB` | `32 × 184.31 GiB = 5,897.9 GiB` |
| **Trainer 侧分配** | **16 颗 v7x 物理芯片**（32 逻辑 TensorCore，`FSDP2=8, EP=4` 或 `FSDP2=32`） | **16 张 GB200 GPU**（4 节点，`FSDP2=16` 或 `FSDP2=4, EP=4`） |
| **Rollout 侧分配** | **16 颗 v7x 物理芯片**（32 逻辑 TensorCore，4 × vLLM replica `TP=8`） | **16 张 GB200 GPU**（4 节点，4 × vLLM replica `TP=4`） |

---

## 3. 结果记录表（占位 / Placeholder — 待填实测数据）

### 3.1 Tier A（8 物理芯片：`8 × TPU v7x` vs `8 × NVIDIA GB200`）性能与精度对照表

| 指标 | NVIDIA GB200（`8 × GPU`，`g2-retool-rl-25step-01` 已完成） | TPU v7x（`8 × 物理芯片 / 16 逻辑核`，TODO） | 比值 / 结论（TODO） |
| --- | --- | --- | --- |
| **Run 名称 / 代码 SHA** | `g2-retool-rl-25step-01` (`64ef634b`) | *TODO（待 PR #10 Raiden 修复后运行）* | - |
| **单次权重同步耗时 `timing_s/update_weights`（Step 3–25 均值）** | **`4.38 s`**（Step 1–25 均值 `4.35 s`） | *TODO* | *TODO* |
| **单步 Actor 更新耗时 `timing_s/update_actor`（Step 3–25 稳态均值）** | **`15.20 s`**（`12.89 s – 18.95 s`） | *TODO* | *TODO* |
| **单步端到端耗时 `timing_s/step`（Step 2–25 / Step 3–25 均值）** | **`39.84 s`** / **`38.59 s`**（全重叠步 `18.55 s – 24.63 s`） | *TODO* | *TODO* |
| **25 步训练总 Tokens / 平均吞吐 (`tok/s`)** | `13,825,419 tokens` / 峰值 `3,582 tok/s` | *TODO* | *TODO* |
| **25 步训练本体墙钟耗时（不含首尾评估）** | **`18m 24s` (`1,104.5 s`)** | *TODO* | *TODO* |
| **Step 0 / Step 25 AIME 2024 `acc/mean@30`** | `23.78% -> 28.78%` (`strict=False`) / `24.22% -> 29.67%` (`strict=True`) | *TODO* | *TODO* |
| **Step 0 / Step 25 AIME 2025 `acc/mean@30`** | `17.00% -> 17.89%` (`strict=False`) / `17.11% -> 18.00%` (`strict=True`) | *TODO* | *TODO* |

### 3.2 Tier B（32 物理芯片：`32 × TPU v7x` vs `32 × NVIDIA GB200`）与 Askhat `64 × TPU v6e` 对照表

| 指标 | Askhat 基线（`64 × TPU v6e 物理芯片`） | TPU v7x（`32 × 物理芯片 / 64 逻辑核`，TODO） | NVIDIA GB200（`32 × GPU`，TODO） |
| --- | --- | --- | --- |
| **物理芯片 / GPU 数** | `64` (`32 trainer + 32 rollout`) | `32` (`16 trainer + 16 rollout`) | `32` (`16 trainer + 16 rollout`) |
| **单次权重同步耗时 `timing_s/update_weights`** | `~8–12 s` (Raiden DCN) | *TODO* | *TODO* |
| **单步 Actor 更新耗时 `timing_s/update_actor`** | `~35–50 s` | *TODO* | *TODO* |
| **单步端到端耗时 `timing_s/step`** | `~70–110 s` | *TODO* | *TODO* |
| **Step 0 -> Step 25 AIME 2024 `acc/mean@30`** | `~16.7% -> ~26.7%` (Step 20) | *TODO* | *TODO* |

---

## 4. 执行前置条件清单（Checklist）

- [x] 完成 NVIDIA GB200 `8 × GPU`（`4 trainer + 4 rollout`）25 步基线实测与原始日志归档（PR #9）
- [x] 定位 TPU v7x `Stage P1c` Step-0 Raiden P2P 权重同步截断根因（PR #10，`tpu_sync` P2P 传输在 `layers.3.mlp.experts.up_proj` 处截断）
- [ ] 修复 TPU v7x `Qwen3-30B-A3B` Step-0 权重同步并通过 `VERIFY_PARITY=all` 全 435 张量校验
- [ ] 运行 Tier A TPU v7x `8 × 物理芯片`（`4 trainer + 4 rollout`）25 步实验并填入表 3.1
- [ ] 运行 Tier B TPU v7x `32 × 物理芯片`（`16 trainer + 16 rollout`）与 GB200 `32 × GPU` 对照实验并填入表 3.2
