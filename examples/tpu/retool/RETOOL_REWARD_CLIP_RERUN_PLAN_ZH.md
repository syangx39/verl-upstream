# ReTool reward 与 clipping 修复及 GPU/TPU 重跑方案

状态：**待 review**（2026-11-03）。本文只是方案，还没有改代码，也没有提交新的运行。

相关：TPU 结果和 scorer 证据见 PR #7（[`results/p3-retool-rl-11/`](https://github.com/syangx39/verl-upstream/tree/retool-tpu7x-exp-results/examples/tpu/retool/results/p3-retool-rl-11)）；GPU `G2` 的记录见 [`gb200/logs/g2-retool-rl-25step-01/`](gb200/logs/g2-retool-rl-25step-01/)；两边解析后配置的对比见 [`results/config-diff-p3-retool-rl-11-vs-g2/`](results/config-diff-p3-retool-rl-11-vs-g2/README.md)。

## 0. 目标与边界

- **目标**：在 GPU 和 TPU 上，用代码中预期的 ReTool reward 跑同一套配置，并做对照。预期 reward 是 `strict_box_verify=True` 加 tool-use shaping。
- **不做**：不为比较现有 GPU/TPU 的 reward 一致性而重跑。现有 TPU `p3-retool-rl-11` 和 GPU `G2` 已经是同一配置下的对照基线：默认 scorer，有效 clipping `max_norm=1.0`。这只说明这两项语义一致；其他配置、评估和性能口径仍要逐项对齐（第 6 节）。
- **公平性要求**：新配置两边都从共享的 P2 checkpoint 重跑。不拿新的 TPU run 和旧 `G2` 当同配置对照。
- **归因边界**：Askhat 的脚本写了 `clip_grad=0.088`，但他实际是否生效、是否用了 shaped scorer，要看他的 resolved config 或日志才能确认。在那之前，新 run 只称"预期 ReTool 配置"，不称与 Askhat 的实际运行严格一致。

## 1. 现有基线 A（保留，不覆盖）

| | TPU `p3-retool-rl-11` | GPU `G2`（`g2-retool-rl-25step-01`） |
|---|---|---|
| 代码 / 镜像 | `aa10dfed`（通过 tarball 下发），`verl-tpu@sha256:d92a2b3d...` | `64ef634b`（镜像内 `verl-pin`），`verl-torchtitan-cu130-arm64@sha256:a139e2b4...` |
| 资源 | 1 trainer + 1 rollout `2x2x4` 切片（32 chips） | 2 节点 × 4 GB200（trainer `FSDP2=4` + rollout `TP=4`） |
| 起点 | `p2-retool-sft-01/global_step_372/huggingface` | 同一个 checkpoint |
| Reward | 默认 `math_dapo`，`strict_box_verify=False`，无 shaping（`driver.log` L946：`reward.custom_reward_function.path: None`） | 相同（[`head.log` L899](https://github.com/syangx39/verl-upstream/blob/48812b043942a03859f841d886936e0c968702b5/examples/tpu/retool/gb200/logs/g2-retool-rl-25step-01/head.log#L899)） |
| 有效 clipping | TorchTitan `training.max_norm=1.0`（`driver.log` L1588） | 相同（[`head.log` L1444](https://github.com/syangx39/verl-upstream/blob/48812b043942a03859f841d886936e0c968702b5/examples/tpu/retool/gb200/logs/g2-retool-rl-25step-01/head.log#L1444)） |
| AIME 2024 mean@30（默认 / strict），Step 0 → 25 | 25.56% → 26.78% / 25.78% → 27.78% | 23.78% → 28.78% / 24.22% → 29.67% |
| AIME 2025 mean@30（默认 / strict），Step 0 → 25 | 17.33% → 17.78% / 17.44% → 17.78% | 17.00% → 17.89% / 17.11% → 18.00% |

TPU 的 `driver.log` 在 GCS：`gs://ubench-logs/xiaotongyang/retool-v7x/runs/p3-retool-rl-11/attempts/bef2992e/logs/driver.log`。

> [!NOTE]
> 在基线 A 下，AIME 2024 的提升 GPU 是 +5.00 pp，TPU 只有 +1.22 pp。不过同一个 checkpoint 的 Step-0 分数在三次运行中就是 23.78%（`G2`）、23.78%（`p3-retool-rl-10`）和 25.56%（`p3-retool-rl-11`），运行之间的差异就有约 2 pp。所以比较时要用第 5 节的配对统计，不能只看容差。

## 2. 修复内容

### 2.1 Reward 配置（GPU 和 TPU 两个脚本都改）

- **根因**：
  - 两个启动脚本都在 `trainer.use_v1=True` 的同时设置顶层 `custom_reward_function.*`：TPU 脚本 [L212-L213](https://github.com/syangx39/verl-upstream/blob/48812b043942a03859f841d886936e0c968702b5/examples/tpu/retool/run_qwen3_30b_a3b_dapo_torchtitan.sh#L212-L213)，GPU 脚本 [L305-L306](https://github.com/syangx39/verl-upstream/blob/48812b043942a03859f841d886936e0c968702b5/examples/tpu/retool/gb200/run_qwen3_30b_a3b_dapo_torchtitan_gb200.sh#L305-L306)。
  - V1 的 reward loader 读的是 `config.reward.custom_reward_function`（[`reward.py` L70](https://github.com/syangx39/verl-upstream/blob/48812b043942a03859f841d886936e0c968702b5/verl/trainer/ppo/reward.py#L70)）。
  - 把顶层配置迁到 `reward` 下的 `migrate_legacy_reward_impl`（[`reward_loop.py` L39-L60](https://github.com/syangx39/verl-upstream/blob/48812b043942a03859f841d886936e0c968702b5/verl/experimental/reward_loop/reward_loop.py#L39-L60)）只在 [`fully_async_main.py` L237](https://github.com/syangx39/verl-upstream/blob/48812b043942a03859f841d886936e0c968702b5/verl/experimental/fully_async_policy/fully_async_main.py#L237) 调用。
  - 结果两边解析后的配置都是 `path: None`，于是 [`default_compute_score`](https://github.com/syangx39/verl-upstream/blob/48812b043942a03859f841d886936e0c968702b5/verl/utils/reward_score/__init__.py#L59-L62) 把 `math_dapo` 和 `aime_*` 数据都交给 `math_dapo.compute_score`，用的是默认的 `strict_box_verify=False`。
  - `gb200-retool-runtime`（`a2cbde47`）上也是同样的情况。
- **修改**：
  - 两个脚本都改为设置 `reward.custom_reward_function.path=pkg://examples.tpu.retool.retool_tpu` 和 `reward.custom_reward_function.name=compute_score`。
  - 加一个开关 `REWARD_FN=retool|default`。默认 `retool`；`default` 用来复现基线 A。
- **防静默回退**：在 `main_ppo` 的 V1 路径里，如果顶层 `custom_reward_function.path` 有值、而 `reward.custom_reward_function.path` 为空，就直接报错。配一个单元测试。
- **前提已核实**：naive reward manager 会把 `__num_turns__` 写进 `extra_info["num_turns"]`（[`naive.py` L49-L51](https://github.com/syangx39/verl-upstream/blob/48812b043942a03859f841d886936e0c968702b5/verl/experimental/reward_loop/reward_manager/naive.py#L49-L51)），所以 ReTool [`compute_score`](https://github.com/syangx39/verl-upstream/blob/48812b043942a03859f841d886936e0c968702b5/examples/tpu/retool/retool_tpu.py#L90-L103) 的 shaping 拿得到轮数。smoke 里还会再确认一次。

### 2.2 D7：让 `optim.clip_grad` 真正生效

- **根因**：`TorchTitanEngine` 构造 `TrainingConfig` 时没有传 `max_norm`（[`transformer_impl.py` L264-L274](https://github.com/syangx39/verl-upstream/blob/48812b043942a03859f841d886936e0c968702b5/verl/workers/engine/torchtitan/transformer_impl.py#L264-L274)），所以 torchtitan 用默认值 1.0；`optim.clip_grad` 只出现在 [L305-L309](https://github.com/syangx39/verl-upstream/blob/48812b043942a03859f841d886936e0c968702b5/verl/workers/engine/torchtitan/transformer_impl.py#L305-L309) 的一条日志里。GPU 分支也是一样。
- **修改**：
  - 把 `optim.clip_grad` 传给 `TrainingConfig(max_norm=...)`，和 FSDP 引擎的语义保持一致。
  - 日志改为打印实际生效的阈值和它的来源。
  - 加单元测试。
- **影响**：verl 的 `optim.clip_grad` 默认就是 1.0，所以只有显式设了别的值的配置才会变化。两个脚本当前默认 `CLIP_GRAD=0.088`（TPU [L111](https://github.com/syangx39/verl-upstream/blob/48812b043942a03859f841d886936e0c968702b5/examples/tpu/retool/run_qwen3_30b_a3b_dapo_torchtitan.sh#L111)，GPU [L185](https://github.com/syangx39/verl-upstream/blob/48812b043942a03859f841d886936e0c968702b5/examples/tpu/retool/gb200/run_qwen3_30b_a3b_dapo_torchtitan_gb200.sh#L185)），修复后这个值会立刻生效。因此每个实验都要显式设置 `CLIP_GRAD`。

### 2.3 两边代码和镜像的前置工作

- **TPU**：修复合进 `tpu-main`，运行时用 `CODE_REF` 指定，不需要重建镜像。
- **GPU**：代码烤在镜像的 `/workspace/verl-pin` 里。需要把修复 cherry-pick 到 `gb200-retool-runtime` 分支（当前推荐版本是 `a2cbde47`），再用 `Dockerfile.gb200.torchtitan` 重建镜像，并记录镜像 digest 和 `EXPECT_GIT_SHA`。
- **每个 run 都要记录**：代码 SHA、镜像 digest、解析后的配置，以及 reward 和 clipping 的生效证据（第 4 节）。

## 3. 实验矩阵

| ID | Scorer | 有效 clipping | 用途 |
|---|---|---|---|
| **A**（已有） | 默认 `math_dapo`，`strict=False`，无 shaping | 1.0 | 基线：`p3-retool-rl-11` / `G2` |
| **B** | ReTool `compute_score`：`strict=True` + shaping | 1.0 | 只换 scorer：B vs A 看 reward 修复本身的影响；B-GPU vs B-TPU 看预期 reward 下的 GPU/TPU 对照 |
| **C** | ReTool `compute_score` | 0.088 | 完整的预期配置：C vs B 看 clipping 的影响 |

- **推荐顺序**：B 的 smoke（两边）→ B 的 25 步（两边）→ 决定是否做 C。做 C 时先跑 C 的 smoke，只改 clipping。
- **快速路线（不推荐）**：跳过 B 直接跑 C，可以更快得到完整配置的结果，但分不清变化来自 scorer 还是 clipping。
- **同一个配置在两边必须一致**：
  - 起点 checkpoint：`global_step_372`，按 SHA-256 清单核对。
  - 数据集和 AIME 源文件：按 digest 核对。
  - 超参数：`lr=1.4e-6`、`bs=64`、`n=8`、`max_turns=8`、`max_response_length=8192`、`n_val=30`、async 设置（`num_warmup_batches=2`、staleness 上限 4）。
  - 有效 clipping 阈值相同。

## 4. Smoke：先证明配置真的生效

- **规模**：每个配置、每一侧各跑一次。
  - 训练 3 步；只做 Step-0 验证，`N_RESP_PER_PROMPT_VAL=2`（共 120 条），够检验验证集的 scorer。
  - rollout 每步抽查 8 个 session。
- **验收条件**（扩展 PR #7 里的 `verify_p3_results.py` 自动检查）：
  1. 解析后的配置里 `reward.custom_reward_function` 是 `{path: pkg://examples.tpu.retool.retool_tpu, name: compute_score}`。
  2. clipping 日志打印的生效阈值等于设定值：B 是 1.0，C 是 0.088。
  3. 训练抽查：每个 session 的记录分数都等于 ReTool `compute_score`（strict + 按 `num_turns` shaping）重算的结果；并且至少出现一个 shaped 分数，也就是落在 (−1, −0.6] 区间的值。
  4. 验证结果：每一行的 `acc` 都等于 `math_dapo(strict_box_verify=True)` 重算的结果；分数只可能是 1 或落在 [−1, −0.6]；答错的多轮 session 里出现 shaped 分数。
  5. 没有异常（比如缺 `num_turns` 导致的 `KeyError`），没有出现"grad norm 非有限、本步跳过"的告警。
  6. TPU：Step-0 逐元素比较通过，后续 numel/L1 检查通过。GPU：NCCL 同步正常。
- **放行条件**：两边全部通过，才开始 25 步。

## 5. 25 步重跑与评估口径

- **评估流程和 A 相同**：Step 0 和 Step 25 各评估一次 AIME 2024/2025，`n=30`。先用 `summarize_validation_dump` 检查完整性：预期题目从源数据算出，每题 30 个 session。
- **每份评估结果都报三个数**：
  1. 默认 scorer（`strict=False`）下的 `acc`，离线重算，用来和旧实验比较；
  2. strict scorer（`strict=True`）下的 `acc`，B 和 C 的运行时记录就是它；
  3. shaped reward 均值（`score`），单独列出，不当作准确率。
- **统计**：用逐题配对 bootstrap，对 Step 0 → 25 的变化，以及 GPU vs TPU、B vs A 的差异给出区间。数据就是每个数据集 30 道题的逐题均值，`validation_summaries.json` 里已经有。容差带只作参考，不当作置信区间。
- **训练指标**：reward、`grad_norm`（clip 前）、KL、`ppl_ratio`、staleness、`num_turns`、response 长度。B/C 的 reward 用了 shaping，和 A 的 reward 不能直接比。
- **性能指标**：沿用 A 的定义（两次同步之间的实际时间、`timing_s/step`）。两边资源规模不同（32 chips vs 8 GB200），所以这里只做记录；硬件对等的性能比较按 PR #11 的方案另做。

## 6. 其他配置对齐清单（`p3-retool-rl-11` 与 `G2` 解析后配置的 diff）

完整列表见 [`results/config-diff-p3-retool-rl-11-vs-g2/config_diff.tsv`](results/config-diff-p3-retool-rl-11-vs-g2/config_diff.tsv)。两份配置展开后共有 108 个键不同：

| 类别 | 键数 | 例子 | 处理 |
|---|---|---|---|
| 平台和并行方式 | 67 | trainer `CP=4/DP=8` vs `FSDP2=4`；rollout `TP=32/EP=32` vs `TP=4`；`attn_type` varlen vs flex；activation checkpoint full vs selective；torch compile；vLLM 显存和 batching；同步方式 raiden vs nccl；transfer_queue 后端 | 记录即可，不影响算法语义 |
| 路径和命名 | 16 | 运行目录、模型暂存路径、实验名；数据集是同一批文件，只是路径前缀不同 | 记录即可 |
| 动态 batch 的 token 预算 | 5 | `*_max_token_len_per_gpu` 4608 vs 16384；`torchtitan.max_seq_len` 18432 vs 16384 | 只改变 micro-batch 怎么切；确认 loss 聚合方式一致后记录 |
| 代码和配置结构版本 | 20 | 量化配置对象（两边都关闭）；`rollout.topk_log_probs`（GPU 为 0）；`rollout_correction.score_centering`（GPU 为 False，TPU 没有这个键）；distillation 参数（未使用） | **需要核对**：对比两个 SHA（GPU `64ef634b`，基于上游 `3fdf2b548`；TPU `aa10dfed`）的 `core_algos`、rollout correction、agent loop 的 tool 解析和 reward manager，确认算法语义一致 |

学习率、`optim.clip_grad` 配置值、batch 大小、`n`、`max_turns`、response 长度、KL 设置、clip ratio、rollout correction、`n_val`、采样参数和 reward 配置在两份配置里都相同（不在 diff 中）。

## 7. 资源与时间估计

| | Smoke | 25 步 |
|---|---|---|
| TPU（32 chips） | 约 45 分钟，含预热和排队 | 约 95 分钟，加排队（`p3-retool-rl-11` head 运行了 5,634 秒） |
| GPU（2 × 4 GB200） | 约 30 分钟，另加镜像构建 | 约 30 分钟（`G2`：训练 1,104 秒，两次评估约 370 秒） |

## 8. 风险

- **shaping 改变 reward 的尺度**：答错的多轮 rollout 会得到 −0.9 到 −0.6，GRPO 的 advantage 随之改变，可能鼓励多用工具，从而影响轮数、response 长度和每步耗时。
- **strict 判分**：答案不在 `\boxed{}` 里就判错。DAPO 训练 prompt 里同时写了"Answer: ..."和 `\boxed{}` 两种格式要求。在 `p3-retool-rl-11` 的数据里，strict 和默认的判定在验证集上只差 3–9 行（共 1,800 行），在训练抽查里差 33 个 session（共 200 个）。
- **clipping 0.088**：`p3-retool-rl-11` 的 `grad_norm` 在 0.193–0.324 之间，改成 0.088 后每步更新会被缩小约 2.2–3.7 倍，25 步内学习会变慢。
- **代码不同步**：GPU 和 TPU 是两条代码线，修复必须两边都合进去，并核对版本。

## 9. 交付物

1. **修复 PR**（合进 `tpu-main`）：两个启动脚本的 reward 配置、V1 防回退检查、TorchTitan clipping 传参、单元测试。
2. **GPU 侧**：cherry-pick 到 `gb200-retool-runtime`，构建新的 GB200 镜像。
3. **扩展验证脚本**：
   - 可以处理任意 run 目录；
   - 两种 scorer 都输出；
   - 检查 reward 配置和 clipping 日志；
   - 计算配对 bootstrap。
4. **新运行作为新记录**：命名带配置标签，例如 `p3b-retool-rl-*` / `g3b-*`。每个 run 有自己的证据目录；基线 A 不改动。

## 10. 需要 review 决定

1. 按 **B → C** 的顺序（推荐），还是直接跑 **C**？
2. smoke 的验证集用 `n=2` 是否可以？
3. 防静默回退：在 `main_ppo` 里直接报错（推荐），还是调用迁移函数？迁移函数会删除旧配置键，影响面更大。
4. GPU 镜像由谁构建，什么时候构建？
