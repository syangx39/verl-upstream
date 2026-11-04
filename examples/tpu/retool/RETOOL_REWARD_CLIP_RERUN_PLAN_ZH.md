# ReTool reward 与 clipping 修复及 GPU/TPU 重跑方案

状态：**待 review（rev 3）**（2026-11-04）。本文只是方案，还没有改代码，也没有提交新的运行。

- rev 3：B 改为在基线 A 的代码和镜像上运行，只多一个 reward override，所以删掉 A2；GPU 的 B 和 C 都不需要重建镜像；GPU 上的 B 改为按需运行（第 10 节）。
- rev 2：smoke 加固定样例检查 shaping；补充评估口径；配置差异按系统差异处理，并列出核对项；clipping 的影响改为需要实测，并写明 strict scorer 的限制。

相关：TPU 结果和 scorer 证据见 PR #7（[`results/p3-retool-rl-11/`](https://github.com/syangx39/verl-upstream/tree/retool-tpu7x-exp-results/examples/tpu/retool/results/p3-retool-rl-11)）；GPU `G2` 的记录见 [`gb200/logs/g2-retool-rl-25step-01/`](gb200/logs/g2-retool-rl-25step-01/)；两边解析后配置的对比见 [`results/config-diff-p3-retool-rl-11-vs-g2/`](results/config-diff-p3-retool-rl-11-vs-g2/README.md)。

## 0. 目标与边界

- **目标**：用代码中预期的 ReTool reward 训练，看 reward 修复本身的影响；需要时在 GPU 上跑同一配置，做 GPU/TPU 对照。预期 reward 是 `strict_box_verify=True` 加 tool-use shaping。
- **不做**：不为比较现有 GPU/TPU 的 reward 一致性而重跑。现有 TPU `p3-retool-rl-11` 和 GPU `G2` 已经是同一配置下的对照基线：默认 scorer，有效 clipping `max_norm=1.0`。这只说明这两项语义一致；其他配置、评估和性能口径仍要逐项对齐（第 6 节）。
- **公平性要求**：
  - 新配置从共享的 P2 checkpoint 开始。不拿新的 TPU run 和旧 `G2` 当同配置对照。
  - B 用和基线 A 完全相同的代码和镜像，只多一个 reward override，所以 B vs A 是只换 scorer 的对照。它仍然只是两次单独 run 之间的比较（第 5 节）。
- **归因边界**：Askhat 的脚本写了 `clip_grad=0.088`，但他实际是否生效、是否用了 shaped scorer，要看他的 resolved config 或日志才能确认。在那之前，新 run 只称"预期 ReTool 配置"，不称与 Askhat 的实际运行严格一致。

## 1. 现有基线 A（保留，不覆盖）

| | TPU `p3-retool-rl-11` | GPU `G2`（`g2-retool-rl-25step-01`） |
|---|---|---|
| 代码 / 镜像 | `aa10dfed`（通过 tarball 下发），`verl-tpu@sha256:d92a2b3d...` | `64ef634b`（pod 内按 `JOBENV_EXPECT_GIT_SHA` checkout），`verl-torchtitan-cu130-arm64@sha256:a139e2b4...` |
| 资源 | 1 trainer + 1 rollout `2x2x4` 切片（32 chips） | 2 节点 × 4 GB200（trainer `FSDP2=4` + rollout `TP=4`） |
| 起点 | `p2-retool-sft-01/global_step_372/huggingface` | 同一个 checkpoint |
| Reward | 默认 `math_dapo`，`strict_box_verify=False`，无 shaping（`driver.log` L946：`reward.custom_reward_function.path: None`） | 相同（[`head.log` L899](https://github.com/syangx39/verl-upstream/blob/48812b043942a03859f841d886936e0c968702b5/examples/tpu/retool/gb200/logs/g2-retool-rl-25step-01/head.log#L899)） |
| 有效 clipping | TorchTitan `training.max_norm=1.0`（`driver.log` L1588） | 相同（[`head.log` L1444](https://github.com/syangx39/verl-upstream/blob/48812b043942a03859f841d886936e0c968702b5/examples/tpu/retool/gb200/logs/g2-retool-rl-25step-01/head.log#L1444)） |
| AIME 2024 mean@30（默认 / strict），Step 0 → 25 | 25.56% → 26.78% / 25.78% → 27.78% | 23.78% → 28.78% / 24.22% → 29.67% |
| AIME 2025 mean@30（默认 / strict），Step 0 → 25 | 17.33% → 17.78% / 17.44% → 17.78% | 17.00% → 17.89% / 17.11% → 18.00% |

TPU 的 `driver.log` 在 GCS：`gs://ubench-logs/xiaotongyang/retool-v7x/runs/p3-retool-rl-11/attempts/bef2992e/logs/driver.log`。

> [!NOTE]
> - **AIME**：在基线 A 下，AIME 2024 的提升 GPU 是 +5.00 pp，TPU 只有 +1.22 pp。但同一个 checkpoint 的 Step-0 分数在三次运行中分别是 23.78%（`G2`）、23.78%（`p3-retool-rl-10`）和 25.56%（`p3-retool-rl-11`），仅评估噪声就有约 2 pp。按逐题配对 bootstrap，两边提升之差（两年合并 60 题）是 +2.1 pp，95% 区间 [−2.1, +6.2]，看不出差别。
> - **训练 reward**：25 步内两边都基本持平（第 1–5 步 → 第 21–25 步：TPU −0.270 → −0.247，GPU −0.284 → −0.206；每步斜率的 95% 区间都包含 0）。TPU v5p 参考 `fcze3xlx` 在 23 步内从 −0.52 升到 +0.66，但它的配置不同（16K、16 轮、`n=16`、`lr=2e-6`、thinking 打开），scorer 也还没确认。B 可以检验换成预期 reward 后这一点是否改变。

## 2. 修复内容

### 2.1 Reward 配置

- **根因**：
  - 两个启动脚本都在 `trainer.use_v1=True` 的同时设置顶层 `custom_reward_function.*`：TPU 脚本 [L212-L213](https://github.com/syangx39/verl-upstream/blob/48812b043942a03859f841d886936e0c968702b5/examples/tpu/retool/run_qwen3_30b_a3b_dapo_torchtitan.sh#L212-L213)，GPU 脚本 [L305-L306](https://github.com/syangx39/verl-upstream/blob/48812b043942a03859f841d886936e0c968702b5/examples/tpu/retool/gb200/run_qwen3_30b_a3b_dapo_torchtitan_gb200.sh#L305-L306)。
  - V1 的 reward loader 读的是 `config.reward.custom_reward_function`（[`reward.py` L70](https://github.com/syangx39/verl-upstream/blob/48812b043942a03859f841d886936e0c968702b5/verl/trainer/ppo/reward.py#L70)）。
  - 把顶层配置迁到 `reward` 下的 `migrate_legacy_reward_impl`（[`reward_loop.py` L39-L60](https://github.com/syangx39/verl-upstream/blob/48812b043942a03859f841d886936e0c968702b5/verl/experimental/reward_loop/reward_loop.py#L39-L60)）只在 [`fully_async_main.py` L237](https://github.com/syangx39/verl-upstream/blob/48812b043942a03859f841d886936e0c968702b5/verl/experimental/fully_async_policy/fully_async_main.py#L237) 调用。
  - 结果两边解析后的配置都是 `path: None`，于是 [`default_compute_score`](https://github.com/syangx39/verl-upstream/blob/48812b043942a03859f841d886936e0c968702b5/verl/utils/reward_score/__init__.py#L59-L62) 把 `math_dapo` 和 `aime_*` 数据都交给 `math_dapo.compute_score`，用的是默认的 `strict_box_verify=False`。
  - `gb200-retool-runtime`（`a2cbde47`）上也是同样的情况。
- **B 不需要改代码**：启动时多加一个 Hydra override 即可：`reward.custom_reward_function.path=pkg://examples.tpu.retool.retool_tpu`。`name` 的默认值已经是 `compute_score`（[`head.log` L899](https://github.com/syangx39/verl-upstream/blob/48812b043942a03859f841d886936e0c968702b5/examples/tpu/retool/gb200/logs/g2-retool-rl-25step-01/head.log#L899)）。已核实：
  - 两个启动脚本最后都把额外参数原样传给 `main_ppo`（TPU [L314](https://github.com/syangx39/verl-upstream/blob/aa10dfed7b64d78e166c484f4649f62d7880b8c9/examples/tpu/retool/run_qwen3_30b_a3b_dapo_torchtitan.sh#L314)，GPU [L350](https://github.com/syangx39/verl-upstream/blob/64ef634b173c69ea31eff598caeee81d9a9caea1/examples/tpu/retool/gb200/run_qwen3_30b_a3b_dapo_torchtitan_gb200.sh#L350)）。两个启动器也会把自己收到的额外参数传给启动脚本（TPU [`run_ray_jobset_v7x_nap.sh` L19-L23](https://github.com/syangx39/verl-upstream/blob/6d074673ed540f2bfc2b0ba85732f883af4bd6c9/examples/tpu/gke/run_ray_jobset_v7x_nap.sh#L19-L23)，GPU [`run_ray_jobset_gb200.sh` L374](https://github.com/syangx39/verl-upstream/blob/64ef634b173c69ea31eff598caeee81d9a9caea1/examples/tpu/retool/gb200/run_ray_jobset_gb200.sh#L374)）。
  - 基线代码 `aa10dfed`（TPU）和 `64ef634b`（GPU）中，`reward.py`、`naive.py`、`tool_agent_loop.py`、`retool_tpu.py`、`math_dapo.py` 和 `reward_score/__init__.py` 与 `48812b04` 逐字节相同。下面引用的行号对三个提交都成立。
  - naive reward manager 会把 `__num_turns__` 写进 `extra_info["num_turns"]`（[`naive.py` L49-L51](https://github.com/syangx39/verl-upstream/blob/48812b043942a03859f841d886936e0c968702b5/verl/experimental/reward_loop/reward_manager/naive.py#L49-L51)），所以 ReTool [`compute_score`](https://github.com/syangx39/verl-upstream/blob/48812b043942a03859f841d886936e0c968702b5/examples/tpu/retool/retool_tpu.py#L90-L103) 的 shaping 拿得到轮数。
  - `num_turns = user_turns + assistant_turns + 1`（[`tool_agent_loop.py` L203](https://github.com/syangx39/verl-upstream/blob/48812b043942a03859f841d886936e0c968702b5/verl/experimental/agent_loop/tool_agent_loop.py#L203)）。ReTool scorer 在答错时加 `(num_turns - 2) / 2 * 0.1`，上限 −0.6（[`retool_tpu.py` L95-L98](https://github.com/syangx39/verl-upstream/blob/48812b043942a03859f841d886936e0c968702b5/examples/tpu/retool/retool_tpu.py#L95-L98)）。正常结束的 session 每多一次工具调用，`num_turns` 加 2，所以答错时每次工具调用加 0.1，4 次及以上都是 −0.6。
  - smoke 里用固定样例再确认一次（第 4 节）。
- **修复 PR（给 C 和以后的 run 用，不阻塞 B）**：
  - 两个脚本都改为设置 `reward.custom_reward_function.path` 和 `name`，并加开关 `REWARD_FN=retool|default`。默认 `retool`；`default` 用来复现 A。
  - 防静默回退：在 `main_ppo` 的 V1 路径里，如果顶层 `custom_reward_function.path` 有值、而 `reward.custom_reward_function.path` 为空，就直接报错。配一个单元测试。
  - B 在旧代码上跑，没有这个检查，由第 4 节的验收条件 1 和 3 把关。

### 2.2 D7：让 `optim.clip_grad` 真正生效（只有 C 需要）

- **根因**：`TorchTitanEngine` 构造 `TrainingConfig` 时没有传 `max_norm`（[`transformer_impl.py` L264-L274](https://github.com/syangx39/verl-upstream/blob/48812b043942a03859f841d886936e0c968702b5/verl/workers/engine/torchtitan/transformer_impl.py#L264-L274)），所以 torchtitan 用默认值 1.0；`optim.clip_grad` 只出现在 [L305-L309](https://github.com/syangx39/verl-upstream/blob/48812b043942a03859f841d886936e0c968702b5/verl/workers/engine/torchtitan/transformer_impl.py#L305-L309) 的一条日志里。GPU 分支也是一样。B 的有效阈值仍是 1.0，不需要这个修复。
- **修改**：
  - 把 `optim.clip_grad` 传给 `TrainingConfig(max_norm=...)`，和 FSDP 引擎的语义保持一致。
  - 日志改为打印实际生效的阈值和它的来源。
  - 加单元测试。
- **影响**：verl 的 `optim.clip_grad` 默认就是 1.0，所以只有显式设了别的值的配置才会变化。两个脚本当前默认 `CLIP_GRAD=0.088`（TPU [L111](https://github.com/syangx39/verl-upstream/blob/48812b043942a03859f841d886936e0c968702b5/examples/tpu/retool/run_qwen3_30b_a3b_dapo_torchtitan.sh#L111)，GPU [L185](https://github.com/syangx39/verl-upstream/blob/48812b043942a03859f841d886936e0c968702b5/examples/tpu/retool/gb200/run_qwen3_30b_a3b_dapo_torchtitan_gb200.sh#L185)），修复后这个值会立刻生效。因此每个实验都要显式设置 `CLIP_GRAD`。

### 2.3 代码和镜像

- **B：和 A 完全相同**。
  - TPU：`CODE_REF=aa10dfed`，镜像 `verl-tpu@sha256:d92a2b3d...`。启动器用 `tpu-main` 上的版本（含 #14 的 watch 修复）；训练代码、manifest 和 pod 的启动脚本都来自 `aa10dfed` 的代码包。提交前先用 `DRY_RUN=1` 确认两者兼容。
  - GPU：镜像 `@sha256:a139e2b4...`，`JOBENV_EXPECT_GIT_SHA=64ef634b`，和 `G2` 一样。启动器也用 `64ef634b` 里的版本：当前启动器要求 sandbox 支持 `--url`，`64ef634b` 还没有（见 [`gb200/README.md`](gb200/README.md)）。
- **C：A 的代码加修复，镜像不变**。
  - TPU：修复合进 `tpu-main` 后用 `CODE_REF` 指定。`aa10dfed` 到当前 `tpu-main`（`6d074673`）之间，`verl/` 和 `examples/tpu/retool/` 下的代码与脚本都没有改动，只改了启动器（`examples/tpu/gke/`）、它的测试和一份诊断文档，所以 C 和 B 的训练代码只差修复本身。
  - GPU：**不需要重建镜像**。GPU 启动器会在 pod 里 `git fetch` 并 checkout `JOBENV_EXPECT_GIT_SHA`（[`run_ray_jobset_gb200.sh` L178-L180](https://github.com/syangx39/verl-upstream/blob/64ef634b173c69ea31eff598caeee81d9a9caea1/examples/tpu/retool/gb200/run_ray_jobset_gb200.sh#L178-L180)），镜像只提供按 `uv.lock` 校验过的依赖。`G2` 就是这样在 `23113cb2` 构建的镜像上跑了 `64ef634b`（[precheck](gb200/logs/g2-retool-rl-25step-01/env/g2-retool-rl-25step-01-head-precheck.json)：`git HEAD SHA verified: 64ef634b...`，`uv.lock` 校验通过）。所以把修复 cherry-pick 到基于 `64ef634b` 的新分支，用同一个镜像跑，C 和 B 在 GPU 上也只差修复本身。只有 `uv.lock` 变化时才需要重建镜像。
- **每个 run 都要记录**：代码 SHA、镜像 digest、解析后的配置，以及 reward 和 clipping 的生效证据（第 4 节）。

## 3. 实验矩阵

| ID | Scorer | 有效 clipping | 代码 / 镜像 | 用途 |
|---|---|---|---|---|
| **A**（已有） | 默认 `math_dapo`，`strict=False`，无 shaping | 1.0 | TPU `aa10dfed`；GPU `64ef634b` + 镜像 `a139e2b4` | 基线 `p3-retool-rl-11` / `G2` |
| **B** | ReTool `compute_score`：`strict=True` + shaping | 1.0 | 同 A，只加 reward override | B vs A：同代码、同镜像下只换 scorer。B-GPU vs B-TPU：预期 reward 下的 GPU/TPU 对照 |
| **C** | ReTool `compute_score` | 0.088 | A 的代码加修复，镜像不变 | 完整的预期配置：C vs B 看 clipping 的影响 |

- **顺序**：TPU B 的 smoke → TPU B 的 25 步 → 根据结果决定是否做 C。GPU B 是否跑见第 10 节；跑的话可以和 TPU 并行，同样先跑 smoke。做 C 时先跑 C 的 smoke，只改 clipping。
- 以上比较都只有一次 run，结论能覆盖的范围见第 5 节。
- **同一个配置在两边必须一致**：
  - 起点 checkpoint：`global_step_372`，按 SHA-256 清单核对。
  - 数据集和 AIME 源文件：按 digest 核对。
  - 超参数：`lr=1.4e-6`、`bs=64`、`n=8`、`max_user_turns=8`、`max_assistant_turns=8`（两个脚本都由 `MAX_TURNS` 同时设置：TPU [L271-L272](https://github.com/syangx39/verl-upstream/blob/48812b043942a03859f841d886936e0c968702b5/examples/tpu/retool/run_qwen3_30b_a3b_dapo_torchtitan.sh#L271-L272)，GPU [L364-L365](https://github.com/syangx39/verl-upstream/blob/48812b043942a03859f841d886936e0c968702b5/examples/tpu/retool/gb200/run_qwen3_30b_a3b_dapo_torchtitan_gb200.sh#L364-L365)）、`max_response_length=8192`、`n_val=30`、async 设置（`num_warmup_batches=2`、staleness 上限 4）。
  - 有效 clipping 阈值相同。

## 4. Smoke：先证明配置真的生效

- **规模**：每个要跑的配置（先 B，之后的 C）在要跑的每一侧各跑一次 smoke。
  - 训练 3 步；只做 Step-0 验证，`N_RESP_PER_PROMPT_VAL=2`（共 120 条），够检验验证集的 scorer。
  - rollout 每步抽查 8 个 session。
- **固定样例检查（必须通过）**：随机抽到的样本里不一定有可检查的多轮错题，所以另用一组固定输入检查 shaping。
  - 方法：用该 run 解析后的配置调用 [`get_custom_reward_fn`](https://github.com/syangx39/verl-upstream/blob/48812b043942a03859f841d886936e0c968702b5/verl/trainer/ppo/reward.py#L50)，取得实际生效的 reward 函数（返回 `None` 时就是 `default_compute_score`），用训练和验证数据实际的 `data_source` 各跑一次下表的样例，逐项比较。再通过 naive reward manager 跑同一组样例（构造带 `__num_turns__` 的 batch），确认轮数确实传进了 `extra_info`。
  - 同一组样例也写进修复 PR 的单元测试。

  | 样例（标准答案 `42`） | `num_turns` | B / C 预期 | 默认 scorer（A） |
  |---|---|---|---|
  | 结尾是 `\boxed{42}` | 2；6 | 1.0；1.0 | 1.0；1.0 |
  | 结尾是 `\boxed{41}` | 2 | −1.0 | −1.0 |
  | 同上 | 3 | −0.95 | −1.0 |
  | 同上 | 4 | −0.9 | −1.0 |
  | 同上 | 6 | −0.8 | −1.0 |
  | 同上 | 10；16 | −0.6；−0.6 | −1.0；−1.0 |
  | 只有 `Answer: 42`，没有 `\boxed{}` | 2 | −1.0（strict 判错） | 1.0 |
  | `\boxed{42}` 之后还有超过 100 个字符 | 2 | −1.0 | −1.0 |

  表中的预期值已用 `math_dapo.py` 和 ReTool 的 shaping 公式实际算过（这两个文件在 `aa10dfed`、`64ef634b` 和 `48812b04` 中相同）。默认 scorer 一列用来确认这些样例能区分两种 scorer。

- **验收条件**（扩展 PR #7 里的 `verify_p3_results.py` 自动检查）：
  1. 解析后的配置里 `reward.custom_reward_function` 是 `{path: pkg://examples.tpu.retool.retool_tpu, name: compute_score}`。
  2. 有效 clipping 阈值等于设定值：B 是 1.0（现有日志：TPU `driver.log` 里打印的 TorchTitan `training.max_norm`，GPU 同 [`head.log` L1444](https://github.com/syangx39/verl-upstream/blob/48812b043942a03859f841d886936e0c968702b5/examples/tpu/retool/gb200/logs/g2-retool-rl-25step-01/head.log#L1444)）；C 是 0.088（看修复后的新日志）。
  3. 固定样例全部符合预期。
  4. 训练抽查：每个 session 的记录分数都等于用 ReTool scorer（strict + 按 `num_turns` shaping）重算的结果。随机样本里出现了多少个 shaped 分数只记录，不作为放行条件。
  5. 验证结果：每一行的 `acc` 都等于 `math_dapo(strict_box_verify=True)` 重算的结果；分数只可能是 1 或落在 [−1, −0.6]。
  6. 没有异常（比如缺 `num_turns` 导致的 `KeyError`），没有出现"grad norm 非有限、本步跳过"的告警。
  7. TPU：Step-0 逐元素比较通过，后续 numel/L1 检查通过。GPU：NCCL 同步正常。
- **放行条件**：这一侧全部通过，才开始这一侧的 25 步。

## 5. 25 步重跑与评估口径

- **评估流程和 A 相同**：Step 0 和 Step 25 各评估一次 AIME 2024/2025，`n=30`。先用 `summarize_validation_dump` 检查完整性：预期题目从源数据算出，每题 30 个 session。
- **每份评估结果都报三个数**：
  1. 默认 scorer（`strict=False`）下的 `acc`，离线重算，用来和旧实验比较；
  2. strict scorer（`strict=True`）下的 `acc`，B 和 C 的运行时记录就是它；
  3. shaped reward 均值（`score`），单独列出，不当作准确率。
- **统计**：
  - 用逐题配对 bootstrap 给出区间：Step 0 → 25 的变化，以及 B vs A、GPU vs TPU 的差异。输入是每个数据集 30 道题的逐题均值，`validation_summaries.json` 里已经有。
  - 这个区间只描述在这批题目上、这两个具体 checkpoint 之间的评估差异（包括题目抽样和每题 30 次采样带来的噪声）。它不覆盖训练 run 本身的随机性：同一配置再训练一次，Step-25 的 checkpoint 会不一样。所以 B vs A、GPU vs TPU 都只是单次 run 的结论。
  - 要估计训练随机性，需要每个配置用不同 seed 重复几次 run。本方案默认不做，列为可选项。第 1 节里 Step-0 跨 run 约 2 pp 的差异只是评估噪声，不能代替它。
  - 容差带只作参考，不当作置信区间。
- **训练指标**：reward、`grad_norm`（clip 前）、clip 系数（第 8 节）、KL、`ppl_ratio`、staleness、`num_turns`、response 长度。B/C 的 reward 用了 shaping，和 A 的 reward 不能直接比；B 的训练 reward 曲线和 `fcze3xlx` 的前 23 步并列，只作参考（配置不同）。
- **性能指标**：沿用 A 的定义（两次同步之间的实际时间、`timing_s/step`）。两边资源规模不同（32 chips vs 8 GB200），所以这里只做记录；硬件对等的性能比较按 PR #11 的方案另做。

## 6. 其他配置对齐清单（`p3-retool-rl-11` 与 `G2` 解析后配置的 diff）

完整列表见 [`results/config-diff-p3-retool-rl-11-vs-g2/config_diff.tsv`](results/config-diff-p3-retool-rl-11-vs-g2/config_diff.tsv)。两份配置展开后共有 108 个键不同。B/C 里两边仍是各自的平台和代码线，这些差异会保留，所以 GPU vs TPU 的结论都要带上它们。

| 类别 | 键数 | 例子 | 处理 |
|---|---|---|---|
| 平台和并行方式 | 67 | trainer `CP=4`、`DP shard=8` vs `FSDP2=4`；attention（TPU `varlen` + splash，GPU `flex`）；activation checkpoint full vs selective；torch compile 开 vs 关；SPMD backend；精度相关的键（`mixed_precision_param`、`fp32_logits` 只在 TPU 配置里有）；rollout `TP=32/EP=32` vs `TP=4`；vLLM 显存比例 0.62 vs 0.75、`max_num_batched_tokens` 4096 vs 8192；同步方式 raiden vs nccl；transfer_queue 后端 | **系统差异，不预先当作没有影响**。attention kernel、精度和并行切分会改变数值（归约顺序、logits 精度）；vLLM 的显存和 batching 设置会改变请求调度和批的组成，从而改变采样时的数值。记录下来，并按下面的核对项检查 |
| 动态 batch 的 token 预算 | 5 | `*_max_token_len_per_gpu` 4608 vs 16384；`torchtitan.max_seq_len` 18432 vs 16384 | 改变 micro-batch 的切分。要确认 loss 聚合和切分方式无关（核对项 1）；归约顺序带来的数值差异仍然存在 |
| 路径和命名 | 16 | 运行目录、模型暂存路径、实验名；数据集是同一批文件，只是路径前缀不同 | 记录即可 |
| 代码和配置结构版本 | 20 | 量化配置对象（两边都关闭）；`rollout.topk_log_probs`（GPU 为 0）；`rollout_correction.score_centering`（GPU 为 False，TPU 没有这个键）；distillation 参数（未使用） | **需要核对**（核对项 4） |

学习率、`optim.clip_grad` 配置值、batch 大小、`n`、`max_user_turns` / `max_assistant_turns`、response 长度、KL 设置、clip ratio、rollout correction、`n_val`、采样参数（训练 `temperature=1.0`、`top_p=1`；验证 `top_p=0.6`）、`loss_agg_mode=token-mean` 和 reward 配置在两份配置里都相同（不在 diff 中）。

**核对项**（在报告 GPU vs TPU 结论之前完成，结果写进报告）：

1. **Loss 聚合**：`agg_loss` 的 `token-mean` 在两个 SHA 里实现相同：本 rank 的 token loss 之和除以全局 `batch_num_tokens`，再乘 `dp_size`（[TPU `aa10dfed`](https://github.com/syangx39/verl-upstream/blob/aa10dfed7b64d78e166c484f4649f62d7880b8c9/verl/trainer/ppo/core_algos.py#L1140-L1175)，[GPU `64ef634b`](https://github.com/syangx39/verl-upstream/blob/64ef634b173c69ea31eff598caeee81d9a9caea1/verl/trainer/ppo/core_algos.py#L1140-L1175)）。还要核对调用方：动态 batch 下传入的是整个 mini-batch 的全局 token 数，TPU `CP=4` 时 `dp_size` 取值正确，保证梯度等于整个 mini-batch 的 token-mean，与 micro-batch 怎么切无关。
2. **精度**：从两边的日志或代码确认 trainer 实际使用的参数、梯度和 logits 的 dtype。GPU 配置里没有 `mixed_precision_param` / `fp32_logits` 这两个键，要查清它实际的默认行为。
3. **Logprob 和采样**：比较两边的 `actor/rollout_corr/ppl_ratio` 和 `k3_kl`（训练侧和 rollout 侧 logprob 的一致性），以及 response 长度和 `num_turns` 的分布。采样参数相同，但 kernel 和 batching 不同，结果不会逐位一致。
4. **代码版本**：对比 GPU `64ef634b`（基于上游 `3fdf2b548`）和 TPU `aa10dfed` 的 `core_algos`、rollout correction、agent loop 的 tool 解析和 reward manager，确认算法语义一致。

## 7. 资源与时间估计

| | Smoke | 25 步 |
|---|---|---|
| TPU（32 chips） | 约 45 分钟，含预热和排队 | 约 95 分钟，加排队（`p3-retool-rl-11` head 运行了 5,634 秒） |
| GPU（2 × 4 GB200，按需） | 约 30 分钟 | 约 30 分钟（`G2`：训练 1,104 秒，两次评估约 370 秒） |

B 和 C 都不需要构建镜像。做 C 时，每一侧再加一次 smoke 和一次 25 步。

## 8. 风险

- **shaping 改变 reward 的尺度**：答错的多轮 rollout 会得到 −0.95 到 −0.6，GRPO 的 advantage 随之改变，可能鼓励多用工具，从而影响轮数、response 长度和每步耗时。
- **strict 判分的限制**：
  - `math_dapo.compute_score` 先只保留末尾 300 个字符（[L267](https://github.com/syangx39/verl-upstream/blob/48812b043942a03859f841d886936e0c968702b5/verl/utils/reward_score/math_dapo.py#L267)）；strict 路径再只看其中末尾 100 个字符（[L211](https://github.com/syangx39/verl-upstream/blob/48812b043942a03859f841d886936e0c968702b5/verl/utils/reward_score/math_dapo.py#L211)），取最后一个 `\boxed{}` 的内容，和标准答案做字符串完全相等的比较（[L217](https://github.com/syangx39/verl-upstream/blob/48812b043942a03859f841d886936e0c968702b5/verl/utils/reward_score/math_dapo.py#L217)），不做归一化，也不判断数学上是否等价。
  - 所以答案不在末尾 100 个字符的 `\boxed{}` 里，或者写法和标准答案不同（例如 `\boxed{x=42}`、`\boxed{042}`、多余的空格），都会判错。
  - 默认 scorer 先按 `Answer: ...` 提取，归一化后比较字符串，提取不到才退回同样的 `\boxed{}` 检查；它也不判断数学等价。
  - DAPO 训练 prompt 里同时写了"Answer: ..."和 `\boxed{}` 两种格式要求。在 `p3-retool-rl-11` 的数据里，strict 和默认的判定在验证集上只差 3–9 行（共 1,800 行）；训练抽查的 200 个 session 里，默认判对 81 个，其中 33 个 strict 会判错。
- **clipping 0.088**：
  - `p3-retool-rl-11` 的 25 步 `grad_norm`（clip 前）在 0.193–0.324 之间，全部高于 0.088。相同梯度下，clip 0.088 每一步都会触发，梯度被缩到原来的 27%–46%（约 2.2–3.7 倍）。
  - 这不能推出参数更新或学习速度也缩小 2.2–3.7 倍。AdamW 用梯度的二阶矩做归一化：如果每步都按同一比例缩放，更新只在 `eps`（1e-8）这一项上有差别。按 A 的 `grad_norm` 推断，每步都会被裁到同一个范数，效果更接近按步重新加权，再加上 `eps` 的相对作用变大。
  - 实际影响要用 C vs B 实测：每步的 clip 系数 `min(1, max_norm / grad_norm)`；在 staleness 设置相同时，rollout 策略和训练策略之间的 KL（`actor/rollout_corr/k3_kl`、`ppl_ratio`）可以间接反映每步策略变化的大小；还有 reward 曲线和 Step-25 评估。如果需要直接看参数更新的大小，可以在修复 PR 里加一个可选的参数更新范数日志。
- **代码线**：B 不改代码。C 的修复要同时进 TPU（`tpu-main`）和 GPU（基于 `64ef634b` 的新分支）两条代码线，并核对版本。

## 9. 交付物

1. **B 的启动参数**：两边的 reward override、`CODE_REF` / `JOBENV_EXPECT_GIT_SHA` 和镜像 digest，写进运行记录。
2. **扩展验证脚本**（B 之前完成）：
   - 可以处理任意 run 目录；
   - 两种 scorer 都输出；
   - 检查 reward 配置、clipping 日志和固定样例；
   - 计算配对 bootstrap。
3. **修复 PR**（C 之前完成，合进 `tpu-main`）：两个启动脚本的 reward 配置和 `REWARD_FN` 开关、V1 防回退检查、TorchTitan clipping 传参和日志、单元测试（包括第 4 节的固定样例）。
4. **GPU（只有做 C 时）**：把修复 cherry-pick 到基于 `64ef634b` 的新分支并推送；不重建镜像。
5. **第 6 节核对项的结果**，写进报告。
6. **新运行作为新记录**：B 用 `p3b-retool-rl-*` / `g3b-*`，C 用 `p3c-retool-rl-*` / `g3c-*`。每个 run 有自己的证据目录；基线 A 不改动。

## 10. Review 决定

已定：

1. 顺序：**B → C**。
2. smoke 的验证集用 `n=2`，另加固定样例检查 shaping（第 4 节）。
3. 不做 A2：B 用基线 A 的代码和镜像，只加 reward override。GPU 的 B 和 C 都不重建镜像。

待定：

4. GPU 的 B 跑不跑？推荐先跑 TPU B；需要预期 reward 下的 GPU/TPU 对照时再跑 GPU B（smoke 加 25 步约 1 小时，不用建镜像）。
5. 防静默回退（修复 PR，做 C 之前定）：在 `main_ppo` 里直接报错（推荐），还是调用迁移函数？迁移函数会删除旧配置键，影响面更大。
