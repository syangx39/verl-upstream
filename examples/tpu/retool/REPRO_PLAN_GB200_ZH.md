# ReTool：GB200 复现与 TPU v7x 对照计划

## 目标和当前基线

主线让 GB200 和 TPU v7x 从**同一个 TPU P2 SFT checkpoint** 出发，完成相同的 ReTool 多轮工具调用 RL，对比**训练行为与最终结果的 parity**、**端到端及分阶段性能**。GPU 重新训练 SFT 和完整 SFT → RL 对照是可选后续实验，不阻塞主线。GPU 训练必须使用 veRL 的 **TorchTitan engine**，配合 veRL 和 GPU vLLM；不是选择 veRL 原生 FSDP2 engine。

本计划冻结的 TPU 文档快照是 [PR #7 commit `6fa65e4`](https://github.com/syangx39/verl-upstream/tree/6fa65e49094c04a9b10cb766ecb1ba7c0976cc5a/examples/tpu/retool)。P2 SFT 已完成 372/372 步。按运行者最新汇报，P1a `p1a-moe-smoke-09` 已进入双 slice 预热/运行，P2 `global_step_372/huggingface` 已完成 100% 静态对账；P1a 最终状态和后续 TPU RL/AIME 结果仍待实际产出。静态对账不代表 GPU 和 TPU 的 forward、RL 更新或采样行为已经一致。P1a 是 GSM8K 系统 smoke，并非完整 ReTool RL。

### 规模建议

| 阶段 | TPU 已有/计划规模 | GB200 首轮建议 | 目的 |
| --- | --- | --- | --- |
| SFT（可选） | P2：1 slice，16 物理 chips，32 逻辑 TensorCore | 如后续开展，先试 4 GPU | 比较各自 SFT，仅在需要 SFT 性能或端到端复现时执行 |
| RL | 2 slices，32 物理 chips，64 逻辑 TensorCore；训练和 rollout 各 16 chips | **8 GPU 总计，4 trainer + 4 rollout** | 先验证运行可行，再比较相同工作量 |

8 卡是第一个工程实验规模，不是 32 个 TPU chip 的等算力换算，也不是 8 张 trainer 加 8 张 rollout。以上 SFT 与 RL 分别比较，不能把 RL 的 TPU 规模套到 SFT。记录每个角色的机器数、互联、HBM 和峰值占用；以现场 `nvidia-smi` 和拓扑为准，不假设 8 卡同机。4+4 能否放下 30B MoE、8K response 和目标并发须先实测。容量不足时先减 per-device microbatch、rollout 并发并保持全局训练语义；如仍不足，另试 6+2 或更多 GPU，单独标注配置和资源，不混进 8 卡结果。

### 主线和可选扩展

1. **隔离 RL 的主对照**：TPU 和 GPU 都从同一个 TPU P2 `global_step_372/huggingface` 启动，固定数据、工具、算法和评估。先验证 HF 文件清单/哈希、加载和同输入 forward，再对比 RL 训练。
2. **可选端到端复现**：GB200 从与 TPU P2 相同的 base、数据和 SFT 配置独立训练至 step 372，导出 GPU SFT HF checkpoint；比较两侧 SFT，再分别用各自的 SFT checkpoint 启动相同的 RL。此轨的 RL 差异包含 SFT 差异，单列报告。

先完成主对照，无需 GPU SFT；只有要比较 SFT 或验证从 base model 开始的完整 recipe 时才启动第二轨。明确每个 run 的 checkpoint 来源，避免把两轨结果合并。

## 0. 冻结输入和实现

- 记录 TPU 及 GPU 的 git SHA、实际 resolved Hydra config、环境镜像 digest、Python/PyTorch/CUDA/NCCL/Ray/vLLM/verl 版本、模型和 tokenizer revision、精度、attention/MoE kernel、所有随机种子、启动命令及 host/GPU 分配。记录 TPU 的实际运行日志，不仅是计划默认值；如 P1a 或 P3 最终配置变化，更新对照表。
- Base model revision 为 `ad44e777bcd18fa416d9da3bd8f70d33ebb85d39`。ReTool SFT parquet 2,000 行，SHA256 `92a29cea03f72dc79066e1697f4d7423d91bd6a02bd1aac50f5c9dfd89b27faa`；train 和 validation 使用同一文件，故 validation 是 **in-sample**。复制 P2 checkpoint 所有 safetensors、index、config、tokenizer 和 chat template，逐文件生成 SHA256 清单，并在两平台重新校验。
- RL DAPO parquet 为 1,791,700 行（17,917 prompts × 100），SHA256 `22e266f54502ce271ee08713feb8b30caf97a4ea948aca0e0506f58c4cdd0545`。保持原顺序/重复、预处理和 shuffle 逻辑，不做去重。保存输入样本 ID、截断、token IDs 和 loss mask；同一 seed 也要实测数据顺序。
- 固定 sandbox 服务源码/版本、镜像、执行超时、资源限制、并发、网络和工具返回格式；固定 tokenizer、prompt、tool schema、parser、reward 和 AIME 题目。复用 [`retool_tpu.py`](https://github.com/syangx39/verl-upstream/blob/6fa65e49094c04a9b10cb766ecb1ba7c0976cc5a/examples/tpu/retool/retool_tpu.py) 与其 tool 配置前，先在 GPU worker 实际导入和执行。配置中的 `127.0.0.1` 是**各执行节点自己的** loopback，须证明调用进程能访问对应 sandbox，不能只在 head 上验证。
- 训练 correctness 使用原始 `acc` 或逐题判分；shaped reward 包含工具轮次调整，不能当作正确率。

## 1. 先准备 GPU launcher，再做预检

这是计划，不声称现有 TPU 脚本能在 GPU 原样运行。建议新增并版本化 GPU RL launcher、环境锁定和记录脚本；GPU SFT launcher 仅为可选实验准备。实现从仓库现有路径开始：

- 可选 SFT 参考 [多轮 FSDP SFT 示例](https://github.com/syangx39/verl-upstream/blob/6fa65e49094c04a9b10cb766ecb1ba7c0976cc5a/examples/sft/multiturn/run_qwen2_5_0_5b_fsdp.sh)，改为 Qwen3-30B-A3B、ReTool P2 的数据/模板和下述有效配置。
- RL 参考 [v1 separate-async GPU 测试](https://github.com/syangx39/verl-upstream/blob/6fa65e49094c04a9b10cb766ecb1ba7c0976cc5a/tests/special_e2e/run_v1_separate_async.sh) 的独立资源池、vLLM 和 NCCL 权重同步，结合 [GPU TorchTitan 小模型测试](https://github.com/syangx39/verl-upstream/blob/6fa65e49094c04a9b10cb766ecb1ba7c0976cc5a/tests/special_e2e/run_ppo_trainer_torchtitan.sh) 的 engine 选择，以及 [Qwen3-30B-A3B GPU 示例](https://github.com/syangx39/verl-upstream/blob/6fa65e49094c04a9b10cb766ecb1ba7c0976cc5a/examples/grpo_trainer/run_qwen3_30b_a3b_fsdp.sh) 的模型支持，以及 [TPU ReTool recipe](https://github.com/syangx39/verl-upstream/blob/6fa65e49094c04a9b10cb766ecb1ba7c0976cc5a/examples/tpu/retool/run_qwen3_30b_a3b_dapo_torchtitan.sh) 的数据和语义。这些例子各自只覆盖部分路径；原 separate-async 测试没有 TorchTitan actor 分支，需新写 launcher 并验证组合，不能直接宣称已验证 30B + 多轮工具 + 长上下文。
- RL 使用 `trainer.use_v1=True`、`trainer.v1.trainer_mode=separate_async`、`actor_rollout_ref.hybrid_engine=False`，使用 `model_engine=torchtitan` 和 `actor_rollout_ref.actor.strategy=torchtitan`，初始 TorchTitan trainer 4 GPU、vLLM rollout TP=4 / 4 GPU、NCCL 同步。TorchTitan 内部可以用 FSDP2 做分片，这和选择 veRL 原生 FSDP2 engine 不同。确认实际 Ray placement 和版本兼容；不要把另一套 fully-async trainer 或 colocated 运行当作同一模式。
- 先采集 `nvidia-smi -L`、`nvidia-smi topo -m`、显存、主机 CPU/内存、节点角色与网络。确认每个 worker 看到正确设备，actor 和 rollout 不重叠，跨节点通信及 NCCL 正常。
- 加载 base 和共享 HF checkpoint，验证 tokenizer、参数 key/shape、文件哈希、权重分片和 MoE expert 映射；分别试短样本和接近长度上限的样本，记录 HBM 峰值。先小并发后扩大；发生 OOM 就记录实际配置及失败点，不悄悄缩短上下文。
- 测一次 actor → rollout 完整同步：固定输入比较同步前后权重映射与输出，至少检查初始权重和一次更新后权重，尤其 MoE expert。NCCL 调用成功不等于权重被正确加载。

### 镜像与 veRL 版本：推荐方案

**结论：做一个新的、固定版本的 GB200 派生镜像；不需要从 CUDA 开始全部重建。** 用户的[旧 Dockerfile](https://github.com/syangx39/verl/blob/main/examples/grpo_trainer/my_scripts_GKE_gsm8k_disagg/Dockerfile) 用 arm64 基础镜像 digest、固定 veRL SHA 和 uv lock，是可复用的构建模式；但它只 sync `fsdp + vllm`，没有锁入/验证 TorchTitan，且代码不是本次 ReTool 冻结版本。**源码中存在 TorchTitan engine 不代表镜像已经安装兼容的 TorchTitan。** 不要在旧环境中直接 `pip install -U torchtitan`，这可能使 torch、vLLM 或 CUDA 扩展的 ABI 组合失配。

优先评估较新的 [verl upstream 候选 `3fdf2b548882a8aeccf0842c2150d9c9f1790a55`](https://github.com/verl-project/verl/blob/3fdf2b548882a8aeccf0842c2150d9c9f1790a55/pyproject.toml)：其声明 `torch==2.13.0`、`vllm==0.32.0`、`torchtitan==0.3.0`，并提供共同 lock 与 `vllm + fsdp + torchtitan` extras；这是**待验证候选，不是已通过本 recipe 的镜像**。将 PR #7 需要的 ReTool 数据/工具/reward、异步行为、Qwen3 MoE 及权重映射等实现移植或逐项核对到 GPU 工作分支，再固定合并后的完整 SHA 和 lock。升级只影响 GPU 工作分支，不改正在运行的 TPU 工作负载；升级本身是跨平台差异，需做第 2 节的语义/数值对账。若上述迁移不通过，备用方案是基于 TPU 冻结的 veRL SHA 建立**单独的兼容 GPU lock**，而不是把新版 extra 直接套到旧 SHA。

版本不能混搭：TPU P0.2 实测 TorchTitan 0.1.0 @ [`f880cc73725de77d892ecab888244e7718e5ad65`](https://github.com/jialei777/torchtitan/commit/f880cc73725de77d892ecab888244e7718e5ad65)、`torch 2.13.0+cpu` 和 TPU `vllm 0.29.0`，这些是 TPU 环境记录，不是 GPU wheel 组合。PR #7 冻结版本的 GPU lock 则声明 torch 2.11 / vLLM 0.24、没有 torchtitan extra；其 [TorchTitan 文档](https://github.com/syangx39/verl-upstream/blob/6fa65e49094c04a9b10cb766ecb1ba7c0976cc5a/docs/workers/torchtitan_workers.rst) 又给出一套另一版本的 GPU nightly 组合。新旧 lock、nightly 和 TPU image 不能互相拼装；备用路线需明确 Titan 来源/commit、GPU PyTorch/CUDA/vLLM wheel 的 ABI 和 aarch64 可用性，完成独立 lock 与实测。

**建议构建流程：**

1. 在候选新版上建立 GPU 工作分支，把必要移植纳入同一 commit，审计配置/reward/tool/async 差异并保留补丁清单；固定 `<GPU_CODE_SHA>` 和其 `pyproject.toml`、`uv.lock`。不要对浮动 `main` 建生产对照镜像。
2. 使用该 SHA 对应的 [官方 CUDA 13 uv Dockerfile](https://github.com/verl-project/verl/blob/3fdf2b548882a8aeccf0842c2150d9c9f1790a55/docker/Dockerfile.uv.cu130) 在原生 arm64 构建，或沿用旧 Dockerfile 的固定 arm64 base digest 做派生层；先确认基础 manifest、驱动和目标 GPU 兼容。官方 Dockerfile 从**构建上下文**复制 checkout 并预取 uv cache，不等于已经提供含目标三 extras 的固定 venv。可在派生层 materialize 统一 venv，并确保 head 与每个 Ray worker 用同一路径和相同 extras。不要让运行时挂载的另一份源码覆盖固定 checkout。
3. 示意命令如下，须替换为已移植并 lock 的实际 commit；这是构建方案，非现成已测试镜像：

```bash
git checkout --detach <GPU_CODE_SHA>   # 包含 ReTool 移植与该 commit 的 uv.lock
docker buildx build --platform linux/arm64 \
  -f docker/Dockerfile.uv.cu130 -t retool-gb200:build --load .
# 在基于该镜像的派生层/容器中，在同一 checkout 的 .venv 内执行：
uv sync --frozen --all-packages --extra fsdp --extra vllm --extra torchtitan
```

   `--extra fsdp` 在这里提供依赖组合，**不表示选择 veRL FSDP2 actor engine**。实际发布前固定基础镜像与最终镜像 digest，并归档构建日志、`uv.lock` 哈希、源码 SHA、torch/CUDA/NCCL/vLLM/TorchTitan 版本和 wheel 架构；构建如需网络，提前确认所需 aarch64 wheel 可用。官方文档提醒 GB200 预构建镜像的可用性需逐个核验；不能凭 tag 假设有 arm64 或包含本配方。
4. 扩展旧镜像的 `verify_venv_lock.py`：验证 TorchTitan 和三 extras 的实际解析、import 路径及 Ray workers 的 Python；检查无意间覆盖 torch 和 CUDA 扩展。目标机验证 `torch.cuda.is_available()`、GPU 型号/compute capability、vLLM CUDA kernel 和 TorchTitan Qwen3 MoE import。Import 成功只是预检。
5. 做一轮模型级门槛：共享 HF checkpoint 的 TorchTitan Qwen3 MoE 全量加载及 router/expert 映射、短与近 10,240 token forward/backward、一次 optimizer step，之后分别测初始和更新后的 TorchTitan → NCCL → vLLM 权重同步与固定输入输出；再做真实工具调用、4+4 separate-async 的 G0/G1。记录每一项 PASS/FAIL、显存峰值与实际版本。旧有 FSDP2 separate-async 和 TorchTitan 小模型测试不能代替该组合的实测。

Sandbox 是独立进程/服务，**不必单独构建镜像**。复用并固定 TPU 的 server 和工具调用语义，核实 GPU 各请求节点可访问服务、运行依赖和实际生效的资源/超时限制；若另行容器化，作为显式差异记录。

## 2. 固定输入的数值和工具对账

以共享 P2 checkpoint，在 TPU/GPU 上使用同一短样本、长样本和一批真实 ReTool transcript；保存输入 ID、padding/attention/loss mask、工具调用文本、输出和版本。

1. 比较分词、模板、停止条件、tool request/response、截断位置、answer extraction、`acc` 与 shaped reward。固定输入下离散解析和评分应一致；不一致先定位 tokenizer、沙箱或实现版本。
2. teacher-forcing 比较逐 token logits/log-prob、top-k 和 NLL 的误差分布；记录 dtype、误差分位数及最大误差。BF16、MoE router 近似相等的分数和不同 kernel 可能放大局部偏差，不能要求逐 token 位相同。
3. 对同一固定 rollout 和 old log-prob 比较 advantage、importance weight、loss 聚合、clip fraction、梯度范数以及一次 optimizer step 后的参数/固定输入输出；记录归约、梯度裁剪和 optimizer 的实际语义。阈值在运行前按精度及测量噪声约定并报告，超出时先调查；缺少参考或阈值时标为 inconclusive。
4. 在真实异步路径记录每条 trajectory 的生成版本、训练消费版本、queue age、importance-weight 分布；若一条轨迹可跨策略版本，保留逐段版本。随机自由生成不要求逐条相同，比较统计分布。

## 3. GPU SFT（可选，不阻塞 RL）：复现 TPU P2

只有需要独立比较 SFT 或完整端到端复现时，才采用 Qwen3-30B-A3B 同一 base 和上述 SFT parquet，从 4 GPU 启动；其训练 engine 也应按本实验的 TorchTitan 要求选定并验证；先 3 步 smoke，再完整 6 epochs / 372 steps。长度 16,384、global batch 32、per-device microbatch 初设 1，保持 no-padding、多轮 mask 和右侧截断语义。参考 TPU P2：AdamW、LR `1e-5`、warmup ratio `0.1`、cosine 降至初始值的 `0.1`、betas `(0.9, 0.95)`、weight decay `0.01`、有效 grad max norm `1.0`。保存 step 186 和 372，并导出完整可加载 HF 权重及 SHA256 清单。

检查真实 global batch / drop-last / shuffle / token normalization、每步有效 tokens、学习率曲线、optimizer 和 gradient clipping；不同并行方式允许 microbatch/gradient accumulation 不同，但不能改变全局损失归约。TPU P2 用 SimpleFSDP8 × CP4、PROMOTE_F32；GPU TorchTitan 的序列并行、精度和 kernel 按可行实现选择并登记。不要直接搬 TPU 的每设备 token 上限，需确保 16K 实际可运行。

记录每步 train loss、learning rate、grad norm、token 数、显存、吞吐、step wall time 和验证 loss；验证采样与 TPU 对齐并明确同训练数据。比较 TPU step 372 loss `0.035875`、末 10 步均值 `0.039265`、validation `0.026332` 时，附每步数据和 batch 顺序，不能把单点接近视作全部训练 parity。

## 4. GPU RL：先 smoke，再跑同工作量

### 4.1 要对齐的语义

以 TPU 的**实际 resolved 配置与日志**为最终基准，首轮按下表准备。实际 TPU P3 启动参数若不同，应先重冻结两边，不把相异 run 写成 parity。

| 项目 | 首轮对照 |
| --- | --- |
| 数据与起点 | 原始 DAPO parquet；主轨使用同一 TPU P2 step-372 HF；端到端轨分别用各自 P2 HF |
| 一次训练更新 | 64 prompts × 每题 8 responses，nominal 512 trajectories；PPO mini-batch 64；核对过滤、丢弃及实际更新数 |
| 长度和轮次 | prompt 最大 2,048，response 最大 8,192；assistant/user 各最多 8 turns；保持同一 tool schema 和 token mask |
| 算法 | GRPO；clip low/high `0.2/0.28`，clip c `10`；reward KL 和 KL loss 关闭，entropy coefficient 0 |
| optimizer | LR `1.4e-6`；逐项核对 optimizer、betas、epsilon、weight decay、schedule、global normalization 和有效 gradient clipping |
| 异步和校正 | separate_async；warmup batches 2、每 step 同步、max off-policy threshold 4；rollout correction bypass、token IS threshold 2 |
| 评估 | AIME 2024/2025 各 30 题，temperature `1.0`、top-p `0.6`、每题 `n=30`；step 0 与 step 25，保留每题原始输出和判分 |

特别核对 TPU TorchTitan 的实际梯度裁剪：脚本中 `actor.optim.clip_grad=0.088` 未必是 TorchTitan 有效的 `training.max_norm`。以运行代码和日志证据为准，再设置 GPU TorchTitan 对应的有效值，记录两者和差异。两个平台的 TorchTitan 版本、optimizer 实现和配置解析也可能不同；尤其核对 betas、scheduler 和 min LR。不要直接照搬 TPU context parallel 下的 `ppo_max_token_len_per_gpu=4608` 到 GPU，须验证最大完整序列及 loss 语义。冻结版本的 `actor_rollout_ref.actor.torchtitan.max_seq_len` 若为 null 会回退至 2,048：GPU launcher 应根据实际 packing/CP 显式设置足够长度并测试近上限样本，在新版本核对对应字段和实际 RoPE 缓存。

### 4.2 分阶段执行与门槛

1. **G0，基础 smoke**：4+4 启动，先用少量固定请求检查 sandbox、工具循环、权重同步、打分及日志，含接近最大长度的案例。确认无死锁、OOM，角色位置正确；保存峰值显存和 trace。
2. **G1，短 RL smoke**：共享 P2 checkpoint，按 TPU P1c 的诊断缩小 workload：train batch 8、mini-batch 8、每题 4 条、2 steps，关闭训练前验证。确认梯度、optimizer、同步、策略版本以及 HF checkpoint 可导出并重新加载；这是诊断，不与全规模性能直接比较。HF 权重重载不等于完整断点续训：如需续训，另行验证 optimizer、scheduler 和异步队列状态；主对照优先连续从头运行。
3. **G2，主对照**：共享 P2 checkpoint，采用上表 workload 运行至 step 25，训练前与 step 25 均做完整 AIME24/25。保留所有 25 步训练指标，确认 step 25 的最终或显式 checkpoint 已保存；逐步对照同配置 TPU P3。固定并对账两边的 scheduler horizon 和前 25 步逐步 LR：若把 total steps 改为 25 会改变调度，就保持双方相同的完整 horizon 并在 step 25 评估、有序停止；不要把不同 schedule 混为一谈。两次评估合计 `2 × 2 × 30 × 30 = 3,600` 条样本，提前计入时间和 sandbox 压力。若 TPU P3 尚无可用结果，先保存 GPU 数据并将跨平台结论标为待对照。
4. **G3，可选端到端轨**：GPU 自己的 SFT step-372 checkpoint，以 G2 相同配置跑 RL；与 TPU 自身 SFT → RL 对照。算力允许时增加独立训练 seed；不要将重复 rollout 采样误认为独立训练重复。

每阶段通过后再扩展：记录完成步数、失败/重试、实际样本数、有效版本和输出位置；任何改动都重新归档 resolved config。Askhat 的历史 v5p RL 使用不同 response length、turn、n、LR 等，而且在 step 23 中断；只能作为上下文，不能当成上述 8K/8-turn run 的严格 parity 基准。

## 5. 结果怎么比

### Parity

- SFT（仅开展可选实验时）：相同样本和有效 tokens 的训练 loss、grad norm、LR、末尾 loss 分布、相同 in-sample validation；比较各自的 step-372 checkpoint，并保留前述固定输入数值对账。
- RL：按 step 和实际 consumed trajectories 对比 correctness、原始 reward/组件、format/tool failures、timeout、turn 数、response 长度、截断率、entropy、actor KL、clip fraction、grad norm、importance weights 和策略滞后。保存对应的逐样本数据，允许异步执行顺序和随机轨迹不同。
- AIME：对比 step 0/25 的每题正确率及总体 pass@1 估计，并核对计算口径；按题做配对 bootstrap 区间，保留每题 30 条原始样本；另评估每题 rollout 抽样的不确定性（例如按题、再按样本分层重采样），区分题目与采样波动。多训练 seed 时另报告 seed 间变化；若缺少原始轨迹、基线或事前约定容差，结论写 evidence insufficient，不以单次分数相近就断言 parity。
- 汇总显式差异：平台、物理拓扑、parallelism、precision/kernel、实际 optimizer、数据顺序、工具运行环境和异步版本。出现轨迹分叉时，先用固定输入测试定位，再分析自由运行统计。

### 性能

固定 prompt/生成长度分布、全局 batch、并发、工具 CPU 资源、cache 与日志/评估开销，记录实际偏差。至少给出：

- 端到端 wall time（含启动、加载、编译/预热、验证、存储）以及排除这些开销的稳态 step 中位数/P90；逐项写明时间边界，尤其 P2 的整作业时间与训练 step 时间不能直接比较。
- rollout 完成轨迹数/秒、输入/输出及有效训练 tokens/秒、每 step 实际 token 数；训练、生成、工具、同步、checkpoint 和 eval 的可观测耗时。异步阶段会重叠，不将各阶段耗时简单相加为总时间。
- trainer、rollout 分别及合计的物理设备数、利用率、HBM 峰值、accelerator-seconds/完成轨迹及 accelerator-seconds/有效训练 token。TPU 记录物理 chip 和逻辑设备；GPU 记录卡数。可额外记录实际价格/能耗，缺数据不估算。
- 单独展示固定工作量绝对用时和设备资源消耗；只有学习曲线及目标指标可比时才报告 time-to-target。不要用 `32/8` 直接推断理论或实际性能倍数，也不把不同长度分布和不同成功率的吞吐当等价。

## 6. 交付清单

- 两平台每个 run 的 SHA、launcher、resolved config、环境/硬件/placement、seed、输入/模型文件清单与哈希、启动命令和实际变更。
- G0/G1 的可行性与同步/数值对账报告，G2（以及若执行 G3）的 checkpoint、逐步指标、逐条轨迹、AIME 逐题结果、性能原始日志和可复现的汇总脚本。
- 一张 TPU vs GB200 表：RL 主线及可选 SFT 分行，注明规模、工作量、成功步数、parity 指标、稳态与总 wall time、总 accelerator-hours、重要配置差异及未解决问题。未运行、失败、数据不可比均如实标注。

参考：[TPU 结果记录](https://github.com/syangx39/verl-upstream/blob/6fa65e49094c04a9b10cb766ecb1ba7c0976cc5a/examples/tpu/retool/REPRO_RESULTS_TPU7X.md) · [Askhat GPU PR #108](https://github.com/jialei777/verl-upstream/pull/108) · [verl v1 async 文档](https://verl.readthedocs.io/en/latest/advance/v1_async_trainer.html)。
