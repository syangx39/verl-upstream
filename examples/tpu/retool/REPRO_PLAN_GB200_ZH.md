# ReTool：GB200 复现与 TPU v7x 对照计划

## 目标和当前基线

主线让 GB200 和 TPU v7x 从**同一个 TPU P2 SFT checkpoint** 出发，完成相同的 ReTool 多轮工具调用 RL，对比**训练行为与最终结果的 parity**、**端到端及分阶段性能**。GPU 重新训练 SFT 和完整 SFT → RL 对照是可选后续实验，不阻塞主线。GPU 训练必须使用 veRL 的 **TorchTitan engine**，配合 veRL 和 GPU vLLM；不是选择 veRL 原生 FSDP2 engine。

本计划冻结的 TPU 基线代码为 [PR #7 commit `6fa65e4`](https://github.com/syangx39/verl-upstream/tree/6fa65e49094c04a9b10cb766ecb1ba7c0976cc5a/examples/tpu/retool)，各阶段（`P1a`、`P1c`、`P2`、`P3`）实时执行状态与指标统一以 [`REPRO_RESULTS_TPU7X.md` 第 2 节状态表](https://github.com/syangx39/verl-upstream/blob/retool-tpu7x-exp-results/examples/tpu/retool/REPRO_RESULTS_TPU7X.md)为准，不在本计划中写死单次 attempt 编号。当前 P2 SFT 已完成 372/372 步，且 6 个 epoch 的 train/validation loss 曲线与 Askhat 的 TPU v5p 参考（`myaf5pal`）逐 epoch 误差均在 `±0.0005` 内，`global_step_372/huggingface` 全部 `18,867 / 18,867` 个 tensor key 与 base model 完成 100% 静态对账；静态对账不代表 GPU 和 TPU 的 forward、RL 更新或采样行为已经一致，`P1a`（GSM8K 系统 smoke）亦不能替代 `P1c`/`P3` 多轮 ReTool RL 实测。

### 规模建议与 GB200 静态显存账本

| 阶段 | TPU 已有/计划规模 | GB200 首轮主方案（8 GPU 总计） | 6+2 方案状态（非直接替换） | 目的 |
| --- | --- | --- | --- | --- |
| SFT（可选） | P2：1 slice，16 物理 chips，32 逻辑 TensorCore | 如后续开展，先试 4 GPU | 另行评估 | 比较各自 SFT，仅在需要 SFT 性能或端到端复现时执行 |
| RL | 2 slices，32 物理 chips，64 逻辑 TensorCore；训练和 rollout 各 16 chips | **4 trainer (`FSDP=4`) + 4 rollout (`TP=4`)** | **待专门设计验证方案**（见下方整除约束说明） | 先验证运行可行，再比较相同工作量 |

8 卡是第一个工程实验规模，不是 32 个 TPU chip 的等算力换算，也不是 8 张 trainer 加 8 张 rollout。以上 SFT 与 RL 分别比较，不能把 RL 的 TPU 规模套到 SFT。记录每个角色的机器数、互联、HBM 和峰值占用；以现场 `nvidia-smi` 和拓扑为准，不假设 8 卡同机。

**统一单位（严格区分十进制 `GB = 10^9 B` 与二进制 `GiB = 2^30 B`）的 `4 Trainer + 4 Rollout` 静态显存估算：**

- **硬件容量（官网标称 vs 实测 CUDA 可见容量）**：NVIDIA GB200 开发者文档标称单卡 HBM3e 为 `186 GB`（约 `173.2 GiB`）；在目标 `aarch64` 4×GB200 节点上实测 `torch.cuda.get_device_properties(i).total_memory` 为 **`197.90 GB`（`184.31 GiB` / 卡，SM `10.0`，NCCL `2.29.7`）**。下文保守按 `173.2~184.3 GiB` 区间核算。
- **Trainer 侧（`4 × GB200`，`FSDP=4`）**：`Qwen3-30B-A3B` 约 `30.53B`（`30.53 × 10^9`）参数；按 fp32 master params + fp32 grads + fp32 Adam `m, v`（`16 bytes/param`）计，纯静态状态总计约 **`488.5 GB`（`454.9 GiB`）**，4 卡均分约 **`122.1 GB / 卡`（`113.7 GiB / 卡`）**。该数字尚未计入 `10,240` token 序列激活、FSDP unshard 临时峰值、NCCL 同步 buffer，以及潜在的 `actor_rollout_ref.ref` 副本（若为记录 `actor/ppo_kl` 在 4 张 Trainer 卡上额外加载 `ref`，会再增加约 `14.2~28.4 GiB / 卡`，使总占用逼近 `173.2 GiB` 上限）。
- **Rollout 侧（`4 × GB200`，`TP=4`）**：bf16 模型权重总计约 **`61.1 GB`（`56.9 GiB`）**，4 卡均分约 **`15.3 GB / 卡`（`14.2 GiB / 卡`）**；实际可用 KV cache 还须扣除 vLLM `gpu_memory_utilization` 上限、PyTorch/CUDA 分配器保留内存、运行时 workspace buffer、CUDA graph 捕获内存池及多会话并发开销，**一切以目标 GB200 节点实测峰值显存（`nvidia-smi` 与 `torch.cuda.max_memory_allocated`）为准，不预先断言宽裕度**。
- **为什么 `6 Trainer + 2 Rollout` 不能直接作为 `FSDP=6` 的 OOM 备选**：候选 `verl`（[`trainer_base.py:1830-1833`](https://github.com/verl-project/verl/blob/3fdf2b548882a8aeccf0842c2150d9c9f1790a55/verl/trainer/ppo/v1/trainer_base.py#L1830-L1833)、[`engine_workers.py:296-305`](https://github.com/verl-project/verl/blob/3fdf2b548882a8aeccf0842c2150d9c9f1790a55/verl/workers/engine_workers.py#L296-L305)）会将 actor mini-batch 展开为 `64 × 8 = 512` 条 trajectories（`G1` smoke 为 `8 × 4 = 32` 条），并断言其必须被 `data_parallel_size` 整除；直接设 `FSDP=6` 会因 `512 % 6 = 2`（及 `32 % 6 = 2`）触发断言失败。
- **OOM 处置顺序**：优先在 **`4+4`** 主方案上核查 `activation_checkpoint`、确认未实例化冗余 `ref` 副本、调整 per-device microbatch / token packing 及 vLLM 并发与 `gpu_memory_utilization`（保持全局 batch 和 `8,192` 最大响应长度不变，绝不静默缩短上下文）；若仍需非 4 卡 Trainer 规模，必须单独设计并验证满足 DP 整除约束的并行 mesh（例如引入 CP/TP 组合使 `DP` 仍整除 `512`/`32`，或使用 2 的幂次卡数）、跨机 placement 与权重同步映射，并单列报告，不混入 `4+4` 主表。

### 主线和可选扩展

1. **隔离 RL 的主对照**：TPU 和 GPU 都从同一个 TPU P2 `global_step_372/huggingface` 启动，固定数据、工具、算法和评估。先验证 HF 文件清单/哈希、加载和同输入 forward，再对比 RL 训练。
2. **可选端到端复现**：GB200 从与 TPU P2 相同的 base、数据和 SFT 配置独立训练至 step 372，导出 GPU SFT HF checkpoint；比较两侧 SFT，再分别用各自的 SFT checkpoint 启动相同的 RL。此轨的 RL 差异包含 SFT 差异，单列报告。

先完成主对照，无需 GPU SFT；只有要比较 SFT 或验证从 base model 开始的完整 recipe 时才启动第二轨。明确每个 run 的 checkpoint 来源，避免把两轨结果合并。

## 0. 冻结输入和实现

- 记录 TPU 及 GPU 的 git SHA、实际 resolved Hydra config、环境镜像 digest、Python/PyTorch/CUDA/NCCL/Ray/vLLM/verl/TorchTitan/TransferQueue 版本、模型和 tokenizer revision、精度、attention/MoE kernel、所有随机种子、启动命令及 host/GPU 分配。记录 TPU 的实际运行日志，不仅是计划默认值；如 P1a/P1c 或 P3 最终配置变化，同步更新对照表。
- Base model revision 为 `ad44e777bcd18fa416d9da3bd8f70d33ebb85d39`。ReTool SFT parquet 2,000 行，SHA256 `92a29cea03f72dc79066e1697f4d7423d91bd6a02bd1aac50f5c9dfd89b27faa`；train 和 validation 使用同一文件，故 validation 是 **in-sample**。复制 P2 checkpoint 所有 safetensors（16 shards，共 `61,064,245,248` 字节）、index、config、tokenizer 和 chat template，逐文件生成 SHA256 清单，并在两平台重新校验。
- RL DAPO parquet 为 1,791,700 行（17,917 prompts × 100），SHA256 `22e266f54502ce271ee08713feb8b30caf97a4ea948aca0e0506f58c4cdd0545`。保持原顺序/重复、预处理和 shuffle 逻辑，不做去重。保存输入样本 ID、截断、token IDs 和 loss mask；同一 seed 也要实测数据顺序。
- 固定 sandbox 服务源码/版本、镜像、执行超时、资源限制、并发、网络和工具返回格式；固定 tokenizer、prompt、tool schema、parser、reward 和 AIME 题目。复用 [`retool_tpu.py`](https://github.com/syangx39/verl-upstream/blob/6fa65e49094c04a9b10cb766ecb1ba7c0976cc5a/examples/tpu/retool/retool_tpu.py) 与其 tool 配置前，先在 GPU worker 实际导入和执行。配置中的 `127.0.0.1` 是**各执行节点自己的** loopback，须证明调用进程能访问对应 sandbox，不能只在 head 上验证。
- 训练 correctness 使用原始 `acc` 或逐题判分；shaped reward 包含工具轮次调整，不能当作正确率。

## 1. 先准备 GPU launcher 与派生镜像，再做预检

这是计划，不声称现有 TPU 脚本能在 GPU 原样运行。本 PR 已随附用于 GB200 派生镜像构建与两级环境校验的 [`gb200/Dockerfile.gb200.torchtitan`](./gb200/Dockerfile.gb200.torchtitan) 与 [`gb200/verify_venv_lock.py`](./gb200/verify_venv_lock.py)；在此基础上新增并版本化 GPU RL launcher（GPU SFT launcher 仅为可选实验准备）：

- 可选 SFT 同时参考 [TPU ReTool TorchTitan SFT 脚本](https://github.com/syangx39/verl-upstream/blob/6fa65e49094c04a9b10cb766ecb1ba7c0976cc5a/examples/tpu/retool/run_qwen3_30b_a3b_sft_torchtitan.sh)（`engine=torchtitan` 字段语义）与 [多轮 SFT 示例](https://github.com/syangx39/verl-upstream/blob/6fa65e49094c04a9b10cb766ecb1ba7c0976cc5a/examples/sft/multiturn/run_qwen2_5_0_5b_fsdp.sh)（统一 `-m verl.trainer.sft_trainer` 入口），将 `trainer.device=cuda` 并对齐 Qwen3-30B-A3B 与 ReTool P2 数据/模板，避免误搬 `engine=fsdp` 专属字段（如 `ulysses_sequence_parallel_size` vs `engine.context_parallel_size`）。
- RL 参考 [v1 separate-async GPU 测试](https://github.com/syangx39/verl-upstream/blob/6fa65e49094c04a9b10cb766ecb1ba7c0976cc5a/tests/special_e2e/run_v1_separate_async.sh) 的独立资源池、vLLM 和 NCCL 权重同步，结合 [GPU TorchTitan 小模型测试](https://github.com/syangx39/verl-upstream/blob/6fa65e49094c04a9b10cb766ecb1ba7c0976cc5a/tests/special_e2e/run_ppo_trainer_torchtitan.sh) 的 engine 选择，以及 [Qwen3-30B-A3B GPU 示例](https://github.com/syangx39/verl-upstream/blob/6fa65e49094c04a9b10cb766ecb1ba7c0976cc5a/examples/grpo_trainer/run_qwen3_30b_a3b_fsdp.sh) 的模型支持，以及 [TPU ReTool recipe](https://github.com/syangx39/verl-upstream/blob/6fa65e49094c04a9b10cb766ecb1ba7c0976cc5a/examples/tpu/retool/run_qwen3_30b_a3b_dapo_torchtitan.sh) 的数据和语义。这些例子各自只覆盖部分路径；原 separate-async 测试没有 TorchTitan actor 分支，需新写 launcher 并验证组合，不能直接宣称已验证 30B + 多轮工具 + 长上下文。
- RL 使用 `trainer.use_v1=True`、`trainer.v1.trainer_mode=separate_async`、`actor_rollout_ref.hybrid_engine=False`，使用 `model_engine=torchtitan` 和 `actor_rollout_ref.actor.strategy=torchtitan`，初始 TorchTitan trainer 4 GPU、vLLM rollout TP=4 / 4 GPU、NCCL 同步。TorchTitan 内部可以用 FSDP2 做分片，这和选择 veRL 原生 FSDP2 engine 不同。确认实际 Ray placement 和版本兼容；不要把另一套 fully-async trainer 或 colocated 运行当作同一模式。
- 在目标 GB200 节点上先执行严格宿主机预检：
  ```bash
  .venv/bin/python /workspace/verify_venv_lock.py /workspace/verl-pin \
    --extras fsdp,vllm,torchtitan \
    --check-all-locked \
    --require-async \
    --require-torchtitan \
    --require-rl-moe-helper \
    --require-gb200 \
    --min-gpus 4
  ```
  `--require-gb200` 会强制校验 `aarch64` 架构、Blackwell SM `>= 10.0`、单卡 HBM `>= 165.0 GiB`、设备名匹配 `GB200|B200` 以及可用 `torch.cuda.nccl.version()`（非目标 GPU 或缺失 NCCL 将直接非零退出），同时采集 `nvidia-smi -L`、`nvidia-smi topo -m`、主机 CPU/内存、节点角色与网络。明确区分“构建期包锁通过”与“GB200 parity 运行条件就绪”：即使宿主机预检通过，仍必须继续完成下方模型加载、MoE 路由一致性与 actor → NCCL → vLLM 权重同步实测。
- 加载 base 和共享 HF checkpoint，验证 tokenizer、参数 key/shape、文件哈希、权重分片和 MoE expert 映射；分别试短样本和接近长度上限（10,240 tokens）的样本，记录 HBM 峰值。先小并发后扩大；发生 OOM 就记录实际配置及失败点，不悄悄缩短上下文。
- 测一次 actor → rollout 完整同步：固定输入比较同步前后权重映射、top-k expert routing 与输出，至少检查初始权重和一次更新后权重，尤其 MoE expert。NCCL 调用成功不等于权重被正确加载。

### 镜像与 veRL 版本：推荐方案

**结论：不需要从裸 CUDA 重建，但必须换掉旧 `2026-08-17` 基础镜像；基于上游 `2026-10-23` 发布的原生 `linux/arm64` 预缓存基础镜像 `verlai/verl:uv.cu130.dev5` 做轻量派生镜像（见 [`gb200/Dockerfile.gb200.torchtitan`](./gb200/Dockerfile.gb200.torchtitan)）。上游依赖环境验证镜像（`VERL_PIN=3fdf2b548882a8aeccf0842c2150d9c9f1790a55`，`uv.lock` SHA256 `df508db0b957fa181c229fb75885cd3863cb651fd5233d0fa5af95e990bda0be`）已在原生 `aarch64` 构建完成并通过 4×GB200 节点 `--require-gb200 --min-gpus 4` 预检（镜像 digest `sha256:ec7c927495a06279b128b47424ecd7f84eb70434848e82e46ba26747b5b805f1`；实测 `uv sync` 热缓存耗时 `12 s` / 冷缓存 `103 s`，完整构建+快照 `202 s`；全部 264 个锁包版本、`flash_attn 2.8.3`、`PPOTrainerSeparateAsync`、TorchTitan `0.3.0` slotted `MoE.Config`、4×GB200 SM `10.0` `184.31 GiB/卡` 及 NCCL `2.29.7` 校验通过）；待 `<GPU_CODE_SHA>` 分支完成 ReTool 代码与 TorchTitan `0.3.0` MoE 修复移植后，再按同一流程固化正式运行镜像。**

新旧 GPU 基础镜像与 TPU P0.2 环境逐项对照如下：

| 维度 | 旧 GPU 基础镜像 (`2026-08-17`) | **推荐新基础镜像 (`verlai/verl:uv.cu130.dev5`, `2026-10-23`)** | TPU v7x 冻结环境 (`P0.2`) |
| --- | --- | --- | --- |
| **镜像 Digest** | `verlai/verl@sha256:094943c1909590564feda8122bd090ba0a93aa42345245111a05225c1a7972f1` (`uv-cu130-arm64`) | Multi-arch index: `sha256:21e75451badbd7080a221f4cf408306ce637b77d5bb9df80a37e4387b5310146`<br>**`linux/arm64` (GB200) digest**: **`sha256:6d7c328b1e7aedd2dece2f939a827704295b13f4bb75f2f17a962a91a19376b9`** (`11.90 GB`) | `us-west2-docker.pkg.dev/tpu-pytorch/raycluster/verl-tpu@sha256:dcece4bc...` |
| **PyTorch** | `2.11.x+cu130` | **`2.13.0+cu130`** (`manylinux_2_28_aarch64` 原生 wheel) | **`2.13.0+cpu`**（主次版本一致） |
| **TorchTitan** | 无（未锁入 `uv.lock`） | **`torchtitan==0.3.0`** + `torch-checkpointing==0.1.0` | `0.1.0 @ f880cc73` |
| **vLLM** | `0.24.x` | **`vllm==0.32.0`** (`cp38-abi3` `aarch64` wheel，`VLLM_USE_V1=1`) | `0.29.0` (TPU，`VLLM_USE_V1=0`) |
| **TransferQueue** | 旧版 | **`TransferQueue==0.1.11`** | **`0.1.8`**（冻结 P0.2 两镜像实测均为 `0.1.8`） |

不要在旧环境中直接 `pip install -U torchtitan`（会导致 torch、vLLM 与 CUDA 扩展 ABI 失配），也不要混拼 TPU lock、nightly wheel 和新 lock。

**建立 `<GPU_CODE_SHA>` 工作分支时的必适配与跨版本核对清单（关键！）：**

对比上游候选 [`verl@3fdf2b548882a8aeccf0842c2150d9c9f1790a55`](https://github.com/verl-project/verl/blob/3fdf2b548882a8aeccf0842c2150d9c9f1790a55/pyproject.toml)、[`torchtitan v0.3.0`](https://github.com/pytorch/torchtitan/blob/v0.3.0/torchtitan/models/common/moe.py#L364-L370) 与 `tpu-main`（`6fa65e49`）源码，以下差异必须针对 **TorchTitan `0.3.0` 原生 API** 实现并验证，不能原样 cherry-pick 旧函数：

1. **[必适配 · 针对 TorchTitan `0.3.0` 关闭会导致 Router 分叉的 Load Balancing]**：
   - 目标：TorchTitan 默认 Qwen3 MoE 配置开启 `moe.load_balance_coeff = 1e-3`，每步 optimizer 更新后通过 hook 更新 `expert_bias_E` 改变 top-8 路由；**但 `Qwen3StateDictAdapter.to_hf` 同步权重到 vLLM 或导出 HF 时会丢弃 `expert_bias_E`**，导致第 2 步起 actor 与 vLLM expert 路由静默分叉。
   - **API 适配要求**：在 TorchTitan `0.3.0` 中，`torchtitan.models.common.moe.MoE.Config` 是 `@dataclass(slots=True)`（slotted dataclass），**已移除 `aux_loss` 字段**；若原样照搬 `tpu-main`（基于 `0.1.0`）的 `moe.aux_loss = None` 会直接触发 `AttributeError`。GPU 分支必须按 `0.3.0` `MoE.Config` schema 编写并单测适配函数（显式将 `moe.load_balance_coeff = None`，且不访问不存在的 `aux_loss` 字段，同时保持 GPU 原生 grouped_mm 而非 Pallas）。
   - **必测门槛**：单测和 `G0` 必须同时验证 (a) 构造出的 Qwen3 MoE model config 实际 `load_balance_coeff is None` 且无 hook 注册；(b) 完成 1 次 optimizer 更新并经 NCCL 同步至 vLLM 后，固定输入在 actor 与 vLLM 两侧的 top-k expert routing 及输出对齐。
2. **[先核对再决定 · 在线 NCCL 权重同步 vs 离线 HF Checkpoint 导出路径]**：
   - 候选上游 `3fdf2b54` 已包含部分 expert stack 批量处理逻辑。不要盲目整段搬运 `tpu-main` 的导出改动，而是将**在线 `actor → NCCL → vLLM` 权重流式同步**与**离线 `61 GB` HF checkpoint 导出**两条路径分开核对与计时：先验证 `3fdf2b54` 现有实现下 `Qwen3StateDictAdapter` 3D expert 权重（`w1/w2/w3`）shape/transpose 在两条路径上的正确性与耗时，仅在实测存在缺失或性能瓶颈时做最小补丁。
3. **[必移植 · ReTool 任务与审计脚本]**：移植 `examples/tpu/retool/`（`retool_tpu.py`、tool schema、rollout audit / validation dump 与 `training.max_norm` 实效日志），升级仅作用于 GPU 分支，不影响正在运行的 TPU 负载。
4. **[必核对 · `torchtitan 0.1.0` → `0.3.0` 调度器与导入路径]**：核对 `LRSchedulersContainer` 模块路径（`torchtitan.components.lr_scheduler` vs `torchtitan.components.optimizer`）及 `torchtitan.__file__` 实际导入位置（已内置于 `verify_venv_lock.py` 自动校验）。

**标准构建与校验命令：**

```bash
(
  set -euo pipefail
  cd examples/tpu/retool/gb200
  docker buildx build --platform linux/arm64 --progress=plain \
    --build-arg VERL_REPO=https://github.com/syangx39/verl-upstream.git \
    --build-arg VERL_PIN=<GPU_CODE_SHA> \
    --build-arg EXPECT_LOCK_SHA256=<UV_LOCK_SHA256> \
    --build-arg FLASH_ATTN_FROM_SOURCE=0 \
    -f Dockerfile.gb200.torchtitan \
    -t retool-gb200-torchtitan:<GPU_CODE_SHA> --load . 2>&1 | tee build_arm64.log
)
```

- `Dockerfile.gb200.torchtitan` 在 `uv sync --frozen --all-packages --extra fsdp --extra vllm --extra torchtitan` 前后强校验 `uv.lock` 的 SHA256 与 `VERL_PIN`；若 `flash-attn` 在 `aarch64` 上需源码构建，传 `--build-arg FLASH_ATTN_FROM_SOURCE=1` 会同时启用 `FLASH_ATTENTION_FORCE_BUILD=TRUE` 与 `--no-binary flash-attn`（防止 `pip`/`uv` 或 `setup.py` 静默回退预编译 wheel），并须归档 `build_arm64.log` 与 `flash_attn` 扩展 import 结果。
- `--extra fsdp` 仅提供依赖组合，**不表示选择 veRL FSDP2 actor engine**。统一设置 `VIRTUAL_ENV=/workspace/verl-pin/.venv` 与 `PYTHONPATH=/workspace/verl-pin`，确保 Ray head 与所有 worker 使用同一 checkout 与 `.venv`，严禁运行时挂载另一份源码覆盖。
- Sandbox 是独立进程/服务，**不必单独构建镜像**。复用并固定 TPU 的 server 和工具调用语义，核实 GPU 各请求节点可访问服务、运行依赖和实际生效的资源/超时限制；若另行容器化，作为显式差异记录。

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

### 4.1 要对齐的语义与显式引擎差异

以 TPU 的**实际 resolved 配置与日志**为最终基准，首轮按下表准备。实际 TPU P3 启动参数若不同，应先重冻结两边，不把相异 run 写成 parity。

| 项目 | 首轮对照 |
| --- | --- |
| 数据与起点 | 原始 DAPO parquet；主轨使用同一 TPU P2 step-372 HF；端到端轨分别用各自 P2 HF |
| 一次训练更新 | 64 prompts × 每题 8 responses，nominal 512 trajectories；PPO mini-batch 64；核对过滤、丢弃及实际更新数 |
| 长度、轮次与模板 | prompt 最大 2,048，response 最大 8,192；assistant/user 各最多 8 turns；**显式固定 `+data.apply_chat_template_kwargs.enable_thinking=False`**（与 TPU 脚本一致，区别于 Askhat `fcze3xlx` 的 `true`）；保持同一 tool schema 和 token mask |
| vLLM 引擎版本 | **显式记录已知跨平台差异**：TPU 脚本设置 `VLLM_USE_V1=0`（vLLM `0.29.0`），GPU 镜像使用 `VLLM_USE_V1=1`（vLLM `0.32.0` V1 engine），重点核对 tool-call parser、chunked prefill 与 rollout logprob 行为 |
| 算法 | GRPO；clip low/high `0.2/0.28`，clip c `10`；reward KL 和 KL loss 关闭，entropy coefficient 0 |
| optimizer | LR `1.4e-6`；逐项核对 optimizer、betas、epsilon、weight decay、schedule、global normalization 和有效 gradient clipping |
| 异步和校正 | separate_async；warmup batches 2、每 step 同步、max off-policy threshold 4；rollout correction bypass、token IS threshold 2 |
| 评估 | AIME 2024/2025 各 30 题，temperature `1.0`、top-p `0.6`、每题 `n=30`；step 0 与 step 25，保留每题原始输出和判分 |

特别核对 TPU TorchTitan 的实际梯度裁剪：脚本中 `actor.optim.clip_grad=0.088` 未必是 TorchTitan 有效的 `training.max_norm`。以运行代码和日志证据为准，再设置 GPU TorchTitan 对应的有效值，记录两者和差异。两个平台的 TorchTitan 版本、optimizer 实现和配置解析也可能不同；尤其核对 betas、scheduler 和 min LR。不要直接照搬 TPU context parallel 下的 `ppo_max_token_len_per_gpu=4608` 到 GPU，须验证最大完整序列及 loss 语义。冻结版本的 `actor_rollout_ref.actor.torchtitan.max_seq_len` 若为 null 会回退至 2,048：GPU launcher 应根据实际 packing/CP 显式设置足够长度并测试近上限样本，在新版本核对对应字段和实际 RoPE 缓存。

### 4.2 分阶段执行与门槛

1. **G0，基础 smoke**：首选 `4+4` 启动，先用少量固定请求检查 sandbox、工具循环、权重同步（含更新后 actor vs vLLM top-k expert routing 对齐）、打分及日志，含接近最大长度的案例。确认无死锁、OOM，角色位置正确；保存峰值显存和 trace。
2. **G1，短 RL smoke**：共享 P2 checkpoint，按 TPU P1c 的诊断缩小 workload：train batch 8、mini-batch 8、每题 4 条（32 trajectories，`32 % 4 == 0`）、2 steps，关闭训练前验证。确认梯度、optimizer、同步、策略版本以及 HF checkpoint 可导出并重新加载；这是诊断，不与全规模性能直接比较。HF 权重重载不等于完整断点续训：如需续训，另行验证 optimizer、scheduler 和异步队列状态；主对照优先连续从头运行。
3. **G2，主对照**：共享 P2 checkpoint，采用上表 workload 运行至 step 25，训练前与 step 25 均做完整 AIME24/25。保留所有 25 步训练指标，确认 step 25 的最终或显式 checkpoint 已保存；逐步对照同配置 TPU P3。固定并对账两边的 scheduler horizon 和前 25 步逐步 LR：若把 total steps 改为 25 会改变调度，就保持双方相同的完整 horizon 并在 step 25 评估、有序停止；不要把不同 schedule 混为一谈。两次评估合计 `2 × 2 × 30 × 30 = 3,600` 条样本，提前计入时间和 sandbox 压力。若 TPU P3 尚无可用结果，先保存 GPU 数据并将跨平台结论标为待对照。
4. **G3，可选端到端轨**：GPU 自己的 SFT step-372 checkpoint，以 G2 相同配置跑 RL；与 TPU 自身 SFT → RL 对照。算力允许时增加独立训练 seed；不要将重复 rollout 采样误认为独立训练重复。

每阶段通过后再扩展：记录完成步数、失败/重试、实际样本数、有效版本和输出位置；任何改动都重新归档 resolved config。Askhat 的历史 v5p RL（`fcze3xlx`）使用不同 response length（16K vs 8K）、turn（16 vs 8）、n（16 vs 8）、LR（`2e-6` vs `1.4e-6`）及 `enable_thinking=true`，而且在 step 23 中断；只能作为上下文，不能当成上述 8K/8-turn run 的严格 parity 基准。

## 5. 结果怎么比

### Parity

- SFT（仅开展可选实验时）：相同样本和有效 tokens 的训练 loss、grad norm、LR、末尾 loss 分布、相同 in-sample validation；比较各自的 step-372 checkpoint，并保留前述固定输入数值对账。
- RL：按 step 和实际 consumed trajectories 对比 correctness、原始 reward/组件、format/tool failures、timeout、turn 数、response 长度、截断率、entropy、actor KL、clip fraction、grad norm、importance weights 和策略滞后。保存对应的逐样本数据，允许异步执行顺序和随机轨迹不同。
- AIME：对比 step 0/25 的每题正确率及总体 pass@1 估计，并核对计算口径；按题做配对 bootstrap 区间，保留每题 30 条原始样本；另评估每题 rollout 抽样的不确定性（例如按题、再按样本分层重采样），区分题目与采样波动。多训练 seed 时另报告 seed 间变化；若缺少原始轨迹、基线或事前约定容差，结论写 evidence insufficient，不以单次分数相近就断言 parity。
- 汇总显式差异：平台、物理拓扑、parallelism、precision/kernel、`VLLM_USE_V1`（`0` vs `1`）、TorchTitan 版本（`0.1.0` vs `0.3.0`）、TransferQueue 版本（`0.1.8` vs `0.1.11`）、实际 optimizer、数据顺序、工具运行环境和异步版本。出现轨迹分叉时，先用固定输入测试定位，再分析自由运行统计。

### 性能

固定 prompt/生成长度分布、全局 batch、并发、工具 CPU 资源、cache 与日志/评估开销，记录实际偏差。至少给出：

- 端到端 wall time（含启动、加载、编译/预热、验证、存储）以及排除这些开销的稳态 step 中位数/P90；逐项写明时间边界，尤其 P2 的整作业时间与训练 step 时间不能直接比较。
- rollout 完成轨迹数/秒、输入/输出及有效训练 tokens/秒、每 step 实际 token 数；训练、生成、工具、同步、checkpoint 和 eval 的可观测耗时。异步阶段会重叠，不将各阶段耗时简单相加为总时间。
- trainer、rollout 分别及合计的物理设备数、利用率、HBM 峰值（统一标注 `GB` 与 `GiB`）、accelerator-seconds/完成轨迹及 accelerator-seconds/有效训练 token。TPU 记录物理 chip 和逻辑设备；GPU 记录卡数。可额外记录实际价格/能耗，缺数据不估算。
- 单独展示固定工作量绝对用时和设备资源消耗；只有学习曲线及目标指标可比时才报告 time-to-target。不要用 `32/8` 直接推断理论或实际性能倍数，也不把不同长度分布和不同成功率的吞吐当等价。

## 6. 交付清单

- 两平台每个 run 的 SHA、launcher、resolved config、环境/硬件/placement、seed、输入/模型文件清单与哈希、启动命令和实际变更。
- GB200 派生镜像与环境锁校验文件：[`gb200/Dockerfile.gb200.torchtitan`](./gb200/Dockerfile.gb200.torchtitan)、[`gb200/verify_venv_lock.py`](./gb200/verify_venv_lock.py)。
- G0/G1 的可行性与同步/数值对账报告，G2（以及若执行 G3）的 checkpoint、逐步指标、逐条轨迹、AIME 逐题结果、性能原始日志和可复现的汇总脚本。
- 一张 TPU vs GB200 表：RL 主线及可选 SFT 分行，注明规模、工作量、成功步数、parity 指标、稳态与总 wall time、总 accelerator-hours、重要配置差异及未解决问题。未运行、失败、数据不可比均如实标注。

参考：[TPU 结果记录](https://github.com/syangx39/verl-upstream/blob/6fa65e49094c04a9b10cb766ecb1ba7c0976cc5a/examples/tpu/retool/REPRO_RESULTS_TPU7X.md) · [Askhat GPU PR #108](https://github.com/jialei777/verl-upstream/pull/108) · [verl v1 async 文档](https://verl.readthedocs.io/en/latest/advance/v1_async_trainer.html)。
