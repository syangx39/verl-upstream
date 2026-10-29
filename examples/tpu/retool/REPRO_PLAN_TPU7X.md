# Reproduction plan: ReTool (Qwen3-30B-A3B) on TPU 7x

Status: **rev 1, proposed** (2026-10-29). Not started. Owner: @syangx39.

This plan reproduces the ReTool recipe added in
[jialei777/verl-upstream#108](https://github.com/jialei777/verl-upstream/pull/108) (this directory, see
[`README.md`](README.md)) on TPU 7x (Ironwood). It reuses the recipe scripts unchanged and only switches the topology
settings to TPU 7x. The layout follows our earlier Qwen3-4B GRPO reproduction on tpu7x: decisions, acceptance criteria
with tolerances, phases with exit criteria, queue strategy, tracking and risks.

## 1. Goal

1. Run both stages of the recipe end to end on TPU 7x under our own accounts:
   - Stage 1: multi-turn cold-start SFT of `Qwen/Qwen3-30B-A3B` on ReTool-SFT (372 steps), exporting a HuggingFace
     checkpoint.
   - Stage 2: multi-turn async DAPO RL with the loopback Python sandbox, starting from the Stage 1 checkpoint.
2. Match the v6e results reported in #108 within the tolerances of section 3.
3. Record TPU 7x step time and stage split for the same logical configuration.

Out of scope for rev 1: performance tuning, more seeds, the 16K / 16-turn schedule, and the Qwen3-32B dense variant
from [askhat-g/verl-recipe#1](https://github.com/askhat-g/verl-recipe/pull/1). All of them are P5.

## 2. What we reproduce

| | v6e reference (#108) | TPU 7x mapping |
|---|---|---|
| Hardware | 2 x v6e-32: 32 trainer chips + 32 rollout chips | 2 x tpu7x `2x2x4` slices: 16 chips = 32 TensorCore devices each |
| Trainer | TorchTitan, SimpleFSDP=8 x Ulysses CP=4, TP=1, EP=1, `TPU_EAGER_MODE=DEFER_NEVER`, splash attention | same, over 32 devices: `TPU_ACCELERATOR_TYPE=tpu7x NNODES_TRAINER=4 N_CHIPS_TRAINER=8` |
| Rollout | one vLLM server, TP=32, EP=32, Raiden P2P weight sync | same: `NNODES_ROLLOUT=4 N_CHIPS_ROLLOUT=8` (TP = EP = 32) |
| Stage 1 SFT | 372 steps, batch 32, `max_length=16384`, lr 1e-5 cosine; `train/loss` 0.78 -> 0.25, `val/loss` 0.214; ~18.6 s/step | same hyperparameters, one `2x2x4` slice |
| Stage 2 RL | 64 prompts x 8 responses, 8K response, 8 turns, `separate_async`, bypass mode; ~108 s/step (`update_actor` 94.2 s, Raiden sync 13.2 s) | same hyperparameters |
| RL accuracy | AIME 2024 mean@30 22.44% -> 30.00%, AIME 2025 mean@30 14.17% -> 18.33% at step 25 (400-step run) | section 3 |

Every TPU 7x chip exposes its 2 TensorCores as 2 devices with 96 GiB HBM each
([`platform_tpu.py`](../../../verl/plugin/platform/platform_tpu.py)), so 32 v6e chips map to 32 TPU 7x devices
(16 chips). Each device has 3x the HBM of a v6e chip, so the v6e parallelism fits with headroom. We still keep it
unchanged until P3 passes (D3).

## 3. Acceptance criteria

The v6e references are single runs, and #108 reports the 400-step run only up to step 25. The tolerances are
therefore provisional; we tighten them once the full v6e logs arrive (section 9).

| # | Check | Phase | Statistic | v6e reference | Pass if |
|---|---|---|---|---|---|
| R1 | TPU 7x numerics flag active | every run | effective `LIBTPU_INIT_ARGS` of trainer **and** rollout workers | - | contains `--xla_tpu_bf16_emission_mode=PROMOTE_F32` on both sides (D4) |
| R2 | Topology | P1 | startup logs | - | 4D mesh for 32 trainer devices (no single-device fallback), vLLM TP=32 / EP=32, Raiden sync completes |
| R3 | SFT step-1 loss | P1 | `train/loss` at step 1 (same weights, same data order) | 0.7941 (#108 smoke) | within +-0.02 |
| R4 | SFT final losses | P2 | mean `train/loss` over steps 363-372; `val/loss` at step 372 | ~0.25; 0.214 | within +-10% |
| R5 | HF export | P1, P2 | `global_step_N/huggingface` | 16 bf16 shards, 56.9 GiB | shards, index, config and tokenizer complete; vLLM loads it |
| R6 | Step-0 AIME of our SFT model | P3 | `val-core/aime_2024/acc/mean@30`, `val-core/aime_2025/acc/mean@30` at step 0 | 22.44%, 14.17% | within +-4.5 pp (about 3 SD of the n=30 sampling noise) |
| R7 | Early RL numerics | P3 | median `actor/ppo_kl` and `actor/rollout_corr/k3_kl`, steps 3-10 | pending v6e logs | within x0.5 to x2 of v6e |
| R8 | RL learning by step 25 | P3 | AIME 2024 / 2025 mean@30 at step 25 | 30.00%, 18.33% | AIME 2024 at least step-0 + 3 pp and within +-5 pp of v6e; AIME 2025 within +-5 pp of v6e |
| R9 | Step time | P2, P3 | median `timing_s/step` and stage medians (`gen`, `update_actor`, `update_weights`) | SFT ~18.6 s; RL ~108 s | informational; slower than v6e needs an explanation |
| R10 | Completeness | P2, P3 | eval points and fields | - | steps 0 and 25 for both AIME sets (mean / maj / best@30) plus tool-call stats; a missing point makes the result incomplete, not passed |

Notes:

- R6: with 30 problems x 30 samples, the SD of mean@30 is at most ~1.4 pp. Variance between SFT runs comes on top.
- If we get the v6e SFT checkpoint (section 9), we also evaluate it at step 0 on TPU 7x. Within +-3 pp of the v6e
  numbers, that separates rollout numerics from SFT run-to-run variance.

## 4. Decisions

| # | Decision |
|---|---|
| D1 | **Target**: #108's Qwen3-30B-A3B recipe in this directory. The Qwen3-32B dense variant (askhat-g/verl-recipe#1, which needs [jialei777/verl-upstream#77](https://github.com/jialei777/verl-upstream/pull/77)) is P5 |
| D2 | **Code**: `syangx39/verl-upstream@tpu-main` at or after `a075ebc7` (#108 plus the HF export retention fix). Every run record pins the exact SHA. If newer TPU 7x fixes land upstream, we sync before P1. TPU 7x changes go in as separate small commits |
| D3 | **Logical configuration identical to v6e**: only `TPU_ACCELERATOR_TYPE=tpu7x`, `NNODES_*=4` and `N_CHIPS_*=8` change. No tuning of CP, eager mode, memory utilization or bucket size before P3 passes |
| D4 | **Numerics**: `--xla_tpu_bf16_emission_mode=PROMOTE_F32` in `LIBTPU_INIT_ARGS` for every run, segment and retry. This flag was needed to match the H100 and v6e numerics in our Qwen3-4B tpu7x reproduction. verl forwards `LIBTPU_INIT_ARGS` from the driver to the trainer and rollout workers |
| D5 | **Storage**: inputs (model, datasets) are staged once to a GCS prefix and mounted read-only. The model is copied to `/dev/shm` on every TPU host (`MODEL_PATH`). All outputs (HF exports, DCP checkpoints, TensorBoard, logs) go to a separate GCS output prefix. Host `/tmp` is never used for checkpoints, because a tpu7x node disk has less free space than one 57 GiB export |
| D6 | **Resume**: Stage 1 SFT runs in one window without resume. Stage 2 RL beyond one window uses DCP checkpoints plus `trainer.resume_mode=auto` (P4). The RL learning rate is constant (`min_lr_factor=1.0`, no warmup), so segments can be chained by raising `trainer.total_training_steps` |
| D7 | **Keep #108's script semantics**, including two known oddities: `optim.clip_grad` is not wired into TorchTitan (effective `max_norm` 1.0), and Stage 1 does not set `enable_thinking=False`. Both are recorded; ablations are P5 |
| D8 | **Pinned inputs**: `Qwen/Qwen3-30B-A3B`; `JoeYing/ReTool-SFT`, preprocessed with verl-recipe's `retool_sft_preprocess.py` using the tool schema of [`sandbox_fusion_tool_config_tpu.yaml`](sandbox_fusion_tool_config_tpu.yaml); `BytedTsinghua-SIA/DAPO-Math-17k`; `Maxwell-Jia/AIME_2024`; `yentinglin/aime_2025`. HF revisions and file sha256 go into the run record |

## 5. P0 deliverables

| # | Deliverable | Why | Done when |
|---|---|---|---|
| P0.1 | **Launch path** for 2 x `2x2x4` on the shared TPU 7x GKE cluster (`bodaborg-tpu7x-nap`: Kueue + JobSet, node auto-provisioning) | The in-repo TPU 7x manifests ([`ray-tpu-v7x16-2slice.yaml`](../gke/ray-tpu-v7x16-2slice.yaml)) are KubeRay `RayCluster`s for a cluster with the KubeRay operator. The shared cluster runs Kueue-managed JobSets, like [`ray-jobset-v5p-nap.yaml`](../gke/ray-jobset-v5p-nap.yaml) | One of two options exists: a ReTool profile in our tpu7x launch tooling (render, submit, watch, collect), or a TPU 7x port of `ray-jobset-v5p-nap.yaml`. Server-side dry run passes. TPU pods select `gke-tpu-topology: 2x2x4` and the cluster's `2x2x4` placement policy. JobSet names are at most 30 characters. A memory-backed `/dev/shm` of at least 80 GiB is mounted on TPU pods. Pod env sets `TPU_ACCELERATOR_TYPE=tpu7x-32` while the job sets `tpu7x` for verl, as in `ray-tpu-v7x16-2slice.yaml` |
| P0.2 | **Image** | Needs torch_tpu, torchtitan, vllm-torchtpu, tpu-sync-torch, fastapi and uvicorn at versions that match the code SHA | #108's image (`verl-tpu:v20261001-tsync990787257`) or a rebuild from [`Dockerfile.tpu`](../gke/Dockerfile.tpu) pulls on the cluster; package versions recorded |
| P0.3 | **Inputs** (D8) | Pods should not depend on HF Hub access | Model and datasets staged under the input prefix with the directory layout the scripts expect (`dataset/<org>/<name>`, `assets/hf/Qwen3-30B-A3B`, `dataset/ReTool-SFT/data/train-00000-of-00001.parquet`); sha256 recorded |
| P0.4 | **Output persistence** | Pods disappear minutes after completion | `SAVE_PATH`, `CKPTS_DIR` and `TENSORBOARD_DIR` point to the output prefix. The driver log and each worker's effective environment are mirrored to GCS during the run |
| P0.5 | **Host ordering** | Trainer ranks must follow the slice worker index. In our Qwen3-4B port, IP-string order caused a SparseCore halt in the first FSDP collective | Trainer rank-to-host order logged for P1a and matches the worker index. If it doesn't, the host-order fix is applied |
| P0.6 | **Numerics flag record** (R1) | The flag must reach both sides on every run | Effective `LIBTPU_INIT_ARGS` of trainer and rollout workers is written to the output prefix and checked by the extractor |
| P0.7 | **Resumable RL config** (P4) | #108 saves `hf_model` only, so a run cannot resume | `actor_rollout_ref.actor.checkpoint.save_contents=['model','optimizer','extra','hf_model']`, `trainer.resume_mode=auto`, DCP to the output prefix. Old step folders are pruned by us, because TorchTitan's `keep_latest_k` does not prune across `global_step_N` folders in the PPO trainer layout |
| P0.8 | **Metrics extraction** | Section 3 checks | Per-step training metrics and an eval table (mean / maj / best@30 per AIME set) from TensorBoard; prints R1-R10 |

## 6. Phases

| Phase | What | Config (on top of section 11) | Exit criteria | Wall time after admission (estimate) |
|---|---|---|---|---|
| **P0** | Section 5 | - | P0.1-P0.8 done | 1-2 days |
| **P1a** | Platform smoke with the in-repo TPU 7x MoE recipe | [`../grpo/run_qwen3_30b_a3b_torchtitan.sh`](../grpo/run_qwen3_30b_a3b_torchtitan.sh), 3 steps, 2 x `2x2x4` | R1, R2, P0.5 | ~45 min |
| **P1b** | ReTool SFT smoke | `TOTAL_TRAINING_STEPS=3 SAVE_FREQ=3 TEST_FREQ=3`, one `2x2x4` slice | R1, R3, R5 (3-step export) | ~45 min |
| **P1c** | ReTool RL smoke from the P1b export (tests the SFT -> RL hand-off) | `TOTAL_TRAINING_STEPS=2 TRAIN_BATCH_SIZE=8 PPO_MINI_BATCH_SIZE=8 N_RESP_PER_PROMPT=4 VAL_BEFORE_TRAIN=False` | sandbox healthy; tool calls executed (`num_turns` > 2 for some trajectories); Raiden sync; R1, R2 | ~60 min |
| **P2** | Stage 1 SFT, full | 372 steps, script defaults | R4, R5, R9, R10 | ~1.5-2.5 h (v6e: 1.9 h of steps) |
| **P3** | Stage 2 RL to step 25 | script defaults from the P2 export, `TOTAL_TRAINING_STEPS=25 TEST_FREQ=25` (step 0 and step 25 validation) | R6-R10 | ~1.5-2.5 h |
| **P4** | Resume, then longer RL | P0.7 config; segment 1 to step 50, segment 2 to step 100; then a reclaim test (delete a segment after a checkpoint, relaunch under the same name) | global step continues, optimizer and dataloader state restored, LR unchanged, curve continuous across the boundary; then optionally 400 steps | several windows |
| **P5** | Optional | 16K / 16-turn / n=16 schedule; Qwen3-32B dense; tuning (CP=1, `DEFER_AND_FUSE`, rollout memory utilization, multi-replica rollout); ablations (`enable_thinking=False` in SFT, `clip_grad` wiring) | as agreed | as needed |

The wall times are estimates derived from v6e numbers. We replace them with P1 / P2 measurements.

## 7. Queue strategy

Quotas, limits and reservations change daily. Re-check them before every submission with the team-internal queue notes.

| Use | Queue | Why |
|---|---|---|
| P1a-P1c | shared priority queue (per-job wall-clock limit, about 2 h) | short runs; driver timeout = limit - startup - teardown |
| P2, P3 | exclusive reservation window (multi-hour) | one uninterrupted window per stage; SFT needs only one `2x2x4` slice (16 chips), RL needs two (32 chips) |
| P4 | reservation windows, chained by resume | runs longer than one window |
| never | queues owned by other teams without the owner's OK | - |

## 8. Tracking and reporting

- One run-record row per submission: run name, code SHA, image, cluster / namespace, rendered manifest, effective
  `LIBTPU_INIT_ARGS` on both sides, submit / admit / ready / end times (UTC), first and last step, restored-from step,
  median step time, status, output prefix.
- After each phase: run the P0.8 extractor and write the verdict per acceptance row into this file (new revision).
- Final report: SFT loss curves vs v6e, AIME table (mean / maj / best@30) vs v6e, step-time table, every resume
  boundary.

## 9. Open items for the recipe author (@askhat-g)

These block exact inputs or tighter tolerances:

1. The exact SFT parquet. The README cites a ReTool-SFT dataset that is not public on the HF Hub.
2. Full v6e logs (TensorBoard) and the resolved configs for SFT and both RL runs.
3. v6e reference values for `actor/grad_norm`, `actor/ppo_kl` and `actor/rollout_corr/*` (R7).
4. The v6e SFT checkpoint, for cross-evaluation on TPU 7x (R6 note).
5. Whether `VLLM_USE_V1=0` and `VERL_AGENT_LOOP_MAX_INFLIGHT_PROMPTS` had an effect. Nothing under `verl/` reads the
   latter at `f9e896fd`.

## 10. Risks

| Risk | Mitigation |
|---|---|
| TPU 7x bf16 emission numerics shift accuracy | D4 on every run (R1); R6 and R7 as gates. Other numerics knobs only as a diagnosed, recorded fix |
| Multi-host trainer ranks in the wrong host order | P0.5 |
| Node disk too small for the model and exports | D5: model in `/dev/shm`, outputs on GCS; the retention fix in `a075ebc7` keeps at most `max_actor_ckpt_to_keep` exports |
| Wall-clock limits, reclaims, preemption | reservation windows for P2-P4; resumable RL (P0.7) |
| DCP checkpoint size (~370 GB with fp32 optimizer state at 30.5B) and retention | write to the output prefix; prune old step folders ourselves (P0.7) |
| Image not pullable, or version skew with the code SHA | P0.2; record versions; rebuild from `Dockerfile.tpu` |
| Our SFT data differs from the v6e run | D8 regeneration plus checksum; R3 / R4 catch differences; diff against the author's parquet once received |
| Multi-host vLLM (TP=32, EP=32) with multi-turn tool calls not yet exercised on TPU 7x | P1a (same rollout layout, single-turn) before P1c. A multi-replica rollout layout is allowed only as a recorded deviation |
| The sandbox runs model-written Python inside the training pods, with network access and the bucket mounts | inputs mounted read-only, outputs in a separate prefix, loopback-only server |
| Single-run v6e references | provisional tolerances; request logs (section 9) |

## 11. Run commands

Run these on the Ray head of the JobSet. `/data/inputs` is the read-only input mount, `/data/outputs` the read-write
output mount.

```bash
# All phases
export TPU_ACCELERATOR_TYPE=tpu7x
export LIBTPU_INIT_ARGS="--xla_tpu_use_enhanced_launch_barrier=false --xla_tpu_bf16_emission_mode=PROMOTE_F32"
export RAY_DATA_HOME=/data/inputs   # assets/hf/Qwen3-30B-A3B, dataset/ReTool-SFT, dataset/<org>/<name>
OUT=/data/outputs/retool-tpu7x
export TENSORBOARD_DIR="${OUT}/tensorboard/${RUN_NAME}"

# P1b / P2: Stage 1 SFT on one 2x2x4 slice (P1b adds TOTAL_TRAINING_STEPS=3 SAVE_FREQ=3 TEST_FREQ=3 and its own SAVE_PATH)
NNODES_TRAINER=4 N_CHIPS_TRAINER=8 \
MODEL_PATH=/dev/shm/models/Qwen3-30B-A3B \
EXPERIMENT_NAME=multiturn-sft-qwen3-30b-a3b-tpu7x \
SAVE_PATH="${OUT}/checkpoint/multiturn-sft-qwen3-30b-a3b-tpu7x" \
  bash examples/tpu/retool/run_qwen3_30b_a3b_sft_torchtitan.sh

# P3: Stage 2 RL to step 25 on 2 x 2x2x4
NNODES_TRAINER=4 N_CHIPS_TRAINER=8 NNODES_ROLLOUT=4 N_CHIPS_ROLLOUT=8 \
MODEL_SRC_PATH="${OUT}/checkpoint/multiturn-sft-qwen3-30b-a3b-tpu7x/global_step_372/huggingface" \
MODEL_PATH=/dev/shm/models/multiturn-sft-qwen3-30b-a3b-tpu7x \
EXPERIMENT_NAME=qwen3-30b-a3b-retool-dapo-tpu7x-p3 \
CKPTS_DIR="${OUT}/checkpoint/qwen3-30b-a3b-retool-dapo-tpu7x-p3" \
TOTAL_TRAINING_STEPS=25 TEST_FREQ=25 \
  bash examples/tpu/retool/run_qwen3_30b_a3b_dapo_torchtitan.sh
```
