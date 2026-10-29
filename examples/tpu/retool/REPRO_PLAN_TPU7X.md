# Reproduction plan: ReTool (Qwen3-30B-A3B) on TPU 7x

Status: **rev 3** (2026-10-29). Rev 2 was merged as the execution plan (#2); rev 3 was merged in #3. P0 preparation (#4, #5, #6) is complete and hardware execution is tracked in [`REPRO_RESULTS_TPU7X.md`](REPRO_RESULTS_TPU7X.md).
Owner: @syangx39.

Rev 2 folded in the review of rev 1 and the strongest checks from the alternative plan in #1 (now closed). Rev 3 fixes
the stage gates: P0 preparation vs. later validation, R11 evidence, the R13 TransferQueue condition, and SFT-only
gates. The changelog is at the end.

This plan reproduces the ReTool recipe added in
[jialei777/verl-upstream#108](https://github.com/jialei777/verl-upstream/pull/108) (this directory, see
[`README.md`](README.md)) on TPU 7x (Ironwood). It reuses the recipe scripts and keeps the reference algorithm,
hyperparameters and logical parallelism. Only the topology settings change, plus one documented TPU 7x numerics
adaptation (D4). The layout follows our earlier Qwen3-4B GRPO reproduction on tpu7x: decisions, checks, phases with
exit criteria, queue strategy, tracking and risks.

## 1. Goal

1. Run both stages of the recipe end to end on TPU 7x under our own accounts:
   - Stage 1: multi-turn cold-start SFT of `Qwen/Qwen3-30B-A3B` on ReTool-SFT (372 steps), exporting a HuggingFace
     checkpoint.
   - Stage 2: multi-turn async DAPO RL with the loopback Python sandbox, starting from the Stage 1 checkpoint.
2. Compare with the v6e results reported in #108 (section 3).
3. Record TPU 7x step time and stage split for the same logical configuration.

We report three outcomes separately:

1. **Pipeline functional**: every component runs correctly on TPU 7x (P1).
2. **Reference-aligned experiment**: model, data, training and evaluation semantics match #108, and every difference is
   recorded (P2, P3).
3. **Learning reproduction**: curves are compared with the reference under a protocol fixed in advance (P3 as an
   observation, P4b).

A passing smoke test does not establish learning parity. Wherever reference evidence is missing, the comparison is
reported as provisional.

Out of scope for rev 3: performance tuning, more seeds, the 16K / 16-turn schedule, and the Qwen3-32B dense variant
from [askhat-g/verl-recipe#1](https://github.com/askhat-g/verl-recipe/pull/1). All of them are P5.

## 2. What we reproduce

| | v6e reference (#108) | TPU 7x mapping |
|---|---|---|
| Hardware | 2 x v6e-32: 32 trainer chips + 32 rollout chips | 2 x tpu7x `2x2x4` slices: 16 chips = 32 TensorCore devices each |
| Trainer | TorchTitan, SimpleFSDP=8 x Ulysses CP=4, TP=1, EP=1, `TPU_EAGER_MODE=DEFER_NEVER`, splash attention | same, over 32 devices: `TPU_ACCELERATOR_TYPE=tpu7x NNODES_TRAINER=4 N_CHIPS_TRAINER=8` |
| Rollout | one vLLM server, TP=32, EP=32, Raiden P2P weight sync | same: `NNODES_ROLLOUT=4 N_CHIPS_ROLLOUT=8` (TP = EP = 32) |
| Stage 1 SFT | 372 steps, batch 32, `max_length=16384`, lr 1e-5 cosine; `train/loss` 0.78 -> 0.25, `val/loss` 0.214 (in-sample: the default `EVAL_DATA` is the training parquet); ~18.6 s/step | same hyperparameters, one `2x2x4` slice |
| Stage 2 RL | 64 prompts x 8 responses, 8K response, 8 turns, `separate_async`, bypass mode, script `optim.clip_grad=0.088` (effective clipping: D7); ~108 s/step (`update_actor` 94.2 s, Raiden sync 13.2 s) | same hyperparameters |
| RL accuracy | AIME 2024 mean@30 22.44% -> 30.00%, AIME 2025 mean@30 14.17% -> 18.33% at step 25 (400-step run) | section 3 |

Every TPU 7x chip exposes its 2 TensorCores as 2 devices with 96 GiB HBM each
([`platform_tpu.py`](../../../verl/plugin/platform/platform_tpu.py)). So 32 v6e chips map to 32 TPU 7x devices
(16 chips), and `N_CHIPS_*=8` counts logical devices per host, not physical chips. Each device has 3x the HBM of a v6e
chip, so the v6e parallelism fits with headroom. We still keep it unchanged until P3 passes (D3).

## 3. Checks: gates and observations

- **Gates** (R1, R2, R5a, R5b, R10-R13) are pass/fail. A phase does not proceed until its gates pass. Gates that need a
  role only check that role in phases where it exists: SFT runs have trainer workers only.
- **Observations** (R3, R4, R6-R9) are compared with the single-run v6e references using *initial ranges*. The ranges
  are not pass/fail. We fix the final thresholds before P4b, once we have the author's per-question outputs and actual
  evaluation config (section 9).

| # | Check | Kind | Phase | Statistic | v6e reference | Initial criterion |
|---|---|---|---|---|---|---|
| R1 | TPU 7x numerics flag effective | gate | every run | allowlisted env dump (P0.4) of the trainer workers; in RL runs (P1a, P1c, P3, P4) also of the rollout workers | - | `LIBTPU_INIT_ARGS` contains `--xla_tpu_bf16_emission_mode=PROMOTE_F32` on every role that exists in the run (D4) |
| R2 | Topology | gate | P1a, P1c | startup logs and live Ray nodes | - | 4D mesh for 32 trainer devices (no single-device fallback); vLLM TP=32 / EP=32; trainer and rollout on distinct complete slices; trainer rank-to-host order matches the topological order inside the slice (P0.5); Raiden sync completes |
| R3 | SFT step-1 loss | observation | P1b | `train/loss` at step 1 | 0.7941 (#108's 3-step smoke; the full run reports ~0.78 at the start) | within +-0.02 is consistent. It becomes a gate only if initial weights, data and batch order are confirmed identical to the smoke |
| R4 | SFT final losses | observation | P2 | mean `train/loss` over steps 363-372; `val/loss` at step 372 (in-sample) | ~0.25; 0.214 | within +-10% |
| R5a | SFT export integrity | gate | P1b, P2 | `global_step_N/huggingface` written by the SFT trainer | 16 bf16 shards, 56.9 GiB | shards, index, config and tokenizer complete; a fresh process reloads it; selected tensors and fixed-input logits match the in-memory model within a declared tolerance; tied weights handled |
| R5b | RL-side load | gate | P1c, P3 | the export staged for an RL run | - | the staged copy on every trainer and rollout host matches the source hashes (P0.3); TorchTitan and the multi-host vLLM server load it; the first weight sync completes (R12) |
| R6 | Step-0 AIME of our SFT model | observation | P3 | mean@30, maj@30, best@30 and per-question accuracy, AIME 2024 and 2025 | mean@30: 22.44%, 14.17% | mean@30 within +-4.5 pp |
| R7 | Early RL numerics | observation | P3 | median `actor/ppo_kl` and `actor/rollout_corr/k3_kl`, steps 3-10 | pending v6e logs | within x0.5 to x2 of v6e |
| R8 | RL by step 25 | observation | P3 | mean@30 at step 25 and change from step 0 | 30.00% (+7.56 pp), 18.33% (+4.16 pp) | AIME 2024 above step 0 and within +-5 pp of v6e; AIME 2025 within +-5 pp. A rise of a few pp on 30 questions is a signal, not evidence of learning parity |
| R9 | Step time | observation | P2, P3 | median `timing_s/step`; stage medians (`gen`, `update_actor`, `update_weights`) over a declared steady-state window, plus p90 | SFT ~18.6 s; RL ~108 s | informational. Async stage timers overlap and are not summed. Slower than v6e needs an explanation |
| R10 | Completeness | gate | P3 | eval dumps | - | steps 0 and 25 for both AIME sets; every question has 30 distinct sessions, counted by session / trajectory ID from the dump (P0.10), not by JSONL line count (a multi-turn session can produce several rows); mean / maj / best@30; tool-call stats. A missing point makes the result incomplete, not passed |
| R11 | Tool trajectory audit | gate | P1c, P3 | the rollout audit dump (P0.10): raw token IDs with the response / loss mask, assistant and tool segments, tool calls and results, reward inputs and outputs, `num_turns`, trajectory ID | - | tool-result tokens are excluded from the loss (response mask), tool results reach the next turn and are not lost in multi-turn concatenation, and correct / incorrect / malformed / timeout cases score as the pinned `compute_score` defines. A timeout reaches the model as `no stdout here`. Tool error and timeout rates are reported. The standard `rollout_data_dir` JSONL holds decoded text without special tokens, IDs or masks, so it cannot satisfy R11 on its own |
| R12 | Weight-sync version check | gate | P1c, P3 | after a completed Raiden sync | - | the rollout's policy version matches the trainer version it was synced from; selected tensor checksums match; later rollouts use the new version; observed staleness stays within `max_off_policy_threshold=4` |
| R13 | Resume equivalence | gate | P4a | uninterrupted vs stopped-and-resumed control from the same checkpoint | - | see the list below the notes |

R13 checks:

- model, optimizer, scheduler, global step and dataloader cursor are restored;
- in-flight and prefetched data are not lost, duplicated or changed in meaning. This holds in exactly one of two ways:
  - the TransferQueue state is saved, restored and verified (P0.7); or
  - the checkpoint is taken at a verified quiescent boundary (queue empty, no in-flight or prefetched trajectories) and
    the A/B control passes.

  Merely recording that the queue was not saved does not pass. Otherwise R13 is **BLOCKED / FAIL**, and runs over
  several windows cannot be reported as one continuous P4b curve. Missing TransferQueue checkpoint support does not
  affect the independent P1 and P3 runs;
- in-flight trajectories are handled as documented (pending and running prompts are re-issued);
- policy version and staleness after the resume;
- the first update after the resume matches the control within a declared tolerance on the same batch: batch content,
  loss, grad norm, LR and parameter update.

Notes on statistics:

- Under independent sampling, SD(mean@30) = sqrt(sum_i p_i (1 - p_i)) / (30 x sqrt(30)). That is at most 1.67 pp
  (all p_i = 0.5), about 1.4 pp at p ~ 0.22, and lower when question difficulty varies. Comparing two runs adds the
  noise of both sides (about sqrt(2) x), and SFT / RL run-to-run variance comes on top.
- Report per-question accuracy over the 30 samples, paired by question with the reference once we have its
  per-question outputs. Add bootstrap confidence intervals over questions and samples, and repeat evaluation seeds on
  the same checkpoint when the budget allows.
- If we get the v6e SFT checkpoint (section 9), we also evaluate it at step 0 on TPU 7x. That separates rollout
  numerics from SFT run-to-run variance.

## 4. Decisions

| # | Decision |
|---|---|
| D1 | **Target**: #108's Qwen3-30B-A3B recipe in this directory. The Qwen3-32B dense variant (askhat-g/verl-recipe#1, which needs [jialei777/verl-upstream#77](https://github.com/jialei777/verl-upstream/pull/77)) is P5 |
| D2 | **Code**: `syangx39/verl-upstream@tpu-main` at or after `a075ebc7` (#108 plus the HF export retention fix). Every run record pins the exact SHA. If newer TPU 7x fixes land upstream, we sync before P1. TPU 7x changes go in as separate small commits |
| D3 | **Fixed reference semantics**: keep #108's algorithm, hyperparameters and logical parallelism. Only `TPU_ACCELERATOR_TYPE`, `NNODES_*` and `N_CHIPS_*` change, plus the numerics adaptation of D4. No tuning of CP, eager mode, memory utilization or bucket size before P3 passes |
| D4 | **Numerics adaptation**: use `--xla_tpu_bf16_emission_mode=PROMOTE_F32` (in `LIBTPU_INIT_ARGS`) as a documented TPU 7x numerics adaptation. Our Qwen3-4B tpu7x reproduction needed it to match the H100 and v6e numerics. verl forwards `LIBTPU_INIT_ARGS` from the driver to the trainer and rollout workers; R1 verifies the effective value on both sides. Once we have the v6e runs' effective environment, we record the difference. A controlled ablation of the flag is P5 |
| D5 | **Storage**: inputs (model, datasets) are staged once to a GCS prefix and mounted read-only. The model is copied to `/dev/shm` on every TPU host (`MODEL_PATH`) and verified by hash (P0.3). All outputs (HF exports, DCP checkpoints, TensorBoard, dumps, logs) go to a separate GCS output prefix. Host `/tmp` is never used for checkpoints, because a tpu7x node disk has less free space than one 57 GiB export |
| D6 | **Run structure and resume**. See the details below the table |
| D7 | **Recipe behavior as-is**. See the details below the table |
| D8 | **Pinned inputs**: `Qwen/Qwen3-30B-A3B`; `JoeYing/ReTool-SFT`, preprocessed with verl-recipe's `retool_sft_preprocess.py` using the tool schema of [`sandbox_fusion_tool_config_tpu.yaml`](sandbox_fusion_tool_config_tpu.yaml); `BytedTsinghua-SIA/DAPO-Math-17k`; `Maxwell-Jia/AIME_2024`; `yentinglin/aime_2025`. HF revisions and file sha256 go into the run record. Rev 3: staged from the recipe's default data root (`/data/jialei`, the bucket mounted by `ray-tpu-v7x16-2slice.yaml`), i.e. the exact files the scripts reference. The ReTool-SFT parquet has 2,000 rows with `messages` and `tools`, every row has at least one `code_interpreter` call, and it uses the same tool schema |
| D9 | **Explicit SFT checkpoint for RL**: every RL run sets `MODEL_SRC_PATH` explicitly, and the load log and staged digest are checked. When the SFT index file is missing, the DAPO script silently falls back to the base model |

**D6. Run structure and resume.**

- Stage 1 SFT runs in one window, without resume.
- P3 is an **independent pilot** with the recipe's checkpoint semantics: `save_contents=['hf_model']` and
  `resume_mode=disable`. Its 25 steps do not count toward P4. With `TOTAL_TRAINING_STEPS=25`, step 25 is the last step
  and triggers a save, but that save is HF-only, so it cannot seed a resume.
- P4b starts a **new** RL run from the same SFT checkpoint with the resumable configuration (P0.7), and only after the
  P4a control experiment passes R13.
- Option: if P0.7 and R13 are validated before P3, P3 can use the resumable configuration, and P4b can continue from
  P3's step-25 DCP checkpoint. The run record states which option was used.
- A bounded run with `TOTAL_TRAINING_STEPS=25` is expected to match the first 25 steps of the 400-step schedule,
  because the RL learning rate is constant (`min_lr_factor=1.0`, no warmup). We verify that `actor/lr` is 1.4e-6 at
  every step and look for any other behavior that depends on the total step count. If we find one, we use a
  controlled stop instead.

**D7. Recipe behavior as-is.**

- `actor.optim.clip_grad=0.088` is the *script value*. The *effective* clipping is TorchTitan's `training.max_norm`,
  which we read and log at runtime (P0.9).
- At `a075ebc7` the engine does not pass `clip_grad` into TorchTitan's `TrainingConfig`, whose `max_norm` defaults to
  1.0 in upstream torchtitan `f880cc73`. This has to be confirmed against the image's torchtitan build or a runtime
  log.
- Parity keeps the recipe's actual behavior. The wiring fix is a separate ablation (P5).
- Stage 1 does not set `enable_thinking=False`; Stage 2 does. We keep this, and an ablation is P5.

## 5. P0 deliverables

P0 has two kinds of items, so that later validations do not block the start:

- **Preparation** items must be ready before P1a.
- **Validation** items are only measurable on hardware. P0.5 is validated in P1a. P0.7 is validated in P4a and must
  pass before P4b.

| # | Deliverable | Needed by | Why | Done when |
|---|---|---|---|---|
| P0.1 | **Launch path** for 2 x `2x2x4` on the shared TPU 7x GKE cluster (`bodaborg-tpu7x-nap`: Kueue + JobSet, node auto-provisioning) | before P1a | The in-repo TPU 7x manifests ([`ray-tpu-v7x16-2slice.yaml`](../gke/ray-tpu-v7x16-2slice.yaml)) are KubeRay `RayCluster`s for a cluster with the KubeRay operator. The shared cluster runs Kueue-managed JobSets, like [`ray-jobset-v5p-nap.yaml`](../gke/ray-jobset-v5p-nap.yaml) | A TPU 7x JobSet launcher renders and passes a server-side dry run; requirements below the table |
| P0.2 | **Image** | before P1a | Needs torch_tpu, torchtitan, vllm-torchtpu, tpu-sync-torch, TransferQueue, fastapi and uvicorn at versions that match the code SHA | #108's image (`verl-tpu:v20261001-tsync990787257`) or a rebuild from [`Dockerfile.tpu`](../gke/Dockerfile.tpu) is pullable by the cluster's nodes and pinned by digest; package versions recorded, including whether TransferQueue supports checkpoints (version 0.1.9 or later with `save_checkpoint` / `load_checkpoint`) |
| P0.3 | **Inputs** (D8) | before P1a | Pods should not depend on HF Hub access. `stage_model_local.py` decides "already staged" by file names and sizes only | Model and datasets staged under the input prefix in the directory layout the scripts expect (`dataset/<org>/<name>`, `assets/hf/Qwen3-30B-A3B`, `dataset/ReTool-SFT/data/train-00000-of-00001.parquet`), with hashes recorded. Each run then verifies shard hashes on every TPU host after staging (part of R5b) |
| P0.4 | **Output persistence** | before P1a | Pods disappear minutes after completion | `SAVE_PATH`, `CKPTS_DIR` and `TENSORBOARD_DIR` point to the output prefix. The driver log and an **allowlisted, redacted** dump of each worker's effective environment (`LIBTPU_INIT_ARGS`, `XLA_FLAGS`, `TPU_ACCELERATOR_TYPE`, ...) are mirrored to GCS during the run. No tokens or credentials are written |
| P0.5 | **Host ordering** | validated in P1a (R2) | verl orders a slice's hosts by IP string; torch_tpu ranks them by physical position. In our Qwen3-4B port, IP order caused a SparseCore halt in the first FSDP collective | Before P1a: the launcher computes each TPU host's topological rank before Ray starts and exposes it as a Ray node label, and verl orders hosts by that label when every host carries it. In P1a: the logged trainer rank-to-host order matches the topological order |
| P0.6 | **Numerics flag record** (R1) | before P1a | The flag must reach every role on every run | The P0.4 dump covers every TPU worker, and the extractor checks it per role |
| P0.7 | **Resumable RL config** (P4) | validated in P4a (R13); needed before P4b | #108 saves `hf_model` only, so a run cannot resume. In async modes the trainer persists in-flight prompts only through a TransferQueue that supports checkpoints | `actor_rollout_ref.actor.checkpoint.save_contents=['model','optimizer','extra','hf_model']`, `trainer.resume_mode=auto`, DCP to the output prefix. Either the TransferQueue state is saved, restored and verified, or checkpoints are taken at a verified quiescent boundary (R13). Old step folders are pruned by us, because TorchTitan's `keep_latest_k` does not prune across `global_step_N` folders in the PPO trainer layout |
| P0.8 | **Metrics extraction** | before P1a | Section 3 checks | Per-step training metrics from TensorBoard; a per-question eval table from the validation dump, grouped by session ID; prints R1-R13 |
| P0.9 | **Effective clipping** (D7) | before P1a | The script value and the effective value can differ | TorchTitan's resolved `training.max_norm` is logged at engine init |
| P0.10 | **Trajectory and eval dumps** | before P1a | R10, R11, per-question statistics | Three dumps on the output prefix. See the list below the table |

P0.1 requirements:

- Server-side dry run passes.
- One CPU Ray head plus two `2x2x4` slices (trainer `tpu-group-0`, rollout `tpu-group-1`).
- TPU pods select `gke-tpu-topology: 2x2x4` and the cluster's `2x2x4` placement policy.
- Ray workers start with explicit resources `{"TPU": 8, "tpu-group-<n>": 1}` (8 logical devices per host). The pod's
  `TPU_ACCELERATOR_TYPE` is a launcher parameter: the KubeRay manifest uses `tpu7x-32` to avoid a placement deadlock,
  while the JobSet manifests of our Qwen3-4B port used `tpu7x`. P1a confirms which one works with this code.
- Network: the head is reachable from every worker, and each worker strips the JobSet-injected multi-slice
  (`MEGASCALE_*`) variables so that each slice forms its own mesh.
- A memory-backed `/dev/shm` of at least 80 GiB is mounted on TPU pods; the GCS bucket is mounted with GCS FUSE.
- JobSet names are at most 30 characters.
- Queue admission does not prove that both slices were provisioned. Placement is checked from the live Ray nodes.

P0.10 dumps:

- `trainer.validation_data_dir`: every validation sample. It is regrouped by question and session ID, never counted by
  rows.
- `trainer.rollout_data_dir`: decoded training trajectories, sampled in long runs to bound storage.
- **Rollout audit dump** (new instrumentation, for R11), for a deterministic sample of whole sessions per step:
  - raw prompt and response token IDs with the response / loss mask;
  - assistant and tool segments, decoded with special tokens;
  - reward inputs and outputs, and `num_turns`;
  - trajectory and session IDs.

  Enabled with `+trainer.rollout_audit_dir=... +trainer.rollout_audit_sessions=N`.

## 6. Phases

| Phase | What | Config (on top of section 11) | Exit criteria | Wall time after admission (estimate) |
|---|---|---|---|---|
| **P0** | Section 5, preparation items | - | Preparation items ready: the launcher renders and passes a server-side dry run, the image is pullable and matches the code SHA, inputs are staged with hashes, output paths and the env dump are ready, and the instrumentation (P0.9, P0.10) is merged. P0.5 and P0.7 are validated later (P1a, P4a) | 1-2 days |
| **P1a** | Platform smoke with the in-repo TPU 7x MoE recipe. It separates TPU / Ray / MoE platform problems from problems in ReTool's multi-turn tool path | [`../grpo/run_qwen3_30b_a3b_torchtitan.sh`](../grpo/run_qwen3_30b_a3b_torchtitan.sh), 3 steps, 2 x `2x2x4` | R1 (trainer and rollout), R2: actual placement, host ordering (P0.5), trainer and rollout devices, Raiden weight sync | ~45 min |
| **P1b** | ReTool SFT smoke | `TOTAL_TRAINING_STEPS=3 SAVE_FREQ=3 TEST_FREQ=3`, one `2x2x4` slice, its own `SAVE_PATH` | R1 (trainer), R5a (3-step export); R3 recorded | ~45 min |
| **P1c** | ReTool RL smoke from the P1b export (tests the SFT -> RL hand-off) | `TOTAL_TRAINING_STEPS=2 TRAIN_BATCH_SIZE=8 PPO_MINI_BATCH_SIZE=8 N_RESP_PER_PROMPT=4 VAL_BEFORE_TRAIN=False`, dumps on | R1 (trainer and rollout), R2, R5b, R11, R12. Sandbox exercised **from the pod hosting the ExecutionWorker**: arithmetic, Python error and stderr, timeout and process cleanup, concurrent requests | ~60 min |
| **P2** | Stage 1 SFT, full | 372 steps, script defaults | R1 (trainer), R5a; R4 and R9 recorded | ~1.5-2.5 h (v6e: 1.9 h of steps) |
| **P3** | Stage 2 RL to step 25, **independent pilot** (D6) | script defaults from the P2 export (explicit `MODEL_SRC_PATH`), `TOTAL_TRAINING_STEPS=25 TEST_FREQ=25`, dumps on | R1 (trainer and rollout), R5b, R10, R11, R12 gates; R6-R9 recorded; `actor/lr` constant at 1.4e-6 | ~1.5-2.5 h |
| **P4a** | Resume control on a small diagnostic workload | P0.7 config from the same SFT checkpoint. Run A goes uninterrupted to step 2k. Run B saves and stops at step k, then resumes to step 2k. Async scheduling means trajectories can differ, so compare the restored state and the first post-resume update on the same batch | R13 | ~2 x 45 min |
| **P4b** | Resumable RL, new run from the SFT checkpoint. Starts only after P4a passes R13; with R13 BLOCKED, segments are reported as separate runs, not one curve | P0.7 config, segments chained by resume (e.g. to step 50, then 100), then optionally 400 steps | per-segment R1 and R12; curve continuous across boundaries; staleness within config | several windows |
| **P5** | Optional | 16K / 16-turn / n=16 schedule; Qwen3-32B dense; tuning (CP=1, `DEFER_AND_FUSE`, rollout memory utilization, multi-replica rollout); ablations (`PROMOTE_F32` off, `clip_grad` wired into TorchTitan, `enable_thinking=False` in SFT) | as agreed | as needed |

The wall times are estimates derived from v6e numbers. We replace them with P1 / P2 measurements.

## 7. Queue strategy

Quotas, limits and reservations change daily. Re-check them before every submission with the team-internal queue notes.

| Use | Queue | Why |
|---|---|---|
| P1a-P1c, P4a | shared priority queue (per-job wall-clock limit, about 2 h) | short runs. Driver timeout = limit - startup - teardown |
| P2, P3 | exclusive reservation window (multi-hour) | one uninterrupted window per stage. SFT needs only one `2x2x4` slice (16 chips); RL needs two (32 chips) |
| P4b | reservation windows, chained by resume | runs longer than one window |
| never | queues owned by other teams without the owner's OK | - |

## 8. Tracking and reporting

- One run-record row per submission:
  - identity: run name, code SHA, image, cluster / namespace, rendered manifest;
  - environment: effective `LIBTPU_INIT_ARGS` on both sides, effective `max_norm`;
  - timeline: submit / admit / ready / end times (UTC), first and last step, restored-from step;
  - results: median step time, status, output prefix.
- After each phase: run the P0.8 extractor and write the verdict per check into this file (new revision). Gates get
  pass / fail; observations get the measured value next to the initial range.
- Final report:
  - the three outcomes of section 1, each with linked evidence;
  - SFT loss curves vs v6e;
  - AIME tables (mean / maj / best@30 and per-question accuracy) vs v6e, with uncertainty;
  - trajectory audit samples, the step-time table, and every resume boundary.

## 9. Open items for the recipe author (@askhat-g)

These block exact inputs or tighter thresholds:

1. Confirmation that the reference runs used the SFT parquet at the recipe's default data root, which we staged (D8).
   The README cites a ReTool-SFT dataset that is not public on the HF Hub.
2. Full v6e logs (TensorBoard) and the resolved configs for SFT and both RL runs.
3. Per-question evaluation outputs (all samples) and the actual evaluation config for steps 0 and 25.
4. v6e reference values for `actor/grad_norm`, `actor/ppo_kl` and `actor/rollout_corr/*` (R7).
5. The effective environment of the v6e runs (`LIBTPU_INIT_ARGS`, `XLA_FLAGS`) and the TransferQueue version.
6. The v6e SFT checkpoint, for cross-evaluation on TPU 7x (section 3 notes).
7. Whether `VLLM_USE_V1=0` and `VERL_AGENT_LOOP_MAX_INFLIGHT_PROMPTS` had an effect. Nothing under `verl/` reads the
   latter at `f9e896fd`.

## 10. Risks

| Risk | Mitigation |
|---|---|
| TPU 7x bf16 emission numerics shift accuracy | D4 on every run (R1); R6 and R7 compared with v6e. Other numerics knobs only as a diagnosed, recorded fix |
| Multi-host trainer ranks in the wrong host order | P0.5, R2 |
| Node disk too small for the model and exports | D5: model in `/dev/shm`, outputs on GCS. The retention fix in `a075ebc7` keeps at most `max_actor_ckpt_to_keep` exports |
| Stale or partial model copy accepted by the staging cache (names and sizes only) | hash check on every TPU host (P0.3) |
| RL silently starts from the base model | explicit `MODEL_SRC_PATH`, load log and staged digest checked (D9) |
| Wall-clock limits, reclaims, preemption | reservation windows for P2-P4b; resumable RL (P0.7) |
| A resume loses, duplicates or changes in-flight data, or loses other state (optimizer, data cursor) | P3 is an independent pilot. P4a control experiment (R13) before P4b. R13 passes only with a verified TransferQueue save / restore or a verified quiescent boundary; otherwise it is BLOCKED and segments are reported separately |
| DCP checkpoint size (~370 GB with fp32 optimizer state at 30.5B) and retention | write to the output prefix; prune old step folders ourselves (P0.7) |
| Image not pullable, or version skew with the code SHA | P0.2; record versions; rebuild from `Dockerfile.tpu` |
| Our SFT data differs from the v6e run | D8 regeneration plus checksum; R3 and R4 compared; diff against the author's parquet once received |
| Multi-host vLLM (TP=32, EP=32) with multi-turn tool calls not yet exercised on TPU 7x | P1a (same rollout layout, single-turn) before P1c. A multi-replica rollout layout is allowed only as a recorded deviation |
| The sandbox is not an isolation boundary. Model-written Python runs as a subprocess in the training pod with the pod identity's file and network access, including the writable output mount. Loopback binding only limits the HTTP entry point | Run only in a trusted, isolated training environment. Inputs are mounted read-only and outputs go to a separate prefix. Stronger isolation (execution identity, filesystem, network) needs a separate design |
| Credentials leak into logs or env dumps | allowlisted, redacted env dump (P0.4) |
| Single-run v6e references and a 30-question benchmark | observations instead of gates; per-question statistics; final thresholds fixed before P4b (section 3) |

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

# P3: Stage 2 RL to step 25 on 2 x 2x2x4, independent pilot with the recipe's HF-only checkpoints
SFT_HF="${OUT}/checkpoint/multiturn-sft-qwen3-30b-a3b-tpu7x/global_step_372/huggingface"
NNODES_TRAINER=4 N_CHIPS_TRAINER=8 NNODES_ROLLOUT=4 N_CHIPS_ROLLOUT=8 \
MODEL_SRC_PATH="${SFT_HF}" \
MODEL_PATH=/dev/shm/models/multiturn-sft-qwen3-30b-a3b-tpu7x \
EXPERIMENT_NAME=qwen3-30b-a3b-retool-dapo-tpu7x-p3 \
CKPTS_DIR="${OUT}/checkpoint/qwen3-30b-a3b-retool-dapo-tpu7x-p3" \
TOTAL_TRAINING_STEPS=25 TEST_FREQ=25 \
  bash examples/tpu/retool/run_qwen3_30b_a3b_dapo_torchtitan.sh \
    trainer.validation_data_dir="${OUT}/dumps/p3/validation" \
    trainer.rollout_data_dir="${OUT}/dumps/p3/rollout" \
    +trainer.rollout_audit_dir="${OUT}/dumps/p3/rollout_audit" +trainer.rollout_audit_sessions=4

# P4a / P4b: resumable RL, a new run from the same SFT checkpoint. Later overrides win, so these replace the
# script's save_contents=['hf_model'] and resume_mode=disable. Keep CKPTS_DIR fixed across the segments of one run
# and raise TOTAL_TRAINING_STEPS per segment (constant LR, D6).
NNODES_TRAINER=4 N_CHIPS_TRAINER=8 NNODES_ROLLOUT=4 N_CHIPS_ROLLOUT=8 \
MODEL_SRC_PATH="${SFT_HF}" \
MODEL_PATH=/dev/shm/models/multiturn-sft-qwen3-30b-a3b-tpu7x \
EXPERIMENT_NAME=qwen3-30b-a3b-retool-dapo-tpu7x-p4 \
CKPTS_DIR="${OUT}/checkpoint/qwen3-30b-a3b-retool-dapo-tpu7x-p4" \
TOTAL_TRAINING_STEPS="${SEGMENT_END_STEP}" \
  bash examples/tpu/retool/run_qwen3_30b_a3b_dapo_torchtitan.sh \
    "actor_rollout_ref.actor.checkpoint.save_contents=['model','optimizer','extra','hf_model']" \
    trainer.resume_mode=auto \
    trainer.validation_data_dir="${OUT}/dumps/p4/validation"
```

## Changelog

**rev 3 (2026-10-29)**, from the review of rev 2:

- **P0 gates (section 5, P0 row).** P0 is split into preparation items, needed before P1a, and validation items. Host
  ordering (P0.5) is validated in P1a. Resume (P0.7) is validated in P4a and gates P4b. P1a no longer depends on P4a.
- **R11 evidence (P0.10).** New rollout audit instrumentation with raw token IDs, response / loss mask, segments,
  reward I/O, `num_turns` and trajectory IDs. The standard JSONL dumps are not sufficient on their own. R10 counts
  sessions by ID, not JSONL rows.
- **R13.** Passes only with a verified TransferQueue save / restore, or a verified quiescent checkpoint boundary plus
  the A/B control. Otherwise it is BLOCKED / FAIL, and multi-window runs are not reported as one continuous curve.
- **SFT-only gates.** R1 checks the roles that exist in the run. R5 is split into R5a (SFT export integrity: P1b, P2)
  and R5b (RL-side staging, hash check, multi-host vLLM load: P1c, P3).
- **Launcher requirements (P0.1).** Explicit Ray resources, the pod-level `TPU_ACCELERATOR_TYPE` as a parameter to
  confirm in P1a, `MEGASCALE_*` stripping, and a GCS FUSE mount.
- **Inputs (D8).** Staged from the recipe's default data root.

**rev 2 (2026-10-29)**, from the review of rev 1 and the closed alternative plan #1:

- **Run structure (D6).** P3 is now an independent pilot. Its HF-only checkpoint cannot seed P4, so P4b starts a new
  run from the SFT checkpoint after a P4a uninterrupted-vs-resumed control experiment (R13). Running P3 with the
  resumable configuration is an option once it is validated.
- **Checks (section 3).** Split into gates and observations. R3, R4, R6 and R8 become initial observation ranges; final
  thresholds are fixed before P4b from the author's per-question outputs.
- **Statistics.** Fixed the sampling-noise bound to at most 1.67 pp, and added per-question, bootstrap and repeat-seed
  reporting.
- **SFT phase exit.** Removed R10 from P2 (its evaluations only exist in P3). `val/loss` is labeled in-sample.
- **Numerics wording.** D3 / D4 now describe `PROMOTE_F32` as a documented TPU 7x numerics adaptation, not as part of
  a topology-only change.
- **`clip_grad` wording (D7, P0.9).** Separated the script value (0.088) from the effective TorchTitan `max_norm`,
  which is logged at runtime.
- **Sandbox wording.** It is not an isolation boundary. Environment dumps are allowlisted and redacted (P0.4).
- **Checks adopted from #1.**
  - tool trajectory audit (R11);
  - weight-sync policy-version check (R12);
  - stricter HF export verification (R5, now R5a);
  - sandbox tests from the ExecutionWorker pod (P1c);
  - shard-hash check after staging (P0.3);
  - explicit `MODEL_SRC_PATH` (D9);
  - per-sample dumps (P0.10);
  - the three-outcome framing (section 1).
