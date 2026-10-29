# Reproduce Qwen3-30B-A3B ReTool on TPU v7x

**Status:** proposed plan; no TPU v7x run has been performed for this plan.  
**Owner:** Xiaotong / TPU reproduction team.  
**Reference:** [Askhat's PR #108](https://github.com/jialei777/verl-upstream/pull/108), [recipe](https://github.com/jialei777/verl-upstream/tree/266e793a20c7baa6f849476ba72c8f8052da1477/examples/tpu/retool), and [TPU access notes](https://github.com/syangx39/gutianyu-rl/blob/main/queue-access/README.md).

## 1. Goal and claims

Reproduce the **two-stage ReTool workflow** in PR #108 on TPU v7x: cold-start multi-turn SFT of Qwen3-30B-A3B, HF checkpoint export, then separate-async multi-turn RL with a Python interpreter. Start with the default 8K-response / 8-turn RL configuration. Verify correctness and learning before attempting the full 400-step schedule. The 16K/16-turn variant is a separate follow-up, not a prerequisite.

Report three distinct outcomes: (1) **pipeline functional**: all components execute correctly; (2) **reference-aligned experiment**: model, data, training and evaluation semantics are documented and match the reference within stated differences; (3) **learning reproduction**: observed curves are compared with the reference under a predeclared protocol. Passing a smoke test does not establish convergence parity. v7x has a different hardware topology; equal wall-clock time or identical sampled trajectories are not expected. This is a TPU-v7 reproduction, not a GPU comparison.

PR #108 describes v7x overrides but reports its cluster training results on v6e, not v7x. Its reported 400-step schedule includes **step-25** evaluation numbers; it does not by itself establish that the full 400 steps finished. Treat those numbers as author-reported reference observations, not established v7 targets.

## 2. Freeze reference and record deviations

Use [`a075ebc7b23aaeb4eef687fb670f551593d2e964`](https://github.com/syangx39/verl-upstream/commit/a075ebc7b23aaeb4eef687fb670f551593d2e964) as the proposed execution baseline. It contains PR #108 (PR head `266e793a20c7baa6f849476ba72c8f8052da1477`, merge `f9e896fd71b3077a5f93d14e2878520fa7899eda`) plus an HF-export rotation fix. Record the actual checked-out commit, all patches and image digest; do not silently substitute the code bundled in an image. If this plan's branch or a later change is used, record the exact execution SHA and diff from `a075ebc`.

| Item | PR #108 reference / initial reproduction target |
|---|---|
| Model | `Qwen/Qwen3-30B-A3B`; freeze exact HF revision, config, tokenizer and shard hashes. |
| SFT | ReTool-SFT; 16,384 max length, batch 32, 6 epochs / 372 steps for reference data; TorchTitan SimpleFSDP=8, CP=4, TP=1, EP=1; LR 1e-5. Default SFT train and validation point to the **same parquet**. Report its validation loss as in-sample; a held-out evaluation is supplemental. |
| RL data | DAPO-Math-17k training; AIME 2024 and AIME 2025 evaluation; pin dataset revisions, row counts/order, transformations and hashes. |
| RL batch | 64 prompts × 8 responses = 512 trajectories per nominal step; mini-batch 64; 2,048 prompt tokens, 8,192 response tokens, max model length 10,240. User and assistant turn limits are each 8. Confirm actual effective batches and token counts in logs. |
| Algorithm | GRPO advantage estimator; DAPO-style clip low 0.2 / high 0.28, LR 1.4e-6, gradient clip 0.088, no KL loss/reward; `separate_async`, bypass mode, token importance correction threshold 2.0, parameter sync every step, max off-policy threshold 4. Preserve resolved values, loss and reward implementation rather than inferring other DAPO features from the name. |
| Rollout / eval | vLLM async, TP=32 / EP=32 logical devices; multi-turn Hermes format, OpenAI tool schema; eval before training and every 25 steps, 30 samples, temperature 1.0, top-p 0.6. Capture all other resolved decoding parameters, max turns, prompts and verifier versions. |
| Tool / reward | Reuse exact tool adapter and math verifier, including boxed-answer requirement and incorrect-answer `num_turns` shaping capped at -0.6. Log actual turns, tool outputs and scorer inputs; do not replace the scorer mid-comparison. |

Request from Askhat the **actual** resolved Hydra configs, software/image digests, base and SFT checkpoint URIs/checksums, exact dataset snapshots and preprocessing, seeds, AIME prompts and per-sample outcomes, step-0/25 evaluation logs, and timing definition. Until obtained, mark matching items **unverified**. A script default is not proof of what ran. Freeze a manifest before each run and record any material difference. If using Askhat's SFT checkpoint to isolate RL, label it as a separate RL-only track; it does not establish reproduction of our own SFT stage.

## 3. v7x topology and queue admission

The [reference v7x YAML](https://github.com/jialei777/verl-upstream/blob/266e793a20c7baa6f849476ba72c8f8052da1477/examples/tpu/gke/ray-tpu-v7x16-2slice.yaml) describes two `2x2x4` slices:

| Role | Physical resources | Expected Torch/Ray resources |
|---|---|---|
| SFT or RL trainer | 1 slice = 4 hosts × 4 chips = 16 physical chips | 8 Torch TPU devices per host, 32 per slice; FSDP=8 × CP=4 |
| RL rollout | 1 independent slice = 4 hosts × 4 chips = 16 physical chips | 8 devices per host, 32 per slice; vLLM TP=32, EP=32 |
| Total RL | 2 slices = 32 physical chips | 64 logical devices across 8 TPU hosts |

The script variables `N_CHIPS_TRAINER=8` and `N_CHIPS_ROLLOUT=8` refer to per-host **logical devices in this setup**, despite their names; do not request 8 physical chips per host. The reference YAML sets worker Ray resource `TPU: 8` and pod accelerator type `tpu7x-32`, while the recipe process uses `TPU_ACCELERATOR_TYPE=tpu7x`. Verify the exact device exposure and Ray resources on the actual image and queue. Trainer and rollout must land on different complete slices; confirm mesh products, device ownership and cross-slice connectivity from actual actors, not just the YAML.

The reference KubeRay YAML is for a different GKE cluster, service account, image and storage setup, and explicitly omits a Kueue queue label. **Do not apply it unchanged to nap or assume it is admitted.** First design/validate a queue-compliant Kueue JobSet or other supported launcher that starts a CPU Ray head and both multi-host Ray worker groups, reserves the two slices together, and provides service discovery, slice labels, network access, image pull and data access. Validate that Ray, Kueue and the cluster policies support the selected submission model. The existing JAX JobSet launcher is a useful queue example, not a proven TorchTitan/vLLM launcher. A ready-made runnable nap manifest is a prerequisite deliverable, not supplied or validated by this plan.

### Access and scheduling preflight

The [queue-access README](https://github.com/syangx39/gutianyu-rl/blob/main/queue-access/README.md) is a dated snapshot (2026-10-29 UTC), not a live capacity guarantee. It describes `bodaborg-tpu7x-nap / priority-dev` as the normal short-run option with a 120-minute limit, and `go/tpu-book-nap` reservations for longer windows (subject to eligibility and booking policy). It reports `gsc` in `HoldAndDrain` with no TPU nodes at that snapshot. Recheck before each submission; do not assume a booking guarantees topology provisioning or uninterrupted execution. The snapshot says `tpu-priority-dev-team` membership expires **2026-11-04 08:00 UTC**; verify/renew if needed.

Read-only checks from an authenticated environment, using the exact context from the access README:

```bash
NAP=gke_cloud-tpu-shared-capacity_us-central1_bodaborg-tpu7x-nap
kubectl --context="$NAP" auth can-i create jobsets.jobset.x-k8s.io -n priority-dev
kubectl --context="$NAP" auth whoami -o json
kubectl --context="$NAP" get clusterqueue priority-dev -o yaml
kubectl --context="$NAP" get clusterpolicy validate-priority-classes -o yaml
kubectl --context="$NAP" get configmap schedule-calendar -n kueue-system -o yaml
kubectl --context="$NAP" get deploy time-limit-controller -n kueue-system -o yaml
```

Inspect namespace LocalQueue → ClusterQueue mapping, quota, flavor, priority/preemption policy, current reservation, node provisioning and allowed wall time. Queue admission is not proof that the two requested slices are provisioned. Capture workload conditions/events and node placement; distinguish queue wait, provisioning, initialization and execution. Use `medium` / `--allow-preempt` only where supported by the actual launcher and current policy. Confirm whether CPU head and network services are permitted and covered by the quota/placement. Never bypass cluster admission.

## 4. Runtime, data and durable artifacts

1. Pin container **digest**, Python, PyTorch, torch-tpu/libtpu, TorchTitan, Ray, vLLM TPU, tpu-sync/Raiden, datasets, transformers and safetensors versions; record `git rev-parse HEAD`, `git status` and resolved Hydra configuration from each pod. Verify repo/package import path and that Ray workers see the same source; `PYTHONPATH` and runtime environment must reach remote workers.
2. Prepare accessible, versioned copies of the base model, SFT parquet and DAPO/AIME datasets. Confirm licenses/access, schemas, split, row counts, stable order, preprocessing, tokenizer/chat template and content hashes. `RAY_DATA_HOME` / `DATA_ROOT` must exist on every pod; `/data/jialei` and the source YAML's GCS bucket/service account are not assumed available in nap. The custom RL loader identifies AIME datasets from the final two path components: retain `Maxwell-Jia/AIME_2024` and `yentinglin/aime_2025`, or deliberately change and test the loader. Check read/write permissions using the job's actual identity.
3. Stage model files onto each TPU host with `examples/tpu/grpo/stage_model_local.py` once Ray is ready. Its third argument `8` means parallel file copies, **not host count**; it copies full weights to every TPU Ray node and lightweight config to CPU nodes. Its cache check compares names and sizes, not content. Independently verify shard hashes or a trusted immutable object manifest on every TPU host; ensure local disk, memory and export headroom.
4. Choose a durable per-run artifact root accessible from head and workers. Verify write/read and the checkpoint writer's single-writer assumptions before use. Persist logs, resolved configs, code/image identifiers, manifests, job events and checkpoints **during** execution; pod-local `/tmp` and post-job pod logs are not durable. RL defaults to `/tmp/checkpoint/...` and a retention setting of two; the proposed `a075ebc` baseline adds HF export rotation, while original PR #108 did not enforce that setting for HF-only saves. Validate rotation on the target, including behavior across restarts; do not assume local exports survive job termination or that a checkpoint from every evaluation step exists. Avoid recording credentials in artifacts.
5. Keep the sandbox bound to loopback in a trusted isolated job environment. The service executes generated Python in a subprocess; it is not a security boundary for untrusted or multi-tenant workloads. The request default is 1 GiB but the implementation sets `RLIMIT_AS` to at least **4 GiB** when successful. Record actual execution timeout, limits, worker placement, concurrency, failures and pod CPU/memory limits; do not expose the endpoint externally.

## 5. Recovery is a prerequisite for a multi-window run

Both reference scripts use `trainer.resume_mode=disable` and `save_contents=['hf_model']`. An HF export is model weights for loading/inference; it does not by itself establish recovery of optimizer, scheduler, step, RNG, data iterator or separate-async sampler/rollout queue and policy version. The trainer may save other files such as `data.pt`, but their presence does not prove a complete recoverable state. Restarting from HF weights is a **new run**, not a continuation of the original learning curve.

The v6e report gives SFT ~372 × 18.6 s ≈ 1.92 h and RL ~400 × 108.1 s ≈ 12.0 h before accounting for all staging, validation and export overhead. These are context, **not v7 runtime predictions**. A 120-minute queue or a 3/6-hour booking may be insufficient; determine v7 wall time from measured pilot runs and current policy.

Before committing to a formal full run, choose one:

- **Continuous:** obtain a verified uninterrupted window that accommodates initialization, 400 updates, evaluations, saves and margin; record admission, preemption exposure and actual duration.
- **Resumable:** implement and test full-state checkpoint/restart for both relevant trainers and async execution on persistent storage. At a declared safe boundary, compare a short uninterrupted control and a stopped/resumed run: restored model/optimizer/scheduler/global step, data cursor and RNG; actor/rollout policy versions, in-flight/queued rollout treatment and staleness; then next input IDs, LR, loss/gradients, update and eval under an explicitly documented numerical tolerance. Determine whether queue draining/quiescing changes semantics and document it. Verify a new allocation can read all ranks' checkpoint files. Do not claim strict continuity if any required state is absent or untested.

If neither option is available, run the bounded smoke/step-25 experiment and explicitly report the full run as **blocked**. Do not concatenate independent HF-only restarts as one 400-step run.

## 6. Execution phases and gates

### P0 — CPU / queue / image preflight

- Review the pinned recipe and changes since PR #108; run relevant existing CPU tests and syntax checks in the target image. Save exact commands and results; CPU checks do not validate TPU execution.
- Validate data paths, hashes, enough host-local capacity, shared storage behavior and per-pod image/source consistency. Render and dry-run the proposed Kueue launcher using the target namespace's current rules. Submit only after confirming authorization, quota and two-slice availability.
- Start a minimal Ray deployment: inspect live nodes and resources, place one probe on each TPU host, record physical chip and logical device counts, TPU topology and connectivity. Verify one CPU head and two distinct complete slices; show trainer/rollout actor placements, not merely aggregate counts.
- **Gate:** all prerequisites have evidence; no repeated device assignment, actor placement deadlock, missing package, unavailable data, or accidental external sandbox exposure.

### P1 — SFT smoke and export

Run one trainer slice with the v7 overrides below, using the **same model and 16K SFT semantics** as the reference, for 3 updates. Use a unique run name and durable output root; a smoke step limit is diagnostic only and must not change the formal run's 6-epoch schedule. Record actual unpadded lengths and verify at least one near-16K example in a separate targeted stress probe if the smoke batches do not contain one; a 16K maximum alone does not exercise that length. Check data masks, finite loss/gradients, optimizer updates, stable memory and validation execution. Trigger an HF export within the smoke window using a dedicated save frequency; verify index, shard list, hashes, tokenizer/config, weight shapes/dtypes and tied weights, then reload in a fresh process. Compare selected weight tensors and fixed-input logits against the pre-export model within a declared tolerance; confirm the checkpoint is accessible to every rollout host. Save before node teardown.

After P1 passes, run full SFT for the reference 6 epochs, subject to the recovery/window gate. Capture actual sample count, steps, loss, LR, timing, memory, checkpoints and evaluation. Confirm the exported final checkpoint corresponds to the actual final step; `global_step_372` is valid only if the dataset/batching really produce 372 steps. If using a separately supplied SFT checkpoint for RL, record its origin and digest and keep the two tracks distinct.

### P2 — tool, reward and multi-turn integration

Test sandbox health **from the pod hosting the ExecutionWorker**, not only from the Ray head: successful arithmetic, Python error/stderr, timeout and process cleanup, and concurrent requests. Verify that the model's OpenAI-schema tool call reaches the local endpoint and its response appears in the next model context. The server reports `TimeLimitExceeded`, while the wrapper returns `no stdout here` for non-`Finished` results: verify the actual model-visible output. Inspect representative trajectory token IDs/messages, tool-call and tool-result boundaries, response masks, truncated responses, `num_turns`, and boxed-answer scorer inputs and output. Check correct, incorrect, malformed and timeout cases against the pinned reward implementation. Persist sanitized traces and counters (tool calls, execution errors, timeouts, latency, turns and response lengths).

**Gate:** at least one real multi-turn trajectory and expected reward/masks verified, no unaccounted tool failures; report failure rates rather than silently treating them as normal rollouts.

### P3 — two-slice RL smoke and synchronization

Explicitly set `MODEL_SRC_PATH` to the verified SFT HF export and verify the staged digest and load logs; do not let the script's missing-index fallback silently select the base model. Provision both slices and run 2–3 actor updates with the reference async/sync semantics and a small diagnostic workload if needed. Then repeat at the **full 64 × 8, 8K/8-turn** workload for enough steps to cover warmup and at least one update/sync. Record all deviations from the formal configuration. Verify real actor parameter updates, Raiden sync success, selected tensor hashes or numerical checks against the corresponding source policy version at a completed sync, and that subsequent rollouts use the new version. Asynchronous trainer and rollout versions need not match at every instant; verify observed staleness against the configured policy. Record `supports_raiden_unbind_weights()` and the resulting `RELEASE_SYNC_BUFFERS` choice. Monitor staleness, queue depth, OOM, recompiles, deadlocks, per-host memory/disk, tool failure and rollout lengths.

**Gate:** complete at least one full-workload update and sync, finite metrics, correct placement, load and tool execution; capture logs/config/weights to durable storage. A diagnostic reduced batch or turn count is not a comparable benchmark.

### P4 — reference-aligned 25-step pilot

Start from the pinned SFT checkpoint with the default workload. Run step-0 evaluation, 25 updates and step-25 evaluation using the frozen prompts, decoding, verifier and data; record seeds and `mean@30` / `maj@30` definitions. A pilot that ends after step-25 evaluation while preserving the 400-step scheduler horizon needs a controlled-stop mechanism; implement and validate that mechanism before launching the bounded pilot. Do not rely on job timeout to finish evaluation and persist artifacts. Preserve per-question/per-sample outputs, tool calls and scores so metrics can be recomputed; report format errors, truncation and timeouts separately. Compare step 0 and step 25 with Askhat's actual logs, with dataset identity and sample counts verified. AIME is small and sampled decoding is noisy: do not set an arbitrary pass/fail threshold from one score; pre-agree tolerances or repeat seeds, and include uncertainty / per-question differences. If reference logs or protocol are missing, report the comparison as provisional.

Default `TEST_FREQ=25` and `SAVE_FREQ=20` do **not** automatically retain a step-25 checkpoint; `MAX_CKPT_TO_KEEP=2` may remove earlier exports. If checkpoint replay is required, arrange and verify an explicit step-25 save and durable copy, documenting any save/timing change. Otherwise preserve the actual evaluation outputs and report that no step-25 checkpoint was saved.

### P5 — full run and optional long-context variant

After P4 and the window/recovery gate, run the reference 400-step schedule; preserve periodic evaluation, checkpoint provenance and any interruptions. Report final completed step, trajectory of accuracy/reward, response lengths, clipping, entropy, gradient norm, KL/importance-correction diagnostics, tool rates, staleness and errors. Compare curves only when inputs, verifier, decoding and algorithm semantics are matched, and distinguish an observed trend from confirmed convergence. Run multiple seeds when available; disclose single-seed limitations. The 16K/16-turn/n=16 variant is a separate experiment with its own batch, LR and gradient clipping; never mix it into the default-run curve.

## 7. Launch sketch (after P0 prerequisites)

These are recipe-level commands **inside the validated Ray job environment**, not nap submission commands. They do not create or reserve slices. Set all paths to verified locations and confirm the actual image exposes 8 logical TPU devices per host first. The SFT smoke must use its own output directory and must not overwrite formal checkpoints.

```bash
export TPU_ACCELERATOR_TYPE=tpu7x
export NNODES_TRAINER=4 N_CHIPS_TRAINER=8
export NNODES_ROLLOUT=4 N_CHIPS_ROLLOUT=8
export RAY_DATA_HOME=/path/to/verified/shared/data
export DATA_ROOT="$RAY_DATA_HOME"
# For SFT, also set TRAIN_DATA, EVAL_DATA, MODEL_SRC_PATH and SAVE_PATH
# explicitly to the verified paths and unique run output.
TOTAL_TRAINING_STEPS=3 SAVE_FREQ=3 \
  bash examples/tpu/retool/run_qwen3_30b_a3b_sft_torchtitan.sh
```

Only after export and P2/P3 checks, set `MODEL_SRC_PATH` to the verified SFT `huggingface` directory and `CKPTS_DIR` to the validated durable output path. The following is a **pilot**; retain the original 400-step learning-rate/scheduler horizon if reducing the executed steps would change it. Verify the resolved schedule rather than assuming `TOTAL_TRAINING_STEPS=25` preserves it. If the implementation derives its scheduler from this value, use a controlled stop at step 25 or document the difference.

```bash
export MODEL_SRC_PATH=/path/to/verified/sft/huggingface
export CKPTS_DIR=/path/to/validated/durable/rl-checkpoints
# Launch the unchanged reference 400-step schedule after all gates are met:
bash examples/tpu/retool/run_qwen3_30b_a3b_dapo_torchtitan.sh
```

Verify that the actual data layout matches the script's expected subpaths and that output filesystem semantics are compatible with rank-gated checkpoint writes. Setting a shared path does not itself make a checkpoint complete or resumable.

## 8. Measurements, artifacts and reporting

- **Performance:** capture per-step start/end times and warmup exclusion, compile/recompile, staging, eval and checkpoint timings, actor update, synchronization, rollout generation and sandbox latency. Report median/mean and p90/p95 for a declared steady-state window, both with and without eval/save, plus successful rollouts/min, generated tokens/s, token-length distribution, chip-hours and wall time. Async component timers can overlap: do not sum them as critical-path time. Compare v7 to v6e only with same workload and timing definitions; physical chips and logical devices are different units.
- **Correctness:** record loss/reward, response and prompt token counts, clipping/truncation rates, turns, tool success/errors/timeouts, sampled outputs and verifier results, finite grad/optimizer state, actor-to-rollout version and sync evidence. Audit raw token IDs and masks on representative trajectories, especially tool-result boundaries.
- **Durable run manifest:** run ID; stage and intent (smoke/formal); git SHA/diff and image digest; cluster/namespace/queue/job ID and admission; physical/logical topology and placement; complete resolved config and CLI; data/model/tokenizer revisions and hashes; seeds; software versions; save/resume policy; timing definitions; artifact URIs; exact start/end and final step; failures and deviations. Include baseline source and links to logs.
- **Final report:** a phase-by-phase PASS / FAIL / BLOCKED table with linked evidence; step-0/25 and later learning curves and per-question results; known gaps against the v6e baseline; performance windows; exact completed steps and recovery evidence. No parity or convergence claim without a defined comparison and supporting artifacts.

## 9. Open decisions / blocking inputs

1. Exact model, SFT export, data snapshots, resolved reference configs, seeds and per-sample evaluation / timing logs from Askhat.
2. A nap-compliant two-slice Ray/JobSet launcher and accessible image, service identity and durable storage; confirm v7 TorchTitan + vLLM + Raiden compatibility on this queue.
3. A sufficiently long continuous window or proven full-state recovery, including separate-async state; establish budget and owner before P5.
4. Predeclare evaluation tolerance, seed count and statistical treatment once the reference protocol and variability are known. Missing baseline evidence means **inconclusive**, not a failed or passed parity claim.
