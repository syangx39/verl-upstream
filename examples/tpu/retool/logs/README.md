# ReTool run evidence (archive)

Logs, metrics, samples and verification scripts archived from the ReTool runs. The documents that cite them are in
[`../exp/`](../exp/); the current guide is [`../REPRO_GUIDE.md`](../REPRO_GUIDE.md). Large raw files (checkpoints, full rollouts, Ray logs) stay in each run's GCS directory; each run
folder's `raw_manifest.json` or `README.md` records where they came from.

| Path | Run | What is here |
| --- | --- | --- |
| [`tpu7x/p3c-retool-25s-02/`](tpu7x/p3c-retool-25s-02/) | 64-chip TPU v7x, 25 steps | Metrics, Step 0/25 validation samples, raw audit sessions, TensorBoard file, Raiden checks, offline verifiers (`verify_evidence.py`, `verify_code_snapshot.py`) |
| [`tpu7x/p3c-retool-25s-01/`](tpu7x/p3c-retool-25s-01/) | 64-chip TPU v7x, first attempt (evicted while saving the step-10 checkpoint) | Gate summaries for steps 0–9 |
| [`tpu7x/p3c-retool-smoke-01/`](tpu7x/p3c-retool-smoke-01/) | 64-chip TPU v7x smoke | Step 0/1 gate summaries, Raiden step-0 parity, reward-shaping fixture check |
| [`tpu7x/p3-retool-rl-11/`](tpu7x/p3-retool-rl-11/) | 32-chip TPU v7x, stage P3 | Derived evidence and `verify_p3_results.py` |
| [`gb200/g64-retool-25s-01/`](gb200/g64-retool-25s-01/) | 64-GPU GB200, 25 steps | `head.log`, metrics, Step 0/25 validation samples, raw audit sessions, TensorBoard file |
| [`gb200/g64-retool-smoke-04/`](gb200/g64-retool-smoke-04/) | 64-GPU GB200, 5-step smoke | Same layout as above |
| [`gb200/g2-retool-rl-25step-01/`](gb200/g2-retool-rl-25step-01/) | 8-GPU GB200, 25 steps | `head.log`, metrics, Step 0/25 validation samples, rescoring script |
| [`gb200/`](gb200/) (scripts) | — | `build_g64_run_evidence.py`, `gen_g64_report_tables.py`, `recompute_g64_metrics.py`, `repro_rollout_is_level.py` |
| [`askhat-wandb-audit/`](askhat-wandb-audit/) | Askhat's TPU v5p runs | Snapshot of selected W&B metrics used by the GB200 report |
