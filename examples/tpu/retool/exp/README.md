# ReTool experiment notes (archive)

Plans, investigation notes and reports written while bringing up the ReTool recipe on TPU v7x and GB200, plus the
summary of Askhat's TPU v5p runs. They are kept for history: commands, paths and numbers in them may be superseded.
The raw run evidence they cite is in [`../logs/`](../logs/). The current guide is [`../REPRO_GUIDE.md`](../REPRO_GUIDE.md); it
does not depend on anything here.

| Path | Contents |
| --- | --- |
| [`tpu7x/REPRO_PLAN_TPU7X.md`](tpu7x/REPRO_PLAN_TPU7X.md) | TPU v7x reproduction plan (stages P0–P4) |
| [`tpu7x/REPRO_RESULTS_TPU7X.md`](tpu7x/REPRO_RESULTS_TPU7X.md) | TPU v7x execution log: SFT (`p2-retool-sft-01`) and the 32-chip RL runs |
| [`tpu7x/RETOOL_REWARD_CLIP_RERUN_PLAN_ZH.md`](tpu7x/RETOOL_REWARD_CLIP_RERUN_PLAN_ZH.md) | Plan and results of the TPU v7x rerun aligned with Askhat's config, incl. the 64-chip 25-step run `p3c-retool-25s-02` (Chinese) |
| [`tpu7x/TPU7X_RAIDEN_WEIGHT_SYNC_DIAGNOSIS.md`](tpu7x/TPU7X_RAIDEN_WEIGHT_SYNC_DIAGNOSIS.md) | Diagnosis of the Raiden step-0 weight-sync mismatch (Chinese) |
| [`gb200/GB200_REPRO_GUIDE_V1.md`](gb200/GB200_REPRO_GUIDE_V1.md) | First GB200 guide (8-GPU `G2` and 64-GPU `G64-D` sections); formerly `gb200/README.md` |
| [`gb200/REPRO_PLAN_GB200_ZH.md`](gb200/REPRO_PLAN_GB200_ZH.md) | GB200 reproduction plan (Chinese) |
| [`gb200/REPRO_RESULTS_GB200_ZH.md`](gb200/REPRO_RESULTS_GB200_ZH.md) | GB200 execution log and audit, incl. the 64-GPU 25-step run `g64-retool-25s-01` (Chinese) |
| [`askhat-v5p/`](askhat-v5p/) | Summary of Askhat's TPU v5p W&B runs with figures (Chinese) |
| [`ISO_CHIP_PERF_BENCHMARK_PLAN.md`](ISO_CHIP_PERF_BENCHMARK_PLAN.md) | Iso-accelerator-count TPU v7x vs. GB200 benchmark plan (Chinese) |
