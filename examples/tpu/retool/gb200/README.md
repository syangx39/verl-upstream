# GB200 runtime files for the ReTool recipe

Files that run the ReTool recipe (`../run_qwen3_30b_a3b_dapo_torchtitan.sh`) on NVIDIA GB200 nodes on GKE
(4 GPUs per node). The reward, dataset and sandbox code is shared with TPU and lives one level up. How to build the
image and launch a run: [`../REPRO_GUIDE.md`](../REPRO_GUIDE.md#2-gb200-setup).

| File | Role |
| --- | --- |
| [`Dockerfile.gb200.torchtitan`](Dockerfile.gb200.torchtitan) | `linux/arm64` image: verl pinned by `VERL_PIN`, `uv.lock` environment (torch cu130, vLLM, TorchTitan) |
| [`run_ray_jobset_gb200.sh`](run_ray_jobset_gb200.sh) | GKE launcher: `ComputeDomain`, head + worker pods, Ray cluster, job submission, log sync and cleanup |
| [`run_qwen3_30b_a3b_dapo_torchtitan_gb200.sh`](run_qwen3_30b_a3b_dapo_torchtitan_gb200.sh) | RL recipe run on the head pod (GPU counterpart of `../run_qwen3_30b_a3b_dapo_torchtitan.sh`) |
| [`verify_venv_lock.py`](verify_venv_lock.py) | Node precheck: package lock, GB200 SM100, NVLink peer access, TorchTitan MoE config |
| [`verify_g0_gb200.py`](verify_g0_gb200.py) | Stages the SFT checkpoint to `/dev/shm`, checks its SHA-256 manifest and the MoE routing round trip |
| [`test_launcher_and_sandbox.py`](test_launcher_and_sandbox.py) | CPU tests for the launcher, the sandbox and the archived GB200 reports |

Archived GB200 plans and reports are in [`../exp/gb200/`](../exp/gb200/); run evidence is in
[`../logs/gb200/`](../logs/gb200/).
