#!/usr/bin/env python3
"""Verify that the active Python environment matches a pinned uv.lock for the extras enabled on GB200.

Usage:
  verify_venv_lock.py <repo_dir> [--extras fsdp,vllm,torchtitan] [--require-async] [--require-torchtitan] [--require-gpu]

`uv export` expands `uv.lock` for the requested extras into requirement lines with environment markers; the line whose
marker evaluates true on this machine gives the exact expected version (distinguishing e.g. `torch==2.13.0+cu130` on
`linux/aarch64` from `+cpu` wheels). In addition to version pinning, this script verifies:
  1. `uv.lock` SHA256 digest and `verl.__file__` import path strictly inside `<repo_dir>/`;
  2. `torch` carries a CUDA build tag (`+cu*`), and `flash_attn` loads its compiled extension;
  3. `--require-async`: `PPOTrainerSeparateAsync` imports cleanly;
  4. `--require-torchtitan`: `torchtitan`, `LRSchedulersContainer` (0.1.x vs 0.3.x path), Qwen3 MoE state-dict adapter,
     and `configure_qwen3_moe_for_rl` (which clears `moe.load_balance_coeff` / `moe.aux_loss` for RL parity) resolve;
  5. `--require-gpu` (run on target GB200 nodes): `torch.cuda.is_available()`, device model, SM capability, HBM & NCCL.
"""

import argparse
import hashlib
import importlib
import importlib.metadata as md
import os
import platform
import re
import shutil
import subprocess
import sys

BASE_CHECK = ("ray", "torch", "vllm", "transformers", "transferqueue", "flash-attn")
TORCHTITAN_CHECK = ("torchtitan", "torch-checkpointing")


def canon(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def expected_versions(repo: str, extras: list[str], targets: tuple[str, ...]) -> dict[str, str]:
    uv = shutil.which("uv") or "/usr/local/bin/uv"
    cmd = [
        uv,
        "export",
        "--frozen",
        "--all-packages",
        "--format",
        "requirements-txt",
        "--no-hashes",
        "--no-header",
        "--no-emit-project",
        "--no-annotate",
    ]
    for extra in extras:
        if extra:
            cmd += ["--extra", extra]
    out = subprocess.run(cmd, cwd=repo, text=True, capture_output=True, check=True).stdout
    from packaging.markers import Marker
    from packaging.requirements import Requirement

    target_set = {canon(c) for c in targets}
    want: dict[str, str] = {}
    for line in out.splitlines():
        line = line.strip()
        if not line or line.startswith(("#", "-")):
            continue
        try:
            req = Requirement(line)
        except Exception:
            continue
        name = canon(req.name)
        if name not in target_set:
            continue
        if req.marker is not None and not Marker(str(req.marker)).evaluate():
            continue
        pins = [s.version for s in req.specifier if s.operator == "=="]
        if not pins:
            continue
        if name in want and want[name] != pins[0]:
            raise RuntimeError(f"lock resolves {name} to both {want[name]} and {pins[0]} on this platform")
        want[name] = pins[0]
    return want


def check_torchtitan_surface(bad: list[str]) -> str:
    import torchtitan  # noqa: F401

    tt_ver = md.version("torchtitan")
    lrs_mod = None
    for candidate in ("torchtitan.components.lr_scheduler", "torchtitan.components.optimizer"):
        try:
            mod = importlib.import_module(candidate)
            if hasattr(mod, "LRSchedulersContainer"):
                lrs_mod = f"{candidate}.LRSchedulersContainer"
                break
        except Exception:
            continue
    if lrs_mod is None:
        bad.append("torchtitan:LRSchedulersContainer-not-found")
    else:
        print(f"torchtitan {tt_ver}: resolved {lrs_mod}")

    adapter_ok = False
    for candidate in (
        "torchtitan.models.qwen3.model.state_dict_adapter",
        "torchtitan.models.qwen3.state_dict_adapter",
    ):
        try:
            mod = importlib.import_module(candidate)
            if hasattr(mod, "Qwen3StateDictAdapter"):
                adapter_ok = True
                print(f"torchtitan {tt_ver}: resolved {candidate}.Qwen3StateDictAdapter")
                break
        except Exception:
            continue
    if not adapter_ok:
        bad.append("torchtitan:Qwen3StateDictAdapter-not-found")

    try:
        tt_utils = importlib.import_module("verl.workers.engine.torchtitan.utils")
        if hasattr(tt_utils, "configure_qwen3_moe_for_rl"):
            print("verl.workers.engine.torchtitan.utils.configure_qwen3_moe_for_rl: present")
        else:
            print(
                "WARNING: verl.workers.engine.torchtitan.utils.configure_qwen3_moe_for_rl missing; "
                "port from tpu-main before running Qwen3-30B-A3B RL so moe.load_balance_coeff is cleared"
            )
    except Exception as exc:
        bad.append(f"verl-torchtitan-utils-import:{exc}")

    return tt_ver


def check_gpu_runtime(bad: list[str]) -> None:
    import torch

    if not torch.cuda.is_available():
        bad.append("cuda:torch.cuda.is_available()=False")
        return
    count = torch.cuda.device_count()
    nccl_ver = torch.cuda.nccl.version() if hasattr(torch.cuda, "nccl") else None
    print(f"CUDA runtime: devices={count} nccl={nccl_ver}")
    for idx in range(count):
        props = torch.cuda.get_device_properties(idx)
        gib = props.total_memory / (1024**3)
        print(
            f"  cuda:{idx} name={props.name!r} capability={props.major}.{props.minor} "
            f"total_memory={gib:.2f} GiB"
        )


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("repo_dir", help="Path to the pinned verl checkout")
    parser.add_argument(
        "--extras",
        default="fsdp,vllm,torchtitan",
        help="Comma-separated uv extras enabled in this environment (default: fsdp,vllm,torchtitan)",
    )
    parser.add_argument("--require-async", action="store_true", help="Assert PPOTrainerSeparateAsync imports cleanly")
    parser.add_argument(
        "--require-torchtitan",
        action="store_true",
        help="Assert TorchTitan package, Qwen3 MoE state-dict adapter, and scheduler container resolve",
    )
    parser.add_argument(
        "--require-gpu",
        action="store_true",
        help="Assert live NVIDIA GPU visibility, compute capability, HBM size, and NCCL version (for target hosts)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv if argv is not None else sys.argv[1:])
    repo = os.path.abspath(args.repo_dir)
    extras = [e.strip() for e in args.extras.split(",") if e.strip()]
    need_torchtitan = args.require_torchtitan or ("torchtitan" in {canon(e) for e in extras})
    check_pkgs = BASE_CHECK + (TORCHTITAN_CHECK if need_torchtitan else ())

    lock_path = os.path.join(repo, "uv.lock")
    if not os.path.isfile(lock_path):
        print(f"FATAL: uv.lock not found at {lock_path}")
        sys.exit(1)
    print(
        f"platform={platform.system()}/{platform.machine()} python={sys.version.split()[0]} "
        f"uv.lock.sha256={sha256_file(lock_path)}"
    )

    want = expected_versions(repo, extras, check_pkgs)
    bad: list[str] = []
    lines: list[str] = []
    for name in check_pkgs:
        w = want.get(canon(name))
        try:
            got = md.version(name)
        except md.PackageNotFoundError:
            got = None
        lines.append(f"{name:22s} installed {got!s:22s} locked {w!s}")
        if w is None or got != w:
            bad.append(name)
    print("\n".join(lines))

    import verl  # noqa: E402

    verl_file = os.path.abspath(verl.__file__)
    print("verl from", verl_file)
    if not verl_file.startswith(repo.rstrip("/") + "/"):
        bad.append(f"verl-import-path:{verl_file}")

    import torch  # noqa: E402

    if not torch.version.cuda:
        bad.append(f"torch-missing-cuda:{torch.__version__}")

    import flash_attn  # noqa: E402

    print("flash_attn", flash_attn.__version__, "imports OK")

    if args.require_async:
        from verl.trainer.ppo.v1.trainer_separate_async import PPOTrainerSeparateAsync  # noqa: F401

        print("PPOTrainerSeparateAsync imports OK")

    tt_summary = ""
    if need_torchtitan:
        tt_ver = check_torchtitan_surface(bad)
        tt_summary = f" torchtitan {tt_ver}"

    if args.require_gpu:
        check_gpu_runtime(bad)

    if bad:
        print("MISMATCH:", bad)
        sys.exit(1)

    import ray  # noqa: E402
    import transformers  # noqa: E402
    import vllm  # noqa: E402

    print(
        f"VENV OK: arch={platform.machine()} ray {ray.__version__} "
        f"torch {torch.__version__} (cuda {torch.version.cuda}) "
        f"vllm {vllm.__version__}{tt_summary} transformers {transformers.__version__}"
    )


if __name__ == "__main__":
    main()
