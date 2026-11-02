#!/usr/bin/env python3
"""Verify Python/uv lock fidelity and target-host runtime preconditions for GB200 ReTool runs.

This script intentionally separates two verification levels and labels them explicitly in output:
  1. Build-time venv & lock check (default):
     - Evaluates `uv export --frozen --all-packages` markers on the current platform (`linux/aarch64` vs `x86_64`,
       `+cu130` vs `+cpu`) and verifies installed versions match `uv.lock` (both the core stack table and, when
       `--check-all-locked` is enabled, every non-editable locked package exported for the selected extras);
     - Verifies git `HEAD` SHA against `--expect-git-sha` (or `/workspace/VERL_PIN`) and `uv.lock` SHA256 against
       `--expect-lock-sha256` (or `/workspace/UV_LOCK_SHA256`) when provided;
     - Verifies `verl.__file__` resolves strictly inside `<repo_dir>/`, `torchtitan.__file__` resolves inside active
       site-packages, `torch` carries a CUDA build tag (`+cu*`), and `flash_attn` loads its compiled extension;
     - Under `--require-torchtitan`, verifies TorchTitan `0.3.x` / `0.1.x` `LRSchedulersContainer` and
       `Qwen3StateDictAdapter` imports, plus slotted `MoE.Config` (`load_balance_coeff` field without legacy `aux_loss`
       assumption).
  2. Target-host GPU / GB200 precheck (`--require-gpu` / `--require-gb200`):
     - Actively enforces `torch.cuda.is_available()`, a working `torch.cuda.nccl.version()`, minimum GPU count
       (`--min-gpus`), minimum compute capability (`--min-sm`), minimum per-GPU memory in GiB (`--min-hbm-gib`), and
       device-name regex (`--expect-gpu-regex`). Under `--require-gb200`, defaults enforce Blackwell SM >= 10.0,
       HBM >= 165.0 GiB (186 GB nominal ~= 173.2 GiB), NCCL present, and device name matching `GB200|B200` — so a
       mismatched or simulated GPU (e.g. A100 40 GiB or missing NCCL) fails with exit code 1.
     - Note: Passing host precheck still does NOT replace G0 live verification of 1-step MoE routing parity and
       actor -> NCCL -> vLLM weight synchronization.
"""

from __future__ import annotations

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
GIB = 1024**3
GB = 10**9


def canon(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git_head_sha(repo: str) -> str | None:
    try:
        out = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True, stderr=subprocess.DEVNULL)
        return out.strip()
    except Exception:
        return None


def expected_versions(
    repo: str,
    extras: list[str],
    targets: tuple[str, ...] | None = None,
) -> dict[str, str]:
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

    target_set = {canon(c) for c in targets} if targets is not None else None
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
        if target_set is not None and name not in target_set:
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


def verify_identity_pins(repo: str, args: argparse.Namespace, bad: list[str]) -> str:
    lock_path = os.path.join(repo, "uv.lock")
    lock_sha = sha256_file(lock_path)
    expect_lock = args.expect_lock_sha256
    if expect_lock is None and os.path.isfile("/workspace/UV_LOCK_SHA256"):
        with open("/workspace/UV_LOCK_SHA256") as f:
            expect_lock = f.read().strip() or None
    if expect_lock is not None:
        if lock_sha != expect_lock:
            bad.append(f"uv.lock-sha256:got={lock_sha}:want={expect_lock}")
        else:
            print(f"uv.lock SHA256 verified: {lock_sha}")
    else:
        print(f"uv.lock SHA256 recorded (no expected pin supplied): {lock_sha}")

    head_sha = git_head_sha(repo)
    expect_git = args.expect_git_sha
    if expect_git is None and os.path.isfile("/workspace/VERL_PIN"):
        with open("/workspace/VERL_PIN") as f:
            expect_git = f.read().strip() or None
    if expect_git is not None:
        if head_sha != expect_git:
            bad.append(f"git-head-sha:got={head_sha}:want={expect_git}")
        else:
            print(f"git HEAD SHA verified: {head_sha}")
    elif head_sha is not None:
        print(f"git HEAD SHA recorded (no expected pin supplied): {head_sha}")

    return lock_sha


def check_torchtitan_surface(bad: list[str], require_rl_moe_helper: bool = False) -> str:
    import torchtitan

    tt_ver = md.version("torchtitan")
    tt_file = os.path.abspath(getattr(torchtitan, "__file__", "") or "")
    print(f"torchtitan {tt_ver} from {tt_file}")
    expected_prefix = os.path.abspath(sys.prefix).rstrip("/") + "/"
    if not tt_file.startswith(expected_prefix):
        bad.append(f"torchtitan-import-outside-venv:{tt_file}")

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

    # Inspect TorchTitan MoE config schema (in 0.3.x MoE.Config is a slotted dataclass with
    # `load_balance_coeff` and no `aux_loss` field; setting `moe.aux_loss = None` blindly raises AttributeError).
    for moe_mod_name in ("torchtitan.models.common.moe", "torchtitan.models.moe"):
        try:
            moe_mod = importlib.import_module(moe_mod_name)
            moe_cls = getattr(moe_mod, "MoE", None)
            cfg_cls = getattr(moe_cls, "Config", None) if moe_cls is not None else None
            if cfg_cls is not None:
                cfg_obj = cfg_cls()
                has_lbc = hasattr(cfg_obj, "load_balance_coeff")
                has_aux = hasattr(cfg_obj, "aux_loss")
                slots = hasattr(cfg_cls, "__slots__")
                print(
                    f"torchtitan MoE.Config ({moe_mod_name}): slotted={slots} "
                    f"has_load_balance_coeff={has_lbc} has_aux_loss={has_aux} "
                    f"default_load_balance_coeff={getattr(cfg_obj, 'load_balance_coeff', 'N/A')!r}"
                )
                if not has_lbc:
                    bad.append("torchtitan:MoE.Config-missing-load_balance_coeff")
                break
        except Exception:
            continue

    try:
        tt_utils = importlib.import_module("verl.workers.engine.torchtitan.utils")
        helper = getattr(tt_utils, "configure_qwen3_moe_for_rl", None)
        if callable(helper):
            print("verl.workers.engine.torchtitan.utils.configure_qwen3_moe_for_rl: present")
        else:
            msg = "verl.workers.engine.torchtitan.utils:missing-configure_qwen3_moe_for_rl"
            if require_rl_moe_helper:
                bad.append(msg)
            else:
                print(
                    f"NOTE: {msg} (must disable moe.load_balance_coeff via 0.3-compatible helper before Qwen3-30B-A3B RL)"
                )
    except Exception as exc:
        bad.append(f"verl-torchtitan-utils-import:{exc}")

    return tt_ver


def check_gpu_runtime(args: argparse.Namespace, bad: list[str]) -> None:
    import torch

    if not torch.cuda.is_available():
        bad.append("cuda:torch.cuda.is_available()=False")
        return

    count = int(torch.cuda.device_count())
    if count < args.min_gpus:
        bad.append(f"cuda:device_count={count}<min_gpus={args.min_gpus}")

    nccl_ver = None
    if hasattr(torch.cuda, "nccl") and callable(getattr(torch.cuda.nccl, "version", None)):
        try:
            nccl_ver = torch.cuda.nccl.version()
        except Exception:
            nccl_ver = None
    if not nccl_ver:
        bad.append("cuda:nccl-unavailable")

    sm_min_major_minor = tuple(int(x) for x in args.min_sm.split(".")) if args.min_sm else None
    name_pat = re.compile(args.expect_gpu_regex, re.IGNORECASE) if args.expect_gpu_regex else None

    print(f"CUDA runtime: devices={count} nccl={nccl_ver}")
    for idx in range(count):
        props = torch.cuda.get_device_properties(idx)
        mem_gib = props.total_memory / GIB
        mem_gb = props.total_memory / GB
        cap = (int(props.major), int(props.minor))
        print(
            f"  cuda:{idx} name={props.name!r} capability={cap[0]}.{cap[1]} "
            f"total_memory={mem_gib:.2f} GiB ({mem_gb:.2f} GB)"
        )
        if sm_min_major_minor is not None and cap < sm_min_major_minor:
            bad.append(f"cuda:{idx}:sm={cap[0]}.{cap[1]}<min_sm={args.min_sm}")
        if args.min_hbm_gib is not None and mem_gib < args.min_hbm_gib:
            bad.append(f"cuda:{idx}:hbm_gib={mem_gib:.2f}<min_hbm_gib={args.min_hbm_gib:.2f}")
        if name_pat is not None and not name_pat.search(str(props.name)):
            bad.append(f"cuda:{idx}:name={props.name!r}!~{args.expect_gpu_regex!r}")


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("repo_dir", help="Path to the pinned verl checkout")
    parser.add_argument(
        "--extras",
        default="fsdp,vllm,torchtitan",
        help="Comma-separated uv extras enabled in this environment (default: fsdp,vllm,torchtitan)",
    )
    parser.add_argument(
        "--check-all-locked",
        action="store_true",
        help="Verify every non-editable locked package exported by uv.lock for --extras, not only the summary table",
    )
    parser.add_argument("--expect-git-sha", default=None, help="Expected full git HEAD commit SHA of repo_dir")
    parser.add_argument("--expect-lock-sha256", default=None, help="Expected SHA256 digest of repo_dir/uv.lock")
    parser.add_argument("--require-async", action="store_true", help="Assert PPOTrainerSeparateAsync imports cleanly")
    parser.add_argument(
        "--require-torchtitan",
        action="store_true",
        help="Assert TorchTitan package path, Qwen3 MoE state-dict adapter, and scheduler container resolve",
    )
    parser.add_argument(
        "--require-rl-moe-helper",
        action="store_true",
        help="Fail if verl.workers.engine.torchtitan.utils.configure_qwen3_moe_for_rl is missing",
    )
    parser.add_argument(
        "--require-gpu",
        action="store_true",
        help="Enforce live NVIDIA GPU visibility, working NCCL, and any --min-*/--expect-gpu-* thresholds",
    )
    parser.add_argument(
        "--require-gb200",
        action="store_true",
        help="Preset strict target-host thresholds for NVIDIA GB200 (aarch64, SM >= 10.0, HBM >= 165 GiB, name ~ GB200|B200, NCCL present)",
    )
    parser.add_argument("--min-gpus", type=int, default=1, help="Minimum visible CUDA device count (default: 1)")
    parser.add_argument("--min-sm", default=None, help="Minimum CUDA compute capability, e.g. 10.0 for Blackwell")
    parser.add_argument(
        "--min-hbm-gib",
        type=float,
        default=None,
        help="Minimum per-GPU total memory in binary GiB (2^30 B); GB200 186 GB nominal is ~173.2 GiB",
    )
    parser.add_argument("--expect-gpu-regex", default=None, help="Regex each CUDA device name must match")
    args = parser.parse_args(argv)
    if args.require_gb200:
        args.require_gpu = True
        if args.min_sm is None:
            args.min_sm = "10.0"
        if args.min_hbm_gib is None:
            args.min_hbm_gib = 165.0
        if args.expect_gpu_regex is None:
            args.expect_gpu_regex = r"(GB200|B200)"
    return args


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

    bad: list[str] = []
    if args.require_gb200 and platform.machine() != "aarch64":
        bad.append(f"arch:got={platform.machine()}:want=aarch64")

    print(f"platform={platform.system()}/{platform.machine()} python={sys.version.split()[0]}")
    verify_identity_pins(repo, args, bad)

    want_all = expected_versions(repo, extras, None if args.check_all_locked else check_pkgs)
    lines: list[str] = []
    for name in check_pkgs:
        w = want_all.get(canon(name))
        try:
            got = md.version(name)
        except md.PackageNotFoundError:
            got = None
        lines.append(f"{name:22s} installed {got!s:22s} locked {w!s}")
        if w is None or got != w:
            bad.append(name)
    print("\n".join(lines))

    if args.check_all_locked:
        extra_mismatches = 0
        for pkg_canon, want_ver in sorted(want_all.items()):
            if pkg_canon in {canon(c) for c in check_pkgs}:
                continue
            try:
                got_ver = md.version(pkg_canon)
            except md.PackageNotFoundError:
                got_ver = None
            if got_ver != want_ver:
                bad.append(f"locked-pkg:{pkg_canon}:got={got_ver}:want={want_ver}")
                extra_mismatches += 1
        print(f"full uv.lock check: verified {len(want_all)} packages (mismatches={extra_mismatches})")

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
        tt_ver = check_torchtitan_surface(bad, require_rl_moe_helper=args.require_rl_moe_helper)
        tt_summary = f" torchtitan {tt_ver}"

    if args.require_gpu:
        check_gpu_runtime(args, bad)

    if bad:
        print("VERIFICATION FAILED:", bad)
        sys.exit(1)

    import ray  # noqa: E402
    import transformers  # noqa: E402
    import vllm  # noqa: E402

    verdict_scope = (
        "GB200 TARGET-HOST PRECHECK OK (hardware/NCCL/venv verified; run G0 1-step MoE routing + weight-sync test next)"
        if args.require_gb200
        else "BUILD-TIME VENV LOCK OK (package versions & imports verified; target-host GPU/MoE/weight-sync check still required)"
    )
    print(
        f"{verdict_scope}: arch={platform.machine()} ray {ray.__version__} "
        f"torch {torch.__version__} (cuda {torch.version.cuda}) "
        f"vllm {vllm.__version__}{tt_summary} transformers {transformers.__version__}"
    )


if __name__ == "__main__":
    main()
