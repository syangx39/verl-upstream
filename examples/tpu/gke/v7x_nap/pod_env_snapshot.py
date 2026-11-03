#!/usr/bin/env python3
# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Record the TPU/XLA/Ray environment of a pod and of its Ray worker processes, allowlisted and redacted.

Evidence for a run's numerics and placement settings (e.g. that ``LIBTPU_INIT_ARGS`` with
``--xla_tpu_bf16_emission_mode=PROMOTE_F32`` reached every trainer and rollout process):

* ``pod --out FILE [--set k=v ...]``: one JSON object with the pod's own environment.
* ``procs --out FILE [--interval S] [--once]``: scans ``/proc`` for Ray processes (``ray::*`` workers, raylet) and
  appends one JSON line per new (pid, environment) to FILE, every S seconds until killed.

Only variables whose names match ``ALLOWED_PREFIXES`` / ``ALLOWED_NAMES`` are kept, and values of names that look
like credentials are replaced by ``<redacted>``. Files are written locally first and then copied to FILE, so FILE can
live on a GCS FUSE mount.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import socket
import sys
import tempfile
import time
from typing import Iterable, Mapping, Optional

ALLOWED_PREFIXES = (
    "TPU_",
    "LIBTPU_",
    "XLA_",
    "MEGASCALE_",
    "JAX_",
    "PJRT_",
    "TORCH_TPU_",
    "CLOUD_TPU_",
    "VERL_",
    "VLLM_",
    "SC_",
    "RAY_EXPERIMENTAL_",
    "RAY_memory",
    "TORCH_DYNAMO_",
)
ALLOWED_NAMES = frozenset(
    {
        "CHIPS_PER_HOST",
        "HOSTNAME",
        "JOB_COMPLETION_INDEX",
        "LOCAL_RANK",
        "LOCAL_WORLD_SIZE",
        "MASTER_ADDR",
        "MASTER_PORT",
        "NODE_NAME",
        "NODE_RANK",
        "POD_NAME",
        "POD_UID",
        "PYTHONPATH",
        "RANK",
        "SKIP_JAX_PRECOMPILE",
        "TORCH_LOGS",
        "WORLD_SIZE",
    }
)
# Substrings that mark a credential even inside an allowed prefix (e.g. a hypothetical VERL_..._TOKEN).
SECRET_MARKERS = ("TOKEN", "SECRET", "PASSWORD", "PASSWD", "CREDENTIAL", "AUTH", "COOKIE", "PRIVATE", "_KEY", "APIKEY")
MAX_VALUE_LEN = 4096
REDACTED = "<redacted>"


def is_allowed(name: str) -> bool:
    """Whether an environment variable name is recorded at all."""
    return name in ALLOWED_NAMES or name.startswith(ALLOWED_PREFIXES)


def looks_secret(name: str) -> bool:
    """Whether a (recorded) name looks like it holds a credential."""
    upper = name.upper()
    return any(marker in upper for marker in SECRET_MARKERS)


def filter_env(env: Mapping[str, str]) -> dict[str, str]:
    """Allowlist ``env`` and redact credential-like names; long values are truncated."""
    out = {}
    for name in sorted(env):
        if not is_allowed(name):
            continue
        value = env[name]
        if looks_secret(name):
            value = REDACTED
        elif len(value) > MAX_VALUE_LEN:
            value = value[:MAX_VALUE_LEN] + f"...<{len(value) - MAX_VALUE_LEN} more chars>"
        out[name] = value
    return out


def parse_environ(raw: bytes) -> dict[str, str]:
    """Parse a ``/proc/<pid>/environ`` blob (NUL-separated ``NAME=value`` entries)."""
    env = {}
    for entry in raw.split(b"\0"):
        if b"=" not in entry:
            continue
        name, _, value = entry.partition(b"=")
        env[name.decode("utf-8", "replace")] = value.decode("utf-8", "replace")
    return env


def is_ray_process(cmdline: list[str]) -> bool:
    """Ray worker processes rename themselves to ``ray::<task or actor>``; also keep the raylet and Ray workers
    that have not been renamed yet."""
    if not cmdline:
        return False
    head = cmdline[0]
    if head.startswith("ray::"):
        return True
    joined = " ".join(cmdline[:3])
    return "/raylet" in joined or "default_worker.py" in joined


def _publish(local_path: str, dest: str) -> None:
    """Copy a locally written file to ``dest`` (which may be on GCS FUSE); failures are reported, not raised."""
    try:
        os.makedirs(os.path.dirname(dest) or ".", exist_ok=True)
        shutil.copyfile(local_path, dest)
    except OSError as e:
        print(f"[pod_env_snapshot] could not write {dest}: {e}", file=sys.stderr, flush=True)


def _base_record() -> dict:
    return {"host": socket.gethostname(), "time": time.time()}


def snapshot_pod(out: str, extra: Iterable[str]) -> dict:
    """Write the pod environment record to ``out`` and return it."""
    record = _base_record()
    record["kind"] = "pod"
    for item in extra:
        key, _, value = item.partition("=")
        record[key] = value
    record["env"] = filter_env(os.environ)
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        json.dump(record, f, indent=1, sort_keys=True)
        local = f.name
    _publish(local, out)
    os.unlink(local)
    return record


def scan_procs(seen: set, proc_root: str = "/proc") -> list[dict]:
    """Records for Ray processes whose (pid, environment) has not been seen yet."""
    records = []
    try:
        pids = [p for p in os.listdir(proc_root) if p.isdigit()]
    except OSError:
        return records
    for pid in pids:
        try:
            with open(os.path.join(proc_root, pid, "cmdline"), "rb") as f:
                cmdline = [a.decode("utf-8", "replace") for a in f.read().split(b"\0") if a]
            if not is_ray_process(cmdline):
                continue
            with open(os.path.join(proc_root, pid, "environ"), "rb") as f:
                env = filter_env(parse_environ(f.read()))
        except OSError:  # the process exited or is not readable
            continue
        digest = hashlib.sha256(json.dumps(env, sort_keys=True).encode()).hexdigest()[:16]
        if (pid, digest) in seen:
            continue
        seen.add((pid, digest))
        record = _base_record()
        record.update({"kind": "proc", "pid": int(pid), "cmd": " ".join(cmdline)[:300], "env_sha": digest, "env": env})
        records.append(record)
    return records


def watch_procs(out: str, interval: float, once: bool) -> None:
    """Append new Ray process records to ``out`` every ``interval`` seconds (forever unless ``once``)."""
    seen: set = set()
    fd, local = tempfile.mkstemp(suffix=".jsonl")
    os.close(fd)
    while True:
        records = scan_procs(seen)
        if records:
            with open(local, "a") as f:
                for r in records:
                    f.write(json.dumps(r, sort_keys=True) + "\n")
            _publish(local, out)
        if once:
            return
        time.sleep(interval)


def main(argv: Optional[list[str]] = None) -> int:
    """CLI entry point (see the module docstring)."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p_pod = sub.add_parser("pod")
    p_pod.add_argument("--out", required=True)
    p_pod.add_argument("--set", action="append", default=[], metavar="KEY=VALUE", help="extra top-level fields")
    p_procs = sub.add_parser("procs")
    p_procs.add_argument("--out", required=True)
    p_procs.add_argument("--interval", type=float, default=60.0)
    p_procs.add_argument("--once", action="store_true")
    args = ap.parse_args(argv)
    if args.cmd == "pod":
        snapshot_pod(args.out, args.set)
    else:
        watch_procs(args.out, args.interval, args.once)
    return 0


if __name__ == "__main__":
    sys.exit(main())
