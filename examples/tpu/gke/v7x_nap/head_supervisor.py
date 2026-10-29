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
"""Main process of the Ray head pod of the TPU 7x NAP JobSet (``run_ray_jobset_v7x_nap.sh``).

1. Starts the Ray head and waits until every host of every TPU slice has joined (``VERL_NAP_EXPECTED_SLICES``).
2. Records the Ray nodes (slice resources, ``verl-tpu-node-rank`` labels) and the image's package versions under
   ``<attempt dir>/cluster/``, and checks that every multi-host slice carries the ranks ``0..hosts-1``.
3. Submits the job of ``$VERL_NAP_RUN_DIR/job_spec.json`` (entrypoint argv + env vars) with the pinned source in
   ``/workspace/verl`` as working_dir, mirrors the driver log to stdout and ``<attempt dir>/logs/driver.log``,
   and stops the job at the driver deadline (``VERL_NAP_DRIVER_TIMEOUT_S``, counted from this process' start).
4. Writes ``verdict.json`` (attempt dir and ``$VERL_NAP_RUN_DIR``) and exits 0 only if the job succeeded, so the
   JobSet's success policy (and its failure policy) follow the job.

The attempt dir is ``$VERL_NAP_RUN_DIR/attempts/<first 8 chars of the head pod UID>``: a Kueue eviction and
re-admission starts a new head pod, and each attempt keeps its own evidence.
"""

from __future__ import annotations

import importlib.metadata
import json
import os
import shlex
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import urllib.request
from typing import Any, Optional

WORKING_DIR = "/workspace/verl"
DASHBOARD = "http://127.0.0.1:8265"
NODE_RANK_LABEL = "verl-tpu-node-rank"
EXCLUDES = [".git", "logs", "*.log", "*.pt", "*.bin", "__pycache__", ".ruff_cache", ".mypy_cache"]
PACKAGES = (
    "ray",
    "torch",
    "torch-tpu",
    "tpu-sync-torch",
    "libtpu",
    "jax",
    "vllm",
    "vllm-torchtpu",
    "torchtitan",
    "TransferQueue",
    "tensordict",
    "transformers",
    "hydra-core",
    "fastapi",
    "uvicorn",
)
LOG_PRINT_INTERVAL_S = 30
LOG_MIRROR_INTERVAL_S = 120
STATUS_POLL_S = 15

START = time.time()
RUN_DIR = os.environ.get("VERL_NAP_RUN_DIR", "")
LOCAL_DIR = tempfile.mkdtemp(prefix="verl-nap-head-")


def log(msg: str) -> None:
    """Print a timestamped supervisor line to the container log."""
    stamp = time.strftime("%H:%M:%S", time.gmtime())
    print(f"[verl-nap head {stamp}Z +{time.time() - START:.0f}s] {msg}", flush=True)


def attempt_dir() -> str:
    """Directory of this head pod's attempt: a Kueue requeue starts a new head pod, which must not overwrite the
    evidence of the previous attempt."""
    return os.path.join(RUN_DIR, "attempts", (os.environ.get("POD_UID") or str(int(START)))[:8])


def copy_out(local: str, dest: str) -> None:
    """Copy a locally written file to ``dest`` on the run dir (GCS FUSE); failures are logged, not raised."""
    try:
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        shutil.copyfile(local, dest)
    except OSError as e:
        log(f"WARNING: could not write {dest}: {e}")


def publish(rel: str, text: str, also_top: bool = False) -> None:
    """Write ``text`` to ``<attempt dir>/<rel>`` (and ``$VERL_NAP_RUN_DIR/<rel>`` if ``also_top``)."""
    local = os.path.join(LOCAL_DIR, rel.replace("/", "__"))
    with open(local, "w") as f:
        f.write(text)
    copy_out(local, os.path.join(attempt_dir(), rel))
    if also_top:
        copy_out(local, os.path.join(RUN_DIR, rel))


def publish_json(rel: str, obj: Any, also_top: bool = False) -> None:
    """``publish`` a JSON document."""
    publish(rel, json.dumps(obj, indent=1, sort_keys=True, default=str) + "\n", also_top)


def parse_expected_slices(spec: str) -> dict[str, int]:
    """``tpu-group-0=4,tpu-group-1=4`` -> ``{"tpu-group-0": 4, "tpu-group-1": 4}``."""
    out = {}
    for item in filter(None, (s.strip() for s in spec.split(","))):
        name, _, hosts = item.partition("=")
        out[name] = int(hosts)
    return out


def node_record(node: dict) -> dict:
    """The parts of a ``ray.nodes()`` entry worth keeping."""
    resources = node.get("Resources", {}) or {}
    keep = {k: v for k, v in resources.items() if k in ("TPU", "CPU") or k.startswith(("tpu-group-", "TPU-"))}
    return {
        "node_id": node.get("NodeID"),
        "ip": node.get("NodeManagerAddress"),
        "hostname": node.get("NodeManagerHostname"),
        "alive": node.get("Alive"),
        "resources": keep,
        "labels": node.get("Labels", {}) or {},
    }


def check_slices(nodes: list[dict], expected: dict[str, int]) -> tuple[bool, bool, list[str]]:
    """Per-slice host counts and node-rank labels; returns (hosts_ok, ranks_ok, report lines)."""
    hosts_ok, ranks_ok, lines = True, True, []
    for name, want in expected.items():
        members = [n for n in nodes if n["alive"] and name in n["resources"]]
        ranks = [n["labels"].get(NODE_RANK_LABEL) for n in members]
        if len(members) != want:
            hosts_ok = False
        if want > 1:
            valid = all(isinstance(r, str) and r.isdigit() for r in ranks)
            if not valid or sorted(int(r) for r in ranks) != list(range(want)):
                ranks_ok = False
        lines.append(f"{name}: {len(members)}/{want} hosts")
        for n in sorted(members, key=lambda m: (str(m["labels"].get(NODE_RANK_LABEL, "")), m["ip"] or "")):
            lines.append(
                f"  {NODE_RANK_LABEL}={n['labels'].get(NODE_RANK_LABEL, '-')} ip={n['ip']} host={n['hostname']}"
            )
    return hosts_ok, ranks_ok, lines


def package_versions() -> dict[str, Optional[str]]:
    """Installed versions of the packages that define the software stack."""
    out = {}
    for name in PACKAGES:
        try:
            out[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            out[name] = None
    return out


class Supervisor:
    """State of one head pod run; ``run()`` returns the process exit code."""

    def __init__(self) -> None:
        env = os.environ
        self.run_name = env["VERL_NAP_RUN_NAME"]
        self.expected = parse_expected_slices(env["VERL_NAP_EXPECTED_SLICES"])
        self.require_rank = env.get("VERL_NAP_REQUIRE_NODE_RANK", "1") == "1"
        self.node_wait_s = int(env.get("VERL_NAP_NODE_WAIT_TIMEOUT_S", "1200"))
        self.driver_timeout_s = int(env.get("VERL_NAP_DRIVER_TIMEOUT_S", "0"))
        attempt = (env.get("POD_UID") or str(int(START)))[:8]
        self.submission_id = f"{self.run_name}-{attempt}"
        self.verdict: dict[str, Any] = {
            "run_name": self.run_name,
            "submission_id": self.submission_id,
            "code_sha": env.get("VERL_NAP_CODE_SHA"),
            "head_pod": env.get("POD_NAME"),
            "head_started_at": START,
            "status": "STARTING",
        }
        self.client = None
        self.submitted = False

    def finish(self, status: str, message: str = "", code: int = 1) -> int:
        """Record the verdict and return the exit code."""
        self.verdict.update({"status": status, "message": message, "finished_at": time.time()})
        self.verdict["duration_s"] = round(time.time() - START)
        publish_json("verdict.json", self.verdict, also_top=True)
        log(f"VERDICT {status}: {message}")
        return code

    def start_ray(self) -> None:
        """Start the Ray head (non-blocking)."""
        cmd = [
            "ray",
            "start",
            "--head",
            "--port=6379",
            "--dashboard-host=0.0.0.0",
            "--dashboard-port=8265",
            f"--num-cpus={os.environ.get('VERL_NAP_HEAD_NUM_CPUS', '12')}",
            "--num-gpus=0",
            "--disable-usage-stats",
        ]
        if os.environ.get("VERL_NAP_HEAD_OBJECT_STORE_BYTES"):
            cmd.append(f"--object-store-memory={os.environ['VERL_NAP_HEAD_OBJECT_STORE_BYTES']}")
        log("starting Ray head: " + shlex.join(cmd))
        subprocess.run(cmd, check=True)

    def wait_for_slices(self) -> Optional[int]:
        """Block until every expected slice has all its hosts; returns an exit code on failure."""
        import ray

        ray.init(address="auto", logging_level="ERROR")
        deadline = time.time() + self.node_wait_s
        want = ", ".join(f"{k}={v}" for k, v in self.expected.items())
        last = None
        try:
            while True:
                nodes = [node_record(n) for n in ray.nodes()]
                hosts_ok, ranks_ok, lines = check_slices(nodes, self.expected)
                summary = "; ".join(line for line in lines if not line.startswith("  "))
                if summary != last:
                    log(f"TPU hosts: {summary} (want {want})")
                    last = summary
                if hosts_ok:
                    break
                if time.time() > deadline:
                    publish_json("cluster/ray_nodes.json", nodes)
                    return self.finish("FAILED", f"TPU hosts did not all join within {self.node_wait_s}s: {summary}")
                time.sleep(10)
            time.sleep(5)  # let late labels / resources settle, then record the final view
            nodes = [node_record(n) for n in ray.nodes()]
        finally:
            ray.shutdown()
        hosts_ok, ranks_ok, lines = check_slices(nodes, self.expected)
        publish_json("cluster/ray_nodes.json", nodes)
        publish("cluster/slices.txt", "\n".join(lines) + "\n")
        for line in lines:
            log(line)
        self.verdict["node_rank_ok"] = ranks_ok
        if not ranks_ok:
            msg = f"multi-host slices without a complete set of {NODE_RANK_LABEL} labels (see cluster/slices.txt)"
            if self.require_rank:
                return self.finish("FAILED", msg + "; set REQUIRE_NODE_RANK=0 to run with verl's IP order")
            log("WARNING: " + msg + "; verl falls back to ordering hosts by IP")
        return None

    def wait_for_dashboard(self, timeout_s: int = 180) -> None:
        """The job API is served by the dashboard, which comes up after the GCS."""
        deadline = time.time() + timeout_s
        while True:
            try:
                with urllib.request.urlopen(f"{DASHBOARD}/api/version", timeout=5):
                    return
            except OSError:
                if time.time() > deadline:
                    raise
                time.sleep(3)

    def submit(self) -> None:
        """Submit the job of ``job_spec.json`` with the pinned source as working_dir."""
        from ray.job_submission import JobSubmissionClient

        with open(os.path.join(RUN_DIR, "job_spec.json")) as f:
            spec = json.load(f)
        entrypoint = shlex.join(spec["entrypoint"])
        runtime_env = {"working_dir": WORKING_DIR, "excludes": EXCLUDES, "env_vars": spec.get("env_vars", {})}
        self.client = JobSubmissionClient(DASHBOARD)
        log(f"submitting {self.submission_id}: {entrypoint}")
        self.client.submit_job(entrypoint=entrypoint, submission_id=self.submission_id, runtime_env=runtime_env)
        self.submitted = True
        self.verdict.update({"submitted_at": time.time(), "entrypoint": entrypoint})

    def follow(self) -> int:
        """Poll the job to completion (or the driver deadline), mirroring its log; returns the exit code."""
        from ray.job_submission import JobStatus

        deadline = START + self.driver_timeout_s if self.driver_timeout_s > 0 else None
        if deadline:
            log(f"driver deadline in {deadline - time.time():.0f}s (VERL_NAP_DRIVER_TIMEOUT_S={self.driver_timeout_s})")
        local_log = os.path.join(LOCAL_DIR, "driver.log")
        printed, last_print, last_mirror, timed_out = 0, 0.0, 0.0, False
        while True:
            info = self.client.get_job_info(self.submission_id)
            terminal = info.status in (JobStatus.SUCCEEDED, JobStatus.FAILED, JobStatus.STOPPED)
            now = time.time()
            if terminal or now - last_print >= LOG_PRINT_INTERVAL_S:
                try:
                    logs = self.client.get_job_logs(self.submission_id)
                except Exception as e:  # the dashboard can hiccup under load; try again next round
                    log(f"WARNING: could not fetch job logs: {e}")
                    logs = None
                if logs is not None:
                    if len(logs) > printed:
                        sys.stdout.write(logs[printed:])
                        sys.stdout.flush()
                        printed = len(logs)
                    with open(local_log, "w") as f:
                        f.write(logs)
                    if terminal or now - last_mirror >= LOG_MIRROR_INTERVAL_S:
                        copy_out(local_log, os.path.join(attempt_dir(), "logs", "driver.log"))
                        last_mirror = now
                last_print = now
            if terminal:
                break
            if deadline and now > deadline and not timed_out:
                timed_out = True
                log(f"driver deadline reached after {now - START:.0f}s; stopping {self.submission_id}")
                self.client.stop_job(self.submission_id)
            time.sleep(STATUS_POLL_S)
        self.verdict["timed_out"] = timed_out
        if info.status == JobStatus.SUCCEEDED:
            return self.finish("SUCCEEDED", info.message or "", 0)
        status = "TIMEOUT" if timed_out else str(info.status.value if hasattr(info.status, "value") else info.status)
        return self.finish(status, info.message or "", 1)

    def on_sigterm(self, signum, frame) -> None:
        """Pod deletion or preemption: stop the job and leave a verdict behind."""
        log(f"received signal {signum}")
        if self.submitted and self.client is not None:
            try:
                self.client.stop_job(self.submission_id)
            except Exception as e:
                log(f"WARNING: stop_job failed: {e}")
        sys.exit(self.finish("TERMINATED", f"head pod received signal {signum}", 143))

    def run(self) -> int:
        """All steps; returns the exit code."""
        signal.signal(signal.SIGTERM, self.on_sigterm)
        if not os.path.isfile(os.path.join(RUN_DIR, "job_spec.json")):
            log(f"ERROR: {RUN_DIR}/job_spec.json not found (the launcher stages it before creating the JobSet)")
            return 2
        publish_json("cluster/packages.json", package_versions())
        self.start_ray()
        code = self.wait_for_slices()
        if code is not None:
            return code
        try:
            self.wait_for_dashboard()
            self.submit()
        except Exception as e:
            return self.finish("SUBMIT_FAILED", f"{type(e).__name__}: {e}", 1)
        return self.follow()


if __name__ == "__main__":
    sys.exit(Supervisor().run())
