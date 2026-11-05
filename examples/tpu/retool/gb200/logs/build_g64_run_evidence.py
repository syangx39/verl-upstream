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
"""Build a G64-D evidence directory from a downloaded GB200 run directory.

Input: a local copy of ``gs://<bucket>/<RUN_ROOT>/<RUN_NAME>/`` (``head.log``, ``exit_code.txt``, ``.launch_lock.json``,
``env/``, ``validation/``, ``rollout/``, ``rollout_audit/``, ``tensorboard/``). Output (in ``--out-dir``):

* ``head.log``, ``exit_code.txt``, ``env/`` (per-node ``*-precheck.json``, ``*-g0-verify.json``, ``*-nvidia-smi-L.txt``,
  ``*-topo.txt``, sandbox report; launch-id-prefixed duplicates are skipped) and ``env/launch_lock.json``;
* ``<metrics-name>``: every ``TaskRunnerV1`` ``step:N`` line of ``head.log`` parsed into ``{step: {metric: value}}``;
* ``validation_step{0,N}_samples.jsonl``: compact validation samples (hashed question id, ``pred``/``acc``/``score``,
  ``num_turns``, ``output_sha256`` prefix and the last 300 output characters, the window ``math_dapo`` scores);
* ``rollout_audit/<step>.jsonl.gz``: the raw token-level audit sessions (``gzip`` of the GCS files, unchanged) and
  ``tensorboard/<scalar event file>``: the TensorBoard scalars, with ``raw_manifest.json`` (GCS URI, size, MD5 as
  in the GCS object metadata, SHA-256 of the uncompressed file);
* ``rollout_and_audit_summary.json``: per-step training-rollout aggregates and the whole-session token-mask audit of
  ``examples/tpu/retool/rollout_mask_audit.py`` (``test_launcher_and_sandbox.py`` recomputes it from
  ``rollout_audit/``);
* ``step_timing_breakdown.json``: wall clock per step split into the ``step`` timer, validation, TensorBoard logging
  and the rest (``examples/tpu/retool/step_timing_breakdown.py``, from the metrics JSON, the TensorBoard event file
  and the ``update_weights done`` lines of ``head.log``).

Usage::

    python3 build_g64_run_evidence.py --src <downloaded run dir> --out-dir g64-retool-25s-01 \
        --metrics-name g64_25s_01_metrics.json --final-step 25 \
        --gcs-run-uri gs://ubench-logs/xiaotongyang/retool-gb200/runs/g64-retool-25s-01
"""

import argparse
import base64
import gzip
import hashlib
import importlib.util
import json
import re
import shutil
from collections import Counter
from pathlib import Path

LOGS_DIR = Path(__file__).resolve().parent
RETOOL_DIR = LOGS_DIR.parents[1]


def _load_tool(name: str):
    spec = importlib.util.spec_from_file_location(name, RETOOL_DIR / f"{name}.py")
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


mask_audit = _load_tool("rollout_mask_audit")
step_timing = _load_tool("step_timing_breakdown")

STEP_LINE_RE = re.compile(r"\bstep:(\d+) - (.*)")
NP_SCALAR_RE = re.compile(r"^np\.\w+\((.*)\)$")
SYNC_DONE_REGEX = r"update_weights done"


def parse_step_metrics(head_log: Path) -> dict[str, dict]:
    metrics: dict[str, dict] = {}
    for line in head_log.read_text(encoding="utf-8", errors="replace").splitlines():
        m = STEP_LINE_RE.search(line)
        if not m:
            continue
        kv: dict = {}
        for part in m.group(2).split(" - "):
            key, sep, value = part.partition(":")
            if not sep:
                continue
            value = NP_SCALAR_RE.sub(r"\1", value.strip())
            try:
                kv[key.strip()] = float(value)
            except ValueError:
                kv[key.strip()] = value
        metrics[m.group(1)] = kv
    return metrics


def compact_validation(path: Path) -> list[str]:
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        prompt, text = str(r.get("input", "")), str(r.get("output", ""))
        out.append(
            json.dumps(
                {
                    "step": int(r["step"]),
                    "data_source": r["data_source"],
                    "question_id": "sha1:" + hashlib.sha1(prompt.encode("utf-8")).hexdigest()[:16],
                    "uid": r["uid"],
                    "gts": str(r["gts"]),
                    "pred": str(r.get("pred", "")),
                    "acc": bool(r["acc"]),
                    "num_turns": int(r["num_turns"]),
                    "score": float(r["score"]),
                    "reward": float(r.get("reward", r["score"])),
                    "tool_response_count": text.count("<tool_response>"),
                    "output_chars": len(text),
                    "output_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest()[:16],
                    "output_tail_300": text[-300:],
                },
                ensure_ascii=False,
            )
        )
    return out


def audit_sessions(path: Path) -> dict:
    """Whole-session token-mask audit of one ``rollout_audit/<step>.jsonl[.gz]`` (see ``rollout_mask_audit.py``)."""
    return mask_audit.audit_records(mask_audit.load_records(path))


def audit_steps(audit_dir: Path, final_step: int, ext: str) -> dict[str, dict]:
    """``{step: whole-session audit counts}`` of ``<audit_dir>/<step><ext>`` for steps ``1..final_step``."""
    return {str(s): audit_sessions(audit_dir / f"{s}{ext}") for s in range(1, final_step + 1)}


def file_digest(data: bytes) -> dict:
    return {
        "bytes": len(data),
        "md5_base64": base64.b64encode(hashlib.md5(data).digest()).decode(),
        "sha256": hashlib.sha256(data).hexdigest(),
    }


def scalar_event_file(tb_dir: Path) -> Path:
    """The event file written by ``trainer.logger`` (the one with the most summary values)."""
    files = sorted(tb_dir.glob("events.out.tfevents.*"))
    return max(files, key=lambda p: len(step_timing.read_scalar_events(p)))


def archive_raw(src: Path, out_dir: Path, gcs_run_uri: str, final_step: int) -> dict:
    """Copy the raw audit sessions (gzip, mtime 0) and the TensorBoard scalars; return the manifest."""
    entries = []
    (out_dir / "rollout_audit").mkdir(exist_ok=True)
    for s in range(1, final_step + 1):
        rel = f"rollout_audit/{s}.jsonl"
        data = (src / rel).read_bytes()
        (out_dir / f"{rel}.gz").write_bytes(gzip.compress(data, compresslevel=9, mtime=0))
        entries.append({"path": f"{rel}.gz", "gzip_of": rel, "gcs_uri": f"{gcs_run_uri}/{rel}", **file_digest(data)})
    tb = scalar_event_file(src / "tensorboard")
    (out_dir / "tensorboard").mkdir(exist_ok=True)
    shutil.copy2(tb, out_dir / "tensorboard" / tb.name)
    rel = f"tensorboard/{tb.name}"
    entries.append({"path": rel, "gcs_uri": f"{gcs_run_uri}/{rel}", **file_digest(tb.read_bytes())})
    manifest = {
        "note": "md5_base64 equals the GCS object md5Hash (checked with `gcloud storage ls --json` when archived); "
        "sha256 is of the uncompressed file",
        "files": entries,
    }
    (out_dir / "raw_manifest.json").write_text(json.dumps(manifest, indent=1) + "\n", encoding="utf-8")
    return manifest


def timing_breakdown(ev_dir: Path, metrics: dict, final_step: int) -> dict:
    """Per-step wall-clock breakdown from the archived metrics, TensorBoard scalars and ``head.log``."""
    (tb,) = sorted((ev_dir / "tensorboard").glob("events.out.tfevents.*"))
    res = step_timing.run_breakdown(
        metrics, tb, final_step, ev_dir / "head.log", SYNC_DONE_REGEX, None, 2, final_step - 1
    )
    res["inputs"] = {"tb_events": f"tensorboard/{tb.name}", "log": "head.log", "sync_done_regex": SYNC_DONE_REGEX}
    return res


def rollout_step(path: Path) -> dict[str, int]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    outputs = [str(r.get("output", "")) for r in rows]
    return dict(
        num_trajectories=len(rows),
        trajectories_with_tool_response=sum("<tool_response>" in o for o in outputs),
        total_tool_responses=sum(o.count("<tool_response>") for o in outputs),
        clip_neg2_count=sum(float(r.get("score", 0.0)) <= -1.99 for r in rows),
    )


def build(src: Path, out_dir: Path, metrics_name: str, final_step: int, gcs_run_uri: str) -> dict:
    env_out = out_dir / "env"
    env_out.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src / "exit_code.txt", out_dir / "exit_code.txt")
    shutil.copy2(src / "head.log", out_dir / "head.log")
    shutil.copy2(src / ".launch_lock.json", env_out / "launch_lock.json")
    launch_prefix = json.loads((src / ".launch_lock.json").read_text(encoding="utf-8"))["launch_id"] + "-"
    for p in sorted((src / "env").iterdir()):
        if p.suffix in (".json", ".txt") and not p.name.startswith(launch_prefix):
            shutil.copy2(p, env_out / p.name)

    metrics = parse_step_metrics(src / "head.log")
    (out_dir / metrics_name).write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    for step in (0, final_step):
        lines = compact_validation(src / "validation" / f"{step}.jsonl")
        (out_dir / f"validation_step{step}_samples.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")

    archive_raw(src, out_dir, gcs_run_uri.rstrip("/"), final_step)
    steps: dict[str, dict] = {}
    totals: dict = {}
    reasons: Counter = Counter()
    total_keys = {
        "num_trajectories": "total_trajectories",
        "trajectories_with_tool_response": "trajectories_with_tool_response",
        "total_tool_responses": "total_tool_responses",
        "clip_neg2_count": "total_clip_neg2_count",
        "audit_sessions": "total_audit_sessions",
    }
    for s, audit in audit_steps(src / "rollout_audit", final_step, ".jsonl").items():
        reasons.update(audit["violation_reasons"])
        row = {"step": int(s), **rollout_step(src / "rollout" / f"{s}.jsonl"), **audit}
        steps[s] = row
        for key, value in row.items():
            if key not in ("step", "violation_reasons"):
                tk = total_keys.get(key, key)
                totals[tk] = totals.get(tk, 0) + value
    totals["violation_reasons"] = dict(sorted(reasons.items()))
    totals["trajectories_with_tool_response_ratio"] = round(
        totals["trajectories_with_tool_response"] / totals["total_trajectories"], 6
    )
    summary = {"steps": steps, "totals": totals}
    (out_dir / "rollout_and_audit_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    timing = timing_breakdown(out_dir, metrics, final_step)
    (out_dir / "step_timing_breakdown.json").write_text(json.dumps(timing, indent=1) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--src", type=Path, required=True, help="downloaded GCS run directory")
    parser.add_argument("--out-dir", type=Path, required=True, help="evidence directory to write")
    parser.add_argument("--metrics-name", required=True, help="file name for the parsed step metrics JSON")
    parser.add_argument("--final-step", type=int, required=True, help="last training step (validated at the end)")
    parser.add_argument("--gcs-run-uri", required=True, help="gs:// URI of the run directory (for the manifest)")
    args = parser.parse_args()
    summary = build(args.src, args.out_dir, args.metrics_name, args.final_step, args.gcs_run_uri)
    print(json.dumps(summary["totals"], indent=2))


if __name__ == "__main__":
    main()
