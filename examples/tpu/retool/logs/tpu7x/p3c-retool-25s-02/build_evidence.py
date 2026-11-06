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
"""Build the evidence directory of a TPU v7x NAP ReTool run from its downloaded artifacts.

Inputs: a local copy of ``gs://<bucket>/<RUN_ROOT>/<RUN_NAME>/`` (``validation/``, ``rollout/``, ``rollout_audit/``,
``tensorboard/``, ``env/``, ``run_record.json``, ``job_spec.json``) and the launcher's driver log
(``/tmp/verl_nap_logs/<run>/head.log``). Outputs (in ``--out-dir``):

* ``head_taskrunner.log``: the ``(TaskRunnerV1 ...)`` driver lines of ``head.log`` with ANSI colors stripped (the
  full log is ~50 MB of vLLM/XLA worker output and is not archived), and ``tqdm_progress.log``: the distinct
  ``Training Progress`` updates;
* ``metrics.json``: every ``step:N`` metrics line parsed into ``{step: {metric: value}}``;
* ``validation_step{0,N}_samples.jsonl``: compact validation samples (hashed question id, ``pred``/``acc``/``score``,
  ``num_turns``, ``output_sha256`` prefix and the last 300 output characters, the window ``math_dapo`` scores);
* ``rollout_audit/<step>.jsonl.gz``: the raw token-level audit sessions (``gzip`` of the GCS files, unchanged) and
  ``tensorboard/<scalar event file>``: the TensorBoard scalars, with ``raw_manifest.json`` (GCS URI, size, MD5 as
  in the GCS object metadata, SHA-256 of the uncompressed file);
* ``rollout_and_audit_summary.json``: per-step training-rollout aggregates and the whole-session token-mask audit of
  ``examples/tpu/retool/rollout_mask_audit.py`` (``verify_evidence.py`` recomputes it from ``rollout_audit/``);
* ``step_timing_breakdown.json``: wall clock per step split into the ``step`` timer, validation, TensorBoard logging
  and the rest (``examples/tpu/retool/step_timing_breakdown.py``, from ``metrics.json``, the TensorBoard event file
  and the Raiden end-of-sync lines of ``head_taskrunner.log``);
* ``raiden_norm_checks.json``: per step, the replicas whose post-sync ``DISTRIBUTED NORM PARITY`` check passed;
* ``eval_summary.json``: Step 0 vs. Step N ``mean@30`` with a two-level paired bootstrap interval
  (10,000 replicates, seed 0, the same procedure as the GB200 G64-D report);
* ``env/``: ``run_record.json``, ``job_spec.json``, ``raiden_exact_parity_step_0.json`` and the per-pod env records.

Usage::

    python3 build_evidence.py --src <downloaded run dir> --head-log /tmp/verl_nap_logs/<run>/head.log \\
        --out-dir . --final-step 25
"""

import argparse
import base64
import gzip
import hashlib
import importlib.util
import json
import random
import re
import shutil
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
RETOOL_DIR = HERE.parents[2]


def _load_tool(name: str):
    spec = importlib.util.spec_from_file_location(name, RETOOL_DIR / f"{name}.py")
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


mask_audit = _load_tool("rollout_mask_audit")
step_timing = _load_tool("step_timing_breakdown")

ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")
STEP_LINE_RE = re.compile(r"\bstep:(\d+) - (.*)")
NP_SCALAR_RE = re.compile(r"^np\.\w+\((.*)\)$")
NORM_RE = re.compile(
    r"\[RAIDEN PARITY VERIFIED \| Step (\d+) \| Replica (\d+)\] 100% DISTRIBUTED NORM PARITY CONFIRMED!"
)
SYNC_DONE_REGEX = r"\[RAIDEN TELEMETRY \| Orchestrator\] Step \d+ Completed"
TQDM_RE = re.compile(r"Training Progress:[^\r\n]*")
BOOTSTRAP_REPS = 10_000
BOOTSTRAP_SEED = 0
DATASETS = ("aime_2024", "aime_2025")


def taskrunner_lines(head_log: Path) -> list[str]:
    text = ANSI_RE.sub("", head_log.read_text(encoding="utf-8", errors="replace"))
    return [line.rstrip("\r") for line in text.splitlines() if line.startswith("(TaskRunnerV1")]


def tqdm_lines(head_log: Path) -> list[str]:
    """Distinct ``Training Progress`` updates (tqdm rewrites the line with ``\\r``) in log order."""
    text = ANSI_RE.sub("", head_log.read_text(encoding="utf-8", errors="replace"))
    out: list[str] = []
    for m in TQDM_RE.finditer(text):
        if not out or out[-1] != m.group(0):
            out.append(m.group(0))
    return out


def parse_step_metrics(lines: list[str]) -> dict[str, dict]:
    metrics: dict[str, dict] = {}
    for line in lines:
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


def parse_norm_checks(lines: list[str]) -> dict[str, list[int]]:
    out: dict[str, list[int]] = defaultdict(list)
    for line in lines:
        m = NORM_RE.search(line)
        if m:
            out[m.group(1)].append(int(m.group(2)))
    return {k: sorted(v) for k, v in sorted(out.items(), key=lambda kv: int(kv[0]))}


def compact_validation(path: Path) -> list[dict]:
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        prompt, text = str(r.get("input", "")), str(r.get("output", ""))
        out.append(
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
            }
        )
    return out


def audit_sessions(path: Path) -> dict:
    """Whole-session token-mask audit of one ``rollout_audit/<step>.jsonl[.gz]`` (see ``rollout_mask_audit.py``)."""
    return mask_audit.audit_records(mask_audit.load_records(path))


def gcs_uri(run_record: dict, rel: str) -> str:
    """GCS URI of a file of the run directory (``run_dir`` is the GCS Fuse path ``/data/<object prefix>``)."""
    prefix = run_record["run_dir"].removeprefix("/data/").strip("/")
    return f"gs://{run_record['gcs_bucket']}/{prefix}/{rel}"


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


def archive_raw(src: Path, out_dir: Path, run_record: dict, final_step: int) -> dict:
    """Copy the raw audit sessions (gzip, mtime 0) and the TensorBoard scalars; return the manifest."""
    entries = []
    (out_dir / "rollout_audit").mkdir(exist_ok=True)
    for s in range(1, final_step + 1):
        rel = f"rollout_audit/{s}.jsonl"
        data = (src / rel).read_bytes()
        (out_dir / f"{rel}.gz").write_bytes(gzip.compress(data, compresslevel=9, mtime=0))
        entries.append({"path": f"{rel}.gz", "gzip_of": rel, "gcs_uri": gcs_uri(run_record, rel), **file_digest(data)})
    tb = scalar_event_file(src / "tensorboard")
    (out_dir / "tensorboard").mkdir(exist_ok=True)
    shutil.copy2(tb, out_dir / "tensorboard" / tb.name)
    rel = f"tensorboard/{tb.name}"
    entries.append({"path": rel, "gcs_uri": gcs_uri(run_record, rel), **file_digest(tb.read_bytes())})
    manifest = {
        "note": "md5_base64 equals the GCS object md5Hash (checked with `gcloud storage ls --json` when archived); "
        "sha256 is of the uncompressed file",
        "files": entries,
    }
    (out_dir / "raw_manifest.json").write_text(json.dumps(manifest, indent=1) + "\n", encoding="utf-8")
    return manifest


def rollout_step(path: Path) -> dict[str, int]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    outputs = [str(r.get("output", "")) for r in rows]
    return dict(
        num_trajectories=len(rows),
        trajectories_with_tool_response=sum("<tool_response>" in o for o in outputs),
        total_tool_responses=sum(o.count("<tool_response>") for o in outputs),
        clip_neg2_count=sum(float(r.get("score", 0.0)) <= -1.99 for r in rows),
    )


def group_by_question(samples: list[dict]) -> dict[str, dict[str, list[dict]]]:
    by_ds_q: dict[str, dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))
    for r in samples:
        by_ds_q[r["data_source"]][r["question_id"]].append(r)
    return by_ds_q


def paired_bootstrap_delta(step0: dict[str, list[dict]], step_n: dict[str, list[dict]]) -> tuple[float, float]:
    """Two-level paired bootstrap of delta mean@30 (resample questions, then the samples of each step).

    The interval covers evaluation sampling noise on this fixed question set only; it says nothing about the
    run-to-run variance of RL training itself.
    """
    rng = random.Random(BOOTSTRAP_SEED)
    qids = sorted(step0)
    acc0 = {q: [1.0 if s["acc"] else 0.0 for s in step0[q]] for q in qids}
    accn = {q: [1.0 if s["acc"] else 0.0 for s in step_n[q]] for q in qids}
    deltas = []
    for _ in range(BOOTSTRAP_REPS):
        total = 0.0
        for q in rng.choices(qids, k=len(qids)):
            a0, an = acc0[q], accn[q]
            total += sum(rng.choices(an, k=len(an))) / len(an) - sum(rng.choices(a0, k=len(a0))) / len(a0)
        deltas.append(total / len(qids))
    deltas.sort()
    return deltas[int(0.025 * BOOTSTRAP_REPS)], deltas[int(0.975 * BOOTSTRAP_REPS) - 1]


def eval_summary(v0: list[dict], vn: list[dict], m0: dict, mn: dict, final_step: int) -> dict:
    g0, gn = group_by_question(v0), group_by_question(vn)
    out: dict = {"final_step": final_step, "bootstrap": {"reps": BOOTSTRAP_REPS, "seed": BOOTSTRAP_SEED}}
    for ds in DATASETS:
        q0, qn = g0[ds], gn[ds]
        assert sorted(q0) == sorted(qn), f"{ds}: question sets differ"
        n = len(next(iter(q0.values())))
        per_q = [sum(s["acc"] for s in qn[q]) - sum(s["acc"] for s in q0[q]) for q in sorted(q0)]
        lo, hi = paired_bootstrap_delta(q0, qn)
        c0 = sum(s["acc"] for q in q0 for s in q0[q])
        cn = sum(s["acc"] for q in qn for s in qn[q])
        row = {
            "questions": len(q0),
            "samples_per_question": n,
            "correct_step0": c0,
            "correct_final": cn,
            "samples": n * len(q0),
            "delta_pp": round(100 * (cn - c0) / (n * len(q0)), 4),
            "paired_bootstrap_95ci_pp": [round(100 * lo, 2), round(100 * hi, 2)],
            "questions_improved": sum(d > 0 for d in per_q),
            "questions_worse": sum(d < 0 for d in per_q),
            "questions_tied": sum(d == 0 for d in per_q),
        }
        for name, key in (
            ("mean_at_n", f"val-core/{ds}/acc/mean@{n}"),
            ("maj_at_n", f"val-core/{ds}/acc/maj@{n}/mean"),
            ("best_at_n", f"val-core/{ds}/acc/best@{n}/mean"),
            ("shaped_reward_mean_at_n", f"val-aux/{ds}/reward/mean@{n}"),
        ):
            row[f"{name}_step0_logged"] = m0.get(key)
            row[f"{name}_final_logged"] = mn.get(key)
        out[ds] = row
    return out


def audit_steps(audit_dir: Path, final_step: int, ext: str) -> dict[str, dict]:
    """``{step: whole-session audit counts}`` of ``<audit_dir>/<step><ext>`` for steps ``1..final_step``."""
    return {str(s): audit_sessions(audit_dir / f"{s}{ext}") for s in range(1, final_step + 1)}


def rollout_audit_summary(src: Path, final_step: int) -> dict:
    """Per-step rollout aggregates of ``rollout/`` plus the whole-session audit of ``rollout_audit/``."""
    steps: dict[str, dict] = {}
    totals: dict = defaultdict(int)
    reasons: Counter = Counter()
    for s, audit in audit_steps(src / "rollout_audit", final_step, ".jsonl").items():
        reasons.update(audit["violation_reasons"])
        steps[s] = {"step": int(s), **rollout_step(src / "rollout" / f"{s}.jsonl"), **audit}
        for key, value in steps[s].items():
            if key not in ("step", "violation_reasons"):
                totals[key] += value
    return {"steps": steps, "totals": {**totals, "violation_reasons": dict(sorted(reasons.items()))}}


def timing_breakdown(out_dir: Path, metrics: dict, final_step: int) -> dict:
    """Per-step wall-clock breakdown from the archived metrics, TensorBoard scalars and driver log."""
    (tb,) = sorted((out_dir / "tensorboard").glob("events.out.tfevents.*"))
    res = step_timing.run_breakdown(
        metrics, tb, final_step, out_dir / "head_taskrunner.log", SYNC_DONE_REGEX, None, 2, final_step - 1
    )
    res["inputs"] = {
        "metrics": "metrics.json",
        "tb_events": f"tensorboard/{tb.name}",
        "log": "head_taskrunner.log",
        "sync_done_regex": SYNC_DONE_REGEX,
    }
    return res


def build(src: Path, head_log: Path, out_dir: Path, final_step: int) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    env_out = out_dir / "env"
    env_out.mkdir(exist_ok=True)
    for name in ("run_record.json", "job_spec.json"):
        shutil.copy2(src / name, env_out / name)
    for p in sorted((src / "env").iterdir()):
        if p.suffix == ".json":
            shutil.copy2(p, env_out / p.name)

    lines = taskrunner_lines(head_log)
    (out_dir / "head_taskrunner.log").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (out_dir / "tqdm_progress.log").write_text("\n".join(tqdm_lines(head_log)) + "\n", encoding="utf-8")
    metrics = parse_step_metrics(lines)
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    norms = parse_norm_checks(lines)
    (out_dir / "raiden_norm_checks.json").write_text(json.dumps(norms, indent=1) + "\n", encoding="utf-8")

    val = {}
    for step in (0, final_step):
        val[step] = compact_validation(src / "validation" / f"{step}.jsonl")
        text = "\n".join(json.dumps(r, ensure_ascii=False) for r in val[step]) + "\n"
        (out_dir / f"validation_step{step}_samples.jsonl").write_text(text, encoding="utf-8")
    ev = eval_summary(val[0], val[final_step], metrics["0"], metrics[str(final_step)], final_step)
    (out_dir / "eval_summary.json").write_text(json.dumps(ev, indent=2) + "\n", encoding="utf-8")

    run_record = json.loads((src / "run_record.json").read_text(encoding="utf-8"))
    archive_raw(src, out_dir, run_record, final_step)
    summary = rollout_audit_summary(src, final_step)
    (out_dir / "rollout_and_audit_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    timing = timing_breakdown(out_dir, metrics, final_step)
    (out_dir / "step_timing_breakdown.json").write_text(json.dumps(timing, indent=1) + "\n", encoding="utf-8")
    return {
        "eval": ev,
        "rollout_totals": summary["totals"],
        "norm_check_steps": len(norms),
        "timing_summary": timing["summary"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--src", type=Path, required=True, help="downloaded GCS run directory")
    parser.add_argument("--head-log", type=Path, required=True, help="launcher driver log (head.log)")
    parser.add_argument("--out-dir", type=Path, required=True, help="evidence directory to write")
    parser.add_argument("--final-step", type=int, required=True, help="last training step (validated at the end)")
    args = parser.parse_args()
    print(json.dumps(build(args.src, args.head_log, args.out_dir, args.final_step), indent=2))


if __name__ == "__main__":
    main()
