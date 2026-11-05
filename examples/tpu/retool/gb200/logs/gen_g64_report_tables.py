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
"""Generate the Stage G64-D report tables from the archived raw artifacts of each run.

Every number in the generated tables is read from files under ``logs/<run>/`` (or the archived Askhat W&B audit);
nothing is typed by hand:

* per-step training / timing / rollout-correction metrics: the ``TaskRunnerV1`` ``step:N - key:value - ...`` lines
  in ``head.log`` (cross-checked against the run's metrics JSON);
* per-step wall clock and its split (``step`` timer, validation, TensorBoard logging, the rest):
  ``step_timing_breakdown.json``, i.e. ``examples/tpu/retool/step_timing_breakdown.py`` on the archived TensorBoard
  event file (the ``Training Progress`` bar is not used: it updates only after each step's TensorBoard logging);
* per-step tool-call, ``score <= -2.0`` and whole-session token-mask audit counts: ``rollout_and_audit_summary.json``
  (built by ``build_g64_run_evidence.py`` from the raw sessions archived in ``rollout_audit/``);
* Step 0 / final-step AIME accuracy, per-question deltas and the paired bootstrap interval:
  ``validation_step{0,N}_samples.jsonl``;
* Askhat references: ``examples/tpu/retool/results/askhat-wandb-audit/askhat_runs_summary.json``.

The markdown blocks in ``gb200/README.md`` and ``gb200/exp/REPRO_RESULTS_GB200_ZH.md`` between the
``AUTO-GENERATED <run>/<table>`` markers are the verbatim generator output (``--write-docs``) and are checked by
``test_launcher_and_sandbox.py``.

Usage::

    python3 examples/tpu/retool/gb200/logs/gen_g64_report_tables.py --run g64-retool-25s-01 --print-markdown
    python3 examples/tpu/retool/gb200/logs/gen_g64_report_tables.py --write-docs  # refresh the blocks in place
"""

import argparse
import difflib
import json
import math
import random
import re
from collections import defaultdict
from pathlib import Path

LOGS_DIR = Path(__file__).resolve().parent
REPO_ROOT = LOGS_DIR.parents[4]
ASKHAT_AUDIT = REPO_ROOT / "examples" / "tpu" / "retool" / "results" / "askhat-wandb-audit" / "askhat_runs_summary.json"
RUNS = {
    "g64-retool-smoke-04": {"metrics": "g64_smoke_04_metrics.json", "final_step": 5},
    "g64-retool-25s-01": {"metrics": "g64_25s_01_metrics.json", "final_step": 25},
}
STEP_LINE_RE = re.compile(r"\bstep:(\d+) - ")
NP_SCALAR_RE = re.compile(r"^np\.\w+\((.*)\)$")
ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")
CONFIG_LINE_RE = re.compile(r"\(TaskRunnerV1 pid=\d+[^)]*\) (.*)$")
BOOTSTRAP_REPS = 10_000
BOOTSTRAP_SEED = 0


def _parse_value(raw: str) -> float:
    """Parse a logged scalar, unwrapping numpy reprs such as ``np.float64(0.1)`` or ``np.int32(32)``."""
    raw = raw.strip()
    m = NP_SCALAR_RE.match(raw)
    if m:
        raw = m.group(1)
    return float(raw)


def parse_head_log_steps(path: Path) -> dict[int, dict]:
    """Return {step: {"line": 1-based line number, "metrics": {key: float}}} from TaskRunnerV1 step lines."""
    steps: dict[int, dict] = {}
    with path.open(encoding="utf-8", errors="replace") as f:
        for lineno, line in enumerate(f, 1):
            m = STEP_LINE_RE.search(line)
            if not m:
                continue
            metrics: dict[str, float] = {}
            for part in line[m.end() :].strip().split(" - "):
                key, sep, value = part.partition(":")
                if not sep:
                    continue
                try:
                    metrics[key.strip()] = _parse_value(value)
                except ValueError:
                    continue
            steps[int(m.group(1))] = {"line": lineno, "metrics": metrics}
    return steps


def resolved_config(path: Path) -> list[str]:
    """Return the Hydra config dump printed by TaskRunnerV1 at startup (ANSI codes stripped)."""
    lines = [ANSI_RE.sub("", line) for line in path.read_text(encoding="utf-8", errors="replace").splitlines()]
    start = next(
        i for i, line in enumerate(lines) if re.search(r"\(TaskRunnerV1 pid=\d+[^)]*\) \{'actor_rollout_ref'", line)
    )
    out: list[str] = []
    for line in lines[start:]:
        m = CONFIG_LINE_RE.match(line)
        if not m:
            continue
        if out and not m.group(1).startswith(" "):
            break
        out.append(m.group(1))
    return out


def resolved_config_diff(run_a: str, run_b: str) -> list[str]:
    """Changed config lines (``-``/``+``) between two runs' resolved Hydra configs."""
    a = resolved_config(LOGS_DIR / run_a / "head.log")
    b = resolved_config(LOGS_DIR / run_b / "head.log")
    return [
        d.strip() for d in difflib.unified_diff(a, b, lineterm="", n=0) if d[:1] in "+-" and d[:3] not in ("+++", "---")
    ]


def check_head_log_against_metrics_json(steps: dict[int, dict], metrics_json: dict) -> int:
    """Assert every numeric key present in both sources has the same value; return the number of compared values."""
    compared = 0
    for step_str, values in metrics_json.items():
        logged = steps[int(step_str)]["metrics"]
        for key, value in values.items():
            if key not in logged or not isinstance(value, int | float):
                continue
            assert math.isclose(logged[key], float(value), rel_tol=1e-9, abs_tol=1e-12), (
                f"step {step_str} {key}: head.log={logged[key]} metrics.json={value}"
            )
            compared += 1
    return compared


def load_validation(path: Path) -> dict[str, dict[str, list[dict]]]:
    by_ds_q: dict[str, dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                by_ds_q[r["data_source"]][r["question_id"]].append(r)
    return by_ds_q


def paired_bootstrap_delta(
    step0: dict[str, list[dict]], step_n: dict[str, list[dict]], reps: int = BOOTSTRAP_REPS, seed: int = BOOTSTRAP_SEED
) -> tuple[float, float]:
    """Two-level paired bootstrap of delta mean@30 (resample questions, then the 30 samples of each step).

    The interval covers evaluation sampling noise on this fixed question set only; it says nothing about the
    run-to-run variance of RL training itself.
    """
    rng = random.Random(seed)
    qids = sorted(step0)
    acc0 = {q: [1.0 if s["acc"] else 0.0 for s in step0[q]] for q in qids}
    accn = {q: [1.0 if s["acc"] else 0.0 for s in step_n[q]] for q in qids}
    deltas = []
    for _ in range(reps):
        total = 0.0
        for q in rng.choices(qids, k=len(qids)):
            a0, an = acc0[q], accn[q]
            total += sum(rng.choices(an, k=len(an))) / len(an) - sum(rng.choices(a0, k=len(a0))) / len(a0)
        deltas.append(total / len(qids))
    deltas.sort()
    return deltas[int(0.025 * reps)], deltas[int(0.975 * reps) - 1]


def eval_summary(run: str, steps: dict[int, dict]) -> dict[str, dict]:
    final = RUNS[run]["final_step"]
    v0 = load_validation(LOGS_DIR / run / "validation_step0_samples.jsonl")
    vn = load_validation(LOGS_DIR / run / f"validation_step{final}_samples.jsonl")
    out = {}
    for ds in ("aime_2024", "aime_2025"):
        q0, qn = v0[ds], vn[ds]
        assert sorted(q0) == sorted(qn) and len(q0) == 30, f"{ds}: question sets differ or != 30"
        per_q = [(sum(s["acc"] for s in qn[q]) - sum(s["acc"] for s in q0[q])) for q in sorted(q0)]
        lo, hi = paired_bootstrap_delta(q0, qn)
        m0, mn = steps[0]["metrics"], steps[final]["metrics"]
        out[ds] = {
            "correct_step0": sum(s["acc"] for q in q0 for s in q0[q]),
            "correct_final": sum(s["acc"] for q in qn for s in qn[q]),
            "samples": 30 * len(q0),
        }
        for name, key in (
            ("mean_at_30", "val-core/{}/acc/mean@30"),
            ("maj_at_30", "val-core/{}/acc/maj@30/mean"),
            ("best_at_30", "val-core/{}/acc/best@30/mean"),
            ("shaped_reward_mean_at_30", "val-aux/{}/reward/mean@30"),
        ):
            out[ds][f"{name}_step0"] = m0[key.format(ds)]
            out[ds][f"{name}_final"] = mn[key.format(ds)]
        out[ds].update(
            questions_improved=sum(d > 0 for d in per_q),
            questions_worse=sum(d < 0 for d in per_q),
            questions_tied=sum(d == 0 for d in per_q),
            delta_mean_at_30_bootstrap_95ci=(lo, hi),
        )
    return out


def _pct(x: float, digits: int = 2) -> str:
    return f"{100.0 * x:.{digits}f}%"


def _pp(x: float) -> str:
    return f"{100.0 * x:+.2f} pp"


def training_table(run: str, steps: dict[int, dict], rollout_summary: dict) -> str:
    final = RUNS[run]["final_step"]
    header = (
        "| Step (`head.log` line) | `timing_s/step` | `timing_s/gen` | `timing_s/update_actor` | "
        "`timing_s/update_weights` | `perf/throughput` (tok/s/GPU) | `critic/score/mean` | `actor/pg_loss` | "
        "`actor/grad_norm` | `response_length/mean` (`max`) | `response_length/clip_ratio` | "
        "`num_turns/mean` (`max`) | trajectories with `<tool_response>` (calls) | `score <= -2.0` | "
        "audit mask violations |"
    )
    rows = [header, "|" + " --- |" * 15]
    for s in range(1, final + 1):
        m = steps[s]["metrics"]
        r = rollout_summary["steps"][str(s)]
        rows.append(
            f"| {s} (L{steps[s]['line']}) | `{m['timing_s/step']:.2f}s` | `{m['timing_s/gen']:.2f}s` | "
            f"`{m['timing_s/update_actor']:.2f}s` | `{m['timing_s/update_weights']:.2f}s` | "
            f"`{m['perf/throughput']:.1f}` | `{m['critic/score/mean']:+.4f}` | `{m['actor/pg_loss']:+.4f}` | "
            f"`{m['actor/grad_norm']:.4f}` | `{m['response_length/mean']:,.1f}` (`{m['response_length/max']:,.0f}`) | "
            f"`{_pct(m['response_length/clip_ratio'])}` | `{m['num_turns/mean']:.2f}` (`{m['num_turns/max']:.0f}`) | "
            f"`{r['trajectories_with_tool_response']} / {r['num_trajectories']}` (`{r['total_tool_responses']:,}`) | "
            f"`{r['clip_neg2_count']}` | `{r['audit_mask_violations']} / {r['audit_sessions']}` |"
        )
    window = list(range(2, final + 1))
    w = [steps[s]["metrics"] for s in window]
    t = rollout_summary["totals"]

    def mean(key: str) -> float:
        return sum(x[key] for x in w) / len(w)

    rows.append(
        f"| Steps 2–{final} mean ({len(window)} steps) | `{mean('timing_s/step'):.2f}s` "
        f"(`{min(x['timing_s/step'] for x in w):.2f}s`–`{max(x['timing_s/step'] for x in w):.2f}s`) | "
        f"`{mean('timing_s/gen'):.3f}s` | `{mean('timing_s/update_actor'):.2f}s` | "
        f"`{mean('timing_s/update_weights'):.2f}s` | `{mean('perf/throughput'):.1f}` | – | – | – | – | – | – | "
        f"total `{t['trajectories_with_tool_response']} / {t['total_trajectories']}` "
        f"(`{t['total_tool_responses']:,}`) | total `{t['total_clip_neg2_count']}` | "
        f"total `{t['audit_mask_violations']} / {t['total_audit_sessions']}` |"
    )
    return "\n".join(rows)


def rollout_corr_table(run: str, steps: dict[int, dict]) -> str:
    keys = (
        ("kl", ".4f"),
        ("k3_kl", ".4f"),
        ("ppl_ratio", ".4f"),
        ("chi2_token", ".4f"),
        ("rollout_is_mean", ".4f"),
        ("rollout_is_ratio_fraction_low", ".4f"),
        ("rollout_is_seq_fraction_low", ".4f"),
        ("rollout_is_max", ".3f"),
        ("rollout_is_min", ".2e"),
    )
    rows = [
        "| Step | " + " | ".join(f"`actor/rollout_corr/{k}`" for k, _ in keys) + " |",
        "|" + " --- |" * (len(keys) + 1),
    ]
    for s in range(1, RUNS[run]["final_step"] + 1):
        m = steps[s]["metrics"]
        rows.append(f"| {s} | " + " | ".join(f"`{m['actor/rollout_corr/' + k]:{fmt}}`" for k, fmt in keys) + " |")
    return "\n".join(rows)


def eval_table(run: str, ev: dict[str, dict]) -> str:
    final = RUNS[run]["final_step"]
    rows = [
        f"| Metric (`n=30` samples per question, 30 questions) | Step 0 | Step {final} | Δ (Step {final} − Step 0) |",
        "| --- | --- | --- | --- |",
    ]
    for ds, label in (("aime_2024", "AIME 2024"), ("aime_2025", "AIME 2025")):
        e = ev[ds]
        lo, hi = e["delta_mean_at_30_bootstrap_95ci"]
        rows += [
            f"| {label} correct samples | `{e['correct_step0']} / {e['samples']}` | "
            f"`{e['correct_final']} / {e['samples']}` | `{e['correct_final'] - e['correct_step0']:+d}` |",
            f"| {label} `acc/mean@30` | `{_pct(e['mean_at_30_step0'])}` | `{_pct(e['mean_at_30_final'])}` | "
            f"`{_pp(e['mean_at_30_final'] - e['mean_at_30_step0'])}` (paired bootstrap 95% CI "
            f"`[{_pp(lo)}, {_pp(hi)}]`) |",
            f"| {label} `acc/maj@30/mean` | `{_pct(e['maj_at_30_step0'])}` | `{_pct(e['maj_at_30_final'])}` | "
            f"`{_pp(e['maj_at_30_final'] - e['maj_at_30_step0'])}` |",
            f"| {label} `acc/best@30/mean` | `{_pct(e['best_at_30_step0'])}` | `{_pct(e['best_at_30_final'])}` | "
            f"`{_pp(e['best_at_30_final'] - e['best_at_30_step0'])}` |",
            f"| {label} shaped `reward/mean@30` | `{e['shaped_reward_mean_at_30_step0']:+.4f}` | "
            f"`{e['shaped_reward_mean_at_30_final']:+.4f}` | "
            f"`{e['shaped_reward_mean_at_30_final'] - e['shaped_reward_mean_at_30_step0']:+.4f}` |",
            f"| {label} questions improved / worse / tied (per-question correct count) | – | – | "
            f"`{e['questions_improved']} / {e['questions_worse']} / {e['questions_tied']}` |",
        ]
    return "\n".join(rows)


def askhat_table(run: str, steps: dict[int, dict], ev: dict[str, dict]) -> str:
    """Our run vs. the archived Askhat v5p Regime 2 (16K + thinking + shaped reward) references at Step 0 / 25."""
    final = RUNS[run]["final_step"]
    audit = json.loads(ASKHAT_AUDIT.read_text(encoding="utf-8"))

    def fmt(x) -> str:
        return "–" if x is None else f"`{_pct(x)}`"

    def fmt_r(x) -> str:
        return "–" if x is None else f"`{x:+.3f}`"

    train = [steps[s]["metrics"]["critic/score/mean"] for s in range(1, final + 1)]
    rows = [
        "| Run | Starts from | eval `n` per question | Step | AIME 2024 `acc` | AIME 2025 `acc` | "
        "train `critic/score/mean`, Steps 1–5 mean | Steps 21–25 mean |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
        f"| `{run}` (this run, 64× GB200) | SFT `p2-retool-sft-01` step 372 | `30` | 0 | "
        f"{fmt(ev['aime_2024']['mean_at_30_step0'])} | {fmt(ev['aime_2025']['mean_at_30_step0'])} | – | – |",
        f"| `{run}` (this run, 64× GB200) | SFT `p2-retool-sft-01` step 372 | `30` | {final} | "
        f"{fmt(ev['aime_2024']['mean_at_30_final'])} | {fmt(ev['aime_2025']['mean_at_30_final'])} | "
        f"{fmt_r(sum(train[:5]) / 5)} | {fmt_r(sum(train[20:25]) / 5 if final >= 25 else None)} |",
    ]
    for r in audit["runs"]:
        if not r["regime"].startswith("Regime_2") or r["config"].get("max_user_turns") != 16:
            continue
        ts = r.get("training_summary", {})
        start = f"resumed from `{r['resumed_from']}`" if r.get("resumed_from") else "Askhat's SFT checkpoint"
        for v in r.get("validations", []):
            if v["step"] in (0, 25):
                rows.append(
                    f"| `{r['run_id']}` (Askhat v5p) | {start} | `{v['n_val']}` | {v['step']} | "
                    f"{fmt(v['aime_2024_acc'])} | {fmt(v['aime_2025_acc'])} | "
                    f"{fmt_r(ts.get('reward_mean_steps_1_to_5'))} | {fmt_r(ts.get('reward_mean_steps_21_to_25'))} |"
                )
    return "\n".join(rows)


def timing_table(run: str, timing: dict) -> str:
    """Per-step wall clock split from ``step_timing_breakdown.json`` (TensorBoard event wall times)."""
    rows = [
        "| Step | wall clock (Δ of TensorBoard log starts) | `timing_s/step` | of which `gen` | of which "
        "`update_actor` | of which `update_weights` | `timing_s/testing` | TensorBoard logging of the previous step "
        "(scalars) | rest outside the timers | weight sync done → `logger.log` |",
        "|" + " --- |" * 10,
    ]
    for r in timing["rows"]:
        rows.append(
            f"| {r['step']} | `{r['wall_s']:.2f}s` | `{r['timing_s_step']:.2f}s` | `{r['timing_s_gen']:.2f}s` | "
            f"`{r['timing_s_update_actor']:.2f}s` | `{r['timing_s_update_weights']:.2f}s` | "
            f"`{r['timing_s_testing']:.2f}s` | `{r['tb_span_prev_step_s']:.2f}s` ({r['tb_scalars_prev_step']}) | "
            f"`{r['residual_s']:.2f}s` | `{r['post_sync_to_log_s']:.2f}s` |"
        )
    t = timing["summary"]
    rows.append(
        f"| Steps {t['steps'][0]}–{t['steps'][1]} mean ({t['n']} steps) | `{t['mean_wall_s']:.2f}s` | "
        f"`{t['mean_timing_s_step']:.2f}s` | `{t['mean_timing_s_gen']:.3f}s` | "
        f"`{t['mean_timing_s_update_actor']:.2f}s` | "
        f"`{t['mean_timing_s_update_weights']:.2f}s` | `{t['mean_timing_s_testing']:.2f}s` | "
        f"`{t['mean_tb_span_prev_step_s']:.2f}s` ({t['mean_tb_scalars_prev_step']:.0f}) | "
        f"`{t['mean_residual_s']:.2f}s` | `{t['mean_post_sync_to_log_s']:.2f}s` |"
    )
    return "\n".join(rows)


def derived_summary(run: str, steps: dict[int, dict], timing: dict) -> dict:
    """Scalar facts quoted in the report prose, derived from the head.log step lines and the timing breakdown."""
    final = RUNS[run]["final_step"]
    train = [steps[s]["metrics"] for s in range(1, final + 1)]
    window = [steps[s]["metrics"] for s in range(2, final + 1)]
    t, last = timing["summary"], timing["rows"][-1]

    def mean(rows: list[dict], key: str) -> float:
        return sum(r[key] for r in rows) / len(rows)

    out = {
        "timing_s_step_step_1": round(train[0]["timing_s/step"], 2),
        "timing_s_step_sum": round(sum(r["timing_s/step"] for r in train), 2),
        "timing_s_step_steps_2_to_final": {
            "mean": round(mean(window, "timing_s/step"), 2),
            "min": round(min(r["timing_s/step"] for r in window), 2),
            "max": round(max(r["timing_s/step"] for r in window), 2),
        },
        f"wall_s_per_step_steps_2_to_{final - 1}": round(t["mean_wall_s"], 2),
        f"timing_s_step_steps_2_to_{final - 1}": round(t["mean_timing_s_step"], 2),
        f"tb_logging_s_per_step_steps_2_to_{final - 1}": round(t["mean_tb_span_prev_step_s"], 2),
        f"tb_logging_s_per_scalar_steps_2_to_{final - 1}": round(
            t["mean_tb_span_prev_step_s"] / (t["mean_tb_scalars_prev_step"] - 1), 3
        ),
        f"rest_outside_timers_s_steps_2_to_{final - 1}": round(t["mean_residual_s"], 2),
        f"weight_sync_done_to_log_s_steps_2_to_{final - 1}": round(t["mean_post_sync_to_log_s"], 2),
        f"step{final}_wall_s": round(last["wall_s"], 2),
        "timing_s_gen_mean_steps_2_to_final": round(mean(window, "timing_s/gen"), 3),
        "timing_s_update_actor_mean_steps_2_to_final": round(mean(window, "timing_s/update_actor"), 2),
        "timing_s_update_weights_mean_steps_2_to_final": round(mean(window, "timing_s/update_weights"), 2),
        "perf_throughput_mean_steps_2_to_final": round(mean(window, "perf/throughput"), 1),
        "perf_total_num_tokens_sum": int(sum(r["perf/total_num_tokens"] for r in train)),
        "response_length_mean_first_last": [
            round(train[0]["response_length/mean"], 1),
            round(train[-1]["response_length/mean"], 1),
        ],
        "clip_ratio_max": round(max(r["response_length/clip_ratio"] for r in train), 6),
        "grad_norm_min_max": [
            round(min(r["actor/grad_norm"] for r in train), 4),
            round(max(r["actor/grad_norm"] for r in train), 4),
        ],
        "score_mean_steps_1_to_5": round(sum(r["critic/score/mean"] for r in train[:5]) / 5, 4),
        "actor_max_memory_allocated_gb_max": round(max(r["actor/perf/max_memory_allocated_gb"] for r in train), 2),
        "actor_max_memory_reserved_gb_max": round(max(r["actor/perf/max_memory_reserved_gb"] for r in train), 2),
        f"step{final}_timing_s_testing": round(steps[final]["metrics"]["timing_s/testing"], 2),
    }
    if final >= 25:
        out["score_mean_steps_21_to_25"] = round(sum(r["critic/score/mean"] for r in train[20:25]) / 5, 4)
    return out


def build_all(run: str) -> dict:
    spec = RUNS[run]
    ev_dir = LOGS_DIR / run
    steps = parse_head_log_steps(ev_dir / "head.log")
    timing = json.loads((ev_dir / "step_timing_breakdown.json").read_text(encoding="utf-8"))
    metrics_json = json.loads((ev_dir / spec["metrics"]).read_text(encoding="utf-8"))
    compared = check_head_log_against_metrics_json(steps, metrics_json)
    rollout_summary = json.loads((ev_dir / "rollout_and_audit_summary.json").read_text(encoding="utf-8"))
    ev = eval_summary(run, steps)
    out = {
        "head_log_vs_metrics_json_values_compared": compared,
        "training_table": training_table(run, steps, rollout_summary),
        "timing_table": timing_table(run, timing),
        "rollout_corr_table": rollout_corr_table(run, steps),
        "eval_table": eval_table(run, ev),
        "eval_summary": ev,
        "derived_summary": derived_summary(run, steps, timing),
    }
    if spec["final_step"] >= 25:
        out["askhat_table"] = askhat_table(run, steps, ev)
    return out


TABLES = ("training_table", "timing_table", "rollout_corr_table", "eval_table", "askhat_table")


DOCS = (LOGS_DIR.parent / "README.md", LOGS_DIR.parent / "exp" / "REPRO_RESULTS_GB200_ZH.md")


def write_docs() -> None:
    """Refresh every ``AUTO-GENERATED <run>/<table>`` block of the two reports in place (markers must exist)."""
    outs = {run: build_all(run) for run in RUNS}
    for doc in DOCS:
        text = doc.read_text(encoding="utf-8")
        for run, out in outs.items():
            for name in TABLES:
                if name not in out:
                    continue
                begin = f"<!-- BEGIN AUTO-GENERATED {run}/{name} (gen_g64_report_tables.py) -->"
                end = f"<!-- END AUTO-GENERATED {run}/{name} -->"
                i, j = text.find(begin), text.find(end)
                assert i >= 0 and j > i, f"{doc.name}: markers of {run}/{name} not found"
                text = text[:i] + f"{begin}\n{out[name]}\n" + text[j:]
        doc.write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run", choices=sorted(RUNS))
    parser.add_argument("--print-markdown", action="store_true", help="print the markdown tables of the run")
    parser.add_argument("--write-docs", action="store_true", help="refresh the AUTO-GENERATED blocks in both reports")
    args = parser.parse_args()
    if args.write_docs:
        write_docs()
        return
    if not args.run:
        parser.error("--run is required unless --write-docs is given")
    out = build_all(args.run)
    if args.print_markdown:
        for name in TABLES:
            if name in out:
                print(f"<!-- {args.run}/{name} -->\n{out[name]}\n")
    else:
        summary = {k: v for k, v in out.items() if not k.endswith("_table")}
        if args.run != "g64-retool-smoke-04":
            summary["resolved_config_diff_vs_g64-retool-smoke-04"] = resolved_config_diff(
                "g64-retool-smoke-04", args.run
            )
        print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
