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
"""Generate the Stage G64-D (``g64-retool-smoke-04``) report tables from the archived raw artifacts.

Every number in the generated tables is read from files in this directory; nothing is typed by hand:

* per-step training / timing / rollout-correction metrics: the ``step:N - key:value - ...`` lines printed by
  ``TaskRunnerV1`` in ``head.log`` (cross-checked against ``g64_smoke_04_metrics.json``);
* per-step tool-call, ``score <= -2.0`` and token-mask audit counts: ``rollout_and_audit_summary.json``;
* Step 0 / Step 5 AIME accuracy, per-question deltas and the paired bootstrap interval:
  ``validation_step{0,5}_samples.jsonl``.

The markdown blocks in ``gb200/README.md`` and ``gb200/exp/REPRO_RESULTS_GB200_ZH.md`` between the
``AUTO-GENERATED`` markers are pasted verbatim from ``--print-markdown`` output and checked by
``test_launcher_and_sandbox.py``.

Usage::

    python3 examples/tpu/retool/gb200/logs/g64-retool-smoke-04/gen_g64_report_tables.py --print-markdown
"""

import argparse
import json
import math
import random
import re
from collections import defaultdict
from pathlib import Path

EV_DIR = Path(__file__).resolve().parent
STEP_LINE_RE = re.compile(r"\bstep:(\d+) - ")
NP_SCALAR_RE = re.compile(r"^np\.\w+\((.*)\)$")
TRAIN_STEPS = (1, 2, 3, 4, 5)
TIMING_WINDOW = (2, 3, 4, 5)
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


def check_head_log_against_metrics_json(steps: dict[int, dict], metrics_json: dict) -> int:
    """Assert every key present in both sources has the same value; return the number of compared values."""
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
    step0: dict[str, list[dict]], step5: dict[str, list[dict]], reps: int = BOOTSTRAP_REPS, seed: int = BOOTSTRAP_SEED
) -> tuple[float, float]:
    """Two-level paired bootstrap of delta mean@30 (resample questions, then the 30 samples of each step).

    The interval covers evaluation sampling noise on this fixed question set only; it says nothing about the
    run-to-run variance of RL training itself.
    """
    rng = random.Random(seed)
    qids = sorted(step0)
    acc0 = {q: [1.0 if s["acc"] else 0.0 for s in step0[q]] for q in qids}
    acc5 = {q: [1.0 if s["acc"] else 0.0 for s in step5[q]] for q in qids}
    deltas = []
    for _ in range(reps):
        total = 0.0
        for _ in range(len(qids)):
            q = qids[rng.randrange(len(qids))]
            a0, a5 = acc0[q], acc5[q]
            m0 = sum(a0[rng.randrange(len(a0))] for _ in range(len(a0))) / len(a0)
            m5 = sum(a5[rng.randrange(len(a5))] for _ in range(len(a5))) / len(a5)
            total += m5 - m0
        deltas.append(total / len(qids))
    deltas.sort()
    return deltas[int(0.025 * reps)], deltas[int(0.975 * reps) - 1]


def eval_summary(val0_path: Path, val5_path: Path, steps: dict[int, dict]) -> dict[str, dict]:
    v0, v5 = load_validation(val0_path), load_validation(val5_path)
    out = {}
    for ds in ("aime_2024", "aime_2025"):
        q0, q5 = v0[ds], v5[ds]
        assert sorted(q0) == sorted(q5) and len(q0) == 30, f"{ds}: question sets differ or != 30"
        correct0 = sum(s["acc"] for q in q0 for s in q0[q])
        correct5 = sum(s["acc"] for q in q5 for s in q5[q])
        per_q = [(sum(s["acc"] for s in q5[q]) - sum(s["acc"] for s in q0[q])) for q in sorted(q0)]
        lo, hi = paired_bootstrap_delta(q0, q5)
        m0, m5 = steps[0]["metrics"], steps[5]["metrics"]
        out[ds] = {
            "correct_step0": correct0,
            "correct_step5": correct5,
            "samples": 30 * len(q0),
            "mean_at_30_step0": m0[f"val-core/{ds}/acc/mean@30"],
            "mean_at_30_step5": m5[f"val-core/{ds}/acc/mean@30"],
            "maj_at_30_step0": m0[f"val-core/{ds}/acc/maj@30/mean"],
            "maj_at_30_step5": m5[f"val-core/{ds}/acc/maj@30/mean"],
            "best_at_30_step0": m0[f"val-core/{ds}/acc/best@30/mean"],
            "best_at_30_step5": m5[f"val-core/{ds}/acc/best@30/mean"],
            "shaped_reward_mean_at_30_step0": m0[f"val-aux/{ds}/reward/mean@30"],
            "shaped_reward_mean_at_30_step5": m5[f"val-aux/{ds}/reward/mean@30"],
            "questions_improved": sum(d > 0 for d in per_q),
            "questions_worse": sum(d < 0 for d in per_q),
            "questions_tied": sum(d == 0 for d in per_q),
            "delta_mean_at_30_bootstrap_95ci": (lo, hi),
        }
    return out


def _pct(x: float, digits: int = 2) -> str:
    return f"{100.0 * x:.{digits}f}%"


def _pp(x: float) -> str:
    return f"{100.0 * x:+.2f} pp"


def training_table(steps: dict[int, dict], rollout_summary: dict) -> str:
    header = (
        "| Step (`head.log` line) | `timing_s/step` | `timing_s/gen` | `timing_s/update_actor` | "
        "`timing_s/update_weights` | `perf/throughput` (tok/s/GPU) | `critic/score/mean` | `actor/pg_loss` | "
        "`actor/grad_norm` | `response_length/mean` (`max`) | `response_length/clip_ratio` | "
        "`num_turns/mean` (`max`) | trajectories with `<tool_response>` (calls) | `score <= -2.0` | "
        "audit mask violations |"
    )
    rows = [header, "|" + " --- |" * 15]
    for s in TRAIN_STEPS:
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
    w = [steps[s]["metrics"] for s in TIMING_WINDOW]
    t = rollout_summary["totals"]

    def mean(key: str) -> float:
        return sum(x[key] for x in w) / len(w)

    rows.append(
        f"| Steps 2–5 mean (4 steps only) | `{mean('timing_s/step'):.2f}s` "
        f"(`{min(x['timing_s/step'] for x in w):.2f}s`–`{max(x['timing_s/step'] for x in w):.2f}s`) | "
        f"`{mean('timing_s/gen'):.3f}s` | `{mean('timing_s/update_actor'):.2f}s` | "
        f"`{mean('timing_s/update_weights'):.2f}s` | `{mean('perf/throughput'):.1f}` | – | – | – | – | – | – | "
        f"total `{t['trajectories_with_tool_response']} / {t['total_trajectories']}` "
        f"(`{t['total_tool_responses']:,}`) | total `{t['total_clip_neg2_count']}` | "
        f"total `{t['audit_mask_violations']} / {t['total_audit_sessions']}` |"
    )
    return "\n".join(rows)


def rollout_corr_table(steps: dict[int, dict]) -> str:
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
    for s in TRAIN_STEPS:
        m = steps[s]["metrics"]
        rows.append(f"| {s} | " + " | ".join(f"`{m['actor/rollout_corr/' + k]:{fmt}}`" for k, fmt in keys) + " |")
    return "\n".join(rows)


def eval_table(ev: dict[str, dict]) -> str:
    rows = [
        "| Metric (`n=30` samples per question, 30 questions) | Step 0 | Step 5 | Δ (Step 5 − Step 0) |",
        "| --- | --- | --- | --- |",
    ]
    for ds, label in (("aime_2024", "AIME 2024"), ("aime_2025", "AIME 2025")):
        e = ev[ds]
        lo, hi = e["delta_mean_at_30_bootstrap_95ci"]
        rows += [
            f"| {label} correct samples | `{e['correct_step0']} / {e['samples']}` | "
            f"`{e['correct_step5']} / {e['samples']}` | `{e['correct_step5'] - e['correct_step0']:+d}` |",
            f"| {label} `acc/mean@30` | `{_pct(e['mean_at_30_step0'])}` | `{_pct(e['mean_at_30_step5'])}` | "
            f"`{_pp(e['mean_at_30_step5'] - e['mean_at_30_step0'])}` (paired bootstrap 95% CI "
            f"`[{_pp(lo)}, {_pp(hi)}]`) |",
            f"| {label} `acc/maj@30/mean` | `{_pct(e['maj_at_30_step0'])}` | `{_pct(e['maj_at_30_step5'])}` | "
            f"`{_pp(e['maj_at_30_step5'] - e['maj_at_30_step0'])}` |",
            f"| {label} `acc/best@30/mean` | `{_pct(e['best_at_30_step0'])}` | `{_pct(e['best_at_30_step5'])}` | "
            f"`{_pp(e['best_at_30_step5'] - e['best_at_30_step0'])}` |",
            f"| {label} shaped `reward/mean@30` | `{e['shaped_reward_mean_at_30_step0']:+.4f}` | "
            f"`{e['shaped_reward_mean_at_30_step5']:+.4f}` | "
            f"`{e['shaped_reward_mean_at_30_step5'] - e['shaped_reward_mean_at_30_step0']:+.4f}` |",
            f"| {label} questions improved / worse / tied (per-question correct count) | – | – | "
            f"`{e['questions_improved']} / {e['questions_worse']} / {e['questions_tied']}` |",
        ]
    return "\n".join(rows)


def derived_summary(steps: dict[int, dict]) -> dict:
    """Scalar facts quoted in the report prose, all derived from the head.log step lines."""
    train = [steps[s]["metrics"] for s in TRAIN_STEPS]
    window = [steps[s]["metrics"] for s in TIMING_WINDOW]

    def mean(rows: list[dict], key: str) -> float:
        return sum(r[key] for r in rows) / len(rows)

    return {
        "timing_s_step_step_1": round(train[0]["timing_s/step"], 2),
        "timing_s_step_sum_steps_1_to_5": round(sum(r["timing_s/step"] for r in train), 2),
        "timing_s_step_steps_2_to_5": {
            "mean": round(mean(window, "timing_s/step"), 2),
            "min": round(min(r["timing_s/step"] for r in window), 2),
            "max": round(max(r["timing_s/step"] for r in window), 2),
        },
        "timing_s_gen_mean_steps_2_to_5": round(mean(window, "timing_s/gen"), 3),
        "timing_s_update_actor_mean_steps_2_to_5": round(mean(window, "timing_s/update_actor"), 2),
        "timing_s_update_weights_per_step_1_to_5": [round(r["timing_s/update_weights"], 2) for r in train],
        "perf_throughput_mean_steps_2_to_5": round(mean(window, "perf/throughput"), 1),
        "perf_total_num_tokens_sum_steps_1_to_5": int(sum(r["perf/total_num_tokens"] for r in train)),
        "response_length_mean_per_step_1_to_5": [round(r["response_length/mean"], 1) for r in train],
        "num_turns_mean_per_step_1_to_5": [round(r["num_turns/mean"], 2) for r in train],
        "actor_max_memory_allocated_gb_max": round(max(r["actor/perf/max_memory_allocated_gb"] for r in train), 2),
        "actor_max_memory_reserved_gb_max": round(max(r["actor/perf/max_memory_reserved_gb"] for r in train), 2),
        "step5_timing_s_testing": round(steps[5]["metrics"]["timing_s/testing"], 2),
    }


def build_all() -> dict:
    steps = parse_head_log_steps(EV_DIR / "head.log")
    metrics_json = json.loads((EV_DIR / "g64_smoke_04_metrics.json").read_text(encoding="utf-8"))
    compared = check_head_log_against_metrics_json(steps, metrics_json)
    rollout_summary = json.loads((EV_DIR / "rollout_and_audit_summary.json").read_text(encoding="utf-8"))
    ev = eval_summary(EV_DIR / "validation_step0_samples.jsonl", EV_DIR / "validation_step5_samples.jsonl", steps)
    return {
        "head_log_vs_metrics_json_values_compared": compared,
        "training_table": training_table(steps, rollout_summary),
        "rollout_corr_table": rollout_corr_table(steps),
        "eval_table": eval_table(ev),
        "eval_summary": ev,
        "derived_summary": derived_summary(steps),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--print-markdown", action="store_true", help="print the three markdown tables")
    args = parser.parse_args()
    out = build_all()
    if args.print_markdown:
        for name in ("training_table", "rollout_corr_table", "eval_table"):
            print(f"<!-- {name} -->\n{out[name]}\n")
    else:
        print(json.dumps({k: v for k, v in out.items() if not k.endswith("_table")}, indent=2))


if __name__ == "__main__":
    main()
