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
"""Generate the markdown tables of the p3c-retool-25s-02 report from the archived evidence.

Usage::

    python3 report_tables.py                 # print every AUTO-GENERATED block
    python3 report_tables.py --check FILE.md # verify the blocks embedded in FILE.md are up to date
"""

import argparse
import json
import re
import statistics
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
RUN = "p3c-retool-25s-02"
FINAL = 25
DATASETS = (("aime_2024", "AIME 2024"), ("aime_2025", "AIME 2025"))


def load(name: str):
    return json.loads((HERE / name).read_text(encoding="utf-8"))


METRICS = load("metrics.json")
EVAL = load("eval_summary.json")
ROLLOUT = load("rollout_and_audit_summary.json")
NORMS = load("raiden_norm_checks.json")
REF = load("reference_step25.json")
TQDM = (HERE / "tqdm_progress.log").read_text(encoding="utf-8").splitlines()


def pct(x: float) -> str:
    return f"{100 * x:.2f}%"


def m(step: int, key: str) -> float:
    return float(METRICS[str(step)][key])


def eval_table() -> str:
    rows = [
        "| 数据集 | Step 0 正确数 / 900（mean@30） | Step 25 正确数 / 900（mean@30） | 变化 | 配对 bootstrap 95% CI | "
        "题目 升 / 降 / 平 | maj@30 Step 0 → 25 | best@30 Step 0 → 25 |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for ds, title in DATASETS:
        e = EVAL[ds]
        lo, hi = e["paired_bootstrap_95ci_pp"]
        rows.append(
            f"| {title} | `{e['correct_step0']}`（`{pct(e['mean_at_n_step0_logged'])}`） | "
            f"`{e['correct_final']}`（`{pct(e['mean_at_n_final_logged'])}`） | `{e['delta_pp']:+.2f}` pp | "
            f"`[{lo:+.2f}, {hi:+.2f}]` pp | "
            f"{e['questions_improved']} / {e['questions_worse']} / {e['questions_tied']} | "
            f"`{pct(e['maj_at_n_step0_logged'])}` → `{pct(e['maj_at_n_final_logged'])}` | "
            f"`{pct(e['best_at_n_step0_logged'])}` → `{pct(e['best_at_n_final_logged'])}` |"
        )
    return "\n".join(rows)


def train_table() -> str:
    rows = [
        "| Step | `score/mean` | `response_length/mean` | `clip_ratio` | `num_turns/mean` | `grad_norm` | "
        "`timing_s/step` | `gen` | `update_actor` | `update_weights` | 含 `<tool_response>` 轨迹 | 掩码审计违规 | "
        "Raiden 范数检查 |",
        "|" + " --- |" * 13,
    ]
    for s in range(1, FINAL + 1):
        r = ROLLOUT["steps"][str(s)]
        reps = NORMS.get(str(s), [])
        rows.append(
            f"| {s} | `{m(s, 'critic/score/mean'):+.4f}` | `{m(s, 'response_length/mean'):,.1f}` | "
            f"`{pct(m(s, 'response_length/clip_ratio'))}` | `{m(s, 'num_turns/mean'):.2f}` | "
            f"`{m(s, 'actor/grad_norm'):.4f}` | "
            f"`{m(s, 'timing_s/step'):.1f}` | `{m(s, 'timing_s/gen'):.1f}` | `{m(s, 'timing_s/update_actor'):.1f}` | "
            f"`{m(s, 'timing_s/update_weights'):.1f}` | "
            f"{r['trajectories_with_tool_response']} / {r['num_trajectories']} | "
            f"{r['audit_mask_violations']} / {r['audit_sessions']} | {len(reps)} / 3 |"
        )
    return "\n".join(rows)


def window_table() -> str:
    def w(key: str, lo: int, hi: int) -> float:
        return statistics.mean(m(s, key) for s in range(lo, hi + 1))

    rows = ["| 指标（5 步均值） | Steps 1–5 | Steps 21–25 |", "| --- | --- | --- |"]
    for key, label, fmt in (
        ("critic/score/mean", "训练 reward `critic/score/mean`", "{:+.3f}"),
        ("response_length/mean", "`response_length/mean`", "{:,.0f}"),
        ("num_turns/mean", "`num_turns/mean`", "{:.2f}"),
        ("response_length/clip_ratio", "`response_length/clip_ratio`", "{:.4f}"),
    ):
        rows.append(f"| {label} | `{fmt.format(w(key, 1, 5))}` | `{fmt.format(w(key, 21, 25))}` |")
    return "\n".join(rows)


def wall_list() -> str:
    last = next(line for line in reversed(TQDM) if f"{FINAL}/{FINAL}" in line)
    elapsed = re.search(r"\[(\d+):(\d+):(\d+)<", last)
    assert elapsed is not None, last
    h, mi, se = (int(x) for x in elapsed.groups())
    secs = 3600 * h + 60 * mi + se
    test_s = m(FINAL, "timing_s/testing")
    train_s = secs - test_s
    steps = [m(s, "timing_s/step") for s in range(2, FINAL + 1)]
    t = ROLLOUT["totals"]
    norm_ok = sum(NORMS.get(str(s)) == [0, 1, 2] for s in range(1, FINAL + 1))
    return "\n".join(
        [
            f"- tqdm 记录的 `25/25` 总耗时 `{h}:{mi:02d}:{se:02d}`（`{secs:,}` s，"
            f"含 Step 25 `n=30` 验证 `timing_s/testing = {test_s:,.1f}` s，"
            f"不含 Step 0 验证）；扣除 Step 25 验证后的 25 步纯训练墙钟耗时为 `{train_s:,.1f}` s"
            f"（平均 `{train_s / FINAL:.1f}` s/步）。",
            f"- Steps 2–25 的 `timing_s/step` 均值 `{statistics.mean(steps):.1f}` s"
            f"（范围 `{min(steps):.1f}`–`{max(steps):.1f}` s）。"
            "V1 异步模式下偶数步承担生成、奇数步的 `gen` 接近 0，所以要看两步平均或 tqdm 总耗时。",
            f"- 训练轨迹 `{t['num_trajectories']:,}` 条，"
            f"其中含 `<tool_response>` 的 `{t['trajectories_with_tool_response']:,}` 条"
            f"（`{pct(t['trajectories_with_tool_response'] / t['num_trajectories'])}`），"
            f"工具返回共 `{t['total_tool_responses']:,}` 次，"
            f"`score <= -1.99` 的轨迹 `{t['clip_neg2_count']}` 条。",
            f"- 掩码审计 `{t['audit_sessions']}` 个会话，违规 `{t['audit_mask_violations']}` 个"
            f"（`<tool_call>` 段 `{t['audit_tool_call_segments']}` 个，"
            f"`<tool_response>` 段 `{t['audit_tool_response_segments']}` 个）。",
            f"- Raiden 训练后分布式范数检查：`{norm_ok}` / `{FINAL}` 步三个副本全部通过。",
        ]
    )


def compare_table() -> str:
    g, a = REF["gpu_g64"], REF["askhat_v5p"]
    rows = [
        "| Run | 硬件 | 每题采样 `n` | Step 0 AIME 2024 / 2025 | "
        "Step 25 AIME 2024 / 2025 | 训练 reward steps 1–5 / 21–25 |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    ours0 = " / ".join(f"`{pct(EVAL[ds]['mean_at_n_step0_logged'])}`" for ds, _ in DATASETS)
    ours25 = " / ".join(f"`{pct(EVAL[ds]['mean_at_n_final_logged'])}`" for ds, _ in DATASETS)
    r15 = statistics.mean(m(s, "critic/score/mean") for s in range(1, 6))
    r2125 = statistics.mean(m(s, "critic/score/mean") for s in range(21, 26))
    rows.append(
        f"| **我们 TPU `{RUN}`** | 64× TPU7x（trainer 16 + rollout 48） | 30 | "
        f"{ours0} | {ours25} | `{r15:+.3f}` / `{r2125:+.3f}` |"
    )
    g0 = " / ".join(f"`{pct(g['step0'][ds]['mean'])}`" for ds, _ in DATASETS)
    g25 = " / ".join(f"`{pct(g['step25'][ds]['mean'])}`" for ds, _ in DATASETS)
    rows.append(
        f"| 我们 GPU `{g['run']}` | 64× GB200（trainer 16 + rollout 48） | 30 | {g0} | {g25} | "
        f"`{g['train_score_mean_steps_1_5']:+.3f}` / `{g['train_score_mean_steps_21_25']:+.3f}` |"
    )
    for line, v in a["lines"].items():
        a25 = " / ".join(f"`{pct(v['step25'][ds]['mean'])}`" for ds, _ in DATASETS)
        hw = "48× v5p（trainer 16 + rollout 32" + ("；前 20 步 rollout 16）" if line == "cp2" else "）")
        rows.append(
            f"| Askhat v5p `{line}`（`{v['step25_run']}`） | {hw} | 8 | –（未评估） | {a25} | "
            f"`{v['train_score_mean_steps_1_5']:+.3f}` / `{v['train_score_mean_steps_21_25']:+.3f}` |"
        )
    return "\n".join(rows)


TABLES = {"eval": eval_table, "wall": wall_list, "window": window_table, "compare": compare_table, "train": train_table}


def block(name: str) -> str:
    begin = f"<!-- BEGIN AUTO-GENERATED {RUN}/{name} (report_tables.py) -->"
    return f"{begin}\n{TABLES[name]()}\n<!-- END AUTO-GENERATED {RUN}/{name} -->"


def check(doc: Path) -> int:
    text = doc.read_text(encoding="utf-8")
    bad = 0
    for name in TABLES:
        b = block(name)
        begin, end = b.splitlines()[0], b.splitlines()[-1]
        i, j = text.find(begin), text.find(end)
        if i < 0 or j < 0 or text[i : j + len(end)] != b:
            print(f"{doc.name}: block {name} is missing or stale")
            bad += 1
    print("all blocks match" if not bad else f"{bad} stale block(s)")
    return 1 if bad else 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--check", type=Path)
    args = parser.parse_args()
    if args.check:
        sys.exit(check(args.check))
    for name in TABLES:
        print(block(name) + "\n")


if __name__ == "__main__":
    main()
