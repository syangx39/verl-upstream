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
    python3 report_tables.py --write FILE.md # refresh those blocks in place
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
TIMING = load("step_timing_breakdown.json")
EXACT = load("env/raiden_exact_parity_step_0.json")
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
    steps = [m(s, "timing_s/step") for s in range(2, FINAL + 1)]
    gens = {s: m(s, "timing_s/gen") for s in range(2, FINAL)}
    odd = [g for s, g in gens.items() if s % 2]
    even = [g for s, g in gens.items() if not s % 2]
    ts, rows = TIMING["summary"], {r["step"]: r for r in TIMING["rows"]}
    f = rows[FINAL]
    t = ROLLOUT["totals"]
    norm_ok = sum(NORMS.get(str(s)) == [0, 1, 2] for s in range(1, FINAL + 1))
    exact_tensors = sum(r["checked"] for r in EXACT["ranks"])
    return "\n".join(
        [
            f"- Steps 2–25 的 `timing_s/step` 均值 `{statistics.mean(steps):.1f}` s"
            f"（范围 `{min(steps):.1f}`–`{max(steps):.1f}` s；不含 Step 25 时 Steps 2–24 为 "
            f"`{ts['mean_timing_s_step']:.1f}` s）。其中 `timing_s/gen`（在 `step` 计时内等 rollout 样本）"
            f"Steps 2–24 均值 `{ts['mean_timing_s_gen']:.1f}` s：奇数步均值 `{statistics.mean(odd):.2f}` s，"
            f"偶数步均值 `{statistics.mean(even):.1f}` s，即 trainer 每两步要等 rollout 一次。",
            f"- 端到端墙钟（相邻两步 TensorBoard 记录起点之差，与 W&B `_timestamp` 一致）Steps 2–24 均值 "
            f"`{ts['mean_wall_s']:.1f}` s = `timing_s/step` `{ts['mean_timing_s_step']:.1f}` s + 上一步 TensorBoard "
            f"记录 `{ts['mean_tb_span_prev_step_s']:.1f}` s（约 `{ts['mean_tb_scalars_prev_step']:.0f}` 个 scalar，"
            f"每个约 `{ts['mean_tb_span_prev_step_s'] / (ts['mean_tb_scalars_prev_step'] - 1):.2f}` s）+ 其余 "
            f"`{ts['mean_residual_s']:.1f}` s（driver log 中同步结束到 `logger.log` 为 "
            f"`{ts['mean_post_sync_to_log_s']:.1f}` s）。",
            f"- Step 25 墙钟 `{f['wall_s']:,.1f}` s = `timing_s/step` `{f['timing_s_step']:.1f}` s + "
            "Step 25 `n=30` 验证 "
            f"`{f['timing_s_testing']:,.1f}` s + 上一步 TensorBoard 记录 `{f['tb_span_prev_step_s']:.1f}` s + 其余 "
            f"`{f['residual_s']:.1f}` s。tqdm 的 `25/25` 总耗时 `{h}:{mi:02d}:{se:02d}`（`{secs:,}` s）在每步 "
            "TensorBoard 记录之后才更新，所以包含 Step 25 验证和全部 25 步的 TensorBoard 记录，不是纯训练时间。",
            f"- 训练轨迹 `{t['num_trajectories']:,}` 条，"
            f"其中含 `<tool_response>` 的 `{t['trajectories_with_tool_response']:,}` 条"
            f"（`{pct(t['trajectories_with_tool_response'] / t['num_trajectories'])}`），"
            f"工具返回共 `{t['total_tool_responses']:,}` 次，"
            f"`score <= -1.99` 的轨迹 `{t['clip_neg2_count']}` 条。",
            f"- 整段掩码审计（[`rollout_mask_audit.py`](../../rollout_mask_audit.py)，从归档的原始 "
            f"[`rollout_audit/`](../../logs/tpu7x/{RUN}/rollout_audit/) 重算）：`{t['audit_sessions']}` 个会话违规 "
            f"`{t['audit_mask_violations']}` 个；`{t['audit_tool_response_blocks']}` 个 `<tool_response>` 块共 "
            f"`{t['audit_tool_response_tokens']:,}` 个 token，"
            f"计入 loss 的 `{t['audit_tool_response_tokens_in_loss']}` 个；"
            f"`{t['audit_mask0_runs']}` 段 `mask=0` 全部恰好是注入的工具轮次（不符的 "
            f"`{t['audit_mask0_runs_not_tool_turn']}` 段）；`{t['audit_tool_call_blocks']}` 个完整 `<tool_call>` 块"
            f"不计入 loss 的 token `{t['audit_tool_call_tokens_not_in_loss']}` 个。",
            f"- Raiden：Step 0 三副本 `{EXACT['num_results']}` 个 rank、`{exact_tensors:,}` 个张量分片逐元素比对，"
            f"不一致 `{EXACT['num_mismatched']}`；Steps 1–25 每步同步后的分布式范数检查 `{norm_ok}` / `{FINAL}` 步"
            "三个副本全部通过（自动判定只看 `numel` 和 `L1`，是范数级检查，不等于逐元素一致）。",
        ]
    )


def timing_table() -> str:
    rows = [
        "| Step | 墙钟（TensorBoard 记录起点间隔） | `timing_s/step` | 其中 `gen` | 其中 `update_actor` | "
        "其中 `update_weights` | `timing_s/testing` | 上一步 TensorBoard 记录（scalar 数） | 计时外其余 | "
        "同步结束 → `logger.log` |",
        "|" + " --- |" * 10,
    ]
    for r in TIMING["rows"]:
        rows.append(
            f"| {r['step']} | `{r['wall_s']:.1f}` | `{r['timing_s_step']:.1f}` | `{r['timing_s_gen']:.1f}` | "
            f"`{r['timing_s_update_actor']:.1f}` | `{r['timing_s_update_weights']:.1f}` | "
            f"`{r['timing_s_testing']:.1f}` | `{r['tb_span_prev_step_s']:.1f}`（{r['tb_scalars_prev_step']}） | "
            f"`{r['residual_s']:.1f}` | `{r['post_sync_to_log_s']:.1f}` |"
        )
    s = TIMING["summary"]
    rows.append(
        f"| Steps {s['steps'][0]}–{s['steps'][1]} 均值（{s['n']} 步） | `{s['mean_wall_s']:.1f}` | "
        f"`{s['mean_timing_s_step']:.1f}` | `{s['mean_timing_s_gen']:.1f}` | `{s['mean_timing_s_update_actor']:.1f}` | "
        f"`{s['mean_timing_s_update_weights']:.1f}` | `{s['mean_timing_s_testing']:.1f}` | "
        f"`{s['mean_tb_span_prev_step_s']:.1f}`（{s['mean_tb_scalars_prev_step']:.0f}） | "
        f"`{s['mean_residual_s']:.1f}` | "
        f"`{s['mean_post_sync_to_log_s']:.1f}` |"
    )
    return "\n".join(rows)


def perf_compare_table() -> str:
    g, gs, ts = REF["gpu_g64"]["timing"], REF["gpu_g64"]["timing"]["steps_2_24"], TIMING["summary"]
    tpu_25 = statistics.mean(m(s, "timing_s/step") for s in range(2, FINAL + 1))
    tok = statistics.mean(m(s, "perf/total_num_tokens") for s in range(2, FINAL))
    per_tok = statistics.mean(m(s, "timing_per_token_ms/update_actor") for s in range(2, FINAL))

    def ratio(a: float, b: float) -> str:
        return f"`{b / a:.1f}×`"

    rows = [
        f"| 指标（秒/步；除首行外为 Steps 2–24 均值） | GPU `{REF['gpu_g64']['run']}` | TPU `{RUN}` | TPU / GPU |",
        "| --- | --- | --- | --- |",
        f"| `timing_s/step`，Steps 2–25 均值 | `{g['timing_s_step_mean_steps_2_25']:.2f}` | `{tpu_25:.2f}` | "
        f"{ratio(g['timing_s_step_mean_steps_2_25'], tpu_25)} |",
        f"| `timing_s/step` | `{gs['mean_timing_s_step']:.2f}` | `{ts['mean_timing_s_step']:.2f}` | "
        f"{ratio(gs['mean_timing_s_step'], ts['mean_timing_s_step'])} |",
        f"| └ `gen`（`step` 计时内等 rollout） | `{gs['mean_timing_s_gen']:.2f}` | "
        f"`{ts['mean_timing_s_gen']:.2f}` | – |",
        f"| └ `update_actor` | `{gs['mean_timing_s_update_actor']:.2f}` | `{ts['mean_timing_s_update_actor']:.2f}` | "
        f"{ratio(gs['mean_timing_s_update_actor'], ts['mean_timing_s_update_actor'])} |",
        f"| └ `update_weights` | `{gs['mean_timing_s_update_weights']:.2f}` | "
        f"`{ts['mean_timing_s_update_weights']:.2f}` | "
        f"{ratio(gs['mean_timing_s_update_weights'], ts['mean_timing_s_update_weights'])} |",
        f"| 端到端墙钟 | `{gs['mean_wall_s']:.2f}` | `{ts['mean_wall_s']:.2f}` | "
        f"{ratio(gs['mean_wall_s'], ts['mean_wall_s'])} |",
        f"| └ 计时外：上一步 TensorBoard 记录 | `{gs['mean_tb_span_prev_step_s']:.2f}` | "
        f"`{ts['mean_tb_span_prev_step_s']:.2f}` | – |",
        f"| └ 计时外：其余（指标、dump 准备、`kv_clear`） | `{gs['mean_residual_s']:.2f}` | "
        f"`{ts['mean_residual_s']:.2f}` | – |",
        f"| 每步 token 数（`perf/total_num_tokens`） | `{g['mean_perf_total_num_tokens_steps_2_24'] / 1e6:.2f}` M | "
        f"`{tok / 1e6:.2f}` M | – |",
        f"| `timing_per_token_ms/update_actor` | `{g['mean_timing_per_token_ms_update_actor_steps_2_24']:.4f}` | "
        f"`{per_tok:.4f}` | {ratio(g['mean_timing_per_token_ms_update_actor_steps_2_24'], per_tok)} |",
    ]
    return "\n".join(rows)


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


TABLES = {
    "eval": eval_table,
    "wall": wall_list,
    "timing": timing_table,
    "perf": perf_compare_table,
    "window": window_table,
    "compare": compare_table,
    "train": train_table,
}


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


def write(doc: Path) -> int:
    """Replace every AUTO-GENERATED block of ``doc`` (the BEGIN / END markers must already be there)."""
    text = doc.read_text(encoding="utf-8")
    for name in TABLES:
        b = block(name)
        begin, end = b.splitlines()[0], b.splitlines()[-1]
        i, j = text.find(begin), text.find(end)
        if i < 0 or j < 0:
            print(f"{doc.name}: markers of block {name} not found")
            return 1
        text = text[:i] + b + text[j + len(end) :]
    doc.write_text(text, encoding="utf-8")
    return check(doc)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--check", type=Path)
    parser.add_argument("--write", type=Path, help="refresh the AUTO-GENERATED blocks of FILE.md in place")
    args = parser.parse_args()
    if args.check:
        sys.exit(check(args.check))
    if args.write:
        sys.exit(write(args.write))
    for name in TABLES:
        print(block(name) + "\n")


if __name__ == "__main__":
    main()
