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
"""Plot and tabulate Askhat's TPU v5p ReTool runs from ``askhat_v5p_runs.json``.

Writes the figures ``fig*.png`` next to this script and prints the markdown tables that ``README.md`` embeds
between ``AUTO-GENERATED`` markers. ``--check README.md`` verifies those blocks match the data.

Usage::

    python3 plot_askhat_v5p.py --figures          # needs matplotlib
    python3 plot_askhat_v5p.py --print-markdown
    python3 plot_askhat_v5p.py --check README.md
"""

import argparse
import json
import re
import statistics
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATA = json.loads((HERE / "askhat_v5p_runs.json").read_text(encoding="utf-8"))
OURS = json.loads((HERE / "ours_g64_retool_25s_01.json").read_text(encoding="utf-8"))
RUNS = {r["run_id"]: r for r in DATA["runs"]}
EXCLUDED = {"yjowd03u"}  # Step 0 only; excluded on request.
BASE_EVAL = "4fl4zl1d"
WANDB_URL = "https://wandb.ai/" + DATA["wandb_project"] + "/runs/{}"

# Training lines: list of (run_id, first_step, last_step); a later segment resumed from a checkpoint.
LINES = {
    "cp2": {"segments": [("fcze3xlx", 1, 20), ("ci0k54pl", 21, 150)], "color": "tab:blue"},
    "cp1-opt": {"segments": [("rq1p980n", 1, 80), ("hsodvfgr", 81, 150)], "color": "tab:orange"},
    "cp1-pr115": {"segments": [("yrcsp4z6", 1, 117)], "color": "tab:green"},
    "cp1-pr115-async": {"segments": [("zyrnkte8", 1, 150)], "color": "tab:red"},
}
DATASETS = (("aime_2024", "AIME 2024"), ("aime_2025", "AIME 2025"))
NOTES = {
    "ci0k54pl": "从 `fcze3xlx` 的 step 20 checkpoint 恢复（见下文“恢复关系”）",
    "hsodvfgr": "从 `rq1p980n` 的 step 80 checkpoint 恢复（见下文“恢复关系”）",
    "myaf5pal": "RL 的 `model_path` 与这个 SFT 实验同名（按名称对应，未核对 checkpoint 哈希）",
    "mtuqkfsy": "同名 SFT 的第一次尝试，39 步后 crashed",
    BASE_EVAL: "原始 `Qwen3-30B-A3B`（无 SFT），`val_only`，`n=8`",
    "fdx3ro0k": "base 模型评估的第一次尝试，无指标",
    "5lo8ruji": "只记录了 step 1",
}


STEP_EX_SAVE = "timing_s/step_minus_save_checkpoint"  # derived: timing_s/step includes the checkpoint save.


def step_of(row: dict) -> int:
    return int(row.get("training/global_step", row.get("_step", 0)))


def value(row: dict, key: str):
    if key == STEP_EX_SAVE:
        return None if "timing_s/step" not in row else row["timing_s/step"] - row.get("timing_s/save_checkpoint", 0.0)
    return row.get(key)


def series(line: str, key: str) -> list[tuple[int, float]]:
    pts = []
    for run_id, lo, hi in LINES[line]["segments"]:
        for row in RUNS[run_id]["history"]:
            s = step_of(row)
            v = value(row, key)
            if lo <= s <= hi and v is not None:
                pts.append((s, float(v)))
    return sorted(pts)


def evals(line: str) -> dict[int, dict]:
    out: dict[int, dict] = {}
    for run_id, lo, hi in LINES[line]["segments"]:
        for row in RUNS[run_id]["history"]:
            s = step_of(row)
            if lo <= s <= hi and any(k.startswith("val-core/") for k in row):
                out[s] = {**row, "_run": run_id}
    return dict(sorted(out.items()))


def base_eval() -> dict:
    return next(r for r in RUNS[BASE_EVAL]["history"] if any(k.startswith("val-core/") for k in r))


def pct(x) -> str:
    return "–" if x is None else f"{100 * x:.2f}%"


def window_mean(pts: list[tuple[int, float]], lo: int, hi: int):
    vals = [v for s, v in pts if lo <= s <= hi]
    return statistics.mean(vals) if len(vals) == hi - lo + 1 else None


def role(run: dict) -> str:
    name = run["display_name"]
    if "sweep-sft" in name:
        return "SFT 吞吐 sweep"
    if "multiturn-sft" in name:
        return "SFT"
    if "eval-base" in name:
        return "base 模型评估"
    return "RL"


def seg_values(line: str, fn) -> str:
    """Per-segment values in chronological order; consecutive duplicates collapsed, changes shown with →."""
    vals: list[str] = []
    for run_id, _, _ in LINES[line]["segments"]:
        v = json.dumps(fn(RUNS[run_id]["config"]))
        if not vals or vals[-1] != v:
            vals.append(v)
    return " → ".join(f"`{json.loads(v)}`" for v in vals)


def inventory_table() -> str:
    rows = [
        "| W&B run | 名称 | 类型 | 状态 | 记录的步数 | 创建时间 (UTC) | 备注 |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for run in DATA["runs"]:
        if run["run_id"] in EXCLUDED:
            continue
        steps = [step_of(r) for r in run["history"]]
        span = f"{min(steps)}–{max(steps)}" if steps else "–"
        rows.append(
            f"| [`{run['run_id']}`]({WANDB_URL.format(run['run_id'])}) | `{run['display_name']}` | {role(run)} | "
            f"{run['state']} | {span} | {run['created_at'][:16].replace('T', ' ')} | {NOTES.get(run['run_id'], '')} |"
        )
    return "\n".join(rows)


def config_table() -> str:
    cols = (
        ("trainer 芯片", lambda c: c["trainer_nnodes"] * c["trainer_n_gpus_per_node"]),
        ("rollout 芯片", lambda c: c["rollout_nnodes"] * c["rollout_n_gpus_per_node"]),
        ("trainer CP", lambda c: c["trainer_cp"]),
        ("trainer DP shard", lambda c: c["trainer_dp_shard"]),
        ("每芯片 token 上限", lambda c: c["ppo_max_token_len_per_gpu"]),
        ("rollout TP", lambda c: c["rollout_tp"]),
        ("rollout EP", lambda c: c["rollout_ep"]),
        ("rollout DP", lambda c: c["rollout_dp"]),
        ("max_num_seqs", lambda c: c["rollout_max_num_seqs"]),
        ("异步 ckpt 保存", lambda c: c["ckpt_async_save"]),
        ("Ray 代码包", lambda c: re.search(r"_ray_pkg_([0-9a-f]+)", c["ray_pythonpath"]).group(1)[:8]),
    )
    rows = ["| 线 | W&B runs | " + " | ".join(label for label, _ in cols) + " |", "|" + " --- |" * (len(cols) + 2)]
    for line, spec in LINES.items():
        run_ids = " → ".join(f"`{s[0]}`" for s in spec["segments"])
        rows.append(f"| `{line}` | {run_ids} | " + " | ".join(seg_values(line, fn) for _, fn in cols) + " |")
    return "\n".join(rows)


SHARED_KEYS = (
    ("model_path", "SFT 起点 `actor_rollout_ref.model.path`"),
    ("train_files", "训练集 `data.train_files`"),
    ("val_files", "验证集 `data.val_files`"),
    ("enable_thinking", "`enable_thinking`"),
    ("max_prompt_length", "`max_prompt_length`"),
    ("max_response_length", "`max_response_length`"),
    ("max_assistant_turns", "`max_assistant_turns`"),
    ("max_user_turns", "`max_user_turns`"),
    ("custom_reward_path", "reward 函数"),
    ("train_batch_size", "`train_batch_size`（prompts）"),
    ("ppo_mini_batch_size", "`ppo_mini_batch_size`"),
    ("rollout_n", "每个 prompt 的 rollout 数 `n`"),
    ("rollout_temperature", "训练 rollout temperature"),
    ("val_n", "验证每题采样数 `val_kwargs.n`"),
    ("val_temperature", "验证 temperature"),
    ("val_top_p", "验证 top_p"),
    ("actor_lr", "actor lr"),
    ("clip_grad", "`optim.clip_grad`"),
    ("loss_mode", "`policy_loss.loss_mode`"),
    ("bypass_mode", "`rollout_correction.bypass_mode`"),
    ("rollout_is", "`rollout_correction.rollout_is`"),
    ("use_v1", "`trainer.use_v1`"),
    ("num_warmup_batches", "`v1.separate_async.num_warmup_batches`"),
    ("max_off_policy_threshold", "`v1.sampler.max_off_policy_threshold`"),
    ("total_training_steps", "`total_training_steps`"),
    ("test_freq", "`test_freq`"),
    ("save_freq", "`save_freq`"),
    ("val_before_train", "`val_before_train`"),
    ("resume_mode", "`resume_mode`"),
)


def shared_table() -> str:
    rl = [RUNS[s[0]]["config"] for spec in LINES.values() for s in spec["segments"]]
    rows = ["| 配置项 | 值 |", "| --- | --- |"]
    for k, label in SHARED_KEYS:
        vals = {json.dumps(c.get(k)) for c in rl}
        val = json.loads(vals.pop()) if len(vals) == 1 else "各 run 不一致"
        if isinstance(val, list):
            val = ", ".join(val)
        rows.append(f"| {label} | `{val}` |")
    return "\n".join(rows)


def eval_table() -> str:
    b = base_eval()
    rows = [
        "| 线 | Step（W&B run） | AIME 2024 mean@8 | maj@8 | best@8 | AIME 2025 mean@8 | maj@8 | best@8 |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]

    def cells(row: dict) -> str:
        out = []
        for ds, _ in DATASETS:
            out += [
                pct(row.get(f"val-core/{ds}/acc/mean@8")),
                pct(row.get(f"val-core/{ds}/acc/maj@8/mean")),
                pct(row.get(f"val-core/{ds}/acc/best@8/mean")),
            ]
        return " | ".join(f"`{c}`" for c in out)

    rows.append(f"| base 模型（无 SFT） | 仅评估（`{BASE_EVAL}`） | {cells(b)} |")
    for line in LINES:
        for s, row in evals(line).items():
            rows.append(f"| `{line}` | {s}（`{row['_run']}`） | {cells(row)} |")
    return "\n".join(rows)


def summary_list() -> str:
    b = base_eval()
    out = []
    for ds, title in DATASETS:
        key = f"val-core/{ds}/acc/mean@8"
        at25 = {line: evals(line)[25][key] for line in LINES}
        late = [(line, s, r[key]) for line in LINES for s, r in evals(line).items() if s >= 50]
        best = max(late, key=lambda x: x[2])
        lo25, hi25 = min(at25.values()), max(at25.values())
        lo, hi = min(x[2] for x in late), max(x[2] for x in late)
        avg = statistics.mean(x[2] for x in late)
        out.append(
            f"- {title} mean@8：Step 25 四条线 `{pct(lo25)}`–`{pct(hi25)}`（极差 `{100 * (hi25 - lo25):.2f}` pp）；"
            f"Step 50–150 全部 {len(late)} 个评估点 `{pct(lo)}`–`{pct(hi)}`，均值 `{pct(avg)}`"
            f"（最高 `{pct(best[2])}`，`{best[0]}` step {best[1]}）；base 模型（无 SFT）`{pct(b[key])}`。"
        )
    for key, label, fmt in (
        ("critic/score/mean", "训练 reward", "{:+.3f}"),
        ("response_length/mean", "`response_length/mean`", "{:,.0f}"),
        ("num_turns/mean", "`num_turns/mean`", "{:.1f}"),
    ):
        first = [window_mean(series(line, key), 1, 5) for line in LINES]
        mid = [window_mean(series(line, key), 21, 25) for line in LINES]
        last = [x for x in (window_mean(series(line, key), 146, 150) for line in LINES) if x is not None]
        out.append(
            f"- {label}（5 步窗口均值）：steps 1–5 `{fmt.format(min(first))}`–`{fmt.format(max(first))}`，"
            f"steps 21–25 `{fmt.format(min(mid))}`–`{fmt.format(max(mid))}`，"
            f"steps 146–150 `{fmt.format(min(last))}`–`{fmt.format(max(last))}`（跑满 150 步的 3 条线）。"
        )
    return "\n".join(out)


def train_table() -> str:
    windows = ((1, 5), (21, 25), (46, 50), (96, 100), (146, 150))
    head = " | ".join(f"Steps {a}–{b}" for a, b in windows)
    rows = []
    for key, label in (
        ("critic/score/mean", "训练 reward `critic/score/mean`"),
        ("response_length/mean", "`response_length/mean`（token）"),
        ("num_turns/mean", "`num_turns/mean`"),
        (STEP_EX_SAVE, "`timing_s/step` − `timing_s/save_checkpoint`（秒；扣掉 checkpoint 保存，验证本来不计入 step）"),
    ):
        rows += [
            f"\n**{label}**（5 步窗口均值；– 表示该窗口没有完整数据）\n",
            f"| 线 | {head} |",
            "|" + " --- |" * (len(windows) + 1),
        ]
        for line in LINES:
            pts = series(line, key)
            fmt = "{:+.3f}" if key == "critic/score/mean" else "{:,.1f}"
            vals = [window_mean(pts, a, b) for a, b in windows]
            rows.append(f"| `{line}` | " + " | ".join("–" if v is None else f"`{fmt.format(v)}`" for v in vals) + " |")
    return "\n".join(rows).strip()


def sft_table() -> str:
    rows = [
        "| W&B run | 状态 | 步数 | lr / warmup 比例 / epochs / batch | train/loss 首 → 末 | val/loss（最小 / 最后） |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for rid in ("myaf5pal", "mtuqkfsy"):
        run = RUNS[rid]
        tl = [(step_of(r), r["train/loss"]) for r in run["history"] if "train/loss" in r]
        vl = [(step_of(r), r["val/loss"]) for r in run["history"] if "val/loss" in r]
        c = run["config"]
        vtxt = f"`{min(v for _, v in vl):.4f}` / `{vl[-1][1]:.4f}`" if vl else "–"
        rows.append(
            f"| `{rid}` | {run['state']} | {tl[0][0]}–{tl[-1][0]} | "
            f"`{c.get('sft_lr')}` / `{c.get('sft_warmup_ratio')}` / `{c.get('sft_epochs')}` / "
            f"`{c.get('sft_train_batch_size')}` | `{tl[0][1]:.4f}` → `{tl[-1][1]:.4f}` | {vtxt} |"
        )
    return "\n".join(rows)


def ours_table() -> str:
    b = base_eval()
    rows = [
        "| Run | 每题采样 `n` | Step 0 AIME 2024 / 2025 | Step 25 AIME 2024 / 2025 | 训练 reward steps 21–25 |",
        "| --- | --- | --- | --- | --- |",
        f"| base 模型 `{BASE_EVAL}`（无 SFT） | 8 | `{pct(b['val-core/aime_2024/acc/mean@8'])}` / "
        f"`{pct(b['val-core/aime_2025/acc/mean@8'])}` | – | – |",
    ]
    for line in LINES:
        e = evals(line).get(25)
        r = window_mean(series(line, "critic/score/mean"), 21, 25)
        e_txt = (
            f"`{pct(e['val-core/aime_2024/acc/mean@8'])}` / `{pct(e['val-core/aime_2025/acc/mean@8'])}`" if e else "–"
        )
        rows.append(
            f"| Askhat `{line}` | 8 | –（未做 Step 0 评估） | {e_txt} | {'–' if r is None else f'`{r:+.3f}`'} |"
        )
    o0, o25 = OURS["eval"]["0"], OURS["eval"]["25"]
    our_r = statistics.mean(x["critic/score/mean"] for x in OURS["train"][20:25])
    rows.append(
        f"| 我们 `{OURS['run']}`（64× GB200，我们自己的 SFT） | 30 | `{pct(o0['val-core/aime_2024/acc/mean@30'])}` / "
        f"`{pct(o0['val-core/aime_2025/acc/mean@30'])}` | `{pct(o25['val-core/aime_2024/acc/mean@30'])}` / "
        f"`{pct(o25['val-core/aime_2025/acc/mean@30'])}` | `{our_r:+.3f}` |"
    )
    return "\n".join(rows)


def resume_table() -> str:
    rows = [
        "| 线 | 前一段 run（记录的步数） | 恢复后的 run（记录的步数） | 共同的 `default_local_dir` | "
        "`resume_mode` / `save_freq` |",
        "| --- | --- | --- | --- | --- |",
    ]
    for line, spec in LINES.items():
        if len(spec["segments"]) < 2:
            continue
        (a, _, _), (b, _, _) = spec["segments"]
        ra, rb = RUNS[a], RUNS[b]
        sa = [step_of(r) for r in ra["history"]]
        sb = [step_of(r) for r in rb["history"]]
        same = ra["config"]["default_local_dir"] == rb["config"]["default_local_dir"]
        rows.append(
            f"| `{line}` | `{a}`（{min(sa)}–{max(sa)}，{ra['state']}） | `{b}`（{min(sb)}–{max(sb)}，{rb['state']}） | "
            f"`{rb['config']['default_local_dir'] if same else '不同'}` | "
            f"`{rb['config']['resume_mode']}` / `{rb['config']['save_freq']}` |"
        )
    return "\n".join(rows)


def ckpt_table() -> str:
    rows = [
        "| 线 | W&B run | `save_freq` | `save_contents` / `load_contents` | `async_save` | 保留个数 | 保存次数 | "
        "`timing_s/save_checkpoint` 均值（最小–最大，秒） | 保存耗时 / wall time |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for line, spec in LINES.items():
        for run_id, _, _ in spec["segments"]:
            run = RUNS[run_id]
            c = run["config"]
            sv = [r["timing_s/save_checkpoint"] for r in run["history"] if "timing_s/save_checkpoint" in r]
            ts = [r["_timestamp"] for r in run["history"] if "timing_s/step" in r]
            t = f"`{statistics.mean(sv):,.1f}`（`{min(sv):,.1f}`–`{max(sv):,.1f}`）" if sv else "–"
            frac = f"`{100 * sum(sv) / (max(ts) - min(ts)):.1f}%`" if sv and len(ts) > 1 else "–"
            rows.append(
                f"| `{line}` | `{run_id}` | `{c['save_freq']}` | "
                f"`{c['ckpt_save_contents']}` / `{c['ckpt_load_contents']}` | "
                f"`{c['ckpt_async_save']}` | `{c['max_actor_ckpt_to_keep']}` | {len(sv)} | {t} | {frac} |"
            )
    return "\n".join(rows)


TABLES = {
    "inventory": inventory_table,
    "config": config_table,
    "shared": shared_table,
    "resume": resume_table,
    "ckpt": ckpt_table,
    "eval": eval_table,
    "summary": summary_list,
    "train": train_table,
    "sft": sft_table,
    "ours": ours_table,
}


def figures() -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    b = base_eval()
    plt.rcParams.update({"figure.dpi": 120, "font.size": 9})

    def resume_marks(ax):
        for spec in LINES.values():
            for _, lo, _ in spec["segments"][1:]:
                ax.axvline(lo - 0.5, color=spec["color"], ls=":", lw=0.8, alpha=0.7)

    # Fig 1/2: evaluation curves.
    for fig_name, metric, label in (
        ("fig1_eval_mean.png", "acc/mean@8", "mean@8"),
        ("fig2_eval_maj.png", "acc/maj@8/mean", "maj@8"),
    ):
        fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharex=True)
        for ax, (ds, title) in zip(axes, DATASETS, strict=True):
            for line, spec in LINES.items():
                ev = evals(line)
                ys = [100 * r[f"val-core/{ds}/{metric}"] for r in ev.values()]
                ax.plot(list(ev), ys, "o-", color=spec["color"], label=line)
            ax.axhline(
                100 * b[f"val-core/{ds}/{metric}"], color="gray", ls="--", lw=1, label="base Qwen3-30B-A3B (no SFT)"
            )
            ax.set_title(f"{title} {label} (n=8 per question, 30 questions)")
            ax.set_xlabel("RL step")
            ax.set_ylabel("%")
            ax.grid(alpha=0.3)
        axes[0].legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(HERE / fig_name)
        plt.close(fig)

    def curve_fig(fig_name: str, panels: list[tuple[str, str]], smooth: int = 5) -> None:
        fig, axes = plt.subplots(1, len(panels), figsize=(5.2 * len(panels), 3.8))
        axes = axes if len(panels) > 1 else [axes]
        for ax, (key, title) in zip(axes, panels, strict=True):
            for line, spec in LINES.items():
                pts = series(line, key)
                xs, ys = [p[0] for p in pts], [p[1] for p in pts]
                ax.plot(xs, ys, color=spec["color"], alpha=0.25, lw=0.8)
                if len(ys) >= smooth:
                    ma = [statistics.mean(ys[max(0, i - smooth + 1) : i + 1]) for i in range(len(ys))]
                    ax.plot(xs, ma, color=spec["color"], lw=1.6, label=line)
            resume_marks(ax)
            ax.set_title(title + f" (thin: raw, bold: {smooth}-step trailing mean)", fontsize=8)
            ax.set_xlabel("RL step")
            ax.grid(alpha=0.3)
        axes[0].legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(HERE / fig_name)
        plt.close(fig)

    curve_fig("fig3_train_reward.png", [("critic/score/mean", "train reward critic/score/mean")])
    curve_fig(
        "fig4_response_turns.png",
        [
            ("response_length/mean", "response_length/mean (tokens)"),
            ("response_length/clip_ratio", "response_length/clip_ratio"),
            ("num_turns/mean", "num_turns/mean"),
        ],
    )
    curve_fig(
        "fig5_optim.png",
        [
            ("actor/grad_norm", "actor/grad_norm"),
            ("actor/ppo_kl", "actor/ppo_kl"),
            ("actor/pg_clipfrac", "actor/pg_clipfrac"),
        ],
    )
    curve_fig(
        "fig6_step_time.png",
        [(STEP_EX_SAVE, "timing_s/step minus save_checkpoint (s)"), ("timing_s/gen", "timing_s/gen (s)")],
    )

    # Fig 7: SFT.
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    for rid, color in (("myaf5pal", "tab:purple"), ("mtuqkfsy", "tab:gray")):
        h = RUNS[rid]["history"]
        tl = [(step_of(r), r["train/loss"]) for r in h if "train/loss" in r]
        axes[0].plot(
            [s for s, _ in tl], [v for _, v in tl], color=color, lw=1, label=f"{rid} train/loss ({RUNS[rid]['state']})"
        )
        vl = [(step_of(r), r["val/loss"]) for r in h if "val/loss" in r]
        if vl:
            axes[0].plot([s for s, _ in vl], [v for _, v in vl], "o--", color="black", ms=3, label=f"{rid} val/loss")
        lr = [(step_of(r), r["train/lr"]) for r in h if "train/lr" in r]
        axes[1].plot([s for s, _ in lr], [v for _, v in lr], color=color, label=rid)
    axes[0].set_title("SFT loss (multiturn-sft-qwen3-30b-a3b-v5p)")
    axes[1].set_title("SFT learning rate")
    for ax in axes:
        ax.set_xlabel("SFT step")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(HERE / "fig7_sft.png")
    plt.close(fig)

    # Fig 8: Askhat vs. our 64x GB200 25-step run (reference only).
    fig, axes = plt.subplots(1, 3, figsize=(15, 3.8))
    for line, spec in LINES.items():
        pts = [p for p in series(line, "critic/score/mean") if p[0] <= 25]
        axes[0].plot([p[0] for p in pts], [p[1] for p in pts], color=spec["color"], lw=1.2, label=f"Askhat {line}")
    axes[0].plot(
        [x["step"] for x in OURS["train"]],
        [x["critic/score/mean"] for x in OURS["train"]],
        "k-",
        lw=2,
        label="ours 64x GB200",
    )
    axes[0].set_title("train reward, steps 1-25")
    axes[0].set_xlabel("RL step")
    for ax, (ds, title) in zip(axes[1:], DATASETS, strict=True):
        for line, spec in LINES.items():
            ev = {s: r for s, r in evals(line).items() if s <= 50}
            ys = [100 * r[f"val-core/{ds}/acc/mean@8"] for r in ev.values()]
            ax.plot(list(ev), ys, "o-", color=spec["color"], label=f"Askhat {line} (n=8)")
        ys = [100 * OURS["eval"][str(s)][f"val-core/{ds}/acc/mean@30"] for s in (0, 25)]
        ax.plot([0, 25], ys, "ks-", lw=2, label="ours 64x GB200 (n=30)")
        ax.axhline(100 * b[f"val-core/{ds}/acc/mean@8"], color="gray", ls="--", lw=1, label="base model (n=8)")
        ax.set_title(f"{title} mean@N, steps 0-50")
        ax.set_xlabel("RL step")
        ax.set_ylabel("%")
    for ax in axes:
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=7)
    axes[1].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(HERE / "fig8_vs_ours.png")
    plt.close(fig)


def block(name: str) -> str:
    begin = f"<!-- BEGIN AUTO-GENERATED {name} (plot_askhat_v5p.py) -->"
    return f"{begin}\n{TABLES[name]()}\n<!-- END AUTO-GENERATED {name} -->"


def check(readme: Path) -> int:
    text = readme.read_text(encoding="utf-8")
    bad = 0
    for name, fn in TABLES.items():
        pattern = (
            rf"<!-- BEGIN AUTO-GENERATED {name} \(plot_askhat_v5p\.py\) -->\n(.*?)\n<!-- END AUTO-GENERATED {name} -->"
        )
        m = re.search(pattern, text, re.S)
        if m is None or m.group(1) != fn():
            print(f"{readme.name}: block {name} is missing or stale")
            bad += 1
    print("all blocks match" if not bad else f"{bad} stale block(s)")
    return 1 if bad else 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--figures", action="store_true")
    parser.add_argument("--print-markdown", action="store_true")
    parser.add_argument("--check", type=Path)
    args = parser.parse_args()
    if args.figures:
        figures()
    if args.print_markdown:
        for name in TABLES:
            print(block(name) + "\n")
    if args.check:
        sys.exit(check(args.check))


if __name__ == "__main__":
    main()
