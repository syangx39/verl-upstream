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
"""Build the tables and figures of ``REPRO_GUIDE.md`` from the snapshot in ``data/``.

``data/`` is written by ``fetch_runs.py`` (W&B + GCS). This script only reads it: every number in the guide's
``AUTO-GENERATED`` blocks comes from there. Tables need only the standard library; ``--figures`` needs matplotlib.

Usage::

    python3 make_report.py --print                  # print every table
    python3 make_report.py --check ../REPRO_GUIDE.md  # fail if a block in the guide is stale
    python3 make_report.py --write ../REPRO_GUIDE.md  # refresh the blocks in place
    python3 make_report.py --figures                # write figures/*.png
"""

import argparse
import json
import math
import statistics
import sys
from collections.abc import Callable, Iterable
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
FIGURES = HERE / "figures"
WANDB_PROJECT = "google-trellis/verl-torchtpu-grpo-test"
MAX_STEP = 25
TIMING_STEPS = range(2, MAX_STEP + 1)  # step 1 includes compilation / warm-up

# One entry per training curve. ``segments`` maps W&B runs to the steps taken from them (a resumed run continues
# the curve of the run it resumed from). ``devices_per_chip``: v7x exposes 2 TensorCores per chip as 2 devices.
RUNS = [
    {
        "key": "v5p-cp1-pr115",
        "platform": "v5p",
        "label": "v5p cp1-pr115",
        "segments": [["yrcsp4z6", 1, 25]],
        "devices_per_chip": 1,
        "eval_n": 8,
    },
    {
        "key": "v5p-cp1-pr115-async",
        "platform": "v5p",
        "label": "v5p cp1-pr115-async",
        "segments": [["zyrnkte8", 1, 25]],
        "devices_per_chip": 1,
        "eval_n": 8,
    },
    {
        "key": "v5p-cp1-opt",
        "platform": "v5p",
        "label": "v5p cp1-opt",
        "segments": [["rq1p980n", 1, 25]],
        "devices_per_chip": 1,
        "eval_n": 8,
    },
    {
        "key": "v5p-cp2",
        "platform": "v5p",
        "label": "v5p cp2",
        "segments": [["fcze3xlx", 1, 20], ["ci0k54pl", 21, 25]],
        "devices_per_chip": 1,
        "eval_n": 8,
    },
    {
        "key": "v7x",
        "platform": "v7x",
        "label": "v7x p3c-retool-25s-02",
        "segments": [["wopup8dq", 0, 25]],
        "devices_per_chip": 2,
        "eval_n": 30,
    },
    {
        "key": "gb200",
        "platform": "gb200",
        "label": "GB200 g64-retool-25s-01",
        "segments": [["skzy1l0y", 0, 25]],
        "devices_per_chip": 1,
        "eval_n": 30,
    },
]
BASE_EVAL_RUN = "4fl4zl1d"  # Qwen3-30B-A3B without SFT, val_only, n=8 (Askhat, v5p)
SFT_V5P_RUN = "myaf5pal"  # Askhat's v5p SFT
PLATFORM_NAMES = {"v5p": "TPU v5p (Askhat)", "v7x": "TPU v7x (ours)", "gb200": "GB200 (ours)"}
COLORS = {
    "v5p-cp1-pr115": "#7fa7d9",
    "v5p-cp1-pr115-async": "#5b8cc9",
    "v5p-cp1-opt": "#9db9e0",
    "v5p-cp2": "#3e6fb0",
    "v7x": "#e8710a",
    "gb200": "#1e8e3e",
}
DATASETS = ("aime_2024", "aime_2025")
DATASET_NAMES = {"aime_2024": "AIME 2024", "aime_2025": "AIME 2025"}

# Algorithm settings that must be identical in every run (checked, not assumed).
INVARIANTS = [
    ("Advantage estimator", "algorithm.adv_estimator"),
    ("Normalize advantage by group std", "algorithm.norm_adv_by_std_in_grpo"),
    ("Policy loss", "actor_rollout_ref.actor.policy_loss.loss_mode"),
    ("Rollout correction: bypass mode", "algorithm.rollout_correction.bypass_mode"),
    ("Rollout correction: IS level / threshold", "algorithm.rollout_correction.rollout_is"),
    (None, "algorithm.rollout_correction.rollout_is_threshold"),
    ("Rollout correction: loss type", "algorithm.rollout_correction.loss_type"),
    ("Clip ratio low / high (DAPO clip-higher)", "actor_rollout_ref.actor.clip_ratio_low"),
    (None, "actor_rollout_ref.actor.clip_ratio_high"),
    ("Dual-clip ratio c", "actor_rollout_ref.actor.clip_ratio_c"),
    ("Loss aggregation", "actor_rollout_ref.actor.loss_agg_mode"),
    ("KL loss / KL in reward / KL coef", "actor_rollout_ref.actor.use_kl_loss"),
    (None, "algorithm.use_kl_in_reward"),
    (None, "actor_rollout_ref.actor.kl_loss_coef"),
    ("Entropy coefficient", "actor_rollout_ref.actor.entropy_coeff"),
    ("Prompts per step / mini-batch / PPO epochs", "data.train_batch_size"),
    (None, "actor_rollout_ref.actor.ppo_mini_batch_size"),
    (None, "actor_rollout_ref.actor.ppo_epochs"),
    ("Responses per prompt (group size)", "actor_rollout_ref.rollout.n"),
    ("Rollout temperature / top_p / top_k", "actor_rollout_ref.rollout.temperature"),
    (None, "actor_rollout_ref.rollout.top_p"),
    (None, "actor_rollout_ref.rollout.top_k"),
    ("Validation temperature / top_p", "actor_rollout_ref.rollout.val_kwargs.temperature"),
    (None, "actor_rollout_ref.rollout.val_kwargs.top_p"),
    ("Max prompt / response length", "data.max_prompt_length"),
    (None, "data.max_response_length"),
    ("Max assistant / user turns", "actor_rollout_ref.rollout.multi_turn.max_assistant_turns"),
    (None, "actor_rollout_ref.rollout.multi_turn.max_user_turns"),
    ("Max tool response length (tokens)", "actor_rollout_ref.rollout.multi_turn.max_tool_response_length"),
    ("Chat template thinking", "data.apply_chat_template_kwargs.enable_thinking"),
    ("Optimizer / LR", "actor_rollout_ref.actor.optim.name"),
    (None, "actor_rollout_ref.actor.optim.lr"),
    ("Adam betas / eps / weight decay", "actor_rollout_ref.actor.optim.betas"),
    (None, "actor_rollout_ref.actor.optim.eps"),
    (None, "actor_rollout_ref.actor.optim.weight_decay"),
    ("LR warm-up steps / ratio / min LR factor", "actor_rollout_ref.actor.optim.lr_warmup_steps"),
    (None, "actor_rollout_ref.actor.optim.lr_warmup_steps_ratio"),
    (None, "actor_rollout_ref.actor.optim.min_lr_factor"),
    ("Trainer mode", "trainer.v1.trainer_mode"),
    ("Async warm-up batches / param sync every N steps", "trainer.v1.separate_async.num_warmup_batches"),
    (None, "trainer.v1.separate_async.parameter_sync_step"),
    ("Max off-policy staleness / strategy", "trainer.v1.sampler.max_off_policy_threshold"),
    (None, "trainer.v1.sampler.max_off_policy_strategy"),
    ("Data shuffle / seed / truncation", "data.shuffle"),
    (None, "actor_rollout_ref.actor.data_loader_seed"),
    (None, "data.truncation"),
    ("Dataset class", "data.custom_cls.path"),
    ("Reward function", "reward.custom_reward_function.path"),
    (None, "reward.custom_reward_function.name"),
    ("Tool config", "actor_rollout_ref.rollout.multi_turn.tool_config_path"),
    ("Trainer / rollout dtype", "actor_rollout_ref.actor.torchtitan.dtype"),
    (None, "actor_rollout_ref.rollout.dtype"),
    ("Validation every N steps", "trainer.test_freq"),
]
# Settings that differ but do not change the algorithm (explained in the guide).
NON_SYSTEM_DIFFS = [
    ("`optim.clip_grad`", "actor_rollout_ref.actor.optim.clip_grad"),
    ("SFT checkpoint `model.path`", "actor_rollout_ref.model.path"),
    ("Validation samples per problem", "actor_rollout_ref.rollout.val_kwargs.n"),
    ("`val_before_train`", "trainer.val_before_train"),
    ("`total_training_steps`", "trainer.total_training_steps"),
    ("`save_freq`", "trainer.save_freq"),
]
# System settings, one column per run.
ADAPTATIONS = [
    ("Trainer data-parallel shard", "actor_rollout_ref.actor.torchtitan.data_parallel_shard_size"),
    ("Trainer context parallel", "actor_rollout_ref.actor.torchtitan.context_parallel_size"),
    ("Trainer attention", "actor_rollout_ref.actor.torchtitan.attn_type"),
    ("Trainer activation checkpointing", "actor_rollout_ref.actor.torchtitan.activation_checkpoint"),
    ("Trainer `torch.compile`", "actor_rollout_ref.actor.torchtitan.use_torch_compile"),
    ("Trainer SPMD backend", "actor_rollout_ref.actor.torchtitan.spmd_backend"),
    ("Token budget per device (train / log-prob)", "actor_rollout_ref.actor.ppo_max_token_len_per_gpu"),
    ("Trainer `max_seq_len`", "actor_rollout_ref.actor.torchtitan.max_seq_len"),
    ("Rollout TP", "actor_rollout_ref.rollout.tensor_model_parallel_size"),
    ("Rollout EP", "actor_rollout_ref.rollout.expert_parallel_size"),
    ("Rollout DP (per server)", "actor_rollout_ref.rollout.data_parallel_size"),
    ("Rollout memory fraction", "actor_rollout_ref.rollout.gpu_memory_utilization"),
    ("Rollout `max_num_seqs`", "actor_rollout_ref.rollout.max_num_seqs"),
    ("Rollout `max_num_batched_tokens`", "actor_rollout_ref.rollout.max_num_batched_tokens"),
    ("Weight sync backend", "actor_rollout_ref.rollout.checkpoint_engine.backend"),
]
HW_KEYS = ("trainer.nnodes", "trainer.n_gpus_per_node", "rollout.nnodes", "rollout.n_gpus_per_node")
CONFIG_KEYS = sorted(
    {k for _, k in INVARIANTS + NON_SYSTEM_DIFFS + ADAPTATIONS}
    | set(HW_KEYS)
    | {"actor_rollout_ref.actor.torchtitan.use_splash_attention", "actor_rollout_ref.actor.torchtitan.use_simple_fsdp"}
)
METRICS = (
    "critic/score/mean",
    "response_length/mean",
    "response_length/clip_ratio",
    "num_turns/mean",
    "actor/grad_norm",
    "actor/pg_clipfrac",
    "actor/ppo_kl",
    "actor/lr",
    "perf/total_num_tokens",
    "timing_s/step",
    "timing_s/gen",
    "timing_s/old_log_prob",
    "timing_s/adv",
    "timing_s/update_actor",
    "timing_s/update_weights",
    "timing_s/save_checkpoint",
    "timing_s/testing",
)
VAL_PREFIX = "val-core/"


def wandb_url(run_id: str) -> str:
    return f"https://wandb.ai/{WANDB_PROJECT}/runs/{run_id}"


def load(name: str) -> dict:
    return json.loads((DATA / name).read_text(encoding="utf-8"))


class Snapshot:
    """The fetched data plus per-curve views (segments merged by step)."""

    def __init__(self) -> None:
        self.wandb = load("wandb_runs.json")
        self.eval_pq = load("eval_per_question.json")
        self.sft = load("sft_loss.json")
        self.prompts = load("train_prompts.json")
        self.tb = load("tb_log_windows.json")
        for spec in RUNS:
            for run_id, _, _ in spec["segments"]:
                if run_id not in self.wandb["runs"]:
                    raise SystemExit(f"data/wandb_runs.json has no run {run_id}; re-run fetch_runs.py")

    def rows(self, spec: dict) -> dict[int, dict]:
        out = {}
        for run_id, first, last in spec["segments"]:
            for row in self.wandb["runs"][run_id]["history"]:
                if first <= row["step"] <= last:
                    out[row["step"]] = row
        return out

    def series(self, spec: dict, metric: str) -> dict[int, float]:
        return {s: r[metric] for s, r in self.rows(spec).items() if isinstance(r.get(metric), int | float)}

    def configs(self, spec: dict) -> list[dict]:
        return [self.wandb["runs"][run_id]["config"] for run_id, _, _ in spec["segments"]]

    def config_value(self, spec: dict, key: str) -> str:
        values = []
        for cfg in self.configs(spec):
            v = fmt_value(cfg.get(key, "<missing>"))
            if v not in values:
                values.append(v)
        return " → ".join(values)

    def eval_value(self, spec: dict, step: int, dataset: str) -> float | None:
        n = spec["eval_n"]
        row = self.rows(spec).get(step, {})
        v = row.get(f"{VAL_PREFIX}{dataset}/acc/mean@{n}")
        return None if v is None else 100.0 * v


def fmt_value(v: object) -> str:
    if isinstance(v, bool):
        return str(v)
    if isinstance(v, float) and v.is_integer() and abs(v) < 1e6:
        return str(int(v))
    if isinstance(v, list):
        return "[" + ", ".join(fmt_value(x) for x in v) + "]"
    return str(v)


def mean(xs: Iterable[float]) -> float:
    xs = list(xs)
    return sum(xs) / len(xs) if xs else math.nan


def window(series: dict[int, float], lo: int, hi: int) -> float | None:
    vals = [series[s] for s in range(lo, hi + 1) if s in series]
    return mean(vals) if len(vals) == hi - lo + 1 else None


def f(v: float | None, spec: str = ".1f", signed: bool = False) -> str:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "–"
    return format(v, ("+" if signed else "") + spec)


def code(s: object) -> str:
    return f"`{s}`"


# ----------------------------------------------------------------------------------------------------------------
# Per-question evaluation statistics (TPU v7x and GB200 only; Askhat's runs log aggregates only).


def per_question(snap: Snapshot, run_key: str, step: int, dataset: str) -> dict[str, list[int]]:
    return snap.eval_pq[run_key]["steps"][str(step)][dataset]


def acc_and_se(pq: dict[str, list[int]]) -> tuple[float, float]:
    """Mean accuracy (%) over problems and its standard error from the finite samples per problem.

    The problems are fixed (the benchmark), so the uncertainty is only the sampling of responses: Var(p_hat_q) =
    p_q (1 - p_q) / n_q, estimated without bias by p_hat (1 - p_hat) / (n - 1).
    """
    q = len(pq)
    p = [c / n for c, n in pq.values()]
    var = sum(pi * (1 - pi) / (n - 1) for pi, (_, n) in zip(p, pq.values(), strict=True))
    return 100.0 * sum(p) / q, 100.0 * math.sqrt(var) / q


def paired_diff(a: dict[str, list[int]], b: dict[str, list[int]]) -> tuple[float, float]:
    """Mean of per-problem accuracy differences a - b (%) and its standard error (independent sampling)."""
    assert set(a) == set(b), "different problem sets"
    q = len(a)
    d = 0.0
    var = 0.0
    for k in a:
        (ca, na), (cb, nb) = a[k], b[k]
        pa, pb = ca / na, cb / nb
        d += pa - pb
        var += pa * (1 - pa) / (na - 1) + pb * (1 - pb) / (nb - 1)
    return 100.0 * d / q, 100.0 * math.sqrt(var) / q


# ----------------------------------------------------------------------------------------------------------------
# Tables


def hw(snap: Snapshot, spec: dict, role: str) -> str:
    vals = []
    for cfg in snap.configs(spec):
        chips = int(cfg[f"{role}.nnodes"]) * int(cfg[f"{role}.n_gpus_per_node"]) // spec["devices_per_chip"]
        if chips not in vals:
            vals.append(chips)
    return " → ".join(str(v) for v in vals)


def _span(vals: list[float], fmt: str, unit: str = "") -> str:
    lo, hi = min(vals), max(vals)
    return f"{lo:{fmt}}{unit}" if format(lo, fmt) == format(hi, fmt) else f"{lo:{fmt}}{unit} to {hi:{fmt}}{unit}"


def hw_summary(snap: Snapshot, specs: list[dict], unit: str) -> str:
    """Most common ``trainer + rollout`` of a platform's runs, with the runs that differ named in parentheses."""
    values = {s["key"]: f"{hw(snap, s, 'trainer')} + {hw(snap, s, 'rollout')}" for s in specs}
    common = max(set(values.values()), key=list(values.values()).count)
    others = [f"{s['label'].split(' ', 1)[1]}: {values[s['key']]}" for s in specs if values[s["key"]] != common]
    return f"{common} {unit}" + (f" ({'; '.join(others)})" if others else "")


def table_summary(snap: Snapshot) -> str:
    plats = ["v5p", "v7x", "gb200"]
    groups = {p: [s for s in RUNS if s["platform"] == p] for p in plats}
    unit = {"v5p": "chips", "v7x": "chips", "gb200": "GPUs"}

    def cell(p: str, fn: Callable[[dict], float | None], fmt: str, suffix: str = "") -> str:
        vals = [fn(spec) for spec in groups[p]]
        if any(v is None for v in vals):
            return "not evaluated"
        return _span(vals, fmt, suffix)

    rows = [
        ("Runs", lambda p: f"{len(groups[p])} ({', '.join(s['label'].split(' ', 1)[1] for s in groups[p])})"),
        ("Trainer + rollout accelerators", lambda p: hw_summary(snap, groups[p], unit[p])),
        ("Validation samples per problem", lambda p: str(groups[p][0]["eval_n"])),
    ]
    for ds in DATASETS:
        for step in (0, MAX_STEP):
            rows.append(
                (
                    f"{DATASET_NAMES[ds]} accuracy, Step {step}",
                    lambda p, ds=ds, step=step: cell(p, lambda s: snap.eval_value(s, step, ds), ".2f", "%"),
                )
            )
    rows += [
        (
            "Train reward, mean of steps 21–25",
            lambda p: cell(p, lambda s: window(snap.series(s, "critic/score/mean"), 21, 25), "+.3f"),
        ),
        (
            "Wall clock per step, mean of steps 2–25 (s)",
            lambda p: cell(p, lambda s: mean(wall_time(snap, s).values()), ",.0f"),
        ),
        (
            "… without TensorBoard logging (s)",
            lambda p: cell(
                p,
                lambda s: mean(wall_time(snap, s).values()) - mean((tb_time(snap, s) or {0: 0.0}).values()),
                ",.0f",
            ),
        ),
    ]
    lines = ["| | " + " | ".join(PLATFORM_NAMES[p] for p in plats) + " |", "| --- | --- | --- | --- |"]
    lines += [f"| {label} | " + " | ".join(fn(p) for p in plats) + " |" for label, fn in rows]
    return "\n".join(lines)


def table_runs(snap: Snapshot) -> str:
    unit = {"v5p": "chips", "v7x": "chips", "gb200": "GPUs"}
    lines = [
        "| Platform | Run | W&B | Trainer + rollout | Steps used | Validation |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for spec in RUNS:
        links = " → ".join(f"[`{r}`]({wandb_url(r)})" for r, _, _ in spec["segments"])
        rows = snap.rows(spec)
        steps = sorted(s for s in rows if s >= 1)
        evals = [s for s in sorted(rows) if snap.eval_value(spec, s, "aime_2024") is not None]
        lines.append(
            f"| {PLATFORM_NAMES[spec['platform']]} | {spec['label'].split(' ', 1)[1]} | {links} | "
            f"{hw(snap, spec, 'trainer')} + {hw(snap, spec, 'rollout')} {unit[spec['platform']]} | "
            f"{steps[0]}–{steps[-1]} | step{'s' if len(evals) > 1 else ''} {', '.join(map(str, evals))}, "
            f"`n={spec['eval_n']}` |"
        )
    return "\n".join(lines)


def table_invariants(snap: Snapshot) -> str:
    rows: list[tuple[str, list[str]]] = []
    for label, key in INVARIANTS:
        values = {snap.config_value(spec, key) for spec in RUNS}
        if len(values) != 1:
            raise SystemExit(f"invariant {key} differs between runs: {sorted(values)}")
        value = values.pop()
        if label is None:
            rows[-1][1].append(value)
        else:
            rows.append((label, [value]))
    lines = [
        f"| Setting | Value (identical in all {len(RUNS)} runs) |",
        "| --- | --- |",
    ]
    lines += [f"| {label} | {' / '.join(code(v) for v in vals)} |" for label, vals in rows]
    return "\n".join(lines)


def table_diffs(snap: Snapshot) -> str:
    plats = ["v5p", "v7x", "gb200"]
    lines = [
        "| Setting | " + " | ".join(PLATFORM_NAMES[p] for p in plats) + " |",
        "| --- | --- | --- | --- |",
    ]
    for label, key in NON_SYSTEM_DIFFS:
        cells = []
        for p in plats:
            vals = []
            for spec in RUNS:
                if spec["platform"] == p:
                    for v in snap.config_value(spec, key).split(" → "):
                        if v not in vals:
                            vals.append(v)
            cells.append(" / ".join(code(v) for v in vals))
        lines.append(f"| {label} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def table_adaptations(snap: Snapshot) -> str:
    lines = [
        "| Setting | " + " | ".join(spec["label"] for spec in RUNS) + " |",
        "| --- |" + " --- |" * len(RUNS),
        "| Trainer chips/GPUs | " + " | ".join(code(hw(snap, s, "trainer")) for s in RUNS) + " |",
        "| Rollout chips/GPUs | " + " | ".join(code(hw(snap, s, "rollout")) for s in RUNS) + " |",
    ]
    for label, key in ADAPTATIONS:
        lines.append(f"| {label} | " + " | ".join(code(snap.config_value(s, key)) for s in RUNS) + " |")
    return "\n".join(lines)


def table_eval(snap: Snapshot) -> str:
    lines = [
        "| Run | `n` | AIME 2024 Step 0 | AIME 2024 Step 25 | Δ | AIME 2025 Step 0 | AIME 2025 Step 25 | Δ |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    base = snap.wandb["runs"][BASE_EVAL_RUN]["history"][0]
    lines.append(
        f"| Qwen3-30B-A3B, no SFT ([`{BASE_EVAL_RUN}`]({wandb_url(BASE_EVAL_RUN)})) | 8 | "
        f"{f(100 * base[VAL_PREFIX + 'aime_2024/acc/mean@8'], '.2f')} | – | – | "
        f"{f(100 * base[VAL_PREFIX + 'aime_2025/acc/mean@8'], '.2f')} | – | – |"
    )
    for spec in RUNS:
        cells = []
        for ds in DATASETS:
            if spec["key"] in snap.eval_pq:
                a0, se0 = acc_and_se(per_question(snap, spec["key"], 0, ds))
                a1, se1 = acc_and_se(per_question(snap, spec["key"], MAX_STEP, ds))
                for step, a in ((0, a0), (MAX_STEP, a1)):  # per-question data must reproduce the W&B number
                    assert abs(a - snap.eval_value(spec, step, ds)) < 1e-6, (spec["key"], step, ds)
                cells += [f"{a0:.2f} ± {1.96 * se0:.2f}", f"{a1:.2f} ± {1.96 * se1:.2f}", f"{a1 - a0:+.2f}"]
            else:
                cells += ["–", f(snap.eval_value(spec, MAX_STEP, ds), ".2f"), "–"]
        lines.append(f"| {spec['label']} | {spec['eval_n']} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def table_eval_paired(snap: Snapshot) -> str:
    lines = [
        "| Comparison (same 30 problems per year) | AIME 2024 | AIME 2025 |",
        "| --- | --- | --- |",
    ]
    for step in (0, MAX_STEP):
        cells = []
        for ds in DATASETS:
            d, se = paired_diff(per_question(snap, "v7x", step, ds), per_question(snap, "gb200", step, ds))
            cells.append(f"{d:+.2f} pp (95% CI {d - 1.96 * se:+.2f} to {d + 1.96 * se:+.2f})")
        lines.append(f"| TPU v7x − GB200, Step {step} | " + " | ".join(cells) + " |")
    for key in ("v7x", "gb200"):
        cells = []
        for ds in DATASETS:
            d, se = paired_diff(per_question(snap, key, MAX_STEP, ds), per_question(snap, key, 0, ds))
            cells.append(f"{d:+.2f} pp (95% CI {d - 1.96 * se:+.2f} to {d + 1.96 * se:+.2f})")
        lines.append(f"| {'TPU v7x' if key == 'v7x' else 'GB200'} Step 25 − Step 0 | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def table_train(snap: Snapshot) -> str:
    lines = [
        "| Run | Reward 1–5 | Reward 21–25 | Response length 1–5 | Response length 21–25 | Turns 1–5 | Turns 21–25 "
        "| Max `grad_norm` 1–25 |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for spec in RUNS:
        r = snap.series(spec, "critic/score/mean")
        rl = snap.series(spec, "response_length/mean")
        t = snap.series(spec, "num_turns/mean")
        g = snap.series(spec, "actor/grad_norm")
        lines.append(
            f"| {spec['label']} | {f(window(r, 1, 5), '+.3f')} | {f(window(r, 21, 25), '+.3f')} | "
            f"{f(window(rl, 1, 5), ',.0f')} | {f(window(rl, 21, 25), ',.0f')} | {f(window(t, 1, 5), '.2f')} | "
            f"{f(window(t, 21, 25), '.2f')} | {f(max(g[s] for s in range(1, MAX_STEP + 1)), '.3f')} |"
        )
    return "\n".join(lines)


def timed_steps(spec: dict) -> list[int]:
    """Steps 2–25 whose previous step was logged by the same W&B run (a resumed run restarts its clock and warms up)."""
    return [s for run_id, first, last in spec["segments"] for s in TIMING_STEPS if first < s <= last]


def step_time(snap: Snapshot, spec: dict) -> dict[int, float]:
    """``timing_s/step`` minus checkpoint saving (validation is not part of ``timing_s/step``)."""
    rows = snap.rows(spec)
    return {s: rows[s]["timing_s/step"] - rows[s].get("timing_s/save_checkpoint", 0.0) for s in timed_steps(spec)}


def wall_time(snap: Snapshot, spec: dict) -> dict[int, float]:
    """Time between consecutive W&B step records, minus checkpoint saving and validation of that step."""
    rows = snap.rows(spec)
    return {
        s: rows[s]["_timestamp"]
        - rows[s - 1]["_timestamp"]
        - rows[s].get("timing_s/save_checkpoint", 0.0)
        - rows[s].get("timing_s/testing", 0.0)
        for s in timed_steps(spec)
    }


def tb_time(snap: Snapshot, spec: dict) -> dict[int, float] | None:
    """TensorBoard logging of step N-1, which happens inside wall(N) (see ``../step_timing_breakdown.py``)."""
    if spec["key"] not in snap.tb:
        return None
    w = snap.tb[spec["key"]]["steps"]
    return {s: w[str(s - 1)]["span"] for s in timed_steps(spec)}


def phase_mean(snap: Snapshot, spec: dict, metric: str) -> float:
    ser = snap.series(spec, metric)
    return mean(ser[s] for s in timed_steps(spec))


def table_wall(snap: Snapshot) -> str:
    lines = [
        "| Run | Chips/GPUs (trainer + rollout) | Wall clock per step | TensorBoard logging per step "
        "| Wall clock without TensorBoard logging | Tokens per step | Wall clock without TensorBoard per 1M tokens |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for spec in RUNS:
        wall = wall_time(snap, spec)
        tb = tb_time(snap, spec) or {s: 0.0 for s in wall}
        tok = snap.series(spec, "perf/total_num_tokens")
        net = {s: wall[s] - tb[s] for s in wall}
        lines.append(
            f"| {spec['label']} | {hw(snap, spec, 'trainer')} + {hw(snap, spec, 'rollout')} | "
            f"{mean(wall.values()):,.1f} | "
            f"{f(mean(tb.values()), ',.1f') if spec['key'] in snap.tb else 'no TensorBoard logger'} | "
            f"{mean(net.values()):,.1f} | {mean(tok[s] for s in wall) / 1e6:.2f} M | "
            f"{mean(net[s] / tok[s] * 1e6 for s in wall):,.1f} |"
        )
    return "\n".join(lines)


def table_step_phases(snap: Snapshot) -> str:
    lines = [
        "| Run | `timing_s/step`, mean (median) | waiting for rollout (`gen`) | `update_actor` | `update_weights` "
        "| rest of the step |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for spec in RUNS:
        st = step_time(snap, spec)
        gen, upd, sync = (
            phase_mean(snap, spec, k) for k in ("timing_s/gen", "timing_s/update_actor", "timing_s/update_weights")
        )
        lines.append(
            f"| {spec['label']} | {mean(st.values()):,.1f} ({statistics.median(st.values()):,.1f}) | {gen:,.1f} | "
            f"{upd:,.1f} | {sync:,.1f} | {mean(st.values()) - gen - upd - sync:,.1f} |"
        )
    return "\n".join(lines)


def sft_series(snap: Snapshot, key: str) -> tuple[dict[int, list[float]], dict[int, float]]:
    d = snap.sft[key]
    return {int(s): v for s, *v in d["train"]}, {int(s): v for s, v in d["val"]}


def table_sft(snap: Snapshot) -> str:
    tr_a, val_a = sft_series(snap, "v5p")
    tr_b, val_b = sft_series(snap, "v7x")
    steps = sorted(set(tr_a) & set(tr_b))
    same_tokens = sum(1 for s in steps if tr_a[s][2] == tr_b[s][2])
    same_lr = sum(1 for s in steps if abs(tr_a[s][1] - tr_b[s][1]) <= 1e-12)
    dl = [tr_b[s][0] - tr_a[s][0] for s in steps]
    lines = [
        "| Quantity | TPU v5p SFT (Askhat) | TPU v7x SFT (ours, used by v7x and GB200) |",
        "| --- | --- | --- |",
        f"| Optimizer steps | `{len(tr_a)}` | `{len(tr_b)}` |",
        f"| Training tokens | `{sum(v[2] for v in tr_a.values()):,.0f}` | `{sum(v[2] for v in tr_b.values()):,.0f}` |",
        f"| Steps with identical token count / identical LR | `{same_tokens}` / `{same_lr}` of `{len(steps)}` "
        "| (same) |",
        f"| `train/loss` step 1 → step {steps[-1]} | `{tr_a[1][0]:.4f}` → `{tr_a[steps[-1]][0]:.4f}` | "
        f"`{tr_b[1][0]:.4f}` → `{tr_b[steps[-1]][0]:.4f}` |",
        f"| `train/loss` difference (v7x − v5p), mean / max abs over steps | `{mean(dl):+.5f}` / "
        f"`{max(map(abs, dl)):.4f}` | |",
    ]
    vsteps = sorted(set(val_a) & set(val_b))
    lines.append(
        "| `val/loss` at steps "
        + ", ".join(map(str, vsteps))
        + " | "
        + ", ".join(f"`{val_a[s]:.4f}`" for s in vsteps)
        + " | "
        + ", ".join(f"`{val_b[s]:.4f}`" for s in vsteps)
        + " |"
    )
    return "\n".join(lines)


def table_prompts(snap: Snapshot) -> str:
    a = {int(s): set(v) for s, v in snap.prompts["v7x"]["steps"].items()}
    b = {int(s): set(v) for s, v in snap.prompts["gb200"]["steps"].items()}
    steps = sorted(set(a) & set(b))
    shared = [len(a[s] & b[s]) for s in steps]
    ua = set().union(*(a[s] for s in steps))
    ub = set().union(*(b[s] for s in steps))
    return (
        f"- Distinct training prompts used in steps {steps[0]}–{steps[-1]}: TPU v7x `{len(ua):,}`, GB200 "
        f"`{len(ub):,}`; used by both runs: `{len(ua & ub):,}` (`{100 * len(ua & ub) / len(ua):.1f}%` of TPU v7x's, "
        f"`{100 * len(ua & ub) / len(ub):.1f}%` of GB200's).\n"
        f"- Same prompt in the same step: `{min(shared)}`–`{max(shared)}` of 64 per step (mean "
        f"`{mean(shared):.1f}`), falling from `{shared[0]}` at step 1 to `{shared[-1]}` at step {steps[-1]}."
    )


TABLES = {
    "summary": table_summary,
    "runs": table_runs,
    "invariants": table_invariants,
    "diffs": table_diffs,
    "adaptations": table_adaptations,
    "eval": table_eval,
    "eval_paired": table_eval_paired,
    "train": table_train,
    "wall": table_wall,
    "step_phases": table_step_phases,
    "sft": table_sft,
    "prompts": table_prompts,
}


def render_blocks(text: str, snap: Snapshot) -> tuple[str, list[str]]:
    """Return (text with every AUTO-GENERATED block refreshed, names of blocks that changed)."""
    changed = []
    for name, fn in TABLES.items():
        begin = f"<!-- BEGIN AUTO-GENERATED {name} (make_report.py) -->"
        end = f"<!-- END AUTO-GENERATED {name} -->"
        i, j = text.find(begin), text.find(end)
        if i < 0 or j < i:
            changed.append(f"{name} (markers missing)")
            continue
        new = f"{begin}\n{fn(snap)}\n"
        if text[i:j] != new:
            changed.append(name)
            text = text[:i] + new + text[j:]
    return text, changed


# ----------------------------------------------------------------------------------------------------------------
# Figures


def figures(snap: Snapshot) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 9, "axes.grid": True, "grid.alpha": 0.3, "savefig.dpi": 130})
    FIGURES.mkdir(exist_ok=True)

    def style(spec: dict) -> dict:
        ours = spec["platform"] != "v5p"
        return {
            "color": COLORS[spec["key"]],
            "lw": 2.2 if ours else 1.1,
            "alpha": 1.0 if ours else 0.85,
            "label": spec["label"],
            "zorder": 3 if ours else 2,
        }

    # Figure 1: AIME accuracy.
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    base = snap.wandb["runs"][BASE_EVAL_RUN]["history"][0]
    for ax, ds in zip(axes, DATASETS, strict=True):
        x = 0
        ticks, names = [], []
        for spec in RUNS:
            if spec["key"] in snap.eval_pq:
                for step, hatch in ((0, "//"), (MAX_STEP, None)):
                    a, se = acc_and_se(per_question(snap, spec["key"], step, ds))
                    ax.bar(x, a, 0.8, color=COLORS[spec["key"]], alpha=0.45 if step == 0 else 1.0, hatch=hatch)
                    ax.errorbar(x, a, yerr=1.96 * se, color="black", capsize=3, lw=1)
                    ax.text(x, a + 1.96 * se + 1.0, f"{a:.1f}", ha="center", fontsize=8)
                    ticks.append(x)
                    names.append(f"{spec['label'].split(' ')[0]} step {step}")
                    x += 1
            else:
                a = snap.eval_value(spec, MAX_STEP, ds)
                ax.bar(x, a, 0.8, color=COLORS[spec["key"]])
                ax.text(x, a + 1.0, f"{a:.1f}", ha="center", fontsize=8)
                ticks.append(x)
                names.append(f"v5p {spec['label'].split(' ', 1)[1]} step 25")
                x += 1
            x += 0.3
        b = 100 * base[f"{VAL_PREFIX}{ds}/acc/mean@8"]
        ax.axhline(b, ls="--", color="grey", lw=1)
        ax.text(x - 0.5, b + 1, f"no-SFT base {b:.1f}", ha="right", fontsize=8, color="grey")
        ax.set_xticks(ticks, names, fontsize=7, rotation=30, ha="right")
        ax.set_title(f"{DATASET_NAMES[ds]} mean accuracy (%)")
        ax.set_ylim(0, 80)
    axes[0].set_ylabel("accuracy (%)")
    fig.suptitle(
        "TPU v5p: step 25, n=8 per problem. TPU v7x / GB200: step 0 (hatched) and 25, n=30, error bar = 95% CI",
        fontsize=9,
    )
    fig.tight_layout()
    fig.savefig(FIGURES / "fig1_aime.png")
    plt.close(fig)

    # Figure 2: training curves.
    fig, axes = plt.subplots(2, 3, figsize=(13, 7))
    panels = [
        ("critic/score/mean", "train reward (critic/score/mean)"),
        ("response_length/mean", "response length (tokens)"),
        ("num_turns/mean", "turns per trajectory"),
        ("response_length/clip_ratio", "fraction of responses hitting the 16K cap"),
        ("actor/grad_norm", "grad norm"),
        ("actor/pg_clipfrac", "PPO clip fraction"),
    ]
    for ax, (metric, title) in zip(axes.flat, panels, strict=True):
        for spec in RUNS:
            ser = snap.series(spec, metric)
            xs = [s for s in range(1, MAX_STEP + 1) if s in ser]
            ax.plot(xs, [ser[s] for s in xs], **style(spec))
        ax.set_title(title)
        ax.set_xlabel("step")
    axes[0][0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(FIGURES / "fig2_training.png")
    plt.close(fig)

    # Figure 3: step time.
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.2), gridspec_kw={"width_ratios": [1.3, 1]})
    for spec in RUNS:
        st = step_time(snap, spec)
        axes[0].plot(sorted(st), [st[s] for s in sorted(st)], marker="o", ms=2.5, **style(spec))
    axes[0].set_yscale("log")
    axes[0].set_ylim(8, 4000)
    axes[0].set_title("timing_s/step minus checkpoint save (log scale)")
    axes[0].set_xlabel("step")
    axes[0].set_ylabel("seconds")
    axes[0].legend(fontsize=7, ncol=3, loc="lower center")
    parts = [
        ("timing_s/gen", "waiting for rollout (gen)"),
        ("timing_s/update_actor", "update_actor"),
        ("timing_s/update_weights", "update_weights"),
    ]
    bottoms = [0.0] * len(RUNS)
    for metric, label in parts:
        vals = [phase_mean(snap, spec, metric) for spec in RUNS]
        axes[1].bar(range(len(RUNS)), vals, bottom=bottoms, label=label)
        bottoms = [b + v for b, v in zip(bottoms, vals, strict=True)]
    for i, spec in enumerate(RUNS):
        total = mean(step_time(snap, spec).values())
        axes[1].text(i, bottoms[i] + 15, f"{total:.0f}s", ha="center", fontsize=8)
    axes[1].set_xticks(range(len(RUNS)), [s["label"] for s in RUNS], fontsize=7, rotation=20, ha="right")
    axes[1].set_title("timing_s/step by phase, mean of steps 2–25 (seconds)")
    axes[1].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(FIGURES / "fig3_step_time.png")
    plt.close(fig)

    # Figure 4: SFT.
    tr_a, val_a = sft_series(snap, "v5p")
    tr_b, val_b = sft_series(snap, "v7x")
    steps = sorted(set(tr_a) & set(tr_b))
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.8))
    for tr, val, color, lw, label in (
        (tr_a, val_a, "#3e6fb0", 2.6, "TPU v5p SFT (Askhat)"),
        (tr_b, val_b, "#e8710a", 0.9, "TPU v7x SFT (ours)"),
    ):
        axes[0].plot(steps, [tr[s][0] for s in steps], color=color, lw=lw, alpha=0.7, label=f"{label} train")
        vs = sorted(val)
        axes[0].plot(vs, [val[s] for s in vs], color=color, marker="o", ls="--", label=f"{label} val")
    axes[0].set_yscale("log")
    axes[0].set_xlabel("step")
    axes[0].set_title("SFT loss (log scale)")
    axes[0].legend(fontsize=7)
    axes[1].plot(steps, [tr_b[s][0] - tr_a[s][0] for s in steps], color="black", lw=0.8)
    axes[1].axhline(0, color="grey", lw=0.8)
    axes[1].set_xlabel("step")
    axes[1].set_title("train/loss difference, v7x − v5p (same tokens every step)")
    fig.tight_layout()
    fig.savefig(FIGURES / "fig4_sft.png")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--print", action="store_true", help="print every table")
    parser.add_argument("--check", type=Path, help="fail if an AUTO-GENERATED block of FILE is stale")
    parser.add_argument("--write", type=Path, help="refresh the AUTO-GENERATED blocks of FILE in place")
    parser.add_argument("--figures", action="store_true", help="write figures/*.png (needs matplotlib)")
    args = parser.parse_args()
    snap = Snapshot()
    if args.print:
        for name, fn in TABLES.items():
            print(f"<!-- {name} -->\n{fn(snap)}\n")
    if args.check:
        _, changed = render_blocks(args.check.read_text(encoding="utf-8"), snap)
        if changed:
            sys.exit(f"{args.check}: stale or missing blocks: {', '.join(changed)}")
        print(f"{args.check}: all {len(TABLES)} blocks match")
    if args.write:
        text, changed = render_blocks(args.write.read_text(encoding="utf-8"), snap)
        args.write.write_text(text, encoding="utf-8")
        print(f"{args.write}: refreshed {', '.join(changed) or 'nothing'}")
    if args.figures:
        figures(snap)


if __name__ == "__main__":
    main()
