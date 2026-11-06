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
"""Recompute the Stage P3 evidence for p3-retool-rl-11 from its raw run files.

Inputs (``--data``), downloaded from the run directory (see README.md):
  driver.log                      attempts/bef2992e/logs/driver.log
  validation/0.jsonl, 25.jsonl    Step-0 / Step-25 AIME validation dumps
  rollout_audit/<step>.jsonl      token-level audit of 8 training sessions per step
  src/AIME_2024.parquet           inputs/dataset/Maxwell-Jia/AIME_2024/data/train-00000-of-00001.parquet
  src/aime_2025.parquet           inputs/dataset/yentinglin/aime_2025/data/train-00000-of-00001.parquet

The checks use the code of the run's commit (``--commit``, read with ``git show`` from ``--repo``):
summarize_validation_dump / validation_question_id (verl/trainer/ppo/v1/rollout_audit.py), math_dapo.compute_score
(verl/utils/reward_score/math_dapo.py) and the AIME prompt mapping (examples/tpu/retool/retool_dataset_utils.py).

Outputs (``--out``): validation_summaries.json, scorer_check.json, train_metrics.json.
"""

import argparse
import collections
import hashlib
import importlib.util
import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import pyarrow.parquet as pq

RUN_COMMIT = "aa10dfed7b64d78e166c484f4649f62d7880b8c9"
SOURCES = {
    "aime_2024": ("Maxwell-Jia/AIME_2024", "AIME_2024.parquet"),
    "aime_2025": ("yentinglin/aime_2025", "aime_2025.parquet"),
}
TRAIN_KEYS = (
    "critic/rewards/mean",
    "actor/grad_norm",
    "actor/ppo_kl",
    "actor/rollout_corr/k3_kl",
    "actor/rollout_corr/ppl_ratio",
    "training/off_policy/trajectory_staleness/max",
    "num_turns/mean",
    "response_length/mean",
    "timing_s/step",
    "timing_s/gen",
    "timing_s/update_actor",
    "timing_s/update_weights",
)
ANSI = re.compile(r"\x1b\[[0-9;]*m")
SPECIAL_TOKENS = re.compile(r"<\|[a-z_]+\|>")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_at_commit(repo: Path, commit: str, path: str, tmp: Path):
    name = path.replace("/", "_").removesuffix(".py")
    target = tmp / f"{name}.py"
    target.write_text(subprocess.check_output(["git", "-C", str(repo), "show", f"{commit}:{path}"], text=True))
    spec = importlib.util.spec_from_file_location(name, target)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def expected_questions(data: Path, audit, ds_utils) -> tuple[dict, dict]:
    """Expected question ids per data source, computed from the source rows exactly as the run's dataset did."""
    expected, ground_truth = {}, {}
    for data_source, (hf_name, file_name) in SOURCES.items():
        ids = []
        for row in pq.read_table(data / "src" / file_name).to_pylist():
            mapped = ds_utils.map_fn(row, data_source=hf_name)
            question_id = audit.validation_question_id(None, mapped["prompt"])
            ids.append(question_id)
            ground_truth[question_id] = mapped["reward_model"]["ground_truth"]
        assert len(set(ids)) == len(ids), f"duplicate question ids in {file_name}"
        expected[data_source] = ids
    return expected, ground_truth


def check_validation(data: Path, audit, math_dapo, expected: dict, ground_truth: dict) -> tuple[dict, dict]:
    summaries, scorer = {}, {}
    for step in (0, 25):
        path = data / "validation" / f"{step}.jsonl"
        rows = [json.loads(line) for line in path.open()]
        summary = audit.summarize_validation_dump(rows, expected_n=30, expected_questions=expected)
        summaries[f"step_{step}"] = {"file": path.name, "sha256": sha256(path), **summary}
        agree = collections.Counter()
        strict_acc = collections.defaultdict(lambda: collections.defaultdict(list))
        for row in rows:
            loose = math_dapo.compute_score(row["output"], str(row["gts"]), strict_box_verify=False)
            strict = math_dapo.compute_score(row["output"], str(row["gts"]), strict_box_verify=True)
            agree["acc"] += bool(loose["acc"]) == bool(row["acc"])
            agree["score"] += float(loose["score"]) == float(row["score"])
            agree["pred"] += str(loose["pred"]) == str(row["pred"])
            agree["strict_acc"] += bool(strict["acc"]) == bool(row["acc"])
            agree["gts_matches_source"] += str(row["gts"]) == ground_truth.get(row["question_id"])
            strict_acc[row["data_source"]][row["question_id"]].append(float(bool(strict["acc"])))
        scorer[f"validation_step_{step}"] = {
            "rows": len(rows),
            "score_values": dict(collections.Counter(float(row["score"]) for row in rows)),
            "rows_matching_math_dapo_strict_box_verify_false": {k: agree[k] for k in ("acc", "score", "pred")},
            "rows_matching_math_dapo_strict_box_verify_true_acc": agree["strict_acc"],
            "rows_whose_gts_matches_the_source_answer": agree["gts_matches_source"],
            "mean_at_30_if_strict_box_verify_true": {
                ds: sum(sum(v) / len(v) for v in qs.values()) / len(qs) for ds, qs in sorted(strict_acc.items())
            },
        }
    return summaries, scorer


def check_training_scores(data: Path, math_dapo) -> dict:
    """Re-score the audited training sessions (special tokens stripped, as the reward function sees the text)."""
    agree, retool_differs, incorrect_multi_turn, total = collections.Counter(), 0, 0, 0
    for path in sorted((data / "rollout_audit").glob("*.jsonl"), key=lambda p: int(p.stem)):
        for line in path.open():
            record = json.loads(line)
            total += 1
            text, answer, logged = (
                SPECIAL_TOKENS.sub("", record["response_text"]),
                str(record["ground_truth"]),
                float(record["score"]),
            )
            loose = math_dapo.compute_score(text, answer, strict_box_verify=False)
            strict = math_dapo.compute_score(text, answer, strict_box_verify=True)
            agree["strict_box_verify_false"] += float(loose["score"]) == logged and str(loose["pred"]) == str(
                record["reward_extra_info"]["pred"]
            )
            agree["strict_box_verify_true"] += float(strict["score"]) == logged
            retool = float(strict["score"])  # examples/tpu/retool/retool_tpu.py compute_score
            if retool < 0:
                incorrect_multi_turn += record["num_turns"] > 2
                retool = min(-0.6, retool + (record["num_turns"] - 2) / 2 * 0.1)
            retool_differs += retool != logged
    return {
        "audited_training_sessions": total,
        "logged_score_and_pred_match_math_dapo_strict_box_verify_false": agree["strict_box_verify_false"],
        "logged_score_matches_math_dapo_strict_box_verify_true": agree["strict_box_verify_true"],
        "sessions_where_retool_compute_score_would_differ": retool_differs,
        "incorrect_multi_turn_sessions": incorrect_multi_turn,
    }


def train_metrics(data: Path) -> dict:
    steps: dict[int, dict] = {}
    for line in (data / "driver.log").open(errors="replace"):
        match = re.search(r"\) step:(\d+) - (.*)$", ANSI.sub("", line))
        if not match:
            continue
        values = {}
        for item in match.group(2).split(" - "):
            key, _, raw = item.partition(":")
            number = re.search(r"\(([-0-9.eE+]+)\)", raw)
            try:
                values[key.strip()] = float(number.group(1) if number else raw)
            except ValueError:
                pass
        steps.setdefault(int(match.group(1)), {}).update(values)
    keep = {}
    for step, values in sorted(steps.items()):
        keys = [k for k in values if k.startswith("val-core/")] if step in (0, 25) else []
        keys += [k for k in TRAIN_KEYS if k in values]
        keep[str(step)] = {k: values[k] for k in sorted(set(keys))}
    return {"source": "driver.log", "sha256": sha256(data / "driver.log"), "steps": keep}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--repo", type=Path, required=True, help="git checkout that contains --commit")
    parser.add_argument("--commit", default=RUN_COMMIT)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    with tempfile.TemporaryDirectory() as tmp:
        audit = load_at_commit(args.repo, args.commit, "verl/trainer/ppo/v1/rollout_audit.py", Path(tmp))
        math_dapo = load_at_commit(args.repo, args.commit, "verl/utils/reward_score/math_dapo.py", Path(tmp))
        ds_utils = load_at_commit(args.repo, args.commit, "examples/tpu/retool/retool_dataset_utils.py", Path(tmp))
        expected, ground_truth = expected_questions(args.data, audit, ds_utils)
        summaries, scorer = check_validation(args.data, audit, math_dapo, expected, ground_truth)
        scorer["training_rollout_audit"] = check_training_scores(args.data, math_dapo)
    scorer["sources"] = {ds: sha256(args.data / "src" / name) for ds, (_, name) in SOURCES.items()}
    args.out.mkdir(parents=True, exist_ok=True)
    for name, obj in (
        ("validation_summaries", summaries),
        ("scorer_check", scorer),
        ("train_metrics", train_metrics(args.data)),
    ):
        (args.out / f"{name}.json").write_text(json.dumps(obj, indent=1, sort_keys=True) + "\n")
    for step in (0, 25):
        s = summaries[f"step_{step}"]
        print(
            f"step {step}: complete={s['complete']} reasons={s['incomplete_reasons']} "
            + " ".join(
                f"{ds}: {d['num_questions']}q x {d['sessions_per_question']} sessions, mean_acc={d['mean_acc']:.4f}"
                for ds, d in s["data_sources"].items()
            )
        )
    print(json.dumps(scorer, indent=1, sort_keys=True))


if __name__ == "__main__":
    main()
