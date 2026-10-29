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
"""CPU-only unit tests for ``verl.trainer.ppo.v1.rollout_audit``.

The fake TransferQueue below stores per-sample dicts and assembles reads with ``list_of_dict_to_tensordict``,
the same helper the agent loop uses to put trajectories, so token fields come back as nested (jagged) tensors when
row lengths differ and as dense tensors when there is a single row.
"""

import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
import torch
from tensordict.tensorclass import NonTensorData

import verl
from verl.utils.tensordict_utils import list_of_dict_to_tensordict


def _load_rollout_audit():
    """Load ``rollout_audit.py`` from its file.

    ``import verl.trainer.ppo.v1.rollout_audit`` runs ``verl/trainer/ppo/v1/__init__.py``, which imports the V1
    trainers and therefore TransferQueue and (through the agent loop) torchtitan; CPU-only runners lack them.
    ``rollout_audit`` itself only needs the standard library.
    """
    path = Path(verl.__file__).parent / "trainer" / "ppo" / "v1" / "rollout_audit.py"
    spec = importlib.util.spec_from_file_location("rollout_audit_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


rollout_audit = _load_rollout_audit()
ROLLOUT_AUDIT_TQ_FIELDS = rollout_audit.ROLLOUT_AUDIT_TQ_FIELDS
build_rollout_audit_records = rollout_audit.build_rollout_audit_records
collect_rollout_audit_records = rollout_audit.collect_rollout_audit_records
compute_mask_segments = rollout_audit.compute_mask_segments
select_audit_keys = rollout_audit.select_audit_keys
split_trajectory_key = rollout_audit.split_trajectory_key
summarize_validation_dump = rollout_audit.summarize_validation_dump
to_jsonable = rollout_audit.to_jsonable
token_rows_from_field = rollout_audit.token_rows_from_field
validation_dump_identities = rollout_audit.validation_dump_identities
validation_dump_identity_columns = rollout_audit.validation_dump_identity_columns
validation_question_id = rollout_audit.validation_question_id
values_from_field = rollout_audit.values_from_field
write_rollout_audit_jsonl = rollout_audit.write_rollout_audit_jsonl


class _FakeTokenizer:
    """Decodes token IDs as ``<t{id}>`` and records the ``skip_special_tokens`` argument of every call."""

    def __init__(self, fail_on=None):
        self.fail_on = fail_on
        self.skip_special_tokens_args = []

    def decode(self, ids, skip_special_tokens=True):
        """Decode like a HF tokenizer; fails on ``fail_on`` to simulate a bad token."""
        self.skip_special_tokens_args.append(skip_special_tokens)
        ids = list(ids)
        if self.fail_on is not None and self.fail_on in ids:
            raise ValueError(f"cannot decode {self.fail_on}")
        return "".join(f"<t{i}>" for i in ids)


def _build(keys, prompts, responses, masks, **overrides):
    n = len(keys)
    kwargs = dict(
        keys=keys,
        prompt_ids=prompts,
        response_ids=responses,
        response_masks=masks,
        scores=[1.0] * n,
        num_turns=[2] * n,
        reward_extra_infos=[{"acc": 1.0, "pred": "42"}] * n,
        data_sources=["aime"] * n,
        ground_truths=["42"] * n,
        tokenizer=_FakeTokenizer(),
        step=3,
    )
    kwargs.update(overrides)
    return build_rollout_audit_records(**kwargs)


def test_split_trajectory_key():
    assert split_trajectory_key("abc_1_2") == ("abc", "abc_1", 2)
    assert split_trajectory_key("a_b_1_2") == ("a_b", "a_b_1", 2)
    assert split_trajectory_key("weird") == ("weird", "weird", None)


def test_compute_mask_segments_splits_contiguous_runs():
    segments = compute_mask_segments([10, 11, 12, 13, 14, 15], [1, 1, 0, 0, 0, 1])
    assert [(s["mask"], s["start"], s["end"], s["num_tokens"]) for s in segments] == [
        (1, 0, 2, 2),
        (0, 2, 5, 3),
        (1, 5, 6, 1),
    ]
    assert compute_mask_segments([], []) == []


def test_record_has_segments_counters_and_text_with_special_tokens():
    tokenizer = _FakeTokenizer()
    (record,) = _build(["q1_0_0"], [[1, 2]], [[10, 11, 12, 13, 14, 15]], [[1, 1, 0, 0, 0, 1]], tokenizer=tokenizer)

    assert record["error"] is None
    assert record["missing_fields"] == []
    assert (record["key"], record["session"], record["question"], record["output_index"]) == (
        "q1_0_0",
        "q1_0",
        "q1",
        0,
    )
    assert record["step"] == 3
    assert (record["data_source"], record["ground_truth"], record["num_turns"]) == ("aime", "42", 2)
    assert record["score"] == 1.0
    assert record["reward_extra_info"] == {"acc": 1.0, "pred": "42"}
    assert record["prompt_ids"] == [1, 2]
    assert record["response_ids"] == [10, 11, 12, 13, 14, 15]
    assert record["response_mask"] == [1, 1, 0, 0, 0, 1]
    assert record["prompt_text"] == "<t1><t2>"
    assert record["response_text"] == "<t10><t11><t12><t13><t14><t15>"
    assert [(s["mask"], s["start"], s["end"], s["num_tokens"], s["text"]) for s in record["segments"]] == [
        (1, 0, 2, 2, "<t10><t11>"),
        (0, 2, 5, 3, "<t12><t13><t14>"),
        (1, 5, 6, 1, "<t15>"),
    ]
    assert record["num_prompt_tokens"] == 2
    assert record["num_response_tokens"] == 6
    assert record["num_loss_tokens"] == 3
    assert record["num_non_loss_tokens"] == 3
    assert record["mask_length_matches"] is True
    # Text is always decoded WITH special tokens.
    assert tokenizer.skip_special_tokens_args and not any(tokenizer.skip_special_tokens_args)


def test_mask_length_mismatch_is_flagged_and_segments_cover_every_token():
    short_mask, long_mask = _build(["q_0_0", "q_1_0"], [[1], [1]], [[5, 6, 7, 8], [5, 6]], [[1, 1, 0], [1, 0, 1]])

    assert short_mask["mask_length_matches"] is False
    assert short_mask["error"] is None
    assert any("3 values for 4 response tokens" in w for w in short_mask["warnings"])
    # The response token without a mask value gets its own segment with mask None.
    assert [(s["mask"], s["start"], s["end"]) for s in short_mask["segments"]] == [(1, 0, 2), (0, 2, 3), (None, 3, 4)]
    assert (short_mask["num_loss_tokens"], short_mask["num_non_loss_tokens"]) == (2, 1)

    assert long_mask["mask_length_matches"] is False
    assert [(s["mask"], s["start"], s["end"]) for s in long_mask["segments"]] == [(1, 0, 1), (0, 1, 2)]


def test_dense_mask_rows_drop_trailing_padding_only():
    padded, not_padding = _build(
        ["q_0_0", "q_1_0"],
        [[1], [1]],
        [[5, 6], [5, 6]],
        [[1, 0, 0, 0], [1, 0, 0, 1]],
        field_layouts={"response_mask": "dense"},
    )

    assert padded["response_mask"] == [1, 0]
    assert padded["mask_length_matches"] is True
    assert any("trailing padding zeros" in w for w in padded["warnings"])
    # A non-zero value past the response is not padding: keep the raw row and flag the mismatch.
    assert not_padding["response_mask"] == [1, 0, 0, 1]
    assert not_padding["mask_length_matches"] is False


def test_bad_records_set_error_without_raising():
    records = _build(
        ["q_0_0", "q_1_0", "q_2_0"],
        [[1], [1], [1]],
        [[5, 6], "not-token-ids", [7, 99]],
        [[1, 0], [1], [1, 1]],
        tokenizer=_FakeTokenizer(fail_on=99),
    )
    good, bad_ids, bad_decode = records

    assert good["error"] is None
    assert "response_ids" in bad_ids["error"]
    assert bad_ids["response_ids"] is None
    assert bad_ids["prompt_ids"] == [1] and bad_ids["prompt_text"] == "<t1>"
    assert bad_ids["mask_length_matches"] is False
    # Token IDs and segments survive a decode failure.
    assert "response_text" in bad_decode["error"]
    assert bad_decode["response_ids"] == [7, 99]
    assert bad_decode["segments"][0]["num_tokens"] == 2 and bad_decode["segments"][0]["text"] is None


def test_missing_and_misaligned_columns_are_reported():
    records = _build(["q_0_0", "q_1_0"], [[1], [1]], [[5], [6]], None, scores=None, ground_truths=["only-one"])

    for record in records:
        assert record["error"] is None
        assert record["missing_fields"] == ["response_mask", "score", "ground_truth"]
        assert record["score"] is None and record["ground_truth"] is None
        assert any("ground_truth: got 1 values for 2 keys" in w for w in record["warnings"])
        assert record["mask_length_matches"] is False
        # Without a mask, the response is one segment with mask None.
        assert [(s["mask"], s["num_tokens"]) for s in record["segments"]] == [(None, 1)]


def test_records_are_json_serializable(tmp_path):
    (record,) = _build(
        ["q_0_0"],
        [[np.int64(1)]],
        [[np.int64(2), np.int64(3)]],
        [[True, False]],
        scores=[float("nan")],
        num_turns=[np.int64(4)],
        reward_extra_infos=[{"acc": np.float32(1.0), "pred": np.str_("42"), "flags": (1, 2), 3: "int key"}],
        ground_truths=[np.array(["7"])],
        extra_fields=[{"turn_scores": torch.tensor([0.5]), "ids": {2, 1}, "reward_extra_info": {"acc": 1.0}}],
        tags=[{"prompt_len": 1, "response_len": 2, "global_steps": np.int64(5)}],
    )

    assert record["error"] is None
    assert record["response_mask"] == [1, 0]
    assert record["score"] == "nan"
    assert record["extra_fields"] == {"turn_scores": [0.5], "ids": [1, 2]}
    assert record["reward_extra_info"]["3"] == "int key"
    assert record["tq_tag"] == {"prompt_len": 1, "response_len": 2, "global_steps": 5}
    json.dumps(record, allow_nan=False)

    path = tmp_path / "audit" / "3.jsonl"
    write_rollout_audit_jsonl([record, record], str(path))
    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    assert json.loads(lines[0]) == json.loads(json.dumps(record))


def test_select_audit_keys_is_deterministic_and_keeps_sessions_whole():
    keys = [f"q{q}_{s}_{o}" for q in range(6) for s in range(4) for o in range(1 + s % 2)]

    selected = select_audit_keys(keys, n=3, step=7)
    assert selected == select_audit_keys(list(reversed(keys)), n=3, step=7)

    sessions = list(dict.fromkeys(split_trajectory_key(key)[1] for key in selected))
    assert len(sessions) == 3
    for session in sessions:
        session_keys = sorted(key for key in keys if split_trajectory_key(key)[1] == session)
        # Every output of a chosen session is selected, contiguously.
        assert [key for key in selected if split_trajectory_key(key)[1] == session] == session_keys
    assert len(selected) == sum(len([k for k in keys if split_trajectory_key(k)[1] == s]) for s in sessions)

    assert len({tuple(select_audit_keys(keys, n=3, step=step)) for step in range(10)}) > 1
    assert select_audit_keys(keys, n=3, step=7, seed=1) != selected
    assert select_audit_keys(keys, n=0, step=7) == []
    assert sorted(select_audit_keys(keys, n=100, step=7)) == sorted(keys)


def test_token_rows_from_field_handles_nested_dense_and_lists():
    nested = torch.nested.as_nested_tensor([torch.tensor([1, 2, 3]), torch.tensor([4])], layout=torch.jagged)
    assert token_rows_from_field(nested) == ([[1, 2, 3], [4]], "nested")
    assert token_rows_from_field(torch.tensor([[1, 2], [3, 4]])) == ([[1, 2], [3, 4]], "dense")
    assert token_rows_from_field([torch.tensor([1]), [2, 3]]) == ([[1], [2, 3]], "list")
    assert token_rows_from_field(torch.tensor([0.5, 1.0])) == ([0.5, 1.0], "scalar")
    assert token_rows_from_field(None) == (None, None)


def test_values_from_field_handles_non_tensor_variants():
    td = list_of_dict_to_tensordict(
        [
            {"ds": "a", "ef": {"acc": 1.0}, "x": torch.tensor([1])},
            {"ds": "b", "ef": {"acc": 0.0}, "x": torch.tensor([2, 3])},
        ]
    )
    assert values_from_field(td["ds"], 2) == ["a", "b"]  # tensordict LinkedList
    assert values_from_field(td.get("ds"), 2) == ["a", "b"]  # NonTensorStack
    assert values_from_field(td.get("ef"), 2) == [{"acc": 1.0}, {"acc": 0.0}]  # NonTensorStack is a Mapping too
    assert to_jsonable(td.get("ef")) == [{"acc": 1.0}, {"acc": 0.0}]
    assert values_from_field(NonTensorData("aime", batch_size=[2]), 2) == ["aime", "aime"]
    assert values_from_field("aime", 2) == ["aime", "aime"]
    assert values_from_field({"ground_truth": "1"}, 2) == [{"ground_truth": "1"}] * 2
    assert values_from_field(np.array([1, 2]), 2) == [1, 2]
    assert values_from_field(torch.tensor([3, 4]), 2) == [3, 4]
    assert values_from_field([NonTensorData({"a": 1})], 1) == [{"a": 1}]
    assert values_from_field(None, 2) is None


def _sample(prompt, response, mask, score, with_extra_fields=True):
    rm_scores = torch.zeros(len(response), dtype=torch.float32)
    rm_scores[-1] = score
    sample = {
        "prompts": torch.tensor(prompt, dtype=torch.int64),
        "responses": torch.tensor(response, dtype=torch.int64),
        "response_mask": torch.tensor(mask, dtype=torch.int64),
        "rm_scores": rm_scores,
        "num_turns": 3,
        "reward_model": {"ground_truth": "7"},
        "data_source": "retool",
        "raw_prompt": [{"role": "user", "content": f"question {prompt[0]}"}],
    }
    if with_extra_fields:
        sample["extra_fields"] = {"reward_extra_info": {"acc": float(score > 0)}, "turn_scores": []}
    return sample


def _fake_kv_batch_get(storage, missing_raises=False):
    """Fake ``tq.kv_batch_get``; a missing field is either left out or raises, as TransferQueue may do either."""
    calls = []

    def kv_batch_get(keys, partition_id, select_fields):
        calls.append({"keys": list(keys), "partition_id": partition_id, "fields": list(select_fields)})
        rows = [storage[key] for key in keys]
        present = [field for field in select_fields if all(field in row for row in rows)]
        if missing_raises and len(present) != len(select_fields):
            raise ValueError(f"fields not found: {sorted(set(select_fields) - set(present))}")
        return list_of_dict_to_tensordict([{field: row[field] for field in present} for row in rows])

    return kv_batch_get, calls


def _batch(with_extra_fields=True):
    storage = {
        "qa_0_0": _sample([1, 2, 3], [10, 11, 12, 13], [1, 0, 0, 1], 1.0, with_extra_fields),
        "qa_1_0": _sample([1, 2, 3], [14, 15], [1, 1], -1.0, with_extra_fields),
        "qb_0_0": _sample([4, 5], [16, 17, 18], [1, 0, 1], -0.5, with_extra_fields),
        "qb_0_1": _sample([4, 5, 16, 17, 18, 6], [19, 20], [1, 1], -0.5, with_extra_fields),
        "pad0_0_0": _sample([0] * 7, [0], [0], 0.0, with_extra_fields),
    }
    keys = list(storage)
    tags = [{"prompt_len": len(s["prompts"]), "response_len": len(s["responses"])} for s in storage.values()]
    tags[-1]["is_padding"] = True
    return storage, keys, tags


def test_collect_reads_only_selected_keys_and_matches_stored_tokens():
    storage, keys, tags = _batch()
    kv_batch_get, calls = _fake_kv_batch_get(storage)

    records = collect_rollout_audit_records(
        kv_batch_get, keys, tags, partition_id="train", tokenizer=_FakeTokenizer(), step=5, num_sessions=10
    )

    # Padding samples are never audited, and only the selected keys are read, in one call.
    assert [r["key"] for r in records] == ["qa_0_0", "qa_1_0", "qb_0_0", "qb_0_1"]
    assert len(calls) == 1
    assert calls[0]["keys"] == ["qa_0_0", "qa_1_0", "qb_0_0", "qb_0_1"]
    assert calls[0]["fields"] == list(ROLLOUT_AUDIT_TQ_FIELDS)
    for record in records:
        sample = storage[record["key"]]
        assert record["error"] is None
        assert record["missing_fields"] == []
        assert record["warnings"] == []
        assert record["prompt_ids"] == sample["prompts"].tolist()
        assert record["response_ids"] == sample["responses"].tolist()
        assert record["response_mask"] == sample["response_mask"].tolist()
        assert record["score"] == pytest.approx(float(sample["rm_scores"].sum()))
        assert record["reward_extra_info"] == sample["extra_fields"]["reward_extra_info"]
        assert record["extra_fields"] == {"turn_scores": []}
        assert (record["ground_truth"], record["data_source"], record["num_turns"]) == ("7", "retool", 3)
        assert record["raw_prompt"] == sample["raw_prompt"]
        assert record["tq_tag"]["response_len"] == len(sample["responses"])
        assert record["tq_layouts"] == {
            "prompts": "nested",
            "responses": "nested",
            "response_mask": "nested",
            "rm_scores": "nested",
        }
        assert record["step"] == 5
    assert [(s["mask"], s["text"]) for s in records[0]["segments"]] == [(1, "<t10>"), (0, "<t11><t12>"), (1, "<t13>")]
    json.dumps(records, allow_nan=False)


def test_collect_single_key_reads_dense_rows_exactly():
    storage, keys, tags = _batch()
    kv_batch_get, _ = _fake_kv_batch_get(storage)

    records = collect_rollout_audit_records(
        kv_batch_get, keys[1:2], tags[1:2], partition_id="train", tokenizer=_FakeTokenizer(), step=1, num_sessions=4
    )

    (record,) = records
    assert record["tq_layouts"]["prompts"] == "dense"
    assert record["prompt_ids"] == [1, 2, 3]
    assert record["response_ids"] == [14, 15]
    assert record["response_mask"] == [1, 1]
    assert record["mask_length_matches"] is True
    assert record["score"] == -1.0


@pytest.mark.parametrize("missing_raises", [False, True])
def test_collect_degrades_gracefully_when_a_field_is_missing(missing_raises):
    storage, keys, tags = _batch(with_extra_fields=False)
    kv_batch_get, calls = _fake_kv_batch_get(storage, missing_raises=missing_raises)

    records = collect_rollout_audit_records(
        kv_batch_get, keys, tags, partition_id="train", tokenizer=_FakeTokenizer(), step=2, num_sessions=10
    )

    assert len(records) == 4
    for record in records:
        assert record["error"] is None
        assert record["missing_fields"] == ["reward_extra_info"]
        assert record["extra_fields"] is None
        assert record["response_ids"] == storage[record["key"]]["responses"].tolist()
        assert record["mask_length_matches"] is True
    # When the batched read raises, every field is retried on its own.
    assert len(calls) == (1 + len(ROLLOUT_AUDIT_TQ_FIELDS) if missing_raises else 1)


def test_collect_never_selects_padding_only_batches():
    storage, keys, tags = _batch()
    kv_batch_get, calls = _fake_kv_batch_get(storage)

    assert collect_rollout_audit_records(kv_batch_get, keys[-1:], tags[-1:], "train", _FakeTokenizer(), 1, 4) == []
    assert calls == []


def test_summarize_validation_dump_counts_distinct_sessions():
    rows = [
        {"uid": f"qa_{s}_0", "data_source": "aime24", "acc": s == 0, "score": 1.0 if s == 0 else -0.6} for s in range(3)
    ]
    rows += [
        # A multi-output session writes two lines; the last output carries the session's reward.
        {"uid": "qb_0_0", "data_source": "aime24", "acc": 0.0, "score": -0.6},
        {"uid": "qb_0_1", "data_source": "aime24", "acc": 1.0, "score": 1.0},
        # qb is missing its third session.
        {"uid": "qb_1_0", "data_source": "aime24", "acc": 1.0, "score": 1.0},
    ]

    summary = summarize_validation_dump(rows, expected_n=3)

    assert summary["complete"] is False
    assert summary["expected_n"] == 3
    aime = summary["data_sources"]["aime24"]
    assert aime["num_rows"] == 6  # JSONL lines ...
    assert aime["num_sessions"] == 5  # ... are not samples
    assert aime["num_questions"] == 2
    assert aime["question_keys_used"] == ["uid"]
    assert aime["sessions_per_question"] == {"min": 2, "max": 3}
    assert aime["outputs_per_session"] == {"min": 1, "max": 2}
    assert aime["questions_with_unexpected_sessions"] == [{"question": "qb", "num_sessions": 2}]
    assert aime["missing_questions"] is None and aime["unexpected_questions"] is None
    assert aime["complete"] is False
    assert aime["per_question"]["qa"] == {
        "num_sessions": 3,
        "num_rows": 3,
        "mean_acc": pytest.approx(1 / 3),
        "mean_score": pytest.approx(-0.2 / 3),
    }
    assert aime["per_question"]["qb"] == {"num_sessions": 2, "num_rows": 3, "mean_acc": 1.0, "mean_score": 1.0}
    assert aime["mean_acc"] == pytest.approx((1 / 3 + 1.0) / 2)
    assert aime["mean_score"] == pytest.approx((-0.2 / 3 + 1.0) / 2)
    json.dumps(summary, allow_nan=False)


def test_summarize_reports_expected_questions_without_rows():
    rows = [
        {"uid": f"u{q}_{s}_0", "question_id": f"index:{q}", "data_source": "aime24", "acc": 1.0}
        for q in (0, 1)
        for s in range(2)
    ]

    # Every session of index:2 is missing, and aime25 has no row at all.
    expected = {"aime24": ["index:0", "index:1", "index:2"], "aime25": ["index:0"]}
    summary = summarize_validation_dump(rows, expected_n=2, expected_questions=expected)

    aime24 = summary["data_sources"]["aime24"]
    assert aime24["missing_questions"] == ["index:2"]
    assert aime24["num_expected_questions"] == 3
    assert aime24["questions_with_unexpected_sessions"] == []  # the questions in the dump look fine ...
    assert aime24["sessions_per_question"] == {"min": 0, "max": 2}  # ... the missing one counts as 0 sessions
    assert aime24["complete"] is False
    aime25 = summary["data_sources"]["aime25"]
    assert (aime25["num_rows"], aime25["missing_questions"], aime25["complete"]) == (0, ["index:0"], False)
    assert summary["complete"] is False

    full = summarize_validation_dump(rows, expected_n=2, expected_questions={"aime24": ["index:0", "index:1"]})
    assert full["complete"] is True
    assert full["data_sources"]["aime24"]["missing_questions"] == []
    # A question that is not expected is listed but does not make the dump incomplete.
    extra = summarize_validation_dump(rows, expected_n=2, expected_questions={"aime24": ["index:0"]})
    assert extra["data_sources"]["aime24"]["unexpected_questions"] == ["index:1"]
    assert extra["complete"] is True


def test_summarize_keeps_data_sources_apart():
    rows = [
        {"uid": f"{uid}_{s}_0", "question_id": "index:0", "data_source": data_source, "acc": data_source == "aime_2024"}
        for uid, data_source in (("u24", "aime_2024"), ("u25", "aime_2025"))
        for s in range(2)
    ]

    summary = summarize_validation_dump(rows, expected_n=2)

    assert sorted(summary["data_sources"]) == ["aime_2024", "aime_2025"]
    for data_source, mean_acc in (("aime_2024", 1.0), ("aime_2025", 0.0)):
        stats = summary["data_sources"][data_source]
        assert (stats["num_questions"], stats["num_sessions"], stats["mean_acc"]) == (1, 2, mean_acc)

    renamed = [{("dataset" if k == "data_source" else k): v for k, v in row.items()} for row in rows]
    by_key = summarize_validation_dump(renamed, expected_n=2, data_source_key="dataset")
    assert sorted(by_key["data_sources"]) == ["aime_2024", "aime_2025"]
    # Without a data source, the two datasets' index:0 questions collapse into one question.
    merged = summarize_validation_dump(renamed)["data_sources"]
    assert list(merged) == ["unknown"]
    assert merged["unknown"]["per_question"]["index:0"]["num_sessions"] == 4


def test_summarize_never_reports_the_reward_score_as_accuracy():
    rows = [{"uid": f"q_{s}_0", "data_source": "retool", "score": score} for s, score in enumerate([1.0, -0.6, -0.55])]

    summary = summarize_validation_dump(rows, expected_n=3)

    retool = summary["data_sources"]["retool"]
    assert retool["mean_acc"] is None
    assert retool["per_question"]["q"]["mean_acc"] is None
    assert retool["mean_score"] == pytest.approx((1.0 - 0.6 - 0.55) / 3)
    assert retool["per_question"]["q"]["mean_score"] == pytest.approx((1.0 - 0.6 - 0.55) / 3)
    assert "never an accuracy" in summary["metric_fields"]["mean_score"]
    assert "'acc' field" in summary["metric_fields"]["mean_acc"]


def test_summarize_completeness_is_unknown_without_expected_questions():
    rows = [{"uid": f"q_{s}_0", "acc": 1.0} for s in range(2)]

    summary = summarize_validation_dump(rows, expected_n=2)
    assert summary["complete"] is None
    assert summary["data_sources"]["unknown"]["complete"] is None
    assert summary["data_sources"]["unknown"]["missing_questions"] is None

    # Session counts cannot be checked without expected_n, so an expected list alone is not enough either.
    without_n = summarize_validation_dump(rows, expected_questions={"unknown": ["q"]})
    assert without_n["complete"] is None
    assert without_n["data_sources"]["unknown"]["questions_with_unexpected_sessions"] is None
    # A wrong session count is a known problem, with or without an expected list.
    assert summarize_validation_dump(rows, expected_n=3)["complete"] is False


def test_summarize_question_identity():
    # The same question in two runs: different per-run uids, one stable question_id.
    rows = [
        {"uid": f"{run}_{s}_0", "question_id": "sha1:abc", "acc": 1.0} for run in ("run1", "run2") for s in range(2)
    ]

    by_id = summarize_validation_dump(rows)["data_sources"]["unknown"]
    assert by_id["question_keys_used"] == ["question_id"]
    assert (by_id["num_questions"], by_id["num_sessions"]) == (1, 4)

    by_uid = summarize_validation_dump(rows, question_key="uid")["data_sources"]["unknown"]
    assert by_uid["question_keys_used"] == ["uid"]
    assert sorted(by_uid["per_question"]) == ["run1", "run2"]

    custom = [{**row, "problem": f"p-{row['uid'][:4]}"} for row in rows] + [{"uid": "x_0_0"}]
    by_custom = summarize_validation_dump(custom, question_key="problem")["data_sources"]["unknown"]
    assert sorted(by_custom["per_question"]) == ["p-run1", "p-run2"]
    assert by_custom["num_rows_without_question"] == 1


def test_validation_question_id_is_stable_across_runs():
    messages = [{"role": "user", "content": "What is 1+1?"}]
    canonical = json.dumps(messages, sort_keys=True, ensure_ascii=False)

    question_id = validation_question_id({}, messages)

    assert question_id == "sha1:" + hashlib.sha1(canonical.encode("utf-8")).hexdigest()[:16]
    assert validation_question_id(None, [{"content": "What is 1+1?", "role": "user"}]) == question_id
    assert validation_question_id({"index": None}, np.array(messages, dtype=object)) == question_id
    assert validation_question_id({}, [{"role": "user", "content": "What is 2+2?"}]) != question_id
    assert validation_question_id({"index": np.int64(7)}, messages) == "index:7"
    assert validation_question_id(None, None) is None


def _collate(rows):
    """Collate dataset rows into numpy object arrays like ``verl.utils.dataset.rl_dataset.collate_fn``."""
    return {key: np.fromiter((row[key] for row in rows), dtype=object, count=len(rows)) for key in rows[0]}


def test_validation_dump_identities_and_columns():
    messages = [{"role": "user", "content": "p1"}]
    batch = _collate(
        [
            {"uid": "u1", "data_source": "aime_2024", "extra_info": {}, "raw_prompt": messages},
            {"uid": "u2", "data_source": "aime_2025", "extra_info": {"index": 5}, "raw_prompt": []},
            {"uid": "u3", "data_source": "aime_2025", "extra_info": {}, "raw_prompt": None},
        ]
    )
    sha1_id = validation_question_id({}, messages)

    identities = validation_dump_identities(batch)

    assert identities == {
        "u1": {"data_source": "aime_2024", "question_id": sha1_id},
        "u2": {"data_source": "aime_2025", "question_id": "index:5"},
        "u3": {"data_source": "aime_2025", "question_id": None},
    }
    assert validation_dump_identity_columns(["u2_0_0", "u1_1_0", "unknown_0_0"], identities) == {
        "data_source": ["aime_2025", "aime_2024", None],
        "question_id": ["index:5", sha1_id, None],
    }
    assert validation_dump_identity_columns(["u1_0_0"], None) == {}
    # Without a data_source column, rows default to "unknown" as in PPOTrainer._validate.
    assert validation_dump_identities({"uid": np.array(["u"], dtype=object)}) == {
        "u": {"data_source": "unknown", "question_id": None}
    }


def test_validate_dump_writes_data_source_and_question_id(tmp_path, monkeypatch):
    """Runs PPOTrainer._validate with fake rollout, replay buffer and TransferQueue calls."""
    pytest.importorskip("transfer_queue")
    pytest.importorskip("torchtitan")  # verl.trainer.ppo.v1 imports the TorchTitan engine via the agent loop
    from types import SimpleNamespace

    from omegaconf import OmegaConf

    from verl.trainer.ppo.v1 import trainer_base

    n = 2
    storage = {}

    class _AgentLoopManager:
        def generate_sequences(self, batch):
            self.keys = []
            uids = batch.get("uid").tolist()
            data_sources = batch.get("data_source").tolist()
            for row, (uid, data_source) in enumerate(zip(uids, data_sources, strict=True)):
                for session in range(n):
                    correct = session == 0
                    response = torch.full((2 + session,), 7)
                    rm_scores = torch.zeros(len(response))
                    rm_scores[-1] = 1.0 if correct else -0.6
                    key = f"{uid}_{session}_0"
                    storage[key] = {
                        "uid": uid,
                        "prompts": torch.arange(1, 4 + row),  # prompt lengths differ, so reads come back nested
                        "responses": response,
                        "rm_scores": rm_scores,
                        "num_turns": 2,
                        "reward_model": {"ground_truth": "1"},
                        "data_source": data_source,
                        "extra_fields": {"reward_extra_info": {"acc": correct, "score": float(rm_scores[-1])}},
                    }
                    self.keys.append(key)

    agent_loop_manager = _AgentLoopManager()

    def kv_batch_get(keys, partition_id, select_fields):
        rows = [storage[key] for key in keys]
        return list_of_dict_to_tensordict([{field: row[field] for field in select_fields} for row in rows])

    monkeypatch.setattr(trainer_base.tq, "kv_batch_put", lambda **kwargs: None, raising=False)
    monkeypatch.setattr(trainer_base.tq, "kv_batch_get", kv_batch_get, raising=False)
    monkeypatch.setattr(trainer_base.tq, "kv_clear", lambda **kwargs: None, raising=False)
    identity_calls = []
    identities = trainer_base.validation_dump_identities
    monkeypatch.setattr(
        trainer_base, "validation_dump_identities", lambda batch: identity_calls.append(1) or identities(batch)
    )

    class _Trainer(trainer_base.PPOTrainer):
        def on_step_end(self):
            pass

        def on_sample_end(self):
            pass

    dataset_rows = [
        {"data_source": "aime_2024", "extra_info": {}, "raw_prompt": [{"role": "user", "content": "p1"}]},
        {"data_source": "aime_2025", "extra_info": {"index": 5}, "raw_prompt": [{"role": "user", "content": "p2"}]},
    ]

    def run_validate(validation_data_dir):
        trainer = _Trainer.__new__(_Trainer)
        trainer.config = OmegaConf.create(
            {
                "trainer": {"validation_data_dir": validation_data_dir, "log_val_generations": 0},
                "actor_rollout_ref": {"rollout": {"val_kwargs": {"n": n}}},
            }
        )
        trainer.global_steps = 0
        trainer.tokenizer = SimpleNamespace(pad_token_id=0, decode=lambda ids, skip_special_tokens: str(ids.tolist()))
        trainer.val_dataloader = [trainer_base.collate_fn([dict(row) for row in dataset_rows])]
        trainer.agent_loop_manager = agent_loop_manager
        trainer.replay_buffer = SimpleNamespace(
            sample=lambda global_steps, partition_id, batch_size: (
                SimpleNamespace(keys=list(agent_loop_manager.keys), partition_id=partition_id),
                {},
            )
        )
        trainer.reward_loop_manager = SimpleNamespace(reward_loop_worker_handles=[object()])
        trainer._init_dump_executor()
        metrics = trainer._validate()
        trainer._shutdown_dump_executor()
        return metrics

    metrics_without_dump = run_validate(None)
    assert identity_calls == []  # nothing extra runs when the dump is off
    metrics_with_dump = run_validate(str(tmp_path))
    assert len(identity_calls) == 1

    # The dump keys are additive: the validation metrics do not change.
    assert metrics_with_dump == pytest.approx(metrics_without_dump, nan_ok=True)
    rows = [json.loads(line) for line in (tmp_path / "0.jsonl").read_text().splitlines()]
    assert len(rows) == 2 * n
    expected_ids = {"aime_2024": validation_question_id({}, dataset_rows[0]["raw_prompt"]), "aime_2025": "index:5"}
    for row in rows:
        assert row["question_id"] == expected_ids[row["data_source"]]
        assert {"input", "output", "gts", "score", "step", "acc", "uid"} <= row.keys()
    summary = summarize_validation_dump(
        rows, expected_n=n, expected_questions={source: [qid] for source, qid in expected_ids.items()}
    )
    assert summary["complete"] is True
    assert {source: stats["mean_acc"] for source, stats in summary["data_sources"].items()} == {
        "aime_2024": 0.5,
        "aime_2025": 0.5,
    }
