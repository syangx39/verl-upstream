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
"""CPU tests for the shared ReTool evidence tools in ``examples/tpu/retool/``.

* ``rollout_mask_audit.py``: the whole-session audit catches mask errors that straddle segments, which the earlier
  per-segment substring check misses;
* ``step_timing_breakdown.py``: the TensorBoard event reader and the per-step wall-clock breakdown.
"""

import gzip
import importlib.util
import json
import struct
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
RETOOL = REPO_ROOT / "examples" / "tpu" / "retool"


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, RETOOL / f"{name}.py")
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


audit = _load("rollout_mask_audit")
timing = _load("step_timing_breakdown")

TEXT = {
    audit.IM_START: "<|im_start|>",
    audit.IM_END: "<|im_end|>",
    audit.TOOL_CALL_OPEN: "<tool_call>",
    audit.TOOL_CALL_CLOSE: "</tool_call>",
    audit.TOOL_RESPONSE_OPEN: "<tool_response>",
    audit.TOOL_RESPONSE_CLOSE: "</tool_response>",
    audit.THINK_OPEN: "<think>",
    audit.THINK_CLOSE: "</think>",
    audit.NEWLINE: "\n",
    audit.DOUBLE_NEWLINE: "\n\n",
    audit.USER: "user",
    audit.ASSISTANT: "assistant",
}


def _session(tokens: list[tuple[int, int]]) -> dict:
    """An audit record for ``(token id, mask)`` pairs, with segments decoded like the recorder does."""
    ids = [t for t, _ in tokens]
    mask = [m for _, m in tokens]
    segs = [
        {"mask": m, "start": a, "end": b, "num_tokens": b - a, "text": "".join(TEXT.get(t, "w") for t in ids[a:b])}
        for m, a, b in audit._runs(mask)
    ]
    return {
        "response_ids": ids,
        "response_mask": mask,
        "segments": segs,
        "response_text": "".join(s["text"] for s in segs),
        "mask_length_matches": True,
        "missing_fields": [],
        "error": None,
    }


MODEL_TURN = [(1, 1), (audit.TOOL_CALL_OPEN, 1), (2, 1), (audit.TOOL_CALL_CLOSE, 1), (audit.IM_END, 1)]
ANSWER = [(7, 1), (8, 1), (audit.IM_END, 1)]


def _tool_turn(think: bool = False) -> list[int]:
    toks = [audit.NEWLINE, audit.IM_START, audit.USER, audit.NEWLINE, audit.TOOL_RESPONSE_OPEN, 5, 6]
    toks += [audit.TOOL_RESPONSE_CLOSE, audit.IM_END, audit.NEWLINE, audit.IM_START, audit.ASSISTANT, audit.NEWLINE]
    return toks + (list(audit.EMPTY_THINK) if think else [])


def test_clean_sessions_pass_with_and_without_thinking():
    for think in (False, True):
        rec = _session(MODEL_TURN + [(t, 0) for t in _tool_turn(think)] + ANSWER)
        res = audit.audit_session(rec)
        assert res["violations"] == [], res
        assert res["audit_tool_response_blocks"] == 1 and res["audit_tool_call_blocks"] == 1
        assert res["audit_mask0_runs"] == 1 and res["audit_mask0_runs_not_tool_turn"] == 0
        assert not audit.per_segment_check(rec)


def test_tool_response_straddling_segments_is_caught_but_missed_per_segment():
    # The tool turn starts as mask 0 but its content, closing tag and the next role header are loss tokens.
    turn = [(t, 0 if i < 6 else 1) for i, t in enumerate(_tool_turn())]
    rec = _session(MODEL_TURN + turn + ANSWER)
    assert not audit.per_segment_check(rec), "the per-segment check misses this leak"
    res = audit.audit_session(rec)
    assert res["audit_mask_violations"] == 1
    assert res["audit_tool_response_tokens_in_loss"] == 2  # content token 6 and </tool_response>
    for reason in (
        "tool_response_tokens_in_loss",
        "char_tool_response_in_loss",
        "im_start_in_loss",
        "mask0_run_not_tool_turn",
        "char_mask0_segment_not_tool_turn",
    ):
        assert reason in res["violations"], res["violations"]


def test_open_tag_in_loss_is_caught_by_both_checks():
    turn = [(t, 1 if i < 5 else 0) for i, t in enumerate(_tool_turn())]
    rec = _session(MODEL_TURN + turn + ANSWER)
    assert audit.per_segment_check(rec)
    assert "tool_response_tokens_in_loss" in audit.audit_session(rec)["violations"]


def test_model_tokens_excluded_from_loss_are_caught():
    # Model text right before the tool turn is masked out; no tag is involved, so only the template check sees it.
    rec = _session(MODEL_TURN[:-1] + [(audit.IM_END, 0), (9, 0)] + [(t, 0) for t in _tool_turn()] + ANSWER)
    assert not audit.per_segment_check(rec)
    res = audit.audit_session(rec)
    assert "mask0_run_not_tool_turn" in res["violations"] and "char_mask0_segment_not_tool_turn" in res["violations"]


def test_masked_tool_call_closing_tag_is_caught():
    model = MODEL_TURN[:3] + [(audit.TOOL_CALL_CLOSE, 0), (audit.IM_END, 1)]
    rec = _session(model + [(t, 0) for t in _tool_turn()] + ANSWER)
    assert not audit.per_segment_check(rec), "'</tool_call>' does not contain '<tool_call>'"
    res = audit.audit_session(rec)
    assert "tool_call_tokens_not_in_loss" in res["violations"] and "char_tool_call_not_in_loss" in res["violations"]


def test_malformed_model_tool_call_tags_are_reported_not_violations():
    model = MODEL_TURN[:4] + [(audit.TOOL_CALL_CLOSE, 1), (audit.IM_END, 1)]  # "</tool_call></tool_call>"
    res = audit.audit_session(_session(model + [(t, 0) for t in _tool_turn()] + ANSWER))
    assert res["violations"] == [] and res["audit_sessions_malformed_tool_call_tags"] == 1


def test_integrity_failures_are_violations():
    rec = _session(MODEL_TURN + [(t, 0) for t in _tool_turn()] + ANSWER)
    rec["response_mask"] = rec["response_mask"][:-1]
    assert audit.audit_session(rec)["violations"] == ["mask_length_mismatch"]
    rec = _session(MODEL_TURN + [(t, 0) for t in _tool_turn()] + ANSWER)
    rec["segments"][0]["end"] -= 1
    assert "segments_disagree_with_mask_runs" in audit.audit_session(rec)["violations"]


def test_audit_files_reads_plain_and_gzip(tmp_path):
    good = _session(MODEL_TURN + [(t, 0) for t in _tool_turn()] + ANSWER)
    bad = _session(MODEL_TURN + [(t, 0 if i < 6 else 1) for i, t in enumerate(_tool_turn())] + ANSWER)
    (tmp_path / "1.jsonl").write_text(json.dumps(good) + "\n" + json.dumps(good) + "\n", encoding="utf-8")
    (tmp_path / "2.jsonl.gz").write_bytes(gzip.compress((json.dumps(bad) + "\n").encode("utf-8")))
    res = audit.audit_files(sorted(tmp_path.iterdir()))
    assert list(res["steps"]) == ["1", "2"]
    assert res["totals"]["audit_sessions"] == 3 and res["totals"]["audit_mask_violations"] == 1
    assert res["steps"]["2"]["violation_reasons"]["tool_response_tokens_in_loss"] == 1


def _varint(n: int) -> bytes:
    out = bytearray()
    while True:
        b, n = n & 0x7F, n >> 7
        out.append(b | (0x80 if n else 0))
        if not n:
            return bytes(out)


def _field(num: int, wire: int, payload: bytes) -> bytes:
    key = _varint((num << 3) | wire)
    return key + (_varint(len(payload)) + payload if wire == 2 else payload)


def _record(event: bytes) -> bytes:
    """TFRecord framing (length, length CRC, data, data CRC); the reader does not check the CRCs."""
    return struct.pack("<Q", len(event)) + b"\0" * 4 + event + b"\0" * 4


def _scalar_record(step: int, wall: float, tag: str) -> bytes:
    value = _field(1, 2, tag.encode()) + _field(2, 5, struct.pack("<f", 1.0))
    event = _field(1, 1, struct.pack("<d", wall)) + (_field(2, 0, _varint(step)) if step else b"")
    return _record(event + _field(5, 2, _field(1, 2, value)))


def test_tensorboard_reader_and_breakdown(tmp_path):
    # Step N logs at log_start(N); its 3 scalars take 0.5 s each, so tb_span(N) = 1.0 s.
    log_start = {0: 1000.0, 1: 1100.0, 2: 1150.0, 3: 1300.0}
    header = _record(_field(1, 1, struct.pack("<d", 999.0)) + _field(3, 2, b"brain.Event:2"))
    records = [_scalar_record(s, t + 0.5 * k, f"m/{k}") for s, t in log_start.items() for k in range(3)]
    path = tmp_path / "events.out.tfevents.test"
    path.write_bytes(header + b"".join(records))
    events = timing.read_scalar_events(path)
    assert len(events) == 12 and events[0] == (0, 1000.0, "m/0") and events[-1] == (3, 1301.0, "m/2")
    windows = timing.tb_log_windows(events)
    assert windows[2] == {"first": 1150.0, "last": 1151.0, "span": 1.0, "scalars": 3}

    metrics = {
        str(s): {
            "timing_s/step": step_t,
            "timing_s/gen": 0.0,
            "timing_s/update_actor": 1.0,
            "timing_s/update_weights": 1.0,
            **({"timing_s/testing": 100.0} if s == 3 else {}),
        }
        for s, step_t in ((1, 90.0), (2, 40.0), (3, 30.0))
    }
    sync = [1097.5, 1148.0, 1196.0]  # end of weight sync in each step, before testing in step 3
    rows = timing.breakdown(metrics, windows, 3, sync)
    assert [r["wall_s"] for r in rows] == [100.0, 50.0, 150.0]
    assert [r["outside_timers_s"] for r in rows] == [10.0, 10.0, 20.0]
    assert [r["residual_s"] for r in rows] == [9.0, 9.0, 19.0]
    assert [r["post_sync_to_log_s"] for r in rows] == [2.5, 2.0, 104.0]
    summary = timing.summarize(rows, 2, 2)
    assert summary["mean_wall_s"] == 50.0 and summary["mean_tb_span_prev_step_s"] == 1.0


def test_log_marker_times_shift_to_utc(tmp_path):
    log = tmp_path / "head.log"
    log.write_text("x INFO:2026-11-04 12:00:00,500:update_weights done\nother 2026-11-04 12:00:01,000\n")
    (t,) = timing.log_marker_times(log, "update_weights done", -8.0)
    assert t == 1793822400.5  # 2026-11-04 20:00:00.5 UTC


def test_infer_log_utc_offset_from_audit_dump_lines(tmp_path):
    windows = {1: {"first": 1793822400.0}, 2: {"first": 1793822500.0}}  # 20:00:00 and 20:01:40 UTC
    log = tmp_path / "head.log"
    log.write_text(
        "INFO:2026-11-04 12:00:00,700:Dumped 8 rollout audit records to /data/r/rollout_audit/1.jsonl\n"
        "INFO:2026-11-04 12:01:40,900:Dumped 8 rollout audit records to /data/r/rollout_audit/2.jsonl\n"
    )
    assert timing.infer_log_utc_offset_hours(log, windows) == -8.0
