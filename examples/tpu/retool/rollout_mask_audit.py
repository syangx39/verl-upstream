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
"""Whole-session token-mask audit of verl V1 rollout audit dumps (``rollout_audit/<step>.jsonl[.gz]``).

Each record (written by ``verl/trainer/ppo/v1/rollout_audit.py``) is one multi-turn session with its
``response_ids``, ``response_mask`` (``1`` = the token is in the policy loss) and ``segments`` (maximal runs of
equal mask, decoded with special tokens). The checks look at the whole response, not one segment at a time, so a
``<tool_response>`` block that straddles two mask segments is caught; a per-segment substring search misses it
(``per_segment_check`` keeps the earlier check so the tests can show the difference).

A session is a violation if any of these fails:

1. integrity: ``len(response_ids) == len(response_mask)``, mask values are 0/1, the recorder reported
   ``mask_length_matches`` and no ``error`` / ``missing_fields``;
2. segments: they are exactly the maximal runs of the mask array, and their texts concatenate to
   ``response_text`` (check 5 relies on this);
3. token level, with the Qwen3 special-token ids below: every ``<tool_response>`` ... ``</tool_response>`` block,
   tags included, is entirely ``mask == 0`` and the tool-response tags are paired and not nested (the agent loop
   writes them); every ``<tool_call>`` / ``</tool_call>`` tag token and every well-formed tool-call block is
   ``mask == 1``; no ``<|im_start|>`` (chat-template role header) is a loss token;
4. template: every maximal ``mask == 0`` run is exactly ``\\n<|im_start|>user\\n``, one or more tool-response blocks
   joined by ``\\n``, then ``<|im_end|>\\n<|im_start|>assistant\\n`` (plus the empty
   ``<think>\\n\\n</think>\\n\\n`` that the Qwen3 chat template appends when thinking is disabled). So the only
   tokens excluded from the loss are the tool turns injected by the agent loop, and every model-generated token is a
   loss token;
5. character level, independent of the token ids: on the concatenated decoded text with a per-character mask,
   every ``<tool_response>`` ... ``</tool_response>`` block is all mask 0, every tool-call tag and well-formed
   tool-call block all mask 1, each ``mask == 0`` segment matches the template of check 4, and each special tag
   occurs as often in the text as its id occurs in ``response_ids`` (this confirms the id table on the data).

Malformed tool-call tags written by the model (a second ``<tool_call>`` before ``</tool_call>``, a stray
``</tool_call>``, or a call cut off by the length limit) are counted in ``audit_sessions_malformed_tool_call_tags``;
they are not mask violations as long as all of those tokens are loss tokens.

Scope: only the sessions in the dump are audited (``trainer.rollout_audit_sessions`` per step), not every training
trajectory.

Usage::

    python3 rollout_mask_audit.py <rollout_audit dir or files ...>   # prints per-step and total counts as JSON
"""

import argparse
import gzip
import json
import re
import sys
from collections import Counter
from pathlib import Path

# Qwen3 tokenizer ids (Qwen/Qwen3-30B-A3B ``tokenizer.json``); check 5 re-derives the tag counts from the decoded
# text, so a wrong id shows up as ``char_token_tag_count_mismatch``.
IM_START, IM_END = 151644, 151645
TOOL_CALL_OPEN, TOOL_CALL_CLOSE = 151657, 151658
TOOL_RESPONSE_OPEN, TOOL_RESPONSE_CLOSE = 151665, 151666
THINK_OPEN, THINK_CLOSE = 151667, 151668
NEWLINE, DOUBLE_NEWLINE, USER, ASSISTANT = 198, 271, 872, 77091
TOOL_TURN_PREFIX = (NEWLINE, IM_START, USER, NEWLINE)
TOOL_TURN_SUFFIX = (IM_END, NEWLINE, IM_START, ASSISTANT, NEWLINE)
EMPTY_THINK = (THINK_OPEN, DOUBLE_NEWLINE, THINK_CLOSE, DOUBLE_NEWLINE)
TAG_IDS = {
    "<think>": THINK_OPEN,
    "</think>": THINK_CLOSE,
    "<|im_start|>": IM_START,
    "<|im_end|>": IM_END,
    "<tool_call>": TOOL_CALL_OPEN,
    "</tool_call>": TOOL_CALL_CLOSE,
    "<tool_response>": TOOL_RESPONSE_OPEN,
    "</tool_response>": TOOL_RESPONSE_CLOSE,
}

TOOL_RESPONSE_TEXT_RE = re.compile(r"<tool_response>.*?(?:</tool_response>|\Z)", re.S)
TOOL_CALL_TAG_TEXT_RE = re.compile(r"</?tool_call>")
TOOL_CALL_BLOCK_TEXT_RE = re.compile(r"<tool_call>(?:(?!<tool_call>).)*?</tool_call>", re.S)
TOOL_TURN_TEXT_RE = re.compile(
    r"\A\n<\|im_start\|>user\n<tool_response>(?:(?!<tool_response>).)*?</tool_response>"
    r"(?:\n<tool_response>(?:(?!<tool_response>).)*?</tool_response>)*<\|im_end\|>\n<\|im_start\|>assistant\n"
    r"(?:<think>\n\n</think>\n\n)?\Z",
    re.S,
)

COUNT_KEYS = (
    "audit_sessions",
    "audit_mask_violations",
    "audit_mask_length_matches",
    "audit_sessions_with_tool_response",
    "audit_tool_response_blocks",
    "audit_tool_response_tokens",
    "audit_tool_response_tokens_in_loss",
    "audit_tool_call_blocks",
    "audit_tool_call_tokens_not_in_loss",
    "audit_sessions_malformed_tool_call_tags",
    "audit_mask0_runs",
    "audit_mask0_runs_not_tool_turn",
    "audit_mask0_tokens",
    "audit_im_start_in_loss",
)


def _runs(mask: list[int]) -> list[tuple[int, int, int]]:
    """Maximal runs of equal mask as ``(mask, start, end)``."""
    out = []
    start = 0
    for i in range(1, len(mask) + 1):
        if i == len(mask) or mask[i] != mask[start]:
            out.append((mask[start], start, i))
            start = i
    return out


def _blocks(ids: list[int], open_id: int, close_id: int) -> tuple[list[tuple[int, int]], list[str]]:
    """Well-formed ``[open, close]`` token spans (inclusive) and the tag problems found on the way.

    A second open tag before the close drops the first one (``nested_open_tag``); a close tag without an open one is
    ``unpaired_close_tag``; an open tag still pending at the end is ``unclosed``.
    """
    blocks: list[tuple[int, int]] = []
    problems: list[str] = []
    start = None
    for i, t in enumerate(ids):
        if t == open_id:
            if start is not None:
                problems.append("nested_open_tag")
            start = i
        elif t == close_id:
            if start is None:
                problems.append("unpaired_close_tag")
            else:
                blocks.append((start, i))
                start = None
    if start is not None:
        problems.append("unclosed")
    return blocks, problems


def _is_tool_turn(ids: list[int]) -> bool:
    """Whether a ``mask == 0`` run is exactly the agent loop's tool turn (module docstring, check 4)."""
    if tuple(ids[-len(EMPTY_THINK) :]) == EMPTY_THINK:
        ids = ids[: -len(EMPTY_THINK)]
    p, s = len(TOOL_TURN_PREFIX), len(TOOL_TURN_SUFFIX)
    if len(ids) < p + 2 + s or tuple(ids[:p]) != TOOL_TURN_PREFIX or tuple(ids[-s:]) != TOOL_TURN_SUFFIX:
        return False
    body = ids[p:-s]
    i = 0
    while True:
        if i >= len(body) or body[i] != TOOL_RESPONSE_OPEN:
            return False
        try:
            j = body.index(TOOL_RESPONSE_CLOSE, i + 1)
        except ValueError:
            return False
        if any(t in (TOOL_RESPONSE_OPEN, IM_START, IM_END) for t in body[i + 1 : j]):
            return False
        if j == len(body) - 1:
            return True
        if body[j + 1] != NEWLINE:
            return False
        i = j + 2


def audit_session(rec: dict) -> dict:
    """Audit one session; returns the ``COUNT_KEYS`` counts and ``violations`` (reasons; empty if clean)."""
    ids = list(rec.get("response_ids") or [])
    mask = list(rec.get("response_mask") or [])
    segs = list(rec.get("segments") or [])
    reasons: list[str] = []
    out: dict = dict.fromkeys(COUNT_KEYS, 0)
    out["audit_sessions"] = 1

    # 1. integrity
    length_ok = bool(rec.get("mask_length_matches")) and len(ids) == len(mask)
    out["audit_mask_length_matches"] = int(length_ok)
    if not length_ok:
        reasons.append("mask_length_mismatch")
    if any(m not in (0, 1) for m in mask):
        reasons.append("mask_value_not_0_1")
    if rec.get("error"):
        reasons.append("recorder_error")
    if rec.get("missing_fields"):
        reasons.append("recorder_missing_fields")
    if reasons:
        out["violations"] = reasons
        out["audit_mask_violations"] = 1
        return out

    # 2. segments are the maximal runs of the mask and decode to response_text
    runs = _runs(mask)
    if [(s.get("mask"), s.get("start"), s.get("end")) for s in segs] != runs:
        reasons.append("segments_disagree_with_mask_runs")
    text = "".join(str(s.get("text", "")) for s in segs)
    if "response_text" in rec and text != rec["response_text"]:
        reasons.append("segment_texts_differ_from_response_text")

    # 3. token-level blocks over the whole response
    tr_blocks, tr_problems = _blocks(ids, TOOL_RESPONSE_OPEN, TOOL_RESPONSE_CLOSE)
    reasons += [f"tool_response_{p}" for p in tr_problems]
    out["audit_tool_response_blocks"] = len(tr_blocks)
    out["audit_sessions_with_tool_response"] = int(bool(tr_blocks))
    for a, b in tr_blocks:
        out["audit_tool_response_tokens"] += b - a + 1
        out["audit_tool_response_tokens_in_loss"] += sum(mask[a : b + 1])
    if out["audit_tool_response_tokens_in_loss"]:
        reasons.append("tool_response_tokens_in_loss")
    tc_blocks, tc_problems = _blocks(ids, TOOL_CALL_OPEN, TOOL_CALL_CLOSE)
    out["audit_tool_call_blocks"] = len(tc_blocks)
    out["audit_sessions_malformed_tool_call_tags"] = int(bool(tc_problems))
    tc_tag_positions = [i for i, t in enumerate(ids) if t in (TOOL_CALL_OPEN, TOOL_CALL_CLOSE)]
    out["audit_tool_call_tokens_not_in_loss"] = sum(1 - mask[i] for i in tc_tag_positions) + sum(
        (b - a + 1) - sum(mask[a : b + 1]) for a, b in tc_blocks
    )
    if out["audit_tool_call_tokens_not_in_loss"]:
        reasons.append("tool_call_tokens_not_in_loss")
    out["audit_im_start_in_loss"] = sum(1 for t, m in zip(ids, mask, strict=True) if t == IM_START and m == 1)
    if out["audit_im_start_in_loss"]:
        reasons.append("im_start_in_loss")

    # 4. every mask == 0 run is exactly an injected tool turn
    for m, a, b in runs:
        if m == 0:
            out["audit_mask0_runs"] += 1
            out["audit_mask0_tokens"] += b - a
            if not _is_tool_turn(ids[a:b]):
                out["audit_mask0_runs_not_tool_turn"] += 1
    if out["audit_mask0_runs_not_tool_turn"]:
        reasons.append("mask0_run_not_tool_turn")

    # 5. character level on the decoded text (independent of the id table)
    char_mask: list[int] = []
    for s in segs:
        char_mask += [int(s.get("mask", -1))] * len(str(s.get("text", "")))
    if any(text.count(tag) != ids.count(tid) for tag, tid in TAG_IDS.items()):
        reasons.append("char_token_tag_count_mismatch")
    if any(any(char_mask[a:b]) for a, b in (mt.span() for mt in TOOL_RESPONSE_TEXT_RE.finditer(text))):
        reasons.append("char_tool_response_in_loss")
    tc_spans = [mt.span() for mt in TOOL_CALL_TAG_TEXT_RE.finditer(text)]
    tc_spans += [mt.span() for mt in TOOL_CALL_BLOCK_TEXT_RE.finditer(text)]
    if any(not all(m == 1 for m in char_mask[a:b]) for a, b in tc_spans):
        reasons.append("char_tool_call_not_in_loss")
    if any(s.get("mask") == 0 and not TOOL_TURN_TEXT_RE.match(str(s.get("text", ""))) for s in segs):
        reasons.append("char_mask0_segment_not_tool_turn")

    out["violations"] = sorted(set(reasons))
    out["audit_mask_violations"] = int(bool(reasons))
    return out


def per_segment_check(rec: dict) -> bool:
    """The earlier per-segment check (a substring search inside each segment); True means "violation".

    It misses a tool-response block whose tags or content fall into different segments, e.g. the content and the
    closing tag in a ``mask == 1`` segment. Kept only so the tests can show what the whole-session audit adds.
    """
    mask = rec.get("response_mask", [])
    bad = not (bool(rec.get("mask_length_matches")) and len(rec.get("response_ids", [])) == len(mask))
    bad = bad or bool(rec.get("error")) or bool(rec.get("missing_fields"))
    for seg in rec.get("segments", []):
        seg_text = seg.get("text", "")
        if any(m != seg["mask"] for m in mask[seg["start"] : seg["end"]]):
            bad = True
        if seg["mask"] == 1:
            bad = bad or "<tool_response>" in seg_text
        else:
            bad = bad or "<tool_call>" in seg_text
    return bad


def load_records(path: Path) -> list[dict]:
    """Records of one ``.jsonl`` or ``.jsonl.gz`` audit file."""
    raw = gzip.decompress(path.read_bytes()) if path.suffix == ".gz" else path.read_bytes()
    return [json.loads(line) for line in raw.decode("utf-8").splitlines() if line.strip()]


def audit_records(records: list[dict]) -> dict:
    """Summed counts over sessions plus ``violation_reasons`` (reason -> number of sessions)."""
    total: dict = dict.fromkeys(COUNT_KEYS, 0)
    reasons: Counter = Counter()
    for rec in records:
        res = audit_session(rec)
        for key in COUNT_KEYS:
            total[key] += res[key]
        reasons.update(res["violations"])
    total["violation_reasons"] = dict(sorted(reasons.items()))
    return total


def step_of(path: Path) -> int:
    return int(path.name.split(".")[0])


def audit_files(paths: list[Path]) -> dict:
    """``{"steps": {step: counts}, "totals": counts}`` over audit files named ``<step>.jsonl[.gz]``."""
    steps: dict[str, dict] = {}
    totals: dict = dict.fromkeys(COUNT_KEYS, 0)
    reasons: Counter = Counter()
    for path in sorted(paths, key=step_of):
        res = audit_records(load_records(path))
        steps[str(step_of(path))] = res
        for key in COUNT_KEYS:
            totals[key] += res[key]
        reasons.update(res["violation_reasons"])
    totals["violation_reasons"] = dict(sorted(reasons.items()))
    return {"steps": steps, "totals": totals}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("paths", nargs="+", type=Path, help="rollout_audit directories or <step>.jsonl[.gz] files")
    args = parser.parse_args()
    files: list[Path] = []
    for p in args.paths:
        files += sorted(p.glob("*.jsonl*")) if p.is_dir() else [p]
    res = audit_files(files)
    print(json.dumps(res, indent=2))
    return 1 if res["totals"]["audit_mask_violations"] else 0


if __name__ == "__main__":
    sys.exit(main())
