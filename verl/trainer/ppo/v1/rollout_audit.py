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
"""Token-level audit records for multi-turn (tool) rollouts.

``trainer.rollout_data_dir`` only keeps text decoded with ``skip_special_tokens=True``. That is not enough to audit
multi-turn tool trajectories: we need the raw token IDs and the response (loss) mask to check that tool-result
tokens get no loss and that tool results reach the next turn.

This module needs only the standard library (tensor-like values are handled by duck typing, tensordict's
``NonTensorData`` is unwrapped if tensordict is installed, and TransferQueue is passed in as a ``kv_batch_get``
callable), so it can be unit tested on CPU:

- :func:`select_audit_keys` picks whole sessions (every output of ``{uid}_{session_id}``) deterministically per step.
- :func:`fetch_rollout_audit_fields` reads the audit fields of those keys from TransferQueue.
- :func:`audit_columns_from_fields` turns TransferQueue field values (nested or dense tensors, non-tensor stacks)
  into plain per-key Python lists.
- :func:`build_rollout_audit_records` builds one JSON-serializable record per key (plain Python lists in).
- :func:`collect_rollout_audit_records` chains the four steps above for one training batch.
- :func:`write_rollout_audit_jsonl` writes the records as JSONL.
- :func:`summarize_validation_dump` counts distinct sessions per question in a ``trainer.validation_data_dir``
  dump. Multi-output sessions write several JSONL lines, so samples per question must not be inferred from line
  counts.

Trajectory keys have the format ``{uid}_{session_id}_{output_index}``: ``uid`` identifies the question (one GRPO
group), ``{uid}_{session_id}`` one sampled trajectory (session), and ``output_index`` one agent-loop output of the
session.
"""

import hashlib
import json
import logging
import math
import operator
import os
from collections.abc import Callable, Mapping, Sequence
from typing import Any, Optional

try:
    from tensordict.tensorclass import NonTensorData as _NonTensorData
except ImportError:  # tensordict is only needed to unwrap NonTensorData items.
    _NonTensorData = None

logger = logging.getLogger(__name__)

AUDIT_RECORD_VERSION = 1
"""Version of the audit record schema, stored in every record as ``audit_version``."""

ROLLOUT_AUDIT_TQ_FIELDS = (
    "prompts",
    "responses",
    "response_mask",
    "rm_scores",
    "num_turns",
    "reward_model",
    "data_source",
    "extra_fields",
    "raw_prompt",
)
"""TransferQueue fields read for the audit."""

# Record fields filled from the builder columns. Missing core columns are reported in ``missing_fields``.
_CORE_COLUMNS = (
    "prompt_ids",
    "response_ids",
    "response_mask",
    "score",
    "num_turns",
    "reward_extra_info",
    "data_source",
    "ground_truth",
)
_OPTIONAL_COLUMNS = ("extra_fields", "tq_tag", "raw_prompt")


def split_trajectory_key(key: str) -> tuple[str, str, Optional[int]]:
    """Split a ``{uid}_{session_id}_{output_index}`` key into ``(question, session, output_index)``.

    Args:
        key: Trajectory key.

    Returns:
        ``question`` is the uid and ``session`` is ``{uid}_{session_id}``. A key that does not have this format is
        its own question and session, with output index ``None``.
    """
    key = str(key)
    parts = key.rsplit("_", 2)
    if len(parts) == 3:
        output_index = int(parts[2]) if parts[2].isdigit() else None
        return parts[0], f"{parts[0]}_{parts[1]}", output_index
    return key, key, None


def _key_order(key: str) -> tuple[str, int, int, str]:
    """Sort key ordering trajectory keys by (uid, session_id, output_index)."""
    key = str(key)
    parts = key.rsplit("_", 2)
    if len(parts) == 3 and parts[1].isdigit() and parts[2].isdigit():
        return parts[0], int(parts[1]), int(parts[2]), key
    return key, -1, -1, key


def select_audit_keys(keys: Sequence[str], n: int, step: int, seed: int = 0) -> list[str]:
    """Deterministically pick ``n`` whole sessions for ``step``.

    Sessions are ranked by a SHA-256 hash of ``(seed, step, session)``. The choice depends only on the set of keys,
    the step and the seed (not on key order or the Python hash seed) and changes from step to step. All outputs of
    a chosen session are returned together.

    Args:
        keys: Trajectory keys of the batch. Callers should drop synthetic padding keys first.
        n: Number of sessions to pick. ``n <= 0`` picks none; ``n`` larger than the number of sessions picks all.
        step: Global step, mixed into the hash.
        seed: Extra seed, mixed into the hash.

    Returns:
        Keys of the chosen sessions, ordered by (uid, session_id, output_index).
    """
    sessions: dict[str, list[str]] = {}
    for key in dict.fromkeys(str(k) for k in keys):
        sessions.setdefault(split_trajectory_key(key)[1], []).append(key)
    if n <= 0 or not sessions:
        return []

    def rank(session: str) -> tuple[str, str]:
        return hashlib.sha256(f"{seed}:{step}:{session}".encode()).hexdigest(), session

    chosen = sorted(sessions, key=rank)[: int(n)]
    chosen.sort(key=lambda session: min(_key_order(key) for key in sessions[session]))
    return [key for session in chosen for key in sorted(sessions[session], key=_key_order)]


def to_jsonable(obj: Any) -> Any:
    """Recursively convert ``obj`` into JSON-serializable built-in types.

    numpy/torch scalars and arrays become Python numbers and lists, dict keys become strings, non-finite floats
    become strings (``"nan"``, ``"inf"``) so the output stays strict JSON, and unknown objects become ``str(obj)``.

    Args:
        obj: Any value.

    Returns:
        A value made of ``dict``, ``list``, ``str``, ``int``, ``float``, ``bool`` and ``None`` only.
    """
    if obj is None or isinstance(obj, bool | int | str):
        return obj
    if isinstance(obj, float):
        return float(obj) if math.isfinite(obj) else str(obj)
    if _NonTensorData is not None and isinstance(obj, _NonTensorData):
        return to_jsonable(obj.data)
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, list | tuple):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, set | frozenset):
        return [to_jsonable(v) for v in sorted(obj, key=repr)]
    if isinstance(obj, bytes):
        return obj.decode("utf-8", errors="replace")
    # Checked before Mapping: tensordict containers are Mappings but convert to their values with tolist().
    if hasattr(obj, "tolist"):
        try:
            return to_jsonable(obj.tolist())
        except Exception:
            pass
    if isinstance(obj, Mapping):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    return str(obj)


def _unwrap_non_tensor(item: Any) -> Any:
    """Return the payload of a tensordict ``NonTensorData`` item, or ``item`` unchanged."""
    if _NonTensorData is not None and isinstance(item, _NonTensorData):
        return item.data
    return item


def _row_to_list(row: Any) -> Any:
    """Convert one row (tensor, array, sequence or scalar) into a Python list or scalar."""
    row = _unwrap_non_tensor(row)
    if hasattr(row, "tolist"):
        return row.tolist()
    if isinstance(row, list | tuple):
        return list(row)
    return row


def token_rows_from_field(value: Any) -> tuple[Optional[list], Optional[str]]:
    """Convert a per-token TransferQueue field into one exact Python list per row.

    Args:
        value: Field value as returned by ``tq.kv_batch_get``. Nested (jagged) tensors are unbound row by row, so
            no padding is added. Dense 2-D tensors are split into rows as they are: TransferQueue returns them when
            every row has the same length (e.g. a single key). Sequences of rows are converted element-wise.

    Returns:
        ``(rows, layout)`` with layout ``"nested"``, ``"dense"``, ``"scalar"`` (a 0-D/1-D tensor, one value per
        row) or ``"list"``; ``(None, None)`` when ``value`` is ``None``.
    """
    if value is None:
        return None, None
    if getattr(value, "is_nested", False):
        return [row.tolist() for row in value.unbind()], "nested"
    ndim = getattr(value, "ndim", None)
    if isinstance(ndim, int) and hasattr(value, "tolist"):
        rows = value.tolist()
        if ndim == 0:
            return [rows], "scalar"
        return rows, "dense" if ndim >= 2 else "scalar"
    if isinstance(value, list | tuple):
        items = list(value)
    elif hasattr(value, "tolist"):
        items = value.tolist()
    else:
        items = list(value)
    return [_row_to_list(row) for row in items], "list"


def values_from_field(value: Any, num_rows: int) -> Optional[list]:
    """Convert a non-tensor TransferQueue field into one plain Python value per row.

    Args:
        value: Field value as returned by ``tq.kv_batch_get``: a ``NonTensorStack``, a tensordict ``LinkedList``
            (a ``list`` subclass), a ``NonTensorData`` or raw value shared by all rows, a numpy array or a tensor.
        num_rows: Number of rows, used to broadcast a value shared by all rows.

    Returns:
        A list of plain values (``NonTensorData`` items are unwrapped), or ``None`` when ``value`` is ``None``.
    """
    if value is None:
        return None
    # A plain dict is one value shared by all rows. (tensordict stacks are Mappings too, so test for dict.)
    if isinstance(value, str | bytes | dict):
        return [value] * num_rows
    if isinstance(value, list | tuple):
        items = list(value)
    elif hasattr(value, "tolist"):
        items = value.tolist()
        if not isinstance(items, list):
            items = [items] * num_rows
    else:
        items = [value] * num_rows
    return [_unwrap_non_tensor(item) for item in items]


def _sum_row(row: Any) -> Optional[float]:
    """Sum a per-token score row (or pass a scalar through); ``None`` if it cannot be summed."""
    if row is None:
        return None
    try:
        if isinstance(row, list | tuple):
            return float(sum(float(v) for v in row))
        return float(row)
    except (TypeError, ValueError):
        return None


def _has_field(data: Any, field: str) -> bool:
    """Whether a ``kv_batch_get`` result (TensorDict or mapping) contains ``field``."""
    keys = data.keys() if hasattr(data, "keys") else data
    return field in keys


def fetch_rollout_audit_fields(
    kv_batch_get: Callable[..., Any],
    keys: Sequence[str],
    partition_id: str,
    fields: Sequence[str] = ROLLOUT_AUDIT_TQ_FIELDS,
) -> dict[str, Any]:
    """Read the audit fields of ``keys`` from TransferQueue, leaving out fields it cannot provide.

    All fields are read in one call. If that call fails (e.g. a field was never written for these keys), each
    field is read on its own and fields that still fail are left out.

    Args:
        kv_batch_get: ``tq.kv_batch_get`` (or a test double) taking ``keys``, ``partition_id``, ``select_fields``.
        keys: Keys to read.
        partition_id: TransferQueue partition.
        fields: Field names to read.

    Returns:
        Field name -> value as returned by TransferQueue (read with ``.get`` so non-tensor fields stay stacks).
    """
    keys = list(keys)
    try:
        data = kv_batch_get(keys=keys, partition_id=partition_id, select_fields=list(fields))
        return {field: data.get(field) for field in fields if _has_field(data, field)}
    except Exception as e:
        logger.warning("Rollout audit: batched TransferQueue read failed (%s); reading fields one by one", e)
    fetched: dict[str, Any] = {}
    for field in fields:
        try:
            data = kv_batch_get(keys=keys, partition_id=partition_id, select_fields=[field])
            if _has_field(data, field):
                fetched[field] = data.get(field)
        except Exception as e:
            logger.warning("Rollout audit: cannot read TransferQueue field %r: %s", field, e)
    return fetched


def audit_columns_from_fields(fields: Mapping[str, Any], num_rows: int) -> dict[str, Any]:
    """Convert TransferQueue fields into keyword arguments for :func:`build_rollout_audit_records`.

    Args:
        fields: Maps the names in :data:`ROLLOUT_AUDIT_TQ_FIELDS` to the values returned by ``tq.kv_batch_get``.
            Absent (or unconvertible) fields become ``None`` columns, which the builder reports in
            ``missing_fields``.
        num_rows: Number of keys that were read.

    Returns:
        ``prompt_ids``, ``response_ids``, ``response_masks``, ``scores`` (sum of ``rm_scores`` per row),
        ``num_turns``, ``reward_extra_infos`` and ``extra_fields`` (from ``extra_fields``), ``data_sources``,
        ``ground_truths`` (from ``reward_model``), ``raw_prompts`` (chat messages before templating) and
        ``field_layouts`` (layout of each token field).
    """

    def convert(name: str, fn: Callable[[Any], Any], default: Any) -> Any:
        value = fields.get(name)
        if value is None:
            return default
        try:
            return fn(value)
        except Exception:
            logger.warning("Rollout audit: cannot convert TransferQueue field %r", name, exc_info=True)
            return default

    def non_tensor(value: Any) -> Optional[list]:
        return values_from_field(value, num_rows)

    prompt_ids, prompts_layout = convert("prompts", token_rows_from_field, (None, None))
    response_ids, responses_layout = convert("responses", token_rows_from_field, (None, None))
    response_masks, mask_layout = convert("response_mask", token_rows_from_field, (None, None))
    rm_scores, rm_scores_layout = convert("rm_scores", token_rows_from_field, (None, None))
    num_turns = convert("num_turns", non_tensor, None)
    reward_models = convert("reward_model", non_tensor, None)
    data_sources = convert("data_source", non_tensor, None)
    extra_fields = convert("extra_fields", non_tensor, None)
    raw_prompts = convert("raw_prompt", non_tensor, None)

    layouts = {
        "prompts": prompts_layout,
        "responses": responses_layout,
        "response_mask": mask_layout,
        "rm_scores": rm_scores_layout,
    }
    reward_extra_infos = ground_truths = None
    if extra_fields is not None:
        reward_extra_infos = [ef.get("reward_extra_info") if isinstance(ef, Mapping) else None for ef in extra_fields]
    if reward_models is not None:
        ground_truths = [rm.get("ground_truth") if isinstance(rm, Mapping) else None for rm in reward_models]
    return {
        "prompt_ids": prompt_ids,
        "response_ids": response_ids,
        "response_masks": response_masks,
        "scores": None if rm_scores is None else [_sum_row(row) for row in rm_scores],
        "num_turns": num_turns,
        "reward_extra_infos": reward_extra_infos,
        "data_sources": data_sources,
        "ground_truths": ground_truths,
        "extra_fields": extra_fields,
        "raw_prompts": raw_prompts,
        "field_layouts": {name: layout for name, layout in layouts.items() if layout is not None},
    }


def compute_mask_segments(response_ids: Sequence[int], response_mask: Sequence[Any]) -> list[dict[str, Any]]:
    """Split a response into contiguous runs of equal mask value.

    Token ``i`` takes ``response_mask[i]``, or ``None`` when the mask is shorter than the response, so the segments
    cover every response token exactly once. Mask entries past the end of the response are ignored here
    (``mask_length_matches`` reports them).

    Args:
        response_ids: Response token IDs.
        response_mask: Response mask (1 = model-generated token with loss, 0 = tool/environment token).

    Returns:
        One ``{"mask", "start", "end", "num_tokens", "text"}`` dict per run; ``end`` is exclusive and ``text`` is
        ``None`` (filled in by :func:`build_rollout_audit_records`).
    """
    segments: list[dict[str, Any]] = []
    num_tokens, num_mask = len(response_ids), len(response_mask)
    start = 0
    while start < num_tokens:
        value = response_mask[start] if start < num_mask else None
        end = start + 1
        while end < num_tokens and (response_mask[end] if end < num_mask else None) == value:
            end += 1
        segments.append({"mask": value, "start": start, "end": end, "num_tokens": end - start, "text": None})
        start = end
    return segments


def _as_row(row: Any, what: str) -> list:
    """Return ``row`` as a list, rejecting strings and non-sequences."""
    if hasattr(row, "tolist") and not isinstance(row, list | tuple):
        row = row.tolist()
    if isinstance(row, str | bytes) or not isinstance(row, Sequence):
        raise TypeError(f"expected a sequence of {what}, got {type(row).__name__}")
    return list(row)


def _int_list(row: Any) -> Optional[list[int]]:
    """Convert a row of token IDs into a list of ints (non-integer IDs raise)."""
    if row is None:
        return None
    return [operator.index(token) for token in _as_row(row, "token IDs")]


def _mask_value(value: Any) -> int | float:
    """Normalize a mask value: bools and integral floats become ints."""
    value = float(value)
    return int(value) if value.is_integer() else value


def _mask_list(row: Any) -> Optional[list]:
    """Convert a mask row into a list of ints (non-integral values are kept as floats)."""
    if row is None:
        return None
    return [_mask_value(value) for value in _as_row(row, "mask values")]


def _guarded(errors: list[str], name: str, fn: Callable[..., Any], *args: Any) -> Any:
    """Call ``fn(*args)``; on failure append a message to ``errors`` and return ``None``."""
    try:
        return fn(*args)
    except Exception as e:
        errors.append(f"{name}: {type(e).__name__}: {e}")
        return None


def _new_record(key: str, step: int, layouts: Mapping[str, str], missing: list[str]) -> dict[str, Any]:
    """Create a record with every field present, so all records share one schema."""
    question, session, output_index = split_trajectory_key(key)
    return {
        "audit_version": AUDIT_RECORD_VERSION,
        "key": str(key),
        "session": session,
        "question": question,
        "output_index": output_index,
        "step": to_jsonable(step),
        "data_source": None,
        "ground_truth": None,
        "num_turns": None,
        "score": None,
        "reward_extra_info": None,
        "extra_fields": None,
        "tq_tag": None,
        "raw_prompt": None,
        "prompt_ids": None,
        "response_ids": None,
        "response_mask": None,
        "prompt_text": None,
        "response_text": None,
        "segments": None,
        "num_prompt_tokens": None,
        "num_response_tokens": None,
        "num_loss_tokens": None,
        "num_non_loss_tokens": None,
        "mask_length_matches": False,
        "tq_layouts": dict(layouts),
        "missing_fields": list(missing),
        "warnings": [],
        "error": None,
    }


def _fill_record(
    record: dict[str, Any],
    value: Callable[[str], Any],
    tokenizer: Any,
    layouts: Mapping[str, str],
    errors: list[str],
) -> None:
    """Fill ``record`` from the per-key column values; failures are appended to ``errors`` and never raised."""
    warnings = record["warnings"]

    def jsonable_value(name: str) -> Any:
        return to_jsonable(value(name))

    # Metadata, reward inputs and reward outputs.
    for name in ("data_source", "ground_truth", "num_turns", "reward_extra_info", "tq_tag", "raw_prompt"):
        record[name] = _guarded(errors, name, jsonable_value, name)
    record["score"] = _guarded(errors, "score", lambda: to_jsonable(_sum_row(value("score"))))
    extra = value("extra_fields")
    if isinstance(extra, Mapping):
        # reward_extra_info already has its own field.
        extra = {k: v for k, v in extra.items() if k != "reward_extra_info"}
    record["extra_fields"] = _guarded(errors, "extra_fields", to_jsonable, extra)

    # Raw token IDs and the response (loss) mask.
    prompt_ids = _guarded(errors, "prompt_ids", _int_list, value("prompt_ids"))
    response_ids = _guarded(errors, "response_ids", _int_list, value("response_ids"))
    mask = _guarded(errors, "response_mask", _mask_list, value("response_mask"))
    if mask is not None and response_ids is not None and layouts.get("response_mask") == "dense":
        tail = mask[len(response_ids) :]
        if tail and all(v == 0 for v in tail):
            # Responses are right-padded in verl, so trailing zeros past the response are padding.
            warnings.append(f"response_mask: dropped {len(tail)} trailing padding zeros of a dense row")
            mask = mask[: len(response_ids)]
    record["prompt_ids"], record["response_ids"], record["response_mask"] = prompt_ids, response_ids, mask

    if prompt_ids is not None:
        record["num_prompt_tokens"] = len(prompt_ids)
    if response_ids is not None:
        record["num_response_tokens"] = len(response_ids)
    if mask is not None:
        num_loss = sum(1 for v in mask if v == 1)
        num_non_loss = sum(1 for v in mask if v == 0)
        record["num_loss_tokens"], record["num_non_loss_tokens"] = num_loss, num_non_loss
        if num_loss + num_non_loss != len(mask):
            warnings.append(f"response_mask: {len(mask) - num_loss - num_non_loss} values are neither 0 nor 1")
    record["mask_length_matches"] = response_ids is not None and mask is not None and len(response_ids) == len(mask)
    if response_ids is not None and mask is not None and not record["mask_length_matches"]:
        warnings.append(f"response_mask has {len(mask)} values for {len(response_ids)} response tokens")

    tag = value("tq_tag")
    if isinstance(tag, Mapping):
        for tag_name, ids in (("prompt_len", prompt_ids), ("response_len", response_ids)):
            expected = tag.get(tag_name)
            if isinstance(expected, int) and ids is not None and expected != len(ids):
                warnings.append(f"TransferQueue tag {tag_name}={expected} but {len(ids)} token IDs were read")

    if response_ids is not None:
        record["segments"] = compute_mask_segments(response_ids, mask if mask is not None else [])

    # Text decoded WITH special tokens (chat template, tool-call and tool-response markers).
    if tokenizer is None:
        return

    def decode(ids: list[int]) -> str:
        return tokenizer.decode(ids, skip_special_tokens=False)

    if prompt_ids is not None:
        record["prompt_text"] = _guarded(errors, "prompt_text", decode, prompt_ids)
    if response_ids is not None:
        record["response_text"] = _guarded(errors, "response_text", decode, response_ids)
        for segment in record["segments"]:
            segment["text"] = _guarded(errors, "segment_text", decode, response_ids[segment["start"] : segment["end"]])


def build_rollout_audit_records(
    keys: Sequence[str],
    prompt_ids: Optional[Sequence[Any]],
    response_ids: Optional[Sequence[Any]],
    response_masks: Optional[Sequence[Any]],
    scores: Optional[Sequence[Any]],
    num_turns: Optional[Sequence[Any]],
    reward_extra_infos: Optional[Sequence[Any]],
    data_sources: Optional[Sequence[Any]],
    ground_truths: Optional[Sequence[Any]],
    tokenizer: Any,
    step: int,
    *,
    extra_fields: Optional[Sequence[Any]] = None,
    tags: Optional[Sequence[Any]] = None,
    raw_prompts: Optional[Sequence[Any]] = None,
    field_layouts: Optional[Mapping[str, str]] = None,
) -> list[dict[str, Any]]:
    """Build one JSON-serializable audit record per trajectory key.

    Every column is a sequence aligned with ``keys``, or ``None`` when the field is unavailable; unavailable or
    misaligned core columns are listed in each record's ``missing_fields``. A failure on one record never raises:
    it is stored in that record's ``error`` field and the fields computed so far are kept.

    Each record holds ``key``, ``session``, ``question``, ``output_index``, ``step``, ``data_source``,
    ``ground_truth``, ``num_turns``, ``score``, ``reward_extra_info``, ``extra_fields`` (without
    ``reward_extra_info``), ``tq_tag``, ``raw_prompt``, ``prompt_ids``, ``response_ids``, ``response_mask``,
    ``prompt_text`` and ``response_text`` (decoded with special tokens), ``segments`` (contiguous runs of equal mask
    value, see :func:`compute_mask_segments`, each with its decoded ``text``), the counters ``num_prompt_tokens``,
    ``num_response_tokens``, ``num_loss_tokens`` (mask == 1) and ``num_non_loss_tokens`` (mask == 0),
    ``mask_length_matches``, ``tq_layouts``, ``missing_fields``, ``warnings`` and ``error``.

    Segment texts are decoded independently, so a multi-byte character split across a segment boundary may decode
    differently than in ``response_text``; the token IDs are authoritative.

    Args:
        keys: Trajectory keys ``{uid}_{session_id}_{output_index}``.
        prompt_ids: Prompt token IDs per key.
        response_ids: Response token IDs per key.
        response_masks: Response mask per key (1 = model-generated token with loss, 0 = tool/environment token).
        scores: Sequence-level score per key, or per-token scores (summed).
        num_turns: Number of turns per key.
        reward_extra_infos: Reward extra info dict per key (e.g. ``acc``, ``pred``, ``score``).
        data_sources: Data source per key.
        ground_truths: Ground truth per key.
        tokenizer: Object with ``decode(ids, skip_special_tokens=...)``; ``None`` skips decoding.
        step: Global step stored in every record.
        extra_fields: Optional agent-loop ``extra_fields`` dict per key.
        tags: Optional TransferQueue tag dict per key (``prompt_len``, ``response_len``, ``min_global_steps``, ...).
        raw_prompts: Optional chat messages per key, before the chat template was applied.
        field_layouts: Optional layout of each TransferQueue token field (see :func:`token_rows_from_field`). A
            ``"dense"`` ``response_mask`` row longer than its response loses its trailing zeros (padding).

    Returns:
        One record per key, in the order of ``keys``.
    """
    keys = [str(key) for key in keys]
    num_keys = len(keys)
    layouts = dict(field_layouts or {})
    columns: dict[str, Any] = {
        "prompt_ids": prompt_ids,
        "response_ids": response_ids,
        "response_mask": response_masks,
        "score": scores,
        "num_turns": num_turns,
        "reward_extra_info": reward_extra_infos,
        "data_source": data_sources,
        "ground_truth": ground_truths,
        "extra_fields": extra_fields,
        "tq_tag": tags,
        "raw_prompt": raw_prompts,
    }
    missing: list[str] = []
    column_warnings: list[str] = []
    for name in (*_CORE_COLUMNS, *_OPTIONAL_COLUMNS):
        column = columns[name]
        if column is not None:
            try:
                column = list(column)
            except TypeError:
                column_warnings.append(f"{name}: column is not a sequence; ignored")
                column = None
        if column is not None and len(column) != num_keys:
            column_warnings.append(f"{name}: got {len(column)} values for {num_keys} keys; ignored")
            column = None
        columns[name] = column
        if column is None and name in _CORE_COLUMNS:
            missing.append(name)

    records = []
    for i, key in enumerate(keys):
        record = _new_record(key, step, layouts, missing)
        record["warnings"].extend(column_warnings)
        errors: list[str] = []

        def value(name: str, i: int = i) -> Any:
            column = columns[name]
            return None if column is None else column[i]

        try:
            _fill_record(record, value, tokenizer, layouts, errors)
        except Exception as e:  # Defensive: _fill_record already guards each step.
            errors.append(f"{type(e).__name__}: {e}")
        record["error"] = "; ".join(dict.fromkeys(errors)) or None
        records.append(record)
    return records


def collect_rollout_audit_records(
    kv_batch_get: Callable[..., Any],
    keys: Sequence[str],
    tags: Optional[Sequence[Any]],
    partition_id: str,
    tokenizer: Any,
    step: int,
    num_sessions: int,
    seed: int = 0,
) -> list[dict[str, Any]]:
    """Select whole sessions of a training batch, read them from TransferQueue and build their audit records.

    Synthetic padding samples (tag ``is_padding``) are never selected. Only the selected keys are read.

    Args:
        kv_batch_get: ``tq.kv_batch_get`` (or a test double).
        keys: Keys of the batch.
        tags: TransferQueue tags aligned with ``keys`` (may be ``None``).
        partition_id: TransferQueue partition of the batch.
        tokenizer: Tokenizer used to decode the token IDs.
        step: Global step.
        num_sessions: Number of sessions to audit.
        seed: Extra seed for :func:`select_audit_keys`.

    Returns:
        Audit records of the selected keys (empty if nothing was selected).
    """
    keys = [str(key) for key in keys]
    tags = list(tags) if tags is not None else []
    tag_by_key = {key: (tags[i] if i < len(tags) else None) for i, key in enumerate(keys)}
    candidates = [
        key for key in keys if not (isinstance(tag_by_key[key], Mapping) and tag_by_key[key].get("is_padding", False))
    ]
    selected = select_audit_keys(candidates, num_sessions, step=step, seed=seed)
    if not selected:
        return []
    fields = fetch_rollout_audit_fields(kv_batch_get, selected, partition_id)
    columns = audit_columns_from_fields(fields, num_rows=len(selected))
    return build_rollout_audit_records(
        keys=selected,
        tokenizer=tokenizer,
        step=step,
        tags=[tag_by_key[key] for key in selected],
        **columns,
    )


def write_rollout_audit_jsonl(records: Sequence[Mapping[str, Any]], path: str) -> None:
    """Write audit records to ``path`` as JSONL (one record per line), creating the parent directory.

    Args:
        records: Records from :func:`build_rollout_audit_records`.
        path: Output file path.
    """
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(to_jsonable(record), ensure_ascii=False) + "\n")


def _metric_value(value: Any) -> Optional[float]:
    """Convert a validation metric (bool, number or numeric string) into a finite float, or ``None``."""
    if value is None:
        return None
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def _min_max(values: Sequence[int]) -> dict[str, Optional[int]]:
    """Return ``{"min", "max"}`` of ``values`` (``None`` for an empty sequence)."""
    return {"min": min(values) if values else None, "max": max(values) if values else None}


def summarize_validation_dump(
    rows: Sequence[Mapping[str, Any]],
    expected_n: Optional[int] = None,
    *,
    default_data_source: str = "unknown",
) -> dict[str, dict[str, Any]]:
    """Count questions and distinct sessions per question in a validation dump.

    Validation dumps (``trainer.validation_data_dir``) write one JSONL line per agent-loop *output*, and a
    multi-turn session can have several outputs, so the number of samples per question must be counted from
    distinct sessions, not lines. Rows are grouped by ``data_source`` (``default_data_source`` when a row has none;
    the V1 trainer's validation dump does not write a ``data_source`` column), then by question
    (``uid.rsplit("_", 2)[0]``) and session (the first two parts).

    The per-session metric is ``acc`` when the data source has it, else ``score``, taken from the session's last
    output (all outputs of a session carry the session's final reward). ``mean_metric`` averages the per-question
    means, like ``mean@N``.

    Args:
        rows: Parsed JSONL rows. Each row has ``uid`` (the trajectory key) and optionally ``data_source``, ``acc``
            and ``score``.
        expected_n: Expected number of sessions per question (e.g. ``val_kwargs.n``). Questions with a different
            count are listed in ``questions_with_unexpected_sessions``; ``None`` skips the check.
        default_data_source: Data source used for rows without one.

    Returns:
        Per data source: ``num_rows``, ``num_rows_without_uid``, ``num_questions``, ``num_sessions``,
        ``sessions_per_question`` and ``outputs_per_session`` (``{"min", "max"}``), ``expected_n``,
        ``questions_with_unexpected_sessions`` (``[{"question", "num_sessions"}]``), ``metric``, ``mean_metric``
        and ``per_question`` (``{question: {"num_sessions", "num_rows", "mean_metric"}}``).
    """
    # data_source -> question -> session -> [(output_index, acc, score)]
    grouped: dict[str, dict[str, dict[str, list[tuple[int, Any, Any]]]]] = {}
    num_rows: dict[str, int] = {}
    rows_without_uid: dict[str, int] = {}
    for row in rows:
        data_source = row.get("data_source")
        data_source = default_data_source if data_source is None else str(data_source)
        num_rows[data_source] = num_rows.get(data_source, 0) + 1
        questions = grouped.setdefault(data_source, {})
        key = row.get("uid")
        if key is None:
            rows_without_uid[data_source] = rows_without_uid.get(data_source, 0) + 1
            continue
        question, session, output_index = split_trajectory_key(str(key))
        output_index = -1 if output_index is None else output_index
        questions.setdefault(question, {}).setdefault(session, []).append(
            (output_index, row.get("acc"), row.get("score"))
        )

    summary: dict[str, dict[str, Any]] = {}
    for data_source in sorted(grouped):
        questions = grouped[data_source]
        outputs = [entry for sessions in questions.values() for entries in sessions.values() for entry in entries]
        if any(acc is not None for _, acc, _ in outputs):
            metric = "acc"
        elif any(score is not None for _, _, score in outputs):
            metric = "score"
        else:
            metric = None

        per_question: dict[str, dict[str, Any]] = {}
        for question in sorted(questions):
            sessions = questions[question]
            session_values = []
            for entries in sessions.values():
                _, acc, score = max(entries, key=lambda entry: entry[0])
                session_value = _metric_value(acc if metric == "acc" else score)
                if session_value is not None:
                    session_values.append(session_value)
            per_question[question] = {
                "num_sessions": len(sessions),
                "num_rows": sum(len(entries) for entries in sessions.values()),
                "mean_metric": sum(session_values) / len(session_values) if session_values else None,
            }

        sessions_per_question = [stats["num_sessions"] for stats in per_question.values()]
        outputs_per_session = [len(entries) for sessions in questions.values() for entries in sessions.values()]
        question_means = [stats["mean_metric"] for stats in per_question.values() if stats["mean_metric"] is not None]
        summary[data_source] = {
            "num_rows": num_rows[data_source],
            "num_rows_without_uid": rows_without_uid.get(data_source, 0),
            "num_questions": len(questions),
            "num_sessions": sum(sessions_per_question),
            "sessions_per_question": _min_max(sessions_per_question),
            "outputs_per_session": _min_max(outputs_per_session),
            "expected_n": expected_n,
            "questions_with_unexpected_sessions": [
                {"question": question, "num_sessions": stats["num_sessions"]}
                for question, stats in per_question.items()
                if expected_n is not None and stats["num_sessions"] != expected_n
            ],
            "metric": metric,
            "mean_metric": sum(question_means) / len(question_means) if question_means else None,
            "per_question": per_question,
        }
    return summary
