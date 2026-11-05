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
"""Per-step wall-clock breakdown of a verl V1 run: what ``timing_s/step`` covers and what it does not.

One iteration of ``RayPPOTrainer.fit`` in ``verl/trainer/ppo/v1/trainer_base.py`` runs, in order:

1. the ``step`` timer: ``on_step_begin``, sampling the batch from the replay buffer (``timing_s/gen``, i.e. the
   wait for rollouts), ``old_log_prob``, ``adv``, ``update_actor`` and ``on_step_end`` (``update_weights``);
2. validation on test steps (``timing_s/testing``, outside the ``step`` timer);
3. ``_compute_metrics``, the rollout / audit dumps (TransferQueue fetch and decode on the driver; the file writes
   run on a background thread), ``tq.kv_clear``;
4. ``logger.log``: W&B first, then TensorBoard ``add_scalar`` once per metric, then the console line; then tqdm.

TensorBoard stores the ``wall_time`` of every ``add_scalar`` call, so its event file times phase 4 independently
of the trainer's own timers: for step ``N``, ``log_start(N)`` is the first scalar's wall time (it equals the W&B
``_timestamp`` of the step) and ``tb_span(N)`` is the last minus the first scalar's wall time. With these:

* ``wall(N) = log_start(N) - log_start(N-1)``: one full iteration (it contains the TensorBoard logging of step
  ``N-1`` and everything of step ``N`` up to its own ``logger.log``);
* ``outside(N) = wall(N) - timing_s/step(N) - timing_s/testing(N)``: time not covered by either timer;
* ``residual(N) = outside(N) - tb_span(N-1)``: phase 3 of step ``N`` plus the last ``add_scalar`` call, the console
  line and tqdm of step ``N-1``.

If a driver-log regex for the end of the weight sync is given (``--sync-done-regex``, timestamps
``YYYY-MM-DD HH:MM:SS,mmm`` in the log's time zone), ``post(N)`` is ``log_start(N)`` minus the last such line before
it, i.e. phase 3 of step ``N`` measured from the log. The log's UTC offset is inferred from the
``Dumped N rollout audit records to .../rollout_audit/N.jsonl`` lines, which the dump thread writes about one second
after ``log_start(N)`` (``--log-utc-offset-hours`` overrides it).

The event-file reader is a minimal TFRecord / protobuf parser (standard library only; CRCs are not checked).

Usage::

    python3 step_timing_breakdown.py --metrics metrics.json --tb-events events.out.tfevents.X \\
        [--log head.log --sync-done-regex 'update_weights done'] [--first 2 --last 24]
"""

import argparse
import datetime as dt
import json
import re
import statistics
import struct
import sys
from pathlib import Path

LOG_TS_RE = re.compile(r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}),(\d{3})")
AUDIT_DUMP_RE = re.compile(r"Dumped \d+ rollout audit records to \S*/rollout_audit/(\d+)\.jsonl")


def _varint(buf: bytes, i: int) -> tuple[int, int]:
    shift = result = 0
    while True:
        b = buf[i]
        i += 1
        result |= (b & 0x7F) << shift
        if not b & 0x80:
            return result, i
        shift += 7


def _fields(buf: bytes):
    """Yield ``(field number, wire type, value)`` of a serialized protobuf message (wire types 0, 1, 2, 5)."""
    i = 0
    while i < len(buf):
        key, i = _varint(buf, i)
        num, wire = key >> 3, key & 7
        if wire == 0:
            val, i = _varint(buf, i)
        elif wire == 1:
            val, i = buf[i : i + 8], i + 8
        elif wire == 2:
            n, i = _varint(buf, i)
            val, i = buf[i : i + n], i + n
        elif wire == 5:
            val, i = buf[i : i + 4], i + 4
        else:
            raise ValueError(f"unsupported protobuf wire type {wire}")
        yield num, wire, val


def read_scalar_events(path: Path) -> list[tuple[int, float, str]]:
    """``(step, wall_time, tag)`` for every summary value in a TensorBoard event file, in file order.

    ``Event``: 1 = wall_time (double), 2 = step (int64), 5 = summary; ``Summary``: 1 = value; ``Value``: 1 = tag.
    """
    data = path.read_bytes()
    out: list[tuple[int, float, str]] = []
    i = 0
    while i < len(data):
        (n,) = struct.unpack_from("<Q", data, i)
        rec = data[i + 12 : i + 12 + n]
        i += 12 + n + 4
        wall, step, tags = None, 0, []
        for num, wire, val in _fields(rec):
            if num == 1 and wire == 1:
                wall = struct.unpack("<d", val)[0]
            elif num == 2 and wire == 0:
                step = val
            elif num == 5 and wire == 2:
                for vnum, vwire, vval in _fields(val):
                    if vnum == 1 and vwire == 2:
                        tags += [t.decode("utf-8") for tn, tw, t in _fields(vval) if tn == 1 and tw == 2]
        out += [(step, wall, tag) for tag in tags]
    return out


def tb_log_windows(events: list[tuple[int, float, str]]) -> dict[int, dict]:
    """Per step: first / last scalar wall time, span and scalar count."""
    per: dict[int, list[float]] = {}
    for step, wall, _tag in events:
        per.setdefault(step, []).append(wall)
    return {
        s: {"first": min(w), "last": max(w), "span": max(w) - min(w), "scalars": len(w)} for s, w in sorted(per.items())
    }


def _naive_utc(line: str) -> float | None:
    """The line's ``YYYY-MM-DD HH:MM:SS,mmm`` timestamp read as if it were UTC (``None`` if it has none)."""
    m = LOG_TS_RE.search(line)
    if not m:
        return None
    t = dt.datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S").replace(tzinfo=dt.timezone.utc)
    return t.timestamp() + int(m.group(2)) / 1000


def infer_log_utc_offset_hours(log: Path, windows: dict[int, dict]) -> float:
    """UTC offset of the driver log's timestamps, rounded to 15 minutes.

    Uses the ``Dumped N rollout audit records`` lines, which follow ``log_start(N)`` by about a second.
    """
    deltas = []
    for line in log.read_text(encoding="utf-8", errors="replace").splitlines():
        m = AUDIT_DUMP_RE.search(line)
        t = _naive_utc(line)
        if m and t is not None and int(m.group(1)) in windows:
            deltas.append(t - windows[int(m.group(1))]["first"])
    if not deltas:
        raise ValueError(f"{log}: no timestamped rollout-audit dump lines to infer the time zone from")
    if max(deltas) - min(deltas) > 60:
        raise ValueError(
            f"{log}: audit dump lines disagree with the TensorBoard log starts by {min(deltas)}..{max(deltas)}"
        )
    return round(statistics.median(deltas) / 900) / 4


def log_marker_times(log: Path, pattern: str, utc_offset_hours: float) -> list[float]:
    """Unix times of the log lines matching ``pattern`` (timestamps in the log's zone, shifted to UTC)."""
    rx = re.compile(pattern)
    out = []
    for line in log.read_text(encoding="utf-8", errors="replace").splitlines():
        if rx.search(line):
            t = _naive_utc(line)
            if t is not None:
                out.append(t - 3600 * utc_offset_hours)
    return sorted(out)


def breakdown(
    metrics: dict,
    windows: dict[int, dict],
    final_step: int,
    sync_done: list[float] | None = None,
) -> list[dict]:
    """One row per training step ``1..final_step`` (see the module docstring for the definitions)."""
    rows = []
    for n in range(1, final_step + 1):
        m = metrics[str(n)]
        wall = windows[n]["first"] - windows[n - 1]["first"]
        step_t = float(m["timing_s/step"])
        testing = float(m.get("timing_s/testing", 0.0))
        outside = wall - step_t - testing
        row = {
            "step": n,
            "wall_s": round(wall, 3),
            "timing_s_step": round(step_t, 3),
            "timing_s_gen": round(float(m["timing_s/gen"]), 3),
            "timing_s_update_actor": round(float(m["timing_s/update_actor"]), 3),
            "timing_s_update_weights": round(float(m["timing_s/update_weights"]), 3),
            "timing_s_testing": round(testing, 3),
            "outside_timers_s": round(outside, 3),
            "tb_span_prev_step_s": round(windows[n - 1]["span"], 3),
            "tb_scalars_prev_step": windows[n - 1]["scalars"],
            "residual_s": round(outside - windows[n - 1]["span"], 3),
        }
        if sync_done is not None:
            inside = [t for t in sync_done if windows[n - 1]["first"] < t < windows[n]["first"]]
            row["post_sync_to_log_s"] = round(windows[n]["first"] - inside[-1], 3) if inside else None
        rows.append(row)
    return rows


def summarize(rows: list[dict], first: int, last: int) -> dict:
    """Means over steps ``first..last`` (inclusive) of every numeric column."""
    sel = [r for r in rows if first <= r["step"] <= last]
    keys = [k for k in sel[0] if k != "step" and all(isinstance(r[k], int | float) for r in sel)]
    out: dict = {"steps": [first, last], "n": len(sel)}
    for k in keys:
        out[f"mean_{k}"] = round(statistics.mean(r[k] for r in sel), 3)
    out["min_timing_s_step"] = round(min(r["timing_s_step"] for r in sel), 3)
    out["max_timing_s_step"] = round(max(r["timing_s_step"] for r in sel), 3)
    return out


def run_breakdown(
    metrics: dict,
    tb_events: Path,
    final_step: int,
    log: Path | None = None,
    sync_done_regex: str | None = None,
    log_utc_offset_hours: float | None = None,
    first: int = 2,
    last: int | None = None,
) -> dict:
    """``{"summary": ..., "rows": [...], "log_utc_offset_hours": ...}`` for one run (see the module docstring)."""
    windows = tb_log_windows(read_scalar_events(tb_events))
    sync = None
    if log is not None and sync_done_regex:
        if log_utc_offset_hours is None:
            log_utc_offset_hours = infer_log_utc_offset_hours(log, windows)
        sync = log_marker_times(log, sync_done_regex, log_utc_offset_hours)
    rows = breakdown(metrics, windows, final_step, sync)
    last = last if last is not None else final_step - 1
    return {"summary": summarize(rows, first, last), "rows": rows, "log_utc_offset_hours": log_utc_offset_hours}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--metrics", type=Path, required=True, help="{step: {metric: value}} JSON")
    parser.add_argument("--tb-events", type=Path, required=True, help="TensorBoard scalar event file of the run")
    parser.add_argument("--final-step", type=int, default=25)
    parser.add_argument("--log", type=Path, help="driver log with timestamped end-of-weight-sync lines")
    parser.add_argument("--sync-done-regex", default=None)
    parser.add_argument("--log-utc-offset-hours", type=float, default=None, help="default: inferred from the log")
    parser.add_argument("--first", type=int, default=2)
    parser.add_argument("--last", type=int, default=None, help="default: final step - 1 (excludes the eval step)")
    args = parser.parse_args()
    metrics = json.loads(args.metrics.read_text(encoding="utf-8"))
    res = run_breakdown(
        metrics,
        args.tb_events,
        args.final_step,
        args.log,
        args.sync_done_regex,
        args.log_utc_offset_hours,
        args.first,
        args.last,
    )
    print(json.dumps(res, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
