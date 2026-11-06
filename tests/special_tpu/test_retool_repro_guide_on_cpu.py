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
"""CPU tests for ``examples/tpu/retool/REPRO_GUIDE.md`` and the files it relies on.

* every ``AUTO-GENERATED`` table of the guide is the output of ``repro_guide/make_report.py`` on the snapshot in
  ``repro_guide/data/``;
* every relative link of the guide resolves, and none points into ``exp/`` or ``logs/`` (the guide must not depend
  on the archived bring-up files), nor does the guide's own code read from there;
* ``prepare_retool_data.py``: the ReTool-SFT conversion and the row digest used by ``--check``.
"""

import importlib.util
import json
import re
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
RETOOL = REPO_ROOT / "examples" / "tpu" / "retool"
GUIDE = RETOOL / "REPRO_GUIDE.md"
ARCHIVE_DIRS = (RETOOL / "exp", RETOOL / "logs")
LINK_RE = re.compile(r"!?\[(?:[^\[\]]|\[[^\[\]]*\])*\]\(([^)\s]+)(?:\s+\"[^\"]*\")?\)")


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _relative_links(md: Path) -> list[tuple[str, Path]]:
    out = []
    for target in LINK_RE.findall(md.read_text(encoding="utf-8")):
        if re.match(r"^[a-z][a-z0-9+.-]*:", target) or target.startswith(("#", "/")):
            continue
        out.append((target, (md.parent / target.split("#", 1)[0]).resolve()))
    return out


def test_guide_tables_match_the_snapshot():
    subprocess.run(
        [sys.executable, str(RETOOL / "repro_guide" / "make_report.py"), "--check", str(GUIDE)],
        check=True,
        cwd=REPO_ROOT,
    )


def test_guide_links_resolve_and_avoid_the_archive():
    links = _relative_links(GUIDE)
    assert links, "no relative links found"
    for target, dest in links:
        assert dest.exists(), f"broken link {target}"
        for archive in ARCHIVE_DIRS:
            assert not dest.is_relative_to(archive.resolve()), f"REPRO_GUIDE.md links into {archive.name}/: {target}"
    for name in ("fig1_aime.png", "fig2_training.png", "fig3_step_time.png", "fig4_sft.png"):
        assert (RETOOL / "repro_guide" / "figures" / name).is_file()


def test_guide_code_does_not_read_the_archive():
    files = [RETOOL / "prepare_retool_data.py", *sorted((RETOOL / "repro_guide").glob("*.py"))]
    for path in files:
        for line in path.read_text(encoding="utf-8").splitlines():
            if "gs://" in line:  # GCS run directories have their own logs/ prefix
                continue
            assert not re.search(r"""["'/ ](exp|logs)["'/]""", line), f"{path.name} refers to exp/ or logs/: {line}"


def test_snapshot_covers_every_run():
    report = _load(RETOOL / "repro_guide" / "make_report.py", "make_report")
    snap = report.Snapshot()
    for spec in report.RUNS:
        steps = sorted(snap.rows(spec))
        assert steps[-1] == report.MAX_STEP and set(range(1, report.MAX_STEP + 1)) <= set(steps), spec["key"]
    for key in ("v7x", "gb200"):
        for step in ("0", str(report.MAX_STEP)):
            per_q = snap.eval_pq[key]["steps"][step]
            assert {ds: len(q) for ds, q in per_q.items()} == {"aime_2024": 30, "aime_2025": 30}
            assert all(n == 30 for q in per_q.values() for _, n in q.values())


def test_acc_and_se_and_paired_diff():
    report = _load(RETOOL / "repro_guide" / "make_report.py", "make_report")
    acc, se = report.acc_and_se({"a": [3, 4], "b": [1, 4]})
    assert acc == pytest.approx(50.0)
    # Var(p_hat) per problem: p (1 - p) / (n - 1) = 0.75 * 0.25 / 3 = 0.0625 for both problems.
    assert se == pytest.approx(100 * (2 * 0.0625) ** 0.5 / 2)
    d, se_d = report.paired_diff({"a": [3, 4], "b": [1, 4]}, {"a": [1, 4], "b": [1, 4]})
    assert d == pytest.approx(25.0)
    assert se_d == pytest.approx(100 * (0.0625 + 0.0625 + 0.0625 + 0.0625) ** 0.5 / 2)


def test_convert_retool_sft_row():
    prep = _load(RETOOL / "prepare_retool_data.py", "prepare_retool_data")
    row = {
        "messages": [
            {"role": "user", "content": "Solve it.\n*user question:*\nWhat is 6*7? <answer>\\boxed{}</answer>"},
            {
                "role": "assistant",
                "content": "Let me compute.<code>```python\nprint(6*7)\n```</code>"
                "<interpreter>42</interpreter>So the answer is<answer>\\boxed{42}</answer>",
            },
        ]
    }
    tools = [{"type": "function", "function": {"name": "code_interpreter"}}]
    out = prep.convert_retool_sft_row(row, tools)
    assert out["tools"] == tools
    assert out["messages"] == [
        {"role": "user", "content": "What is 6*7? \\boxed{}"},
        {
            "role": "assistant",
            "content": "Let me compute.",
            "tool_calls": [
                {"type": "function", "function": {"name": "code_interpreter", "arguments": {"code": "print(6*7)"}}}
            ],
        },
        {"role": "tool", "content": "42"},
        {"role": "assistant", "content": "So the answer is\\boxed{42}"},
    ]


def test_tool_schema_matches_rl_tool_config():
    pytest.importorskip("yaml")
    prep = _load(RETOOL / "prepare_retool_data.py", "prepare_retool_data")
    schema = prep.tool_schema()
    assert schema["function"]["name"] == "code_interpreter"
    assert schema["function"]["parameters"]["required"] == ["code"]


def test_rows_digest_is_order_and_content_sensitive(tmp_path):
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    prep = _load(RETOOL / "prepare_retool_data.py", "prepare_retool_data")
    rows = [{"b": 1, "a": "x"}, {"b": 2, "a": "y"}]
    pq.write_table(pa.Table.from_pylist(rows), tmp_path / "a.parquet")
    pq.write_table(pa.Table.from_pylist([{"a": r["a"], "b": r["b"]} for r in rows]), tmp_path / "b.parquet")
    pq.write_table(pa.Table.from_pylist(rows[::-1]), tmp_path / "c.parquet")
    n, digest = prep.rows_digest(tmp_path / "a.parquet")
    assert n == 2
    assert prep.rows_digest(tmp_path / "b.parquet") == (2, digest)  # column order does not matter
    assert prep.rows_digest(tmp_path / "c.parquet")[1] != digest  # row order does
    expected = (
        __import__("hashlib").sha256(b"".join(json.dumps(r, sort_keys=True).encode() + b"\n" for r in rows)).hexdigest()
    )
    assert digest == expected
