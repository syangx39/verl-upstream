# Copyright 2024 Bytedance Ltd. and/or its affiliates
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
"""Download the ReTool datasets at pinned revisions, lay them out under ``RAY_DATA_HOME`` and verify them.

The recipes read (relative to ``RAY_DATA_HOME`` / ``DATA_ROOT``)::

    dataset/ReTool-SFT/data/train-00000-of-00001.parquet                       SFT data (converted, see below)
    dataset/BytedTsinghua-SIA/DAPO-Math-17k/data/train-00000-of-00001.parquet  RL training prompts
    dataset/Maxwell-Jia/AIME_2024/data/train-00000-of-00001.parquet            validation
    dataset/yentinglin/aime_2025/data/train-00000-of-00001.parquet             validation
    assets/hf/Qwen3-30B-A3B/                                                   base model (``--with-model``)

DAPO-Math-17k and AIME are copied as they are. ReTool-SFT (``JoeYing/ReTool-SFT``) stores each trajectory as one
assistant message with ``<code>``, ``<interpreter>`` and ``<answer>`` tags; ``convert_retool_sft_row`` splits it
into multi-turn tool-calling messages and adds the ``code_interpreter`` tool schema, as
``retool/retool_sft_preprocess.py`` of verl-recipe does (``verl-project/verl-recipe@e805cfd1``).

``--check`` compares the rows of each dataset file with the digests of the files all runs in ``REPRO_GUIDE.md`` read
(byte sizes differ between exports, so the digest is over the rows, not the file). Checking DAPO-Math-17k takes a few
minutes.

Usage::

    python3 examples/tpu/retool/prepare_retool_data.py --out /path/to/inputs [--with-model]
    python3 examples/tpu/retool/prepare_retool_data.py --check /path/to/inputs
"""

import argparse
import hashlib
import json
import re
import shutil
import sys
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
MODEL = {"repo_id": "Qwen/Qwen3-30B-A3B", "revision": "ad44e777bcd18fa416d9da3bd8f70d33ebb85d39"}
MODEL_DIR = "assets/hf/Qwen3-30B-A3B"
_FENCE = re.compile(r"```python(.*?)```", re.DOTALL)
DATASETS = {
    "ReTool-SFT": {
        "repo_id": "JoeYing/ReTool-SFT",
        "revision": "74943ce52f389f16926702302fcff6255875cbb2",
        "filename": "train_2000.parquet",
        "path": "dataset/ReTool-SFT/data/train-00000-of-00001.parquet",
        "convert": True,
        "rows": 2000,
        "row_digest": "da164b358627f5cf260547d2a8cd077225a286820e7a3bea433c851a7883881d",
    },
    "DAPO-Math-17k": {
        "repo_id": "BytedTsinghua-SIA/DAPO-Math-17k",
        "revision": "65877096c24ffa7abc4e4fa5edb95cf3413a5674",
        "filename": "data/dapo-math-17k.parquet",
        "path": "dataset/BytedTsinghua-SIA/DAPO-Math-17k/data/train-00000-of-00001.parquet",
        "convert": False,
        "rows": 1791700,
        "row_digest": "88f40bcd359d6dd8a4f09bd9d54cadd2a6e9aa9b1aa93ce296aa0e18411a2427",
    },
    "AIME_2024": {
        "repo_id": "Maxwell-Jia/AIME_2024",
        "revision": "8d88b2876a82a080e2f172cc9b25d0d9d2cb4792",
        "filename": "aime_2024_problems.parquet",
        "path": "dataset/Maxwell-Jia/AIME_2024/data/train-00000-of-00001.parquet",
        "convert": False,
        "rows": 30,
        "row_digest": "2de5cf977a76cdacbddf2ea5eb34f1827c1d07b243650d07f8d622824fa0871e",
    },
    "aime_2025": {
        "repo_id": "yentinglin/aime_2025",
        "revision": "6f71d77b0b89b9dabe07ab466c51df33f514df7f",
        "filename": "data/train-00000-of-00001-243207c6c994e1bd.parquet",
        "path": "dataset/yentinglin/aime_2025/data/train-00000-of-00001.parquet",
        "convert": False,
        "rows": 30,
        "row_digest": "e21ad0a5d47f6d811613d2616077aee0fe9ab97bbeef6308400ded09f1ebf456",
    },
}


def rows_digest(path: Path) -> tuple[int, str]:
    """Number of rows and SHA-256 over the rows (one JSON object per row, keys sorted)."""
    import pyarrow.parquet as pq

    table = pq.read_table(path)
    cols = sorted(table.column_names)
    h = hashlib.sha256()
    for batch in table.to_batches(max_chunksize=65536):
        for row in batch.to_pylist():
            h.update(json.dumps({c: row[c] for c in cols}, sort_keys=True, default=str).encode())
            h.update(b"\n")
    return table.num_rows, h.hexdigest()


def tool_schema() -> dict:
    """The ``code_interpreter`` schema of the RL tool config, so SFT and RL prompts list the same tool."""
    import yaml

    cfg = yaml.safe_load((HERE / "sandbox_fusion_tool_config_tpu.yaml").read_text(encoding="utf-8"))
    return cfg["tools"][0]["tool_schema"]


def _cut(content: str, start: str, stop: str) -> tuple[str, str, str] | None:
    i = content.find(start)
    if i == -1:
        return None
    j = content.find(stop)
    assert j > i, f"{stop!r} before {start!r}"
    return content[:i], content[i + len(start) : j], content[j + len(stop) :]


def convert_retool_sft_row(row: dict[str, Any], tools: list[dict]) -> dict[str, Any]:
    """One ``JoeYing/ReTool-SFT`` row -> ``{"messages": [...], "tools": tools}`` (multi-turn tool calls)."""
    content = row["messages"][0]["content"]
    marker = "*user question:*"
    i = content.find(marker)
    assert i != -1, "no user question"
    prompt = content[i + len(marker) :].replace("<answer>", "").replace("</answer>", "").strip()
    messages: list[dict[str, Any]] = [{"role": "user", "content": prompt}]
    content = row["messages"][1]["content"]
    role = "assistant"
    while content:
        if role == "assistant":
            cut = _cut(content, "<code>", "</code>")
            if cut is not None:
                before, code, content = cut
                fenced = _FENCE.findall(code)
                code = fenced[0].strip() if fenced else code
                messages.append(
                    {
                        "role": "assistant",
                        "content": before.strip(),
                        "tool_calls": [
                            {"type": "function", "function": {"name": "code_interpreter", "arguments": {"code": code}}}
                        ],
                    }
                )
            else:
                cut = _cut(content, "<answer>", "</answer>")
                assert cut is not None, "assistant turn without <code> or <answer>"
                before, answer, content = cut
                messages.append({"role": "assistant", "content": (before + answer).strip()})
            role = "tool"
        else:
            cut = _cut(content, "<interpreter>", "</interpreter>")
            assert cut is not None, "missing <interpreter> after a tool call"
            _, output, content = cut
            messages.append({"role": "tool", "content": output.strip()})
            role = "assistant"
    return {"messages": messages, "tools": tools}


def prepare(out: Path, with_model: bool) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download, snapshot_download

    for name, src in DATASETS.items():
        local = Path(hf_hub_download(src["repo_id"], src["filename"], revision=src["revision"], repo_type="dataset"))
        dest = out / src["path"]
        dest.parent.mkdir(parents=True, exist_ok=True)
        if src["convert"]:
            tools = [tool_schema()]
            rows = [convert_retool_sft_row(r, tools) for r in pq.read_table(local).to_pylist()]
            pq.write_table(pa.Table.from_pylist(rows), dest)
        else:
            shutil.copyfile(local, dest)
        print(f"{name}: {src['repo_id']}@{src['revision'][:8]}/{src['filename']} -> {dest}")
    if with_model:
        snapshot_download(MODEL["repo_id"], revision=MODEL["revision"], local_dir=out / MODEL_DIR)
        print(f"model: {MODEL['repo_id']}@{MODEL['revision'][:8]} -> {out / MODEL_DIR}")


def check(root: Path) -> bool:
    ok = True
    for name, src in DATASETS.items():
        path = root / src["path"]
        if not path.is_file():
            print(f"MISSING {name}: {path}")
            ok = False
            continue
        rows, digest = rows_digest(path)
        good = rows == src["rows"] and digest == src["row_digest"]
        ok &= good
        print(f"{'OK' if good else 'MISMATCH'} {name}: {rows} rows, row digest {digest[:16]}... ({path})")
    return ok


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--out", type=Path, help="download and convert into this RAY_DATA_HOME, then check it")
    group.add_argument("--check", type=Path, help="only check an existing RAY_DATA_HOME")
    parser.add_argument("--with-model", action="store_true", help=f"also download {MODEL['repo_id']} (~61 GB)")
    args = parser.parse_args()
    if args.out:
        prepare(args.out, args.with_model)
    sys.exit(0 if check(args.out or args.check) else 1)


if __name__ == "__main__":
    main()
