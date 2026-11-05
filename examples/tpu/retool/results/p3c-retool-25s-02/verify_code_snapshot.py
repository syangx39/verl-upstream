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
"""Check that the launcher's code tarball of a TPU v7x NAP run is exactly the tree of the recorded commit.

The launcher (``examples/tpu/gke/run_ray_jobset_v7x_nap.sh``) uploads ``git archive <CODE_SHA>`` as
``<run dir>/src-<sha12>.tar.gz`` and the pods run that tarball. This script compares, file by file, the git blob
hash of every regular file and symlink in the tarball with ``git ls-tree -r <CODE_SHA>`` of a local clone, and
checks the commit id that ``git archive`` stores in the tar's global pax header.

Usage::

    gcloud storage cp gs://ubench-logs/xiaotongyang/retool-v7x/runs/p3c-retool-25s-02/src-348bbff9f4d3.tar.gz /tmp/
    git fetch origin tag run-code/p3c-retool-25s-01-02
    python3 verify_code_snapshot.py --tarball /tmp/src-348bbff9f4d3.tar.gz [--repo .]
"""

import argparse
import hashlib
import json
import subprocess
import sys
import tarfile
from pathlib import Path

HERE = Path(__file__).resolve().parent


def git_blob_sha1(data: bytes) -> str:
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


def tarball_blobs(path: Path) -> tuple[str | None, dict[str, str]]:
    """The pax ``comment`` (commit id written by ``git archive``) and ``{path: git blob sha1}``."""
    blobs: dict[str, str] = {}
    with tarfile.open(path) as tf:
        comment = tf.pax_headers.get("comment")
        for m in tf.getmembers():
            if m.isfile():
                f = tf.extractfile(m)
                assert f is not None
                blobs[m.name] = git_blob_sha1(f.read())
            elif m.issym():
                blobs[m.name] = git_blob_sha1(m.linkname.encode("utf-8"))
    return comment, blobs


def tree_blobs(repo: Path, sha: str) -> dict[str, str]:
    """``{path: blob sha1}`` of the regular files and symlinks of ``sha`` (submodules are not archived)."""
    out = subprocess.run(["git", "-C", str(repo), "ls-tree", "-r", "-z", sha], capture_output=True, check=True).stdout
    blobs = {}
    for entry in out.decode("utf-8").split("\0"):
        if entry:
            meta, path = entry.split("\t", 1)
            mode, kind, obj = meta.split()
            if kind == "blob":
                blobs[path] = obj
    return blobs


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--tarball", type=Path, required=True)
    parser.add_argument("--repo", type=Path, default=HERE.parents[4])
    parser.add_argument("--snapshot-json", type=Path, default=HERE / "env" / "code_snapshot.json")
    args = parser.parse_args()
    snap = json.loads(args.snapshot_json.read_text(encoding="utf-8"))
    sha = snap["code_sha"]
    tar_sha256 = hashlib.sha256(args.tarball.read_bytes()).hexdigest()
    assert tar_sha256 == snap["gcs_tarball"]["sha256"], f"tarball sha256 {tar_sha256} != {snap['gcs_tarball']}"
    comment, tar = tarball_blobs(args.tarball)
    assert comment == sha, f"tar pax comment {comment} != {sha}"
    tree = tree_blobs(args.repo, sha)
    assert tar == tree, (
        f"tarball and {sha} differ: only in tar {sorted(set(tar) - set(tree))[:5]}, only in tree "
        f"{sorted(set(tree) - set(tar))[:5]}, changed {sorted(p for p in tar if p in tree and tar[p] != tree[p])[:5]}"
    )
    assert len(tar) == snap["gcs_tarball"]["files"], f"{len(tar)} files != {snap['gcs_tarball']['files']}"
    print(json.dumps({"code_sha": sha, "tarball_sha256": tar_sha256, "files_matching_commit_tree": len(tar)}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
