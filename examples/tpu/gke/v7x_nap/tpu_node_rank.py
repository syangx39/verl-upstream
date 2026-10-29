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
"""Print this TPU host's topological rank inside its multi-host slice as ``NODE_RANK=<r> COORDS=[...]``.

The Kubernetes job completion index (and the ``TPU_WORKER_ID`` GKE derives from it) is not the host's physical
position in the slice. torch_tpu's own multi-host launcher ranks hosts by the physical coordinates of their TPU
devices, and verl must use the same order for a slice's hosts (``verl-tpu-node-rank`` Ray node label), otherwise
the trainer's collectives pair the wrong hosts.

Run it on every host of the slice at about the same time (as torch_tpu's launcher does), before Ray starts and
before anything changes the TPU environment GKE provides. Usage: ``tpu_node_rank.py <slice topology, e.g. 2x2x4>``.
"""

import os
import sys


def main() -> int:
    """Compute and print the rank; exit non-zero when torch_tpu cannot provide it."""
    if len(sys.argv) != 2:
        print(__doc__, file=sys.stderr)
        return 2
    os.environ.setdefault("PJRT_DEVICE", "TPU")
    from torch_tpu._internal.distributed import tpu_distributed
    from torch_tpu._internal.distributed.launchers import multihost_wrapper

    coords = tpu_distributed.local_device_coords()
    rank = multihost_wrapper.calculate_topological_node_rank(coords, sys.argv[1], None)
    print(f"NODE_RANK={int(rank)} COORDS={list(coords)}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
