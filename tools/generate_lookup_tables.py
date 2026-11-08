"""Rebuild lookup_tables/*.npy from evaluator.get_table and the preflop proxy.

Startup of agent.py only reads those files. Run this after a change to
_build_tables or the evaluator, then commit the .npy files and manifest.json.

    AGENT_BUILD_TABLES=1 is set below before agent is imported.
    python tools/generate_lookup_tables.py

The files are numeric NumPy .npy (allow_pickle=False) plus a JSON manifest.
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
import tracemalloc
from pathlib import Path

os.environ["AGENT_BUILD_TABLES"] = "1"
os.environ.setdefault("OMP_NUM_THREADS", "1")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

import agent


def main() -> None:
    t0 = time.perf_counter()
    built = agent._build_tables()
    build_s = time.perf_counter() - t0
    loaded_now = (
        agent._TBL,
        agent._VAL,
        agent._PRE_PCT,
        agent._RMASK,
        agent._SCNT,
        agent._STRAIGHT_RANK,
    )
    # Two builds must match. The import-time build and this call are separate.
    for a, b in zip(built[:-1], loaded_now[:-1]):
        if not np.array_equal(a, b):
            raise SystemExit("two _build_tables() calls disagreed")
    if int(built[5]) != int(loaded_now[5]):
        raise SystemExit("straight_rank disagreed across builds")
    manifest = agent._write_lookup_tables(built)
    again = agent._load_lookup_tables()
    for a, b in zip(built[:5], again[:5]):
        if not np.array_equal(a, b):
            raise SystemExit("reloaded arrays do not match the build")
    if int(again[5]) != int(built[5]):
        raise SystemExit("reloaded straight_rank mismatch")
    total = sum(spec["nbytes"] for spec in manifest["arrays"].values())
    print(f"build_seconds {build_s:.3f}")
    print(f"uncompressed_bytes {total}")
    print(f"straight_rank {int(built[5])}")
    for name, spec in manifest["arrays"].items():
        print(f"{name} dtype={spec['dtype']} shape={spec['shape']} nbytes={spec['nbytes']} sha256={spec['sha256']}")

    env = os.environ.copy()
    env.pop("AGENT_BUILD_TABLES", None)
    probe = r"""
import time, tracemalloc
tracemalloc.start()
t0 = time.perf_counter()
import agent
full = time.perf_counter() - t0
cur, peak = tracemalloc.get_traced_memory()
t1 = time.perf_counter()
agent._load_lookup_tables()
load = time.perf_counter() - t1
print(f"import_seconds {full:.3f}")
print(f"reload_seconds {load:.3f}")
print(f"tracemalloc_current {cur}")
print(f"tracemalloc_peak {peak}")
"""
    out = subprocess.check_output(
        [sys.executable, "-c", probe], cwd=ROOT, env=env, text=True
    )
    print(out, end="")


if __name__ == "__main__":
    main()
