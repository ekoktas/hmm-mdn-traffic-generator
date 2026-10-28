#!/usr/bin/env python3
"""
Table 4 benchmark of the Dual-HMM generator: 100 flows of 50 packets on one CPU thread.

    python scripts/bench/bench_dualhmm.py --artifacts runs/facebook_video/seed0/artifacts/dualhmm
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import timing  # noqa: E402  (first import: hides the GPU and limits the process to one thread)

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--artifacts", type=Path, default=Path("runs/facebook_video/seed0/artifacts/dualhmm"))
ap.add_argument("--reps", type=int, default=10)
args = ap.parse_args()

import joblib  # noqa: E402
import numpy as np  # noqa: E402

hmm_size = joblib.load(args.artifacts / "dualhmm_hmm_size.pkl")
hmm_time = joblib.load(args.artifacts / "dualhmm_hmm_time.pkl")
discretizer_size = joblib.load(args.artifacts / "dualhmm_discretizer_size.pkl")
discretizer_time = joblib.load(args.artifacts / "dualhmm_discretizer_time.pkl")
N_FLOWS, L = 100, 50


def generate_dualhmm():
    records = []
    for fid in range(N_FLOWS):
        size_bins, _ = hmm_size.sample(L)
        time_bins, _ = hmm_time.sample(L)
        payloads = np.clip(discretizer_size.inverse_transform(size_bins.reshape(-1, 1)).flatten(), 0.0, 1500.0)
        dts = np.maximum(discretizer_time.inverse_transform(time_bins.reshape(-1, 1)).flatten(), 0.0)
        records.extend([[fid, round(p), float(t)] for p, t in zip(payloads, dts)])
    return records


timing.benchmark("Dual-HMM [13]", generate_dualhmm, reps=args.reps,
                 **timing.footprint(hmm_size, hmm_time, discretizer_size, discretizer_time))
