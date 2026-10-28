#!/usr/bin/env python3
"""
Where do the long gaps occur? For the real test flows and each model: the share of the IATs above
--threshold seconds that lie in flows with at most --split packets, per seed, and the mean and
standard deviation over the seeds.

    python scripts/long_gaps.py --datasets ftps facebook_video 5g_netflix --models proposed m20
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluate_all import discover, load_trace  # noqa: E402


def short_share(df: pd.DataFrame, thr: float, split: int):
    size = df.groupby("flow_id")["flow_id"].transform("size")
    big = df["time_diff"] > thr
    n = int(big.sum())
    return (float((big & (size <= split)).sum() / n) if n else np.nan), n


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", type=Path, default=Path("runs"))
    ap.add_argument("--datasets", nargs="*")
    ap.add_argument("--models", nargs="*", default=["proposed", "m20"])
    ap.add_argument("--threshold", type=float, default=1.0, help="seconds")
    ap.add_argument("--split", type=int, default=100, help="short and medium flows have at most this many packets")
    ap.add_argument("--out", type=Path, default=Path("long_gaps.csv"))
    args = ap.parse_args()

    rows = []
    for ds, seed, test_path, model_paths in discover(args.runs, args.datasets, args.models):
        for name, path in [("real", test_path)] + list(model_paths.items()):
            share, n = short_share(load_trace(path), args.threshold, args.split)
            rows.append(dict(dataset=ds, seed=seed, series=name, share_in_short_flows=share, long_gaps=n))
    df = pd.DataFrame(rows)
    out = (df.groupby(["dataset", "series"])
             .agg(share_mean=("share_in_short_flows", "mean"), share_std=("share_in_short_flows", "std"),
                  long_gaps_mean=("long_gaps", "mean"), seeds=("seed", "nunique")).round(3))
    print(f"share of the IATs > {args.threshold:g} s that lie in flows with <= {args.split} packets")
    print(out.to_string())
    df.to_csv(args.out, index=False)
    print(f"written: {args.out}")


if __name__ == "__main__":
    main()
