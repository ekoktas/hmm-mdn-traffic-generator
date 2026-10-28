#!/usr/bin/env python3
"""
Data of the per-flow CDF figures. For every dataset and feature: the average per-flow CDF
(every flow weighted equally) of the real test flows and of each model on a logarithmic grid,
averaged over the seeds. One CSV per dataset and feature with the columns x, real, <model>...

    python scripts/export_cdfs.py --runs runs --out figs_data

A curve can be plotted with pgfplots, e.g.
    \\addplot table [x=x, y=proposed, col sep=comma] {figs_data/facebook_video_time_diff.csv};
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluate_all import discover, load_trace  # noqa: E402

GRIDS = {"payload_length": np.logspace(0, np.log10(1500), 200),
         "time_diff": np.logspace(-6, np.log10(3600), 250)}
DEFAULT_MODELS = ["proposed", "m20", "timegan", "aewgan", "necstgen", "trafficgpt", "dualhmm"]


def avg_flow_cdf(df: pd.DataFrame, feat: str, grid: np.ndarray) -> np.ndarray:
    codes, _ = pd.factorize(df["flow_id"], sort=False)
    vals = df[feat].to_numpy(np.float64)
    order = np.lexsort((vals, codes))
    v, c = vals[order], codes[order]
    acc, n = np.zeros(len(grid)), 0
    for seg in np.split(v, np.flatnonzero(np.diff(c)) + 1):
        acc += np.searchsorted(seg, grid, side="right") / len(seg)
        n += 1
    return acc / n


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", type=Path, default=Path("runs"))
    ap.add_argument("--out", type=Path, default=Path("figs_data"))
    ap.add_argument("--datasets", nargs="*")
    ap.add_argument("--models", nargs="*", default=DEFAULT_MODELS)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    curves = {}                                   # (dataset, feature, series) -> list of per-seed curves
    for ds, seed, test_path, model_paths in discover(args.runs, args.datasets, args.models):
        frames = {"real": load_trace(test_path)}
        frames.update({m: load_trace(p) for m, p in model_paths.items()})
        for feat, grid in GRIDS.items():
            for name, df in frames.items():
                curves.setdefault((ds, feat, name), []).append(avg_flow_cdf(df, feat, grid))
        print(f"done  {ds:>16s}  seed{seed}")

    for ds in sorted({k[0] for k in curves}):
        for feat, grid in GRIDS.items():
            table = {"x": grid}
            for name in ["real"] + args.models:
                if (ds, feat, name) in curves:
                    table[name] = np.mean(curves[(ds, feat, name)], axis=0)
            out = args.out / f"{ds}_{feat}.csv"
            pd.DataFrame(table).round(6).to_csv(out, index=False)
            print(f"wrote {out}  ({len(table) - 1} series)")


if __name__ == "__main__":
    main()
