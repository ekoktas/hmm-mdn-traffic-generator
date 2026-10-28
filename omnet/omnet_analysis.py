#!/usr/bin/env python3
"""
Figure data of the system-level study from the CSV-S exports in omnet_results_csv/:

  tikz_data/throughput_<scenario>.dat     received throughput of one run in 10-s bins (Fig. 8); the run is
                                          the one in which the synthetic traffic delivers the median volume
                                          (lower median), the same run number for all three scenarios
  tikz_data/queue_mb_cdf_<scenario>.dat   MAC queue size in MB, CDF pooled over all runs and UEs (Fig. 9)
  tikz_data/queue_cdf_<scenario>.dat      MAC queue length in packets, pooled CDF
  tikz_data/latency_cdf_<scenario>.dat    end-to-end latency in ms, pooled CDF (Fig. 9)

Run inside the OMNeT++ project folder after run_monte_carlo.py. --fig-run selects another run for Fig. 8.
"""
import argparse
import os
import re

import numpy as np
import pandas as pd

SCENARIOS = ["basic", "real", "ml"]


def read_export(path):
    try:
        return pd.read_csv(path, low_memory=False)
    except FileNotFoundError:
        print(f"warning: {path} not found")
        return None


def run_order(df):
    """Indices of the column pairs ordered by the start time of their run (the runs were executed one after another)."""
    keys = []
    for k in range(len(df.columns) // 2):
        m = re.search(r"-(\d{8}-\d{2}:\d{2}:\d{2})-(\d+)\)\s*$", str(df.columns[2 * k]))
        keys.append((m.group(1), int(m.group(2))) if m else None)
    if any(x is None for x in keys):
        print("warning: run IDs not found in the header; the export order is used")
        return list(range(len(keys)))
    return sorted(range(len(keys)), key=lambda k: keys[k])


def throughput(fig_run):
    ml = read_export("omnet_results_csv/throughput_data_ml.csv")
    if ml is not None:
        order = run_order(ml)
        vol = [pd.to_numeric(ml.iloc[:, 2 * k + 1], errors="coerce").sum() / 1e6 for k in order]
        if fig_run is None:
            fig_run = int(np.argsort(vol)[(len(vol) - 1) // 2])
        print(f"Fig. 8 shows run {fig_run} of {len(vol)}: the synthetic traffic delivered {vol[fig_run]:.2f} MB "
              f"(median {np.median(vol):.2f} MB, range {min(vol):.2f} to {max(vol):.2f} MB)")
    fig_run = fig_run or 0
    for sc in SCENARIOS:
        df = read_export(f"omnet_results_csv/throughput_data_{sc}.csv")
        if df is None:
            continue
        c = 2 * run_order(df)[fig_run]                     # time and value of the same vector, i.e. one run
        pair = df.iloc[:, [c, c + 1]].dropna()
        times = pd.to_numeric(pair.iloc[:, 0]).values
        bytes_val = pd.to_numeric(pair.iloc[:, 1]).values
        if len(times) > 0:
            duration = np.max(times) - np.min(times)
            print(f"{sc}: {np.sum(bytes_val) / 1e6:.2f} MB received, {np.sum(bytes_val) * 8 / max(duration, 1.0) / 1e3:.2f} "
                  f"kbit/s over {duration:.1f} s")
        rate = pd.DataFrame({"time": np.floor(times).astype(int), "bytes": bytes_val}).groupby("time").sum()
        rate["kbps"] = rate["bytes"] * 8 / 1000                   # bytes per 1-s bin -> kbit/s
        rate = pd.DataFrame({"time": np.arange(0, int(rate.index.max()) + 1)}).merge(rate, on="time", how="left").fillna(0)
        rate["time_bin"] = rate["time"] // 10                     # 10-s bins: mean rate, so the area stays the volume
        down = rate.groupby("time_bin").agg({"time": "first", "kbps": "mean"})
        down[["time", "kbps"]].to_csv(f"tikz_data/throughput_{sc}.dat", sep="\t", index=False)


def pooled_cdf(prefix, out_name):
    for sc in SCENARIOS:
        df = read_export(f"omnet_results_csv/{prefix}_{sc}.csv")
        if df is None:
            continue
        vals = []
        for col in range(1, len(df.columns), 2):
            vals.extend(pd.to_numeric(df.iloc[:, col], errors="coerce").dropna().values)
        if not vals:
            continue
        vals = np.sort(vals)
        if "latency" in prefix:
            vals = vals * 1000                                    # s -> ms
        if "queue_bits" in prefix:
            vals = vals / 8 / 1e6                                 # bits -> MB
        if len(vals) > 300:
            if "latency" in prefix:                              # 300 log-spaced points for the heavy-tailed latency
                pos = vals[vals > 0]
                if len(pos) == 0:
                    continue
                grid = np.logspace(np.log10(max(np.min(pos), 1e-3)), np.log10(np.max(pos)), 300)
                p = np.searchsorted(vals, grid, side="right") / len(vals)
                vals = grid
            else:                                                 # 300 evenly spaced order statistics
                p_full = np.arange(1, len(vals) + 1) / len(vals)
                idx = np.linspace(0, len(vals) - 1, 300).astype(int)
                vals, p = vals[idx], p_full[idx]
        else:
            p = np.arange(1, len(vals) + 1) / len(vals)
        pd.DataFrame({out_name: vals, "cdf": p}).to_csv(f"tikz_data/{out_name}_cdf_{sc}.dat", sep="\t", index=False)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fig-run", type=int, default=None, help="run shown in Fig. 8 (default: median-volume run)")
    args = ap.parse_args()
    os.makedirs("tikz_data", exist_ok=True)
    throughput(args.fig_run)
    pooled_cdf("latency_data", "latency")
    pooled_cdf("queue_lengths", "queue")
    pooled_cdf("queue_bits", "queue_mb")
    print("figure data written to tikz_data/")


if __name__ == "__main__":
    main()
