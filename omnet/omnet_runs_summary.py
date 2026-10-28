#!/usr/bin/env python3
"""
Per-run results of the system-level study and their mean and standard deviation over the runs:
offered and delivered volume, delivered share, active time (first to last received packet) and
the median and 99th-percentile end-to-end latency.

Reads the CSV-S exports in omnet_results_csv/ (two columns per vector: time and value; the header
of the time column holds the run ID) and the traces of the last run in real_traces/ and ml_traces/.
Run inside the OMNeT++ project folder after run_monte_carlo.py.
"""
import argparse
import re

import numpy as np
import pandas as pd

SCENARIOS = ["basic", "real", "ml"]


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


def pairs(path):
    df = pd.read_csv(path, low_memory=False)
    out = []
    for k in run_order(df):
        t = pd.to_numeric(df.iloc[:, 2 * k], errors="coerce")
        v = pd.to_numeric(df.iloc[:, 2 * k + 1], errors="coerce")
        ok = t.notna() & v.notna()
        out.append((t[ok].to_numpy(), v[ok].to_numpy()))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n-ues", type=int, default=5)
    args = ap.parse_args()
    rows = []
    for sc in SCENARIOS:
        thr = pairs(f"omnet_results_csv/throughput_data_{sc}.csv")
        lat = pairs(f"omnet_results_csv/latency_data_{sc}.csv")
        if len(thr) != len(lat):
            print(f"warning: {sc}: {len(thr)} throughput vectors but {len(lat)} latency vectors")
        offered = np.nan
        if sc != "basic":       # the volume offered by the trace-driven scenarios is the same in every run
            offered = sum(pd.read_csv(f"{sc}_traces/{sc}_trace_ue{u}.csv")["size_bytes"].sum()
                          for u in range(args.n_ues)) / 1e6
        for run, ((t, b), (_, l)) in enumerate(zip(thr, lat)):
            rows.append({"scenario": sc, "run": run, "offered_MB": offered, "delivered_MB": b.sum() / 1e6,
                         "active_s": t.max() - t.min() if len(t) else np.nan,
                         "latency_median_ms": 1e3 * np.median(l) if len(l) else np.nan,
                         "latency_p99_ms": 1e3 * np.percentile(l, 99) if len(l) else np.nan})
    df = pd.DataFrame(rows)
    df["delivered_share_pct"] = 100 * df["delivered_MB"] / df["offered_MB"]
    df.to_csv("omnet_runs.csv", index=False)
    summary = df.drop(columns="run").groupby("scenario").agg(["mean", "std"]).round(3)
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(summary.to_string())
    print("\nper-run values: omnet_runs.csv")


if __name__ == "__main__":
    main()
