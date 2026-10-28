#!/usr/bin/env python3
"""
Check every dataset listed in run_matrix.py and print the statistics of Table 2.

    python scripts/check_data.py

For each file it checks what the pipeline assumes (the columns exist, the packets of a flow are
contiguous, value ranges) and reports the number of flows and packets, packets per flow, volume,
mean flow duration, median and 99.8th-percentile IAT, their ratio in decades and the share of
flows with at most 10 packets. The statistics are computed on the file as it is, before the
cleaning in data_utils.load_packets. Nothing is modified; the table is written to data_check.csv.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_matrix import DATASETS  # noqa: E402

NEEDED = ["flow_id", "payload_length", "time_diff"]


def check(key: str, path: str) -> dict:
    p = Path(path)
    row = {"dataset": key, "file": path}
    if not p.exists():
        row["status"] = "MISSING"
        return row
    head = pd.read_csv(p, nrows=5)
    missing = [c for c in NEEDED if c not in head.columns]
    if missing:
        row["status"] = f"MISSING COLUMNS {missing} (has {list(head.columns)})"
        return row
    df = pd.read_csv(p, usecols=NEEDED)
    fid = df["flow_id"].to_numpy()
    blocks = int((fid[1:] != fid[:-1]).sum()) + 1          # number of contiguous runs of one flow id
    n_flows = int(df["flow_id"].nunique())
    sizes = df.groupby("flow_id").size()
    iat = df["time_diff"].to_numpy(np.float64)
    iat_pos = iat[iat > 0]
    med, q998 = float(np.median(iat_pos)), float(np.quantile(iat_pos, 0.998))
    row.update({
        "status": "ok",
        "file_size_MiB": round(p.stat().st_size / 2**20, 2),
        "flows": n_flows,
        "packets": len(df),
        "pkts_per_flow": round(len(df) / n_flows, 1),
        "volume_MiB": round(df["payload_length"].clip(lower=0).sum() / 2**20, 2),
        "mean_flow_duration_s": round(float(df.groupby("flow_id")["time_diff"].sum().mean()), 2),
        "median_iat_ms": round(1e3 * med, 3),
        "p99.8_iat_s": round(q998, 4),
        "tail_ratio_decades": round(float(np.log10(q998 / med)), 2) if med > 0 else np.nan,
        "flows_le_10_pkts_pct": round(100 * float((sizes <= 10).mean()), 1),
        "sorted_by_flow_id": bool(df["flow_id"].is_monotonic_increasing),
        "flows_contiguous": blocks == n_flows,
        "payload_gt_1500_pct": round(100 * float((df["payload_length"] > 1500).mean()), 3),
        "iat_ge_3600s": int((iat >= 3600).sum()),
        "zero_pay_zero_iat_rows": int(((df["payload_length"] == 0) & (df["time_diff"] == 0)).sum()),
    })
    return row


def main():
    df = pd.DataFrame([check(k, v) for k, v in DATASETS.items()]).set_index("dataset")
    with pd.option_context("display.max_columns", None, "display.width", 200):
        print(df.T.to_string())
    print()
    for key, r in df[df["status"] == "ok"].iterrows():
        if not r["flows_contiguous"] or not r["sorted_by_flow_id"]:
            print(f"[note] {key}: the file is not sorted by flow_id; load_packets() sorts it (stable sort, "
                  f"the packet order inside each flow is kept).")
        if r["payload_gt_1500_pct"] > 0:
            print(f"[note] {key}: {r['payload_gt_1500_pct']}% of the payloads exceed 1500 B and are clipped.")
    for key, r in df[df["status"] != "ok"].iterrows():
        print(f"[fail] {key}: {r['status']}  ({r['file']})")
    df.to_csv("data_check.csv")
    print("\nwritten: data_check.csv")


if __name__ == "__main__":
    main()
