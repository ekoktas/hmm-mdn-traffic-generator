#!/usr/bin/env python3
"""
Selection of the number of mixture components M (H = 128) from the validation NLL.

Rule: on each dataset, take the smallest configuration whose mean validation NLL over the splits
lies within one standard deviation (that of the best configuration, over the same splits) of the
best configuration. The configuration chosen on most datasets is used for all of them. Only
splits with results for both configurations count.

Reads sweeps_val/<dataset>/seed<k>/artifacts/sw_M16|sw_M32/meta.json, written by the plans
sweep_mh (Facebook video) and sweep_m_all (the other datasets).

    python scripts/select_m.py
"""
import json
from pathlib import Path

import pandas as pd

SWEEPS_ROOT = Path("sweeps_val")
KEYS = {"sw_M16": 16, "sw_M32": 32}
N_DATASETS, N_SEEDS = 5, 3


def main():
    rows, warn = [], []
    for key, M in KEYS.items():
        for p in sorted(SWEEPS_ROOT.glob(f"*/seed*/artifacts/{key}/meta.json")):
            meta = json.loads(p.read_text())
            ds, seed = p.parts[-5], int(p.parts[-4][4:])
            if meta.get("dataset") not in (None, ds) or meta.get("M") != M or meta.get("H") != 128:
                warn.append(f"{p}: dataset, M or H in meta.json do not match the folder")
                continue
            if meta.get("val_nll") is None:
                warn.append(f"{p}: no val_nll")
                continue
            rows.append({"dataset": ds, "seed": seed, "M": M, "val_nll": float(meta["val_nll"])})
    if not rows:
        print(f"no sw_M16/sw_M32 results in {SWEEPS_ROOT}/")
        return
    df = pd.DataFrame(rows)
    out, per_seed = [], []
    for ds, g in df.groupby("dataset"):
        w = g.pivot_table(index="seed", columns="M", values="val_nll").reindex(columns=sorted(KEYS.values())).dropna()
        per_seed.append(w.round(3).assign(dataset=ds).reset_index().set_index(["dataset", "seed"]))
        if len(w) < N_SEEDS:
            out.append({"dataset": ds, "paired splits": len(w), "choice": "incomplete"})
            continue
        mean, std = w.mean(), w.std()
        best = mean.idxmin()
        choice = min(M for M in mean.index if mean[M] <= mean[best] + std[best])
        out.append({"dataset": ds, "paired splits": len(w),
                    **{f"M={M}": f"{mean[M]:.3f} +- {std[M]:.3f}" for M in mean.index},
                    "M=16 lower in": f"{int((w[16] < w[32]).sum())}/{len(w)} splits",
                    "threshold (best + 1 std)": round(mean[best] + std[best], 3), "choice": choice})
    t = pd.DataFrame(out).set_index("dataset")
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print("validation NLL per split (lower is better):\n")
        print(pd.concat(per_seed).to_string())
        print("\nselection per dataset (smallest configuration within one standard deviation of the best):\n")
        print(t.to_string())
    done = t[t["choice"] != "incomplete"]
    if len(done) < N_DATASETS:
        print(f"\nno decision yet: {len(done)} of {N_DATASETS} datasets complete")
    else:
        votes = done["choice"].value_counts()
        print(f"\ndecision: M = {int(votes.idxmax())} for all datasets (chosen on {votes.max()} of {N_DATASETS} datasets)")
    for w in warn:
        print("WARNING:", w)


if __name__ == "__main__":
    main()
