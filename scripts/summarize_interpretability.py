#!/usr/bin/env python3
"""
Compare the output of notebooks/analysis_interpretability.ipynb across the seeds (splits).

Every split has its own HMM, so state numbers do not correspond between splits. States are
matched by role: the long-gap state is the used state with the largest 99th-percentile IAT among
the training packets that the Viterbi path assigns to it. For each split this prints the
fingerprint of that state, the rank of its effective degrees of freedom among the used states
(1 = lowest nu, heaviest tail) and its generated P(IAT > 1 s) for flows of 5 to 5,000 packets.

    python scripts/summarize_interpretability.py --dataset facebook_video --model proposed
"""
import argparse
import json
from pathlib import Path

import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=Path, default=Path("runs"))
    ap.add_argument("--dataset", default="facebook_video")
    ap.add_argument("--model", default="proposed")
    a = ap.parse_args()
    rows, nus, missing = [], [], []
    for d in sorted((a.runs / a.dataset).glob("seed*"), key=lambda p: int(p.name[4:])):
        art = d / "artifacts" / a.model
        fp_p, fl_p = art / "state_fingerprint.csv", art / "flow_length_response.csv"
        if not (fp_p.exists() and fl_p.exists()):
            missing.append(d.name)
            continue
        meta = json.loads((art / "meta.json").read_text())
        fp = pd.read_csv(fp_p, index_col="state")
        fl = pd.read_csv(fl_p)
        used = fp[fp["share"].notna()]
        lg = int(used["iat_p99_s"].idxmax())
        p = fl[fl["state"] == lg].set_index("L")["p_iat_gt_1s"]
        seed = int(d.name[4:])
        rows.append({
            "seed": seed, "M": meta.get("M"), "K": meta.get("K"),
            "idle_init_state": meta["K"] - 1 if meta.get("idle") else "-",
            "long_gap_state": lg,
            "share": round(used.loc[lg, "share"], 3),
            "train_iat_p99_s": round(used.loc[lg, "iat_p99_s"], 2),
            "dwell_pkts": round(used.loc[lg, "dwell_pkts"], 1),
            "nu_eff": round(used.loc[lg, "nu_eff"], 1),
            "nu_rank": f"{int(used['nu_eff'].rank(method='min')[lg])}/{len(used)}",
            **{f"gen_P>1s_L{L}": p.get(L) for L in (5, 50, 500, 5000)},
            "gen_P>1s_max_any_state": fl["p_iat_gt_1s"].max(),
        })
        nus.append(used["nu_eff"].round(1).rename(f"seed{seed}"))
    if not rows:
        print(f"no results in {a.runs / a.dataset}/seed*/artifacts/{a.model}/")
        return
    t = pd.DataFrame(rows).set_index("seed")
    with pd.option_context("display.width", 250, "display.max_columns", 30):
        print(f"long-gap state per seed ({a.dataset}, {a.model}); nu_rank 1 = lowest nu_eff = heaviest tail\n")
        print(t.to_string())
        print("\nnu_eff of every used state (state numbers differ between seeds):\n")
        print(pd.concat(nus, axis=1).to_string())
    print(f"\nthe long-gap state has the lowest nu_eff in {sum(r['nu_rank'].startswith('1/') for r in rows)} "
          f"of {len(rows)} seeds")
    if missing:
        print("no analysis output for: " + ", ".join(missing))


if __name__ == "__main__":
    main()
