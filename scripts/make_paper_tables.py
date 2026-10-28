#!/usr/bin/env python3
"""
Build the tables of the article and a summary of the numbers quoted in its text from the
evaluation outputs. Run from the repository root after the evaluation steps of reproduce.sh:

    python scripts/make_paper_tables.py      # writes paper_tables.tex and results_summary.md

Inputs (a missing input is skipped with a note): eval/main, eval/ablation, eval/tau, eval/flowlen,
eval/mh, eval/bins, sweeps_val/*/seed*/artifacts/*/meta.json, timing_results.jsonl,
bench_fbvideo/scalability.csv, long_gaps.csv and data_check.csv. Only the seven models of
Table 3 are ranked, whatever else eval/main contains.
"""
from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import evaluate_all as ev  # noqa: E402

MAIN = ["proposed", "m20", "timegan", "aewgan", "necstgen", "trafficgpt", "dualhmm"]
METRICS = ["D_marg", "D_temp", "D_div"]
DS = {"google_home_udp": "Google Home (UDP)", "facebook_audio": "Facebook audio", "facebook_video": "Facebook video",
      "ftps": "FTPS", "5g_netflix": "5G Netflix"}
FEAT = {"payload_length": "payload", "time_diff": "IAT"}
tex, md = [], ["# Results summary\n"]

try:
    import tabulate  # noqa: F401
except ImportError:          # DataFrame.to_markdown needs tabulate; fall back to plain text
    pd.DataFrame.to_markdown = lambda self, *a, **k: "```\n" + self.to_string(index=k.get("index", True)) + "\n```"


def have(path):
    if Path(path).exists():
        return True
    md.append(f"_skipped: {path} not found_\n")
    return False


def ranks(s, method="average"):
    s = s[s.metric.isin(METRICS)].copy()
    s["rank"] = s.groupby(["dataset", "metric", "feature"])["mean"].rank(method=method)
    return s


# ------------------------------------------------------------------------------- Table 3
if have("eval/main/metrics_long.csv"):
    long = pd.read_csv("eval/main/metrics_long.csv")
    long = long[long.model.isin(MAIN)]
    ev.MAIN_METRICS = METRICS
    s = ev.summarize(long)
    s["dataset"] = pd.Categorical(s["dataset"], categories=[d for d in DS if d in set(s["dataset"])], ordered=True)
    s = s.sort_values("dataset")
    s["dataset"] = s["dataset"].astype(str)
    rows = ev.latex_main_table(s, ev.DEFAULT_NAMES, MAIN)
    for k, v in DS.items():
        rows = rows.replace(f"& {k} &", f"& {v} &")
    ref = pd.read_csv("eval/main/real_reference.csv")
    ov = ev.overall(s, ref.groupby("dataset")["n_test_packets"].mean().to_dict())
    tex.append("% ===== Table 3: mean and std over the seeds; best value bold, second best underlined =====\n" + rows)
    tex.append("% ===== Table 3, overall block: mean over the datasets (average rank) =====\n"
               + ev.latex_overall_table(ov, ev.DEFAULT_NAMES, MAIN))
    seeds = long.groupby(["dataset", "model"])["seed"].nunique()
    md.append(f"## Table 3\n\nseeds per (dataset, model): min {seeds.min()}, max {seeds.max()}\n")
    ra, rm = ranks(s, "average"), ranks(s, "min")
    allr = ra.groupby("model")["rank"].mean().sort_values()
    md.append("Average rank over all cells: " + ", ".join(f"{ev.DEFAULT_NAMES.get(m, m)} {r:.2f}" for m, r in allr.items()) + "\n")
    win = rm.groupby("model")["rank"].agg(best=lambda r: int((r == 1).sum()), top2=lambda r: int((r <= 2).sum()))
    ncell = rm.groupby(["dataset", "metric", "feature"]).ngroups
    md.append(f"Proposed: best in {win.loc['proposed', 'best']} of {ncell} cells, top-2 in {win.loc['proposed', 'top2']}\n")
    col = ra.assign(c=ra.metric + " " + ra.feature.map(FEAT)).pivot_table(index="model", columns="c", values="rank").reindex(MAIN)
    md.append("Average rank per column:\n\n" + col.round(1).to_markdown() + "\n")
    md.append("Average rank of the proposed model per dataset: " + ", ".join(
        f"{DS[d]} {v:.2f}" for d, v in rm[rm.model == "proposed"].groupby("dataset")["rank"].mean().items()) + "\n")
    w = s[s.metric.isin(METRICS)].pivot_table(index=["dataset", "metric", "feature"], columns="model", values="mean")
    md.append(f"Proposed better than the prior model in {int((w['proposed'] < w['m20']).sum())} of {len(w)} cells\n")
    mm = w.groupby(level=["metric", "feature"]).mean()
    chg = ((mm["proposed"] - mm["m20"]) / mm["m20"] * 100).round(0)
    md.append("Change of the mean over the datasets vs the prior model (%): "
              + ", ".join(f"{m} {FEAT[f]} {v:+.0f}" for (m, f), v in chg.items()) + "\n")
    # every run against the real test flows of its own split
    p = long[long.metric == "p999_IAT_s"].set_index(["dataset", "seed", "model"])["value"]
    r = ref.set_index(["dataset", "seed"])["time_diff_q99.9"]
    lr = np.log10(p.clip(lower=1e-9).to_numpy() / r.reindex(p.index.droplevel("model")).clip(lower=1e-9).to_numpy())
    lr = pd.Series(lr, index=p.index).unstack("model")[MAIN]
    summ = lr.groupby(level="dataset").agg(lambda x: f"{np.median(x):+.2f} [{x.min():+.2f}, {x.max():+.2f}]")
    md.append("99.9th percentile of the IAT, log10(generated / real) per run, median [min, max] over the seeds:\n\n"
              + summ.to_markdown() + "\n")
    tp = ref.pivot_table(index="dataset", columns="seed", values="n_test_packets").astype(int)
    md.append("Test packets per split:\n\n" + tp.to_markdown() + "\n")
    share = ref.groupby("dataset")["n_test_packets"].mean()
    share = share / share.sum()
    pw = ov[ov.model.isin(MAIN)].pivot_table(index="model", columns=["metric", "feature"], values="packet_weighted_mean")
    pwr = pw.rank().mean(axis=1).sort_values()
    md.append("Overall block = mean over the datasets (datasets weighted equally). Share of the test packets: "
              + ", ".join(f"{DS[d]} {v:.1%}" for d, v in share.items())
              + f". Mean rank of the packet-weighted means over the {pw.shape[1]} columns: "
              + ", ".join(f"{ev.DEFAULT_NAMES.get(m, m)} {v:.1f}" for m, v in pwr.items()) + "\n")
    best_cells = rm[rm["rank"] == 1].groupby("dataset").apply(
        lambda g: ", ".join(f"{m} {FEAT[f]}: {mo}" for m, f, mo in zip(g.metric, g.feature, g.model)))
    md.append("Best model per cell:\n\n" + "\n".join(f"- {DS[d]}: {v}" for d, v in best_cells.items()) + "\n")

# ------------------------------------------------------------------------------- Table 5 (ablation)
if have("eval/ablation/metrics_long.csv"):
    a = pd.read_csv("eval/ablation/metrics_long.csv")
    aref = pd.read_csv("eval/ablation/real_reference.csv")
    names = {"proposed": "Full model", "abl_A_noidle": "A: no idle state", "abl_B_noflowlen": "B: no flow-length input",
             "abl_C_gaussian": "C: Gaussian components", "m20": r"Prior model~\cite{koktas2025state}"}
    cols = [("D_marg", "payload_length"), ("D_marg", "time_diff"), ("D_temp", "payload_length"), ("D_temp", "time_diff"),
            ("D_div", "payload_length"), ("D_div", "time_diff")]
    g = a.groupby(["model", "metric", "feature"])["value"].agg(["mean", "std"])
    best = {c: g.xs(c[0], level="metric").xs(c[1], level="feature")["mean"].idxmin() for c in cols}
    lines = []
    for m, lab in names.items():
        if m not in g.index.get_level_values(0):
            continue
        cells = []
        for c in cols:
            mu, sd = g.loc[(m, c[0], c[1])]
            t = f"{mu:.3f}{{\\scriptstyle\\pm{sd:.3f}}}"
            cells.append(f"$\\mathbf{{{t}}}$" if best[c] == m else f"${t}$")
        ext = a[(a.model == m) & a.metric.isin(["p999_IAT_s", "max_IAT_s"])].groupby("metric")["value"].median()
        lines.append(f"{lab} & " + " & ".join(cells) + f" & {ext['p999_IAT_s']:.2f} & {ext['max_IAT_s']:.2f} \\\\")
    lines += [r"\hline", "Real test flows & " + " & ".join(["--"] * len(cols))
              + f" & {aref['time_diff_q99.9'].median():.2f} & {aref['time_diff_max'].median():.2f} \\\\"]
    tex.append("% ===== Table 5, Facebook video: mean and std over the seeds, best value bold\n"
               "% columns: D_marg pay | D_marg IAT | D_temp pay | D_temp IAT | D_div pay | D_div IAT | "
               "99.9% IAT (s) | max IAT (s); the last two are medians over the seeds =====\n" + "\n".join(lines))
    md.append("## Table 5 (ablation, Facebook video)\n\n" + "\n".join(f"- {x}" for x in lines[:-2]) + "\n")
    order = [m for m in names if m in set(a.model)]
    for met, rc in [("p999_IAT_s", "time_diff_q99.9"), ("max_IAT_s", "time_diff_max")]:
        t5 = a[a.metric == met].pivot_table(index="model", columns="seed", values="value").reindex(order)
        t5.loc["real test flows"] = aref.set_index("seed")[rc]
        md.append(f"{met} per seed (the table shows the median):\n\n" + t5.round(2).to_markdown() + "\n")
    md.append("Test packets per seed: " + ", ".join(f"{int(s)}: {int(n)}" for s, n in aref.set_index("seed")["n_test_packets"].items()) + "\n")

# ------------------------------------------------------------------------------- tail threshold
if have("eval/tau/metrics_long.csv"):
    l = pd.read_csv("eval/tau/metrics_long.csv")
    l = l[l.seed.isin([0, 1, 2])]
    keys = [("sw_q0p99", "0.99"), ("sw_q0p995", "0.995"), ("proposed", "0.998 (default)"), ("sw_q0p999", "0.999"),
            ("sw_q0p9995", "0.9995")]
    g = l.groupby(["model", "metric", "feature"])["value"].agg(["mean", "std"])
    lines = []
    for k, lab in keys:
        if k not in g.index.get_level_values(0):
            continue
        cells = [f"${g.loc[(k, m, f), 'mean']:.3f}{{\\scriptstyle\\pm{g.loc[(k, m, f), 'std']:.3f}}}$"
                 for m, f in [("D_marg", "time_diff"), ("D_temp", "time_diff"), ("D_div", "time_diff"), ("D_div", "payload_length")]]
        pq = l[(l.model == k) & (l.metric == "p999_IAT_s")]["value"].median()
        lines.append(f"{lab} & " + " & ".join(cells) + f" & {pq:.2f} \\\\")
    tex.append("% ===== tail-threshold quantile (seeds 0-2); columns D_marg IAT | D_temp IAT | D_div IAT | D_div pay | "
               "99.9% IAT (s, median over the seeds) =====\n" + "\n".join(lines))
    md.append("## Tail-threshold quantile\n\n" + "\n".join(f"- {x}" for x in lines) + "\n")

# ------------------------------------------------------------------------------- flow-length selector
if have("eval/flowlen/metrics_long.csv") and Path("eval/main/metrics_long.csv").exists():
    fl = pd.read_csv("eval/flowlen/metrics_long.csv")
    base = pd.read_csv("eval/main/metrics_long.csv")
    base = base[base.model.isin(MAIN)]
    out = []
    ns = fl.groupby(["dataset", "model"])["seed"].nunique().unstack()
    short = [(d, v, int(ns.loc[d, v])) for d in ns.index for v in ["fl_empirical"]
             if v in ns and ns.loc[d, v] < ns.loc[d, "proposed"]]
    if short:
        out.append("INCOMPLETE: " + ", ".join(f"{d} {v} has {n} seed(s)" for d, v, n in short))
    for v in ["fl_empirical"]:
        x = pd.concat([fl[fl.model == v], base[base.model != "proposed"]])
        x = x[x.metric.isin(METRICS)]
        s2 = x.groupby(["dataset", "metric", "feature", "model"])["value"].mean().reset_index()
        s2["rank"] = s2.groupby(["dataset", "metric", "feature"])["value"].rank(method="average")
        rr = s2.groupby("model")["rank"].mean().sort_values()
        out.append(f"{v}: average rank {rr[v]:.2f}; next best {rr.drop(v).index[0]} {rr.drop(v).iloc[0]:.2f}")
    f2 = fl[fl.metric.isin(METRICS)].groupby(["dataset", "metric", "feature", "model"])["value"].mean().unstack()
    for v in ["fl_empirical"]:
        d = ((f2[v] - f2["proposed"]) / f2["proposed"].abs() * 100).groupby(level="metric").median()
        out.append(f"{v}: median change vs the test-flow lengths (%): " + ", ".join(f"{k} {x:+.1f}" for k, x in d.items()))
    md.append("## Flow-length selector\n\n" + "\n".join(f"- {x}" for x in out) + "\n")

# ------------------------------------------------------------------------------- M and H
metas = [json.load(open(p)) for p in glob.glob("sweeps_val/*/seed*/artifacts/*/meta.json")]
if metas:
    m = pd.DataFrame(metas)
    t = m.pivot_table(index=["dataset", "M", "H"], columns="seed", values="val_nll")
    seeds = list(t.columns)
    t = t.assign(mean=t[seeds].mean(axis=1), std=t[seeds].std(axis=1),
                 mean_rank=t[seeds].groupby(level="dataset").rank().mean(axis=1))
    par = m.groupby(["dataset", "M", "H"])["mdn_params"].first()
    md.append("## M and H, validation NLL (selection: python scripts/select_m.py)\n\n" + t.join(par).round(3).to_markdown() + "\n")
else:
    md.append("_skipped: no sweeps_val meta.json files_\n")
if have("eval/mh/metrics_long.csv"):
    l = pd.read_csv("eval/mh/metrics_long.csv")
    t = l[l.metric.isin(["D_marg", "D_div"]) & (l.feature == "time_diff")].groupby(["dataset", "model", "metric"])["value"].mean().unstack()
    md.append("M/H test metrics (IAT):\n\n" + t.round(3).to_markdown() + "\n")

# ------------------------------------------------------------------------------- bins
if have("eval/bins/bin_sensitivity.csv"):
    from scipy.stats import kendalltau
    b = pd.read_csv("eval/bins/bin_sensitivity.csv")
    b = b[b.model.isin(MAIN)]
    g = b.groupby(["dataset", "feature", "B", "model"])["value"].mean().reset_index()
    taus, rk = [], []
    for (d, f), sub in g.groupby(["dataset", "feature"]):
        base = sub[sub.B == 1000].set_index("model")["value"]
        for B, s3 in sub.groupby("B"):
            s3 = s3.set_index("model")["value"].reindex(base.index)
            taus.append((B, kendalltau(base.values, s3.values)[0]))
            rk.append((d, f, B, int(s3.rank(method="min")["proposed"])))
    t = pd.DataFrame(taus, columns=["B", "tau"]).groupby("B")["tau"].min()
    r = pd.DataFrame(rk, columns=["dataset", "feature", "B", "rank"])
    md.append("## Bins\n\nminimum Kendall tau vs B = 1000: " + ", ".join(f"B={B}: {v:.2f}" for B, v in t.items()) + "\n\n"
              "D_marg rank of the proposed model (min and max over B):\n\n"
              + r.groupby(["dataset", "feature"])["rank"].agg(["min", "max"]).to_markdown() + "\n")

# ------------------------------------------------------------------------------- Table 4, scalability, long gaps, Table 2
if have("timing_results.jsonl"):
    t = pd.DataFrame([json.loads(x) for x in open("timing_results.jsonl")]).drop_duplicates("model", keep="last")
    lines = [f"{r.model} & ${r.mean_s:.4f}{{\\scriptstyle\\pm{r.std_s:.4f}}}$ & {int(r.valid_packets):,} & "
             f"{r.pkt_per_s:,.0f} & {r.footprint_MB:.4f} & {int(r.nn_params):,} \\\\" for r in t.itertuples()]
    tex.append("% ===== Table 4: model | time (s) | valid packets | packets/s | footprint (MB) | NN parameters =====\n"
               + "\n".join(lines))
    md.append("## Table 4\n\n" + "\n".join(f"- {x}" for x in lines) + f"\n\nCPU: {t.cpu.iloc[-1]} | torch {t.torch.iloc[-1]}\n")
if have("bench_fbvideo/scalability.csv"):
    md.append("## Scalability\n\n" + pd.read_csv("bench_fbvideo/scalability.csv").to_markdown(index=False) + "\n")
if have("long_gaps.csv"):
    lg = pd.read_csv("long_gaps.csv").sort_values("seed")
    agg = lg.groupby(["dataset", "series"]).agg(
        share_mean=("share_in_short_flows", "mean"), share_std=("share_in_short_flows", "std"),
        seeds_with_long_gaps=("share_in_short_flows", "count"),
        long_gaps_per_seed=("long_gaps", lambda x: ", ".join(str(int(v)) for v in x)))
    md.append("## Long gaps (share of the IATs > 1 s in flows with <= 100 packets)\n\n"
              "The share is undefined for a run without any IAT > 1 s, so mean and std use only the seeds counted "
              "in seeds_with_long_gaps.\n\n" + agg.round(3).to_markdown() + "\n")
if have("data_check.csv"):
    md.append("## Table 2 statistics\n\n" + pd.read_csv("data_check.csv").to_markdown(index=False) + "\n")

Path("paper_tables.tex").write_text("\n\n".join(tex) + "\n")
Path("results_summary.md").write_text("\n".join(md))
print("wrote paper_tables.tex and results_summary.md")
