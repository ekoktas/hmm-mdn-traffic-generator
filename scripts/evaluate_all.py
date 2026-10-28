#!/usr/bin/env python3
"""
Discrepancy metrics of the article for every (dataset, seed, model) found on disk, aggregated
over the seeds (mean and standard deviation), with the overall block of Table 3 (mean over the
datasets and average rank) and, optionally, the sensitivity of D_marg to the number of bins.

  D_marg  sum over B bins of |mean per-flow PMF of the real flows - that of the generated flows|,
          with B bins whose edges are log-spaced between the minimum and maximum of the real data
  D_temp  RMSE between the mean per-flow autocorrelation functions of log(1 + x), lags 1..W
  D_div   1-D Wasserstein distance between the pooled values of log(1 + x)

Expected layout (all models of a dataset and seed share one split):
    runs/<dataset>/seed<k>/test.csv        real test flows
    runs/<dataset>/seed<k>/<model>.csv     synthetic flows, one file per model
Every CSV has the columns flow_id, payload_length, time_diff, with the packets of each flow in
temporal order.

    python scripts/evaluate_all.py --runs runs --out eval/main --models proposed m20 timegan aewgan necstgen trafficgpt dualhmm
    python scripts/evaluate_all.py --runs runs --out eval/bins --models proposed m20 --no-extras --bins-sweep 100 250 500 1000 2000
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, wasserstein_distance

FEATURES = ("payload_length", "time_diff")
MAIN_METRICS = ["D_marg", "D_temp", "D_div"]
TAIL_QUANTILES = (99.0, 99.9, 99.99)

DEFAULT_NAMES = {                   # LaTeX names in the tables (bib keys of the article)
    "proposed": "Proposed",
    "m20": r"Prior model~\cite{koktas2025state}",
    "timegan": r"TimeGAN~\cite{NEURIPS2019_c9efe5f2}",
    "aewgan": r"AE-WGAN~\cite{9320384}",
    "necstgen": r"NeCSTGen~\cite{10000731}",
    "trafficgpt": r"TrafficGPT (adapted)~\cite{qu2024trafficgpt}",
    "dualhmm": r"Dual-HMM~\cite{9145646}",
}


# ----------------------------------------------------------------------------------------------
# input
# ----------------------------------------------------------------------------------------------
def load_trace(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, usecols=["flow_id", "payload_length", "time_diff"])
    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    df["payload_length"] = df["payload_length"].clip(lower=0)
    df["time_diff"] = df["time_diff"].clip(lower=0)
    return df


def discover(runs: Path, datasets=None, models=None):
    """Yield (dataset, seed, test_path, {model: path}) for every seed folder under `runs`."""
    for ds_dir in sorted(p for p in runs.iterdir() if p.is_dir()):
        if datasets and ds_dir.name not in datasets:
            continue
        for seed_dir in sorted(p for p in ds_dir.iterdir() if p.is_dir()):
            m = re.fullmatch(r"seed(\d+)", seed_dir.name)
            test_path = seed_dir / "test.csv"
            if not m or not test_path.exists():
                continue
            model_paths = {p.stem: p for p in sorted(seed_dir.glob("*.csv"))
                           if p.name != "test.csv" and (not models or p.stem in models)}
            if model_paths:
                yield ds_dir.name, int(m.group(1)), test_path, model_paths


# ----------------------------------------------------------------------------------------------
# metrics
# ----------------------------------------------------------------------------------------------
def _codes(flow_ids: pd.Series):
    codes, uniques = pd.factorize(flow_ids, sort=False)
    return codes.astype(np.int64), len(uniques)


def log1p_edges(lo: float, hi: float, n_bins: int) -> np.ndarray:
    """n_bins + 1 bin edges, equally spaced in log(1 + x) between lo and hi."""
    if not hi > lo:
        hi = lo + 1e-6
    edges = np.expm1(np.linspace(np.log1p(lo), np.log1p(hi), n_bins + 1))
    edges[0], edges[-1] = lo, hi
    return edges


def avg_flow_pmf(values: np.ndarray, codes: np.ndarray, n_flows: int, edges: np.ndarray) -> np.ndarray:
    """Mean of the per-flow PMFs, (1/N) sum_i counts_i(b) / L_i, without a loop over flows."""
    v = np.clip(values, edges[0], edges[-1])
    b = np.clip(np.searchsorted(edges, v, side="right") - 1, 0, len(edges) - 2)
    flow_len = np.bincount(codes, minlength=n_flows).astype(np.float64)
    weights = 1.0 / flow_len[codes]
    return np.bincount(b, weights=weights, minlength=len(edges) - 1) / n_flows


def d_marg(real: pd.DataFrame, synth: pd.DataFrame, feat: str, n_bins: int) -> float:
    r = real[feat].to_numpy(np.float64)
    s = synth[feat].to_numpy(np.float64)
    edges = log1p_edges(r.min(), r.max(), n_bins)
    rc, rn = _codes(real["flow_id"])
    sc, sn = _codes(synth["flow_id"])
    return float(np.abs(avg_flow_pmf(r, rc, rn, edges) - avg_flow_pmf(s, sc, sn, edges)).sum())


def mean_acf(values: np.ndarray, codes: np.ndarray, max_lag: int) -> np.ndarray:
    """Sample ACF at lags 1..max_lag of every flow, averaged over the flows (packet order kept)."""
    order = np.argsort(codes, kind="stable")
    v, c = values[order], codes[order]
    segments = np.split(v, np.flatnonzero(np.diff(c)) + 1)
    ac_sum = np.zeros(max_lag)
    ac_cnt = np.zeros(max_lag)
    for x in segments:
        n = x.size
        if n < 2:
            continue
        x = x - x.mean()
        denom = float(np.dot(x, x))
        if denom <= 0.0:
            continue
        for lag in range(1, min(max_lag, n - 1) + 1):
            ac_sum[lag - 1] += np.dot(x[lag:], x[:-lag]) / denom
            ac_cnt[lag - 1] += 1
    out = np.zeros(max_lag)
    mask = ac_cnt > 0
    out[mask] = ac_sum[mask] / ac_cnt[mask]
    return out


def _log_feature(df: pd.DataFrame, feat: str) -> np.ndarray:
    x = df[feat].to_numpy(np.float64)
    return np.log1p(np.clip(x, 1e-9, None) if feat == "time_diff" else np.clip(x, 0, None))


def d_temp(real: pd.DataFrame, synth: pd.DataFrame, feat: str, max_lag: int = 20) -> float:
    ac_r = mean_acf(_log_feature(real, feat), _codes(real["flow_id"])[0], max_lag)
    ac_s = mean_acf(_log_feature(synth, feat), _codes(synth["flow_id"])[0], max_lag)
    return float(np.sqrt(np.mean((ac_r - ac_s) ** 2)))


def d_div(real: pd.DataFrame, synth: pd.DataFrame, feat: str) -> float:
    return float(wasserstein_distance(np.log1p(real[feat].to_numpy(np.float64)),
                                      np.log1p(synth[feat].to_numpy(np.float64))))


def real_reference(real: pd.DataFrame) -> dict:
    """Maximum and high quantiles of the real test flows (reference row of Table 5)."""
    out = {}
    for feat in FEATURES:
        x = real[feat].to_numpy(np.float64)
        out[f"{feat}_max"] = float(x.max())
        for q, v in zip(TAIL_QUANTILES, np.percentile(x, TAIL_QUANTILES)):
            out[f"{feat}_q{q}"] = float(v)
    return out


def extras(synth: pd.DataFrame) -> dict:
    """Longest and 99.9th-percentile generated IAT (right columns of Table 5)."""
    s_iat = synth["time_diff"].to_numpy(np.float64)
    return {("max_IAT_s", "time_diff"): float(s_iat.max()),
            ("p999_IAT_s", "time_diff"): float(np.percentile(s_iat, 99.9))}


# ----------------------------------------------------------------------------------------------
# aggregation and LaTeX rows
# ----------------------------------------------------------------------------------------------
def summarize(long: pd.DataFrame) -> pd.DataFrame:
    return (long.groupby(["dataset", "model", "metric", "feature"])["value"]
                .agg(mean="mean", std="std", n_seeds="count").reset_index())


def overall(summary: pd.DataFrame, packets_per_dataset: dict) -> pd.DataFrame:
    """Per metric and feature: mean over the datasets, packet-weighted mean and average rank."""
    main = summary[summary["metric"].isin(MAIN_METRICS)].copy()
    main["rank"] = main.groupby(["dataset", "metric", "feature"])["mean"].rank(method="average")
    main["w"] = main["dataset"].map(packets_per_dataset).astype(float)
    rows = []
    for (model, metric, feat), g in main.groupby(["model", "metric", "feature"]):
        rows.append({"model": model, "metric": metric, "feature": feat,
                     "macro_mean": g["mean"].mean(),
                     "packet_weighted_mean": np.average(g["mean"], weights=g["w"]),
                     "avg_rank": g["rank"].mean(), "n_datasets": g["dataset"].nunique()})
    per_col = pd.DataFrame(rows)
    all_rank = main.groupby("model")["rank"].mean().rename("avg_rank_all_metrics").reset_index()
    return per_col.merge(all_rank, on="model")


def _fmt(mean: float, std: float, digits: int = 3) -> str:
    std_txt = "" if not np.isfinite(std) else rf"{{\scriptstyle\pm{std:.{digits}f}}}"
    return rf"{mean:.{digits}f}{std_txt}"


def latex_main_table(summary: pd.DataFrame, names: dict, model_order: list) -> str:
    """Rows of Table 3: mean and std over the seeds, best value bold, second best underlined."""
    cols = [(m, f) for m in MAIN_METRICS for f in FEATURES]
    main = summary[summary["metric"].isin(MAIN_METRICS)].copy()
    main["rank"] = main.groupby(["dataset", "metric", "feature"])["mean"].rank(method="min")
    lines = []
    datasets = list(dict.fromkeys(main["dataset"]))
    for model in model_order:
        sub = main[main["model"] == model]
        if sub.empty:
            continue
        first = True
        for ds in datasets:
            cells = []
            for metric, feat in cols:
                row = sub[(sub["dataset"] == ds) & (sub["metric"] == metric) & (sub["feature"] == feat)]
                if row.empty:
                    cells.append("--")
                    continue
                r = row.iloc[0]
                txt = _fmt(r["mean"], r["std"])
                if r["rank"] == 1:
                    txt = rf"\mathbf{{{txt}}}"
                elif r["rank"] == 2:
                    txt = rf"\underline{{{txt}}}"
                cells.append(f"${txt}$")
            label = names.get(model, model) if first else ""
            lines.append(f"{label} & {ds} & " + " & ".join(cells) + r" \\")
            first = False
        lines.append(r"\hline")
    return "\n".join(lines)


def latex_overall_table(ov: pd.DataFrame, names: dict, model_order: list) -> str:
    """Overall block of Table 3: mean over the datasets with the average rank in parentheses."""
    cols = [(m, f) for m in MAIN_METRICS for f in FEATURES]
    lines = []
    for model in model_order:
        sub = ov[ov["model"] == model]
        if sub.empty:
            continue
        cells = []
        for metric, feat in cols:
            r = sub[(sub["metric"] == metric) & (sub["feature"] == feat)]
            cells.append("--" if r.empty else f"${r.iloc[0]['macro_mean']:.3f}$ ({r.iloc[0]['avg_rank']:.1f})")
        lines.append(f"{names.get(model, model)} & " + " & ".join(cells)
                     + f" & ${sub['avg_rank_all_metrics'].iloc[0]:.2f}$ \\\\")
    return "\n".join(lines)


def bin_sensitivity(pairs, bins_list, ref_B) -> pd.DataFrame:
    """D_marg for several B, and the Kendall tau of the model ranking against B = ref_B."""
    rows = []
    for ds, seed, real, synths in pairs:
        for B in bins_list:
            for model, synth in synths.items():
                for feat in FEATURES:
                    rows.append({"dataset": ds, "seed": seed, "model": model, "feature": feat, "B": B,
                                 "value": d_marg(real, synth, feat, B)})
    df = pd.DataFrame(rows)
    means = df.groupby(["dataset", "feature", "B", "model"])["value"].mean().reset_index()
    taus = []
    for (ds, feat), g in means.groupby(["dataset", "feature"]):
        ref = g[g["B"] == ref_B].set_index("model")["value"]
        for B, gb in g.groupby("B"):
            cur = gb.set_index("model")["value"].reindex(ref.index)
            tau = kendalltau(ref.values, cur.values).statistic if len(ref) > 1 else np.nan
            taus.append({"dataset": ds, "feature": feat, "B": B, "kendall_tau_vs_ref": tau,
                         "best_model": cur.idxmin(), "best_model_ref": ref.idxmin()})
    return means.merge(pd.DataFrame(taus), on=["dataset", "feature", "B"])


# ----------------------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", type=Path, required=True, help="root folder of the runs, e.g. runs")
    ap.add_argument("--out", type=Path, required=True, help="output folder")
    ap.add_argument("--datasets", nargs="*")
    ap.add_argument("--models", nargs="*")
    ap.add_argument("--B", type=int, default=1000, help="number of bins of D_marg")
    ap.add_argument("--W", type=int, default=20, help="maximum lag of D_temp")
    ap.add_argument("--bins-sweep", nargs="*", type=int, default=[], help="values of B for the bin sensitivity")
    ap.add_argument("--no-extras", action="store_true", help="skip the generated-IAT extremes")
    ap.add_argument("--model-order", nargs="*", help="row order of the LaTeX tables")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    rows, pairs, packets, refs = [], [], {}, []
    for ds, seed, test_path, model_paths in discover(args.runs, args.datasets, args.models):
        real = load_trace(test_path)
        packets.setdefault(ds, []).append(len(real))
        refs.append({"dataset": ds, "seed": seed, "n_test_packets": len(real),
                     "n_test_flows": int(real["flow_id"].nunique()), **real_reference(real)})
        synths = {}
        for model, path in model_paths.items():
            synth = load_trace(path)
            if synth.empty:
                print(f"[warn] empty synthetic file: {path}")
                continue
            synths[model] = synth
            for feat in FEATURES:
                rows.append((ds, seed, model, "D_marg", feat, d_marg(real, synth, feat, args.B)))
                rows.append((ds, seed, model, "D_temp", feat, d_temp(real, synth, feat, args.W)))
                rows.append((ds, seed, model, "D_div", feat, d_div(real, synth, feat)))
            if not args.no_extras:
                for (metric, feat), val in extras(synth).items():
                    rows.append((ds, seed, model, metric, feat, val))
            print(f"done  {ds:>16s}  seed{seed}  {model}")
        if args.bins_sweep:
            pairs.append((ds, seed, real, synths))

    if not rows:
        raise SystemExit(f"no runs found under {args.runs}")

    long = pd.DataFrame(rows, columns=["dataset", "seed", "model", "metric", "feature", "value"])
    long.to_csv(args.out / "metrics_long.csv", index=False)
    summary = summarize(long)
    summary.to_csv(args.out / "summary.csv", index=False)
    pd.DataFrame(refs).to_csv(args.out / "real_reference.csv", index=False)
    ov = overall(summary, {k: float(np.mean(v)) for k, v in packets.items()})
    ov.to_csv(args.out / "overall.csv", index=False)

    order = args.model_order or [m for m in DEFAULT_NAMES if m in set(long["model"])] + \
        sorted(set(long["model"]) - set(DEFAULT_NAMES))
    (args.out / "table_main_rows.tex").write_text(latex_main_table(summary, DEFAULT_NAMES, order))
    (args.out / "table_overall_rows.tex").write_text(latex_overall_table(ov, DEFAULT_NAMES, order))

    if args.bins_sweep:
        ref = args.B if args.B in args.bins_sweep else args.bins_sweep[-1]
        bin_sensitivity(pairs, args.bins_sweep, ref).to_csv(args.out / "bin_sensitivity.csv", index=False)

    seeds_per = long.groupby(["dataset", "model"])["seed"].nunique()
    if seeds_per.nunique() > 1:
        print("[warn] unequal number of seeds per (dataset, model):\n", seeds_per.to_string())
    print(f"\nwrote the results to {args.out.resolve()}")


if __name__ == "__main__":
    main()
