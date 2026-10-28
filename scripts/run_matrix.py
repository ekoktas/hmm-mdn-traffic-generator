#!/usr/bin/env python3
"""
Run the experiments of the article. Every job executes one notebook with papermill for one
(dataset, seed, model) and writes the synthetic flows to <root>/<dataset>/seed<k>/<model>.csv.

Plans
  main         7 models x 5 datasets x 5 seeds. Run the proposed model and the prior model first: their runs
               write the split.json and test.csv that all other models of a dataset and seed reuse.
  ablation     variants A, B and C of the proposed model on Facebook video, 5 seeds
  sweeps       tail-threshold quantile on Facebook video, seeds 0-2
  sweep_mh     M and H on Facebook video with validation flows, seeds 0-2 (root sweeps_val/)
  sweep_m_all  M = 16 and M = 32 on the other four datasets with validation flows, seeds 0-2
  genonly      generation with flow lengths drawn from the training flows, from the trained models

A job is skipped when its output CSV exists, so an interrupted plan resumes where it stopped.
Run from the repository root:

    python scripts/run_matrix.py --plan main --dry-run       # list the jobs
    python scripts/run_matrix.py --plan main --only proposed m20 --datasets google_home_udp --seeds 0
    python scripts/run_matrix.py --plan main                 # every job that is not done yet
    python scripts/run_matrix.py --plan main --status        # done / pending / failed
    python scripts/run_matrix.py --plan main --errors        # the exception of every failed job
    python scripts/run_matrix.py --plan main --verify        # damaged split, test or output files
"""
from __future__ import annotations

import argparse
import json
import re
import time
import traceback
from itertools import product
from pathlib import Path

# ----------------------------------------------------------------------------------------------
# configuration (paths relative to the repository root)
# ----------------------------------------------------------------------------------------------
KERNEL = "hmm-mdn"                  # Jupyter kernel of the conda environment, see README.md
TEST_SIZE = 0.3
SEEDS_MAIN = [0, 1, 2, 3, 4]
SEEDS_SWEEP = [0, 1, 2]
RUNS_ROOT = Path("runs")
SWEEPS_ROOT = Path("sweeps_val")    # runs with validation flows (selection of M and H)

DATASETS = {                        # key -> packet CSV with the columns flow_id, payload_length, time_diff
    "google_home_udp": "data/google_home_udp.csv",
    "facebook_audio":  "data/facebook_audio.csv",
    "facebook_video":  "data/facebook_video.csv",
    "ftps":            "data/ftps.csv",
    "5g_netflix":      "data/5g_netflix.csv",
}

DATASET_PARAMS = {                  # core HMM states; the idle state is added on top
    "google_home_udp": {"K_CORE": 2},
    "facebook_audio":  {"K_CORE": 1},
    "facebook_video":  {"K_CORE": 4},
    "ftps":            {"K_CORE": 3},
    "5g_netflix":      {"K_CORE": 4},
}

NB_PROPOSED = "notebooks/proposed_batch.ipynb"      # training and generation
NB_GENERATE = "notebooks/proposed_generate.ipynb"   # generation from saved artifacts

MODELS = {                          # key -> (notebook, fixed parameters)
    "proposed":   (NB_PROPOSED, {}),
    # prior model of the authors (PIMRC 2025): Gaussian components, no idle state, no flow-length input, M = 32
    "m20":        (NB_PROPOSED, {"ABLATION_NO_IDLE": True, "ABLATION_NO_FLOW_LEN": True,
                                 "DIST_FAMILY": "gaussian", "N_MIX_MDN": 32}),
    "timegan":    ("notebooks/timegan_batch.ipynb", {}),
    "aewgan":     ("notebooks/aewgan_batch.ipynb", {}),
    "necstgen":   ("notebooks/necstgen_batch.ipynb", {}),
    "trafficgpt": ("notebooks/trafficgpt_batch.ipynb", {}),
    "dualhmm":    ("notebooks/dualhmm_batch.ipynb", {}),
}

ABLATION = {                        # Facebook video
    "abl_A_noidle":    {"ABLATION_NO_IDLE": True},
    "abl_B_noflowlen": {"ABLATION_NO_FLOW_LEN": True},
    "abl_C_gaussian":  {"DIST_FAMILY": "gaussian"},
}

SWEEP_MH = {                        # 15 % of the training flows are held out for the validation NLL
    **{f"sw_M{m}": {"N_MIX_MDN": m, "HIDDEN_MDN": 128, "VAL_FRACTION": 0.15} for m in (4, 8, 16, 32, 64)},
    **{f"sw_H{h}": {"N_MIX_MDN": 32, "HIDDEN_MDN": h, "VAL_FRACTION": 0.15} for h in (32, 64, 256)},
}                                   # H = 128 at M = 32 is sw_M32

SWEEPS = {                          # quantile of log(1 + IAT) that defines the tail threshold
    f"sw_q{str(q).replace('.', 'p')}": {"TAIL_Q": q} for q in (0.99, 0.995, 0.999, 0.9995)
}

GENONLY = {                         # reuse the artifacts of the "proposed" runs
    "fl_empirical": {"FLOW_LEN_MODE": "empirical"},
}
PLANS = ["main", "ablation", "sweeps", "sweep_mh", "sweep_m_all", "genonly"]
# ----------------------------------------------------------------------------------------------


def jobs_for(plan: str):
    """Yield (dataset, seed, model key, notebook, parameters) for every job of a plan."""
    fbv = {**DATASET_PARAMS["facebook_video"], "RAW_PATH_STR": DATASETS["facebook_video"]}
    if plan == "main":
        for (ds, path), (mk, (nb, extra)), seed in product(DATASETS.items(), MODELS.items(), SEEDS_MAIN):
            ds_params = DATASET_PARAMS.get(ds, {}) if nb == NB_PROPOSED else {}
            yield ds, seed, mk, nb, {**ds_params, **extra, "RAW_PATH_STR": path}
    elif plan == "ablation":
        for (mk, extra), seed in product(ABLATION.items(), SEEDS_MAIN):
            yield "facebook_video", seed, mk, NB_PROPOSED, {**fbv, **extra}
    elif plan == "sweeps":
        for (mk, extra), seed in product(SWEEPS.items(), SEEDS_SWEEP):
            yield "facebook_video", seed, mk, NB_PROPOSED, {**fbv, **extra}
    elif plan == "sweep_mh":
        for (mk, extra), seed in product(SWEEP_MH.items(), SEEDS_SWEEP):
            yield "facebook_video", seed, mk, NB_PROPOSED, {**fbv, **extra, "_root": str(SWEEPS_ROOT)}
    elif plan == "sweep_m_all":
        for (ds, path), mk, seed in product(DATASETS.items(), ("sw_M16", "sw_M32"), SEEDS_SWEEP):
            if ds != "facebook_video":        # Facebook video: these runs belong to sweep_mh
                yield ds, seed, mk, NB_PROPOSED, {**DATASET_PARAMS.get(ds, {}), **SWEEP_MH[mk],
                                                  "RAW_PATH_STR": path, "_root": str(SWEEPS_ROOT)}
    elif plan == "genonly":
        for (mk, extra), seed, (ds, path) in product(GENONLY.items(), SEEDS_MAIN, DATASETS.items()):
            yield ds, seed, mk, NB_GENERATE, {**extra, "RAW_PATH_STR": path, "SOURCE_MODEL_KEY": "proposed"}
    else:
        raise SystemExit(f"unknown plan {plan}")


def root_of(extra: dict) -> Path:
    return Path(extra.get("_root") or RUNS_ROOT)


def read_log(log_path: Path) -> dict:
    """Last log record per (dataset, seed, model)."""
    last = {}
    if log_path.exists():
        for line in log_path.read_text().splitlines():
            r = json.loads(line)
            last[(r["dataset"], r["seed"], r["model"])] = r
    return last


def show_errors(jobs, log_path):
    shown = 0
    for (ds, seed, mk), r in sorted(read_log(log_path).items()):
        job = next((j for j in jobs if j[:3] == (ds, seed, mk)), None)
        if job is None or r["status"] == "ok" or (root_of(job[4]) / ds / f"seed{seed}" / f"{mk}.csv").exists():
            continue
        text = re.sub(r"\x1b\[[0-9;]*m", "", r.get("error") or "")
        lines = [ln for ln in text.splitlines() if ln.strip()]
        where = [ln for ln in lines if "Exception encountered at" in ln]
        if not where and ("didn't respond" in text or "died before replying" in text):
            where = ["the kernel did not start, no cell ran"]
        print("=" * 78)
        print(f"{ds} seed{seed} {mk}  [{r['status']}]  started {r.get('started', '?')}, {r.get('seconds', '?')} s  "
              f"{where[-1].strip() if where else ''}")
        print("\n".join(lines[-6:]))
        shown += 1
    print(f"\n{shown} failed job(s) without output" if shown else "no failed jobs without output")


def verify(jobs):
    import pandas as pd
    problems = 0
    for root in sorted({root_of(j[4]) for j in jobs}):
        for d in sorted(root.glob("*/seed*")):
            sp, tp = d / "split.json", d / "test.csv"
            if not sp.exists():
                continue
            try:
                split = json.loads(sp.read_text())
            except Exception as e:
                print(f"BAD split.json  {d}: {e}")
                problems += 1
                continue
            if tp.exists():
                try:
                    t = pd.read_csv(tp, usecols=["flow_id"])
                    if set(t["flow_id"].unique()) != set(split["test"]):
                        print(f"MISMATCH        {tp}: the flows differ from split.json")
                        problems += 1
                except Exception as e:
                    print(f"BAD test.csv    {tp}: {e}")
                    problems += 1
            for f in sorted(d.glob("*.csv")):
                if f.name == "test.csv":
                    continue
                try:
                    x = pd.read_csv(f, usecols=["flow_id", "payload_length", "time_diff"])
                    if x.empty or x.isna().any().any():
                        print(f"EMPTY/NaN       {f}")
                        problems += 1
                except Exception as e:
                    print(f"BAD output      {f}: {e}")
                    problems += 1
    print("verify: OK" if not problems else f"verify: {problems} problem(s); delete the listed files and rerun")


def status(jobs, log_path):
    last = read_log(log_path)
    done = pending = failed = 0
    for ds, seed, mk, nb, extra in jobs:
        seed_dir = root_of(extra) / ds / f"seed{seed}"
        if (seed_dir / f"{mk}.csv").exists():
            done += 1
        elif last.get((ds, seed, mk), {}).get("status") in ("failed", "no_output"):
            failed += 1
            print(f"FAILED  {ds} seed{seed} {mk}: see {seed_dir / 'executed' / (mk + '.ipynb')}")
        else:
            pending += 1
    print(f"done {done} | pending {pending} | failed {failed}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--plan", required=True, choices=PLANS)
    ap.add_argument("--only", nargs="*", help="restrict to these model keys")
    ap.add_argument("--datasets", nargs="*", help="restrict to these datasets")
    ap.add_argument("--seeds", nargs="*", type=int, help="restrict to these seeds")
    ap.add_argument("--dry-run", action="store_true", help="list the jobs without running them")
    ap.add_argument("--kernel", default=KERNEL, help="Jupyter kernel name")
    ap.add_argument("--start-timeout", type=int, default=600, help="seconds a new kernel may take to start")
    ap.add_argument("--status", action="store_true", help="show done / pending / failed jobs and exit")
    ap.add_argument("--errors", action="store_true", help="print the exception of every failed job and exit")
    ap.add_argument("--verify", action="store_true", help="check split.json, test.csv and outputs and exit")
    args = ap.parse_args()

    jobs = [j for j in jobs_for(args.plan)
            if (not args.only or j[2] in args.only) and (not args.datasets or j[0] in args.datasets)
            and (not args.seeds or j[1] in args.seeds)]
    for m in sorted({j[4]["RAW_PATH_STR"] for j in jobs if not Path(j[4]["RAW_PATH_STR"]).exists()}):
        print(f"[skip] dataset file not found: {m}")
    for m in sorted({j[3] for j in jobs if not Path(j[3]).exists()}):
        print(f"[skip] notebook not found: {m}")
    jobs = [j for j in jobs if Path(j[4]["RAW_PATH_STR"]).exists() and Path(j[3]).exists()]
    print(f"{len(jobs)} runnable jobs in plan '{args.plan}'")
    log_path = RUNS_ROOT / f"run_log_{args.plan}.jsonl"

    if args.errors:
        return show_errors(jobs, log_path)
    if args.verify:
        return verify(jobs)
    if args.status:
        return status(jobs, log_path)

    for ds, seed, mk, nb, extra in jobs:
        root = root_of(extra)
        extra = {k: v for k, v in extra.items() if k != "_root"}
        out_dir = root / ds / f"seed{seed}"
        target = out_dir / f"{mk}.csv"
        params = {"SEED": seed, "DATASET": ds, "OUT_DIR": str(out_dir), "MODEL_KEY": mk,
                  "TEST_SIZE": TEST_SIZE, **extra}
        if target.exists():
            print(f"skip  {target}")
            continue
        print(f"run   {ds:>16s} seed{seed} {mk:<16s} {nb}  {extra}")
        if args.dry_run:
            continue
        import papermill as pm          # imported here so that --dry-run works without papermill
        (out_dir / "executed").mkdir(parents=True, exist_ok=True)
        t0, state, err = time.time(), "ok", None
        started = time.strftime("%Y-%m-%d %H:%M:%S")
        try:
            pm.execute_notebook(nb, str(out_dir / "executed" / f"{mk}.ipynb"), parameters=params,
                                kernel_name=args.kernel, progress_bar=False, cwd=str(Path.cwd()),
                                start_timeout=args.start_timeout)
            if not target.exists():
                state, err = "no_output", f"{target} was not written"
        except Exception:               # keep going; the executed notebook shows the failing cell
            state, err = "failed", traceback.format_exc()
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("a") as fh:
            fh.write(json.dumps({"dataset": ds, "seed": seed, "model": mk, "status": state, "started": started,
                                 "seconds": round(time.time() - t0, 1), "error": err}) + "\n")
        print(f"      -> {state} in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
