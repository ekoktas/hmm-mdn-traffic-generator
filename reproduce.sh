#!/bin/sh
# All experiments of the article, in order. Run from the repository root in the conda environment
# (see README.md) after the five traces are in data/ (see data/README.md). Every step is resumable:
# a run whose output exists is skipped, so the script can be restarted after an interruption.
set -e
mkdir -p logs

# 1  data: statistics of Table 2
python scripts/check_data.py

# 2  selection of M and H on validation flows (15 % of the training flows, seeds 0-2)
python scripts/run_matrix.py --plan sweep_mh                 # Facebook video: M in {4,...,64}, H in {32,...,256}
python scripts/run_matrix.py --plan sweep_m_all              # other datasets: M = 16 and M = 32
python scripts/select_m.py                                   # the rule; N_MIX_MDN in proposed_batch.ipynb is 16
python scripts/evaluate_all.py --runs sweeps_val --out eval/mh

# 3  main runs, seeds 0-4. The proposed model and the prior model write the split of every dataset and seed,
#    so they run first; the five baselines reuse it.
python scripts/run_matrix.py --plan main --only proposed m20
python scripts/run_matrix.py --plan main --only timegan aewgan necstgen trafficgpt dualhmm

# 4  ablation (Facebook video), tail threshold (Facebook video), flow lengths drawn from the training flows
python scripts/run_matrix.py --plan ablation
python scripts/run_matrix.py --plan sweeps
python scripts/run_matrix.py --plan genonly

# 5  evaluation
MODELS="proposed m20 timegan aewgan necstgen trafficgpt dualhmm"
python scripts/evaluate_all.py --runs runs --out eval/main --models $MODELS
python scripts/evaluate_all.py --runs runs --out eval/bins --models $MODELS --no-extras --bins-sweep 100 250 500 1000 2000
python scripts/evaluate_all.py --runs runs --out eval/ablation --datasets facebook_video --models proposed abl_A_noidle abl_B_noflowlen abl_C_gaussian m20
python scripts/evaluate_all.py --runs runs --out eval/tau --datasets facebook_video --models proposed sw_q0p99 sw_q0p995 sw_q0p999 sw_q0p9995
python scripts/evaluate_all.py --runs runs --out eval/flowlen --models proposed fl_empirical
python scripts/long_gaps.py --datasets ftps facebook_video 5g_netflix --models proposed m20
python scripts/export_cdfs.py --runs runs --out figs_data

# 6  generation time and footprint (Table 4) and scalability, one CPU thread, Facebook video model of seed 0
sh scripts/bench/run_all.sh
python scripts/scalability_bench.py --artifacts runs/facebook_video/seed0/artifacts/proposed \
    --flows 1 5 10 50 100 500 1000 --packets-per-flow 200 --reps 5 --out bench_fbvideo

# 7  interpretability of the Facebook video model, all five splits
for s in 0 1 2 3 4; do
    python -m papermill notebooks/analysis_interpretability.ipynb logs/analysis_facebook_video_seed$s.ipynb \
        -k hmm-mdn -p SEED $s -p OUT_DIR runs/facebook_video/seed$s --cwd .
done
python scripts/summarize_interpretability.py > interpretability_summary.txt

# 8  tables and the summary of all numbers
python scripts/make_paper_tables.py
