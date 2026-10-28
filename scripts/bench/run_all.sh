#!/bin/sh
# Table 4: time every model trained on Facebook video, split 0, on one CPU thread.
# Run from the repository root after the main plan has finished for facebook_video, seed 0.
# Every script appends one line to timing_results.jsonl; the table is printed at the end.
set -e
A=runs/facebook_video/seed0/artifacts
python scripts/bench/bench_proposed.py   --artifacts $A/proposed
python scripts/bench/bench_proposed.py   --artifacts $A/proposed --no-reject
python scripts/bench/bench_timegan.py    --artifacts $A/timegan
python scripts/bench/bench_aewgan.py     --artifacts $A/aewgan
python scripts/bench/bench_dualhmm.py    --artifacts $A/dualhmm
python scripts/bench/bench_necstgen.py   --artifacts $A/necstgen
python scripts/bench/bench_trafficgpt.py --artifacts $A/trafficgpt
python -c "import pandas as pd; d = pd.read_json('timing_results.jsonl', lines=True); print(d[['model', 'mean_s', 'std_s', 'valid_packets', 'pkt_per_s', 'nn_params', 'footprint_MB']].to_string(index=False)); print('CPU:', d.cpu.iloc[-1], '| torch', d.torch.iloc[-1])"
