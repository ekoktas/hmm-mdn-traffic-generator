# Packet-level traffic generator with heavy-tailed payload and inter-arrival distributions

Code of the article

> E. Koktas and P. Rost, "Packet-Level Traffic Modeling with Heavy-Tailed Payload and Inter-Arrival Distributions for Digital Twins," arXiv:2512.15432, 2026. https://arxiv.org/abs/2512.15432

The code corresponds to the revised version of the article, which is under review. Its results differ from those of the earlier version on arXiv; the table and figure numbers in this README refer to the revised version.

The generator combines a small hidden Markov model (HMM), whose states describe phases such as bursts, steady streaming and idle periods, with a mixture density network (MDN) that models the joint distribution of payload length and inter-arrival time (IAT) within each state. The mixture components are Student-t distributions, the MDN is conditioned on the length of the flow, and an extra idle state is initialized in the tail of the IAT distribution. The repository contains the training and generation code, the five baselines of the comparison, the evaluation metrics, the scripts that produce the tables and figure data of the article, and the system-level study in OMNeT++.

## Contents

| Path | Content |
|---|---|
| `notebooks/proposed_batch.ipynb` | proposed model: training and generation for one dataset and split |
| `notebooks/proposed_generate.ipynb` | generation from a trained model with flow lengths drawn from the training flows |
| `notebooks/analysis_interpretability.ipynb` | state fingerprint and flow-length response of a trained model |
| `notebooks/{timegan,aewgan,necstgen,trafficgpt,dualhmm}_batch.ipynb` | the five baselines |
| `scripts/hmm_mdn.py` | MDN, Student-t loss, mixture sampler and flow generation |
| `scripts/safe_hmm.py` | hmmlearn's GMMHMM with guards against degenerate EM updates |
| `scripts/data_utils.py`, `scripts/split_rule.py` | cleaning of the traces and the train/test split shared by all models |
| `scripts/run_matrix.py` | runs every experiment (the notebooks, through papermill); resumable |
| `scripts/evaluate_all.py` | the metrics D_marg, D_temp and D_div, aggregation over the splits, ranks |
| `scripts/make_paper_tables.py` | LaTeX rows of the tables and a summary of the numbers in the text |
| `scripts/bench/`, `scripts/timing.py`, `scripts/scalability_bench.py` | generation time, memory footprint, scalability |
| `scripts/check_data.py`, `scripts/tshark_to_flows.py`, `scripts/convert_5g_csv.py` | dataset statistics and conversion of the raw traces |
| `omnet/` | system-level study with OMNeT++/INET (see `omnet/README.md`) |
| `reproduce.sh` | all steps in order |
| `data/README.md` | how to obtain and convert the five traces |

## Installation

```bash
conda env create -f environment.yml          # Python 3.11 and the pinned packages of requirements.txt
conda activate hmm-mdn
python -m ipykernel install --user --name hmm-mdn
```

`scripts/run_matrix.py` executes the notebooks with the Jupyter kernel `hmm-mdn` (`KERNEL` in the script or `--kernel`). The PyTorch wheel from PyPI includes CUDA; for a CPU-only machine install it with `pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu` first. The results of the article were obtained with Python 3.11, PyTorch 2.5.1 and hmmlearn 0.3.3, the models were trained on a GPU, and the timing of Table 4 used one thread of an AMD EPYC 7742.

## Data

The five traces are public but not redistributed here. `data/README.md` describes where to download them and how to convert them into the CSV format of the pipeline (one row per packet with `flow_id`, `payload_length` and `time_diff`).

## Quick start

On the smallest trace (Google Home, 10,000 packets) one split of the proposed model takes a few minutes on a CPU:

```bash
python scripts/check_data.py
python scripts/run_matrix.py --plan main --only proposed --datasets google_home_udp --seeds 0
python scripts/evaluate_all.py --runs runs --out eval/quick --models proposed
```

The synthetic flows are in `runs/google_home_udp/seed0/proposed.csv`, the real test flows in `runs/google_home_udp/seed0/test.csv`, and the trained model in `runs/google_home_udp/seed0/artifacts/proposed/`.

## Reproducing the article

`sh reproduce.sh` runs all steps in order. Every step can be restarted: a run whose output exists is skipped. The FTPS trace (5.7 million packets) takes by far the longest.

| Result in the article | Produced by | Output |
|---|---|---|
| Table 2, dataset statistics | `scripts/check_data.py` | `data_check.csv` |
| Table 3, comparison of all models | plan `main`, `evaluate_all.py --out eval/main` | `paper_tables.tex`, `results_summary.md` |
| Table 4, footprint and generation time | `scripts/bench/run_all.sh` | `timing_results.jsonl` |
| Table 5, ablation study | plan `ablation`, `evaluate_all.py --out eval/ablation` | `paper_tables.tex` |
| Table 6, HMM states | `analysis_interpretability.ipynb`, split 0 | `runs/facebook_video/seed0/artifacts/proposed/state_fingerprint.csv` |
| Figs. 2 to 6, per-flow CDFs | `scripts/export_cdfs.py` | `figs_data/<dataset>_<feature>.csv` |
| Figs. 8 and 9, system-level study | `omnet/` | `omnet/tikz_data/` |
| Selection of M and H | plans `sweep_mh` and `sweep_m_all`, `scripts/select_m.py` | printed |
| Flow lengths drawn from the training flows | plan `genonly`, `evaluate_all.py --out eval/flowlen` | `results_summary.md` |
| Tail threshold | plan `sweeps`, `evaluate_all.py --out eval/tau` | `results_summary.md` |
| Number of bins of D_marg | `evaluate_all.py --out eval/bins --bins-sweep ...` | `results_summary.md` |
| Long gaps in short and long flows | `scripts/long_gaps.py` | `long_gaps.csv` |

## Notes on reproducibility

- All results are means and standard deviations over five repetitions (seeds 0 to 4). The seed fixes the train/test split, the initialization and the sampling.
- The split of a dataset and seed is drawn by the first run of the proposed model and written to `runs/<dataset>/seed<k>/split.json`; all other models read it and check that their test flows match `test.csv`.
- Training on a GPU is not reproducible bit for bit across hardware and library versions. Single runs can differ from those of the article; the means over the splits are the reference.
- The number of core HMM states per dataset is set in `DATASET_PARAMS` of `scripts/run_matrix.py`.
- The runs of the article fitted the HMM with three identical Gaussian components per state, which EM keeps identical; the code here uses one component (`N_MIX_HMM` in `notebooks/proposed_batch.ipynb`). The generated traffic is the same, but the saved HMM is about 1 kB smaller, so the footprint in Table 4 comes out about 0.001 MB lower.

## Citation

```bibtex
@misc{koktas2026packetleveltrafficmodelingheavytailed,
  title         = {Packet-Level Traffic Modeling with Heavy-Tailed Payload and Inter-Arrival Distributions for Digital Twins},
  author        = {Enes Koktas and Peter Rost},
  year          = {2026},
  eprint        = {2512.15432},
  archivePrefix = {arXiv},
  primaryClass  = {cs.NI},
  url           = {https://arxiv.org/abs/2512.15432},
}
```

## License

The code is released under the MIT license (`LICENSE`). The datasets keep the terms of their providers.

## Acknowledgment

This research was supported by the German Federal Ministry of Research, Technology and Space (BMFTR) under grant number 16KIS2259 (SUSTAINET-inNOvAte).
