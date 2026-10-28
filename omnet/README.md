# System-level study (OMNeT++/INET)

Five UEs share an IEEE 802.11ac channel with one access point and send UDP traffic to a server. Three scenarios offer the same volume per UE: the real test flows of the FTPS trace (split 0), synthetic flows of the proposed model trained on the same split, and a Poisson baseline with the mean packet size and the mean IAT of each UE's real trace. Each scenario is simulated in 30 Monte Carlo runs; in every run the order of the real flows of each UE is shuffled and new synthetic flows are generated.

| File | Content |
|---|---|
| `MultiUeWifiScenario.ned` | the network: five UEs, an access point and a server |
| `MlTraceUdpApp.ned`, `MlTraceUdpApp.cc` | INET application that replays a packet trace (CSV: `timestamp_sec`, `size_bytes`, `flow_id`) over UDP |
| `omnetpp.ini` | network parameters and the three scenarios |
| `align_and_generate_multi_ue.py` | real and synthetic traces of one run, Poisson parameters written into `omnetpp.ini` |
| `run_monte_carlo.py` | all runs, export of the recorded vectors with `opp_scavetool`, analysis |
| `omnet_runs_summary.py` | per-run delivered volume, active time and latency, mean and standard deviation over the runs |
| `omnet_analysis.py` | data of Fig. 8 (throughput of the median-volume run) and Fig. 9 (pooled CDFs of the MAC queue size and the latency) |

## Building

The study was run with OMNeT++ 6.0.3 and INET 4.4.1, in a Docker container with INET in `/workspace/inet`. Copy this folder into an OMNeT++ workspace and build it against INET, for example:

```bash
opp_makemake -f --deep -o MlTrafficSimDataRateAligned -KINET_PROJ=/workspace/inet -DINET_IMPORT \
    -I'$(INET_PROJ)/src' -L'$(INET_PROJ)/src' -lINET'$(D)'
make MODE=release
```

## Running

The model of the FTPS trace, split 0, must exist in the repository (`python scripts/run_matrix.py --plan main --only proposed --datasets ftps --seeds 0`). Then, inside this folder:

```bash
python run_monte_carlo.py --runs 30 --raw-path <repo>/runs/ftps/seed0/test.csv \
    --artifacts <repo>/runs/ftps/seed0/artifacts/proposed --inet-src /workspace/inet/src
```

The run summary is written to `omnet_runs.csv` and the figure data to `tikz_data/`.
