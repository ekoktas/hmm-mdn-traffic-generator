#!/usr/bin/env python3
"""
Monte Carlo runs of the system-level study: for every run, new traces (align_and_generate_multi_ue.py),
then the three scenarios in OMNeT++ (Poisson baseline, real traffic, synthetic traffic) with the run
index as seed set, then the export of the recorded vectors with opp_scavetool and the analysis.

Run inside the OMNeT++ project folder that contains omnetpp.ini, the NED files and the compiled
simulation:

    python run_monte_carlo.py --runs 30 --raw-path ../runs/ftps/seed0/test.csv \
        --artifacts ../runs/ftps/seed0/artifacts/proposed --inet-src /workspace/inet/src
"""
import argparse
import glob
import os
import re
import subprocess

SCENARIOS = {"Basic": "Basic_Traffic_Multi_Wifi", "Real": "Real_Traffic_Multi_Wifi", "ML": "ML_Traffic_Multi_Wifi"}
EXTRACTIONS = {
    "throughput_data":    'module =~ "*.server.app[0].io" AND name =~ "packetReceived:vector(packetBytes)"',
    "queue_lengths":      'module =~ "*.ue[*].**.pendingQueue" AND name =~ "queueLength:vector"',
    "queue_bits":         'module =~ "*.ue[*].**.pendingQueue" AND name =~ "queueBitLength:vector"',
    "latency_data":       'module =~ "*.server.app[0].sink" AND name =~ "meanBitLifeTimePerPacket:vector"',
    "contention_windows": 'module =~ "*.ue[*].**" AND name =~ "contentionWindowChanged:vector"',
    "packets_sent":       'module =~ "*.ue[*].app[0]" AND name =~ "packetSent:vector*"',
}


def run_cmd(cmd, env):
    print("running:", " ".join(cmd))
    subprocess.run(cmd, check=True, env=env)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", type=int, default=30)
    ap.add_argument("--raw-path", required=True)
    ap.add_argument("--artifacts", required=True)
    ap.add_argument("--n-ues", type=int, default=5)
    ap.add_argument("--omnet-exec", default="./MlTrafficSimDataRateAligned", help="compiled simulation")
    ap.add_argument("--inet-src", default=os.environ.get("INET_SRC", "/workspace/inet/src"), help="NED path of INET")
    ap.add_argument("--inet-lib", default=None, help="folder of libINET.so, added to LD_LIBRARY_PATH")
    args = ap.parse_args()

    env = os.environ.copy()
    if args.inet_lib:
        env["LD_LIBRARY_PATH"] = f"{args.inet_lib}:" + env.get("LD_LIBRARY_PATH", "")

    for folder in ("results", "omnet_results_csv"):        # results of earlier runs must not be mixed in
        os.makedirs(folder, exist_ok=True)
        for ext in ("*.vec", "*.vci", "*.sca", "*.csv"):
            for path in glob.glob(f"{folder}/{ext}"):
                os.remove(path)

    here = os.path.dirname(os.path.abspath(__file__))
    for i in range(args.runs):
        print(f"\n===== Monte Carlo run {i + 1} of {args.runs} =====")
        run_cmd(["python3", os.path.join(here, "align_and_generate_multi_ue.py"), "--raw-path", args.raw_path,
                 "--artifacts", args.artifacts, "--n-ues", str(args.n_ues), "--run-index", str(i)], env)
        for config in SCENARIOS.values():
            run_cmd([args.omnet_exec, "-u", "Cmdenv", "-n", f".:{args.inet_src}", "-c", config, "--seed-set", str(i),
                     f"--output-vector-file=results/{config}-{i}.vec", f"--output-scalar-file=results/{config}-{i}.sca",
                     "--cmdenv-status-frequency=15s"], env)

    for scenario, config in SCENARIOS.items():
        # sorted by run number, so that the vectors of every export are in run order
        vec_files = sorted(glob.glob(f"results/{config}-*.vec"), key=lambda p: int(re.search(r"-(\d+)\.vec$", p).group(1)))
        if not vec_files:
            print(f"no vector files for {scenario}")
            continue
        for prefix, query in EXTRACTIONS.items():
            run_cmd(["opp_scavetool", "x"] + vec_files + ["-F", "CSV-S", "-o",
                     f"omnet_results_csv/{prefix}_{scenario.lower()}.csv", "-f", query], env)

    run_cmd(["python3", os.path.join(here, "omnet_runs_summary.py"), "--n-ues", str(args.n_ues)], env)
    run_cmd(["python3", os.path.join(here, "omnet_analysis.py")], env)


if __name__ == "__main__":
    main()
