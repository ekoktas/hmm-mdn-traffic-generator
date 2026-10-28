#!/usr/bin/env python3
"""
Packet traces of one Monte Carlo run of the system-level study (Sec. V-B of the article).

1. Real traffic: the real test flows are split into N_UE equal groups of consecutive flows, one
   group per UE. The flow order inside a group is shuffled with the seed of the run, and the
   flows are concatenated into one trace per UE that starts at 2 s.
2. Synthetic traffic: for every UE, flows are generated with a trained model until the UE has
   offered the same number of bytes as its real trace. The length of each synthetic flow is drawn
   uniformly with replacement from the lengths of the real flows of the same UE.
3. Poisson baseline: the mean packet size and the mean IAT of every real UE trace, and the
   simulation time limit, are written into omnetpp.ini.

Output: real_traces/real_trace_ue<k>.csv and ml_traces/ml_trace_ue<k>.csv with the columns
timestamp_sec, size_bytes, flow_id, read by MlTraceUdpApp.

    python align_and_generate_multi_ue.py --raw-path ../runs/ftps/seed0/test.csv \
        --artifacts ../runs/ftps/seed0/artifacts/proposed --n-ues 5 --run-index 0
"""
import argparse
import math
import os
import re
import sys
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from hmm_mdn import GlobalStateLenMDN  # noqa: E402

warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")


def update_omnetpp_ini(ini_path, ue_targets, sim_time_limit):
    """Write the Poisson baseline of every UE and the simulation time limit into omnetpp.ini."""
    if not os.path.exists(ini_path):
        print(f"{ini_path} not found: copy the values above into it by hand")
        return
    with open(ini_path) as fh:
        text = fh.read()

    def replace(pattern, value):
        nonlocal text
        new, n = re.subn(pattern, lambda m: m.group(1) + value, text, flags=re.M)
        if n == 1:
            text = new
        else:
            print(f"warning: {n} lines of {ini_path} match {pattern!r}; left unchanged")

    replace(r"^(sim-time-limit\s*=\s*)\S+", f"{sim_time_limit:.1f}s")
    for ue_idx, t in ue_targets.items():
        key = re.escape(f"*.ue[{ue_idx}].app[0].")
        replace(rf"^({key}messageLength\s*=\s*)\S+", f"{int(t['mean_pkt_size'])}B")
        replace(rf"^({key}sendInterval\s*=\s*)\S+", f"exponential({t['mean_iat']:.5f}s)")
        replace(rf"^({key}stopTime\s*=\s*)\S+", f"{2.0 + t['t_sim']:.1f}s")
    with open(ini_path, "w") as fh:
        fh.write(text)
    print(f"wrote the Poisson baseline and sim-time-limit into {ini_path}")


def run_alignment_pipeline(raw_trace_path, artifacts, num_ues, run_index, ini_path="omnetpp.ini"):
    os.makedirs("real_traces", exist_ok=True)
    os.makedirs("ml_traces", exist_ok=True)

    # 1  real traces, one per UE
    df = pd.read_csv(raw_trace_path)
    df = df.rename(columns={"payload_length": "size_bytes", "payload_len": "size_bytes"})
    df["size_bytes"] = df["size_bytes"].clip(upper=1500)
    df["flow_id"] = pd.factorize(df["flow_id"])[0] + 1
    real_flows = [group for _, group in df.groupby("flow_id")]
    total_flows = len(real_flows)
    flows_per_ue = math.ceil(total_flows / num_ues)

    ue_targets, max_t_sim = {}, 0.0
    for ue_idx in range(num_ues):
        start_idx, end_idx = ue_idx * flows_per_ue, min((ue_idx + 1) * flows_per_ue, total_flows)
        if start_idx >= total_flows:
            print(f"not enough flows for UE {ue_idx}: skipped")
            continue
        ue_specific_flows = real_flows[start_idx:end_idx]          # the same flows in every run
        np.random.seed((run_index * 1000) + ue_idx)               # their order changes from run to run
        shuffled_idx = np.random.permutation(len(ue_specific_flows))
        ue_df = pd.concat([ue_specific_flows[i] for i in shuffled_idx]).reset_index(drop=True)
        ue_df.loc[0, "time_diff"] = 0.0                           # the UE starts sending at 2 s
        ue_df["timestamp_sec"] = ue_df["time_diff"].cumsum() + 2.0
        ue_df["flow_id"] = pd.factorize(ue_df["flow_id"])[0] + 1
        filename = f"real_traces/real_trace_ue{ue_idx}.csv"
        ue_df[["timestamp_sec", "size_bytes", "flow_id"]].to_csv(filename, index=False)

        t_sim = ue_df["timestamp_sec"].iloc[-1] - 2.0
        target_bytes = ue_df["size_bytes"].sum()
        ue_targets[ue_idx] = {
            "t_sim": t_sim,
            "target_bytes": target_bytes,
            "mean_pkt_size": ue_df["size_bytes"].mean(),
            "mean_iat": t_sim / len(ue_df) if len(ue_df) > 0 else 0,
            "flow_lengths": ue_df.groupby("flow_id").size().values,
        }
        max_t_sim = max(max_t_sim, t_sim)
        print(f"real trace UE {ue_idx} -> {filename} | {t_sim:.1f} s | {target_bytes / 1e6:.2f} MB")

    # 2  synthetic traces with the trained model, the same byte budget per UE
    device = torch.device("cpu")
    hmm_model = joblib.load(Path(artifacts) / "hmm.pkl")
    transmat = hmm_model.transmat_
    num_states = transmat.shape[0]
    state_dict = torch.load(Path(artifacts) / "mdn.pth", map_location=device, weights_only=True)
    n_mix, hidden = state_dict["fc_pi.weight"].shape
    model = GlobalStateLenMDN(num_states, hidden=hidden, n_mix=n_mix).to(device)
    model.load_state_dict(state_dict)
    model.eval()
    print(f"model: {num_states} states, M = {n_mix}, H = {hidden}")
    scaler_pkt = joblib.load(Path(artifacts) / "scaler_pkt.pkl")
    scaler_len = joblib.load(Path(artifacts) / "scaler_len.pkl")

    ml_end = {}
    for ue_idx in range(num_ues):
        if ue_idx not in ue_targets:
            continue
        target_bytes = ue_targets[ue_idx]["target_bytes"]
        length_pool = ue_targets[ue_idx]["flow_lengths"]
        if len(length_pool) == 0:
            continue
        selected_flows, current_bytes = [], 0
        with torch.no_grad():
            while current_bytes < target_bytes:
                current_state = np.random.choice(num_states, p=hmm_model.startprob_)
                target_flow_len = np.random.choice(length_pool)
                if target_flow_len == 0:
                    continue
                len_z_val = scaler_len.transform(np.array([[np.log1p(target_flow_len)]]))[0][0]

                states = np.empty(target_flow_len, dtype=np.int64)
                states[0] = current_state
                for t in range(1, target_flow_len):
                    states[t] = np.random.choice(num_states, p=transmat[states[t - 1]])
                x_input = torch.zeros(target_flow_len, num_states + 1)
                x_input[np.arange(target_flow_len), states] = 1.0
                x_input[:, num_states] = float(len_z_val)

                pi, mu, sg, nu = model(x_input)
                m_idx = torch.distributions.Categorical(pi).sample()
                idx2 = m_idx.view(-1, 1, 1).expand(-1, 1, 2)
                z_sample = torch.distributions.StudentT(df=nu.gather(1, m_idx.view(-1, 1, 1)).squeeze(1),
                                                        loc=mu.gather(1, idx2).squeeze(1),
                                                        scale=sg.gather(1, idx2).squeeze(1)).sample()
                real_vals = scaler_pkt.inverse_transform(z_sample.numpy())
                plogs = np.clip(real_vals[:, 0], a_min=0, a_max=20)
                dtlogs = np.clip(real_vals[:, 1], a_min=-20, a_max=20)
                payloads_arr = np.clip(np.expm1(plogs), 14, 1500).astype(int)     # at least 14 B per UDP datagram
                dts_arr = np.maximum(np.expm1(dtlogs), 0.000001)

                bad_mask = (dts_arr > 3600.0) | (~np.isfinite(dts_arr))           # IATs above one hour are drawn again
                retry = 0
                while np.any(bad_mask) and retry < 10:
                    m_bad = torch.distributions.Categorical(pi[bad_mask]).sample()
                    idx2_bad = m_bad.view(-1, 1, 1).expand(-1, 1, 2)
                    z_bad = torch.distributions.StudentT(df=nu[bad_mask].gather(1, m_bad.view(-1, 1, 1)).squeeze(1),
                                                         loc=mu[bad_mask].gather(1, idx2_bad).squeeze(1),
                                                         scale=sg[bad_mask].gather(1, idx2_bad).squeeze(1)).sample()
                    real_bad = scaler_pkt.inverse_transform(z_bad.numpy())
                    dts_arr[bad_mask] = np.maximum(np.expm1(np.clip(real_bad[:, 1], a_min=-20, a_max=20)), 0.000001)
                    bad_mask = (dts_arr > 3600.0) | (~np.isfinite(dts_arr))
                    retry += 1
                dts_arr = np.minimum(dts_arr, 3600.0)

                flow_packets, flow_byte_sum = [], 0
                for i in range(target_flow_len):
                    flow_packets.append((payloads_arr[i], dts_arr[i]))
                    flow_byte_sum += payloads_arr[i]
                    if current_bytes + flow_byte_sum >= target_bytes:
                        break
                if flow_packets:
                    selected_flows.append(flow_packets)
                current_bytes += flow_byte_sum

        trace_data, current_time = [], 2.0
        for flow_idx, flow in enumerate(selected_flows):
            for payload, dt in flow:
                current_time += dt
                trace_data.append({"timestamp_sec": current_time, "size_bytes": payload, "flow_id": flow_idx + 1})
        ml_df = pd.DataFrame(trace_data)
        filename = f"ml_traces/ml_trace_ue{ue_idx}.csv"
        ml_df.to_csv(filename, index=False)
        ml_end[ue_idx] = float(ml_df["timestamp_sec"].iloc[-1]) if len(ml_df) else 2.0
        print(f"synthetic trace UE {ue_idx} -> {filename} | {current_bytes / 1e6:.2f} MB")

    # 3  Poisson baseline and simulation time limit (longest real or synthetic trace + 5 s)
    sim_time_limit = max(2.0 + max_t_sim, max(ml_end.values(), default=0.0)) + 5.0
    print(f"sim-time-limit = {sim_time_limit:.1f}s")
    for ue_idx, t in ue_targets.items():
        print(f"*.ue[{ue_idx}].app[0].messageLength = {int(t['mean_pkt_size'])}B | "
              f"sendInterval = exponential({t['mean_iat']:.5f}s) | stopTime = {2.0 + t['t_sim']:.1f}s")
    update_omnetpp_ini(ini_path, ue_targets, sim_time_limit)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw-path", required=True, help="real test flows, e.g. runs/ftps/seed0/test.csv")
    ap.add_argument("--artifacts", required=True, help="trained model, e.g. runs/ftps/seed0/artifacts/proposed")
    ap.add_argument("--n-ues", type=int, default=5)
    ap.add_argument("--run-index", type=int, default=0, help="Monte Carlo run; seeds the shuffle and the generation")
    ap.add_argument("--ini", default="omnetpp.ini", help="omnetpp.ini to update with the Poisson baseline")
    args = ap.parse_args()
    torch.manual_seed(args.run_index)
    np.random.seed(args.run_index)
    run_alignment_pipeline(args.raw_path, args.artifacts, args.n_ues, args.run_index, args.ini)
