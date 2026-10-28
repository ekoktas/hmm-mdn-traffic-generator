#!/usr/bin/env python3
"""
Scalability of the proposed generator with the number of concurrent flows, on one CPU thread.

Two modes:
  batched    all N flows advance by one packet per step, with one MDN forward pass per step for all
             flows; this is how a digital twin serves many flows at the same time
  per-flow   one flow after another, one forward pass per flow (as in the generation notebooks)

Reported per number of flows: throughput in packets per second, mean and 99th-percentile latency of
one step (batched: one packet in every flow; per-flow: one whole flow), the growth of the resident
memory, the model size and the state kept per flow.

    python scripts/scalability_bench.py --artifacts runs/facebook_video/seed0/artifacts/proposed \
        --flows 1 5 10 50 100 500 1000 --packets-per-flow 200 --reps 5 --out bench_fbvideo

Writes <out>/scalability.csv and <out>/table_scalability.tex.
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import pickle
import sys
import time
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")      # before torch is imported

import joblib  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from hmm_mdn import GlobalStateLenMDN, sample_mixture  # noqa: E402


def rss_bytes() -> int:
    """Current resident set size (Linux); 0 where /proc is not available."""
    try:
        for line in Path("/proc/self/status").read_text().splitlines():
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) * 1024
    except OSError:
        pass
    return 0


class Generator:
    """MDN forward pass and sampler of a trained run."""

    def __init__(self, artifacts: Path):
        meta = json.loads((artifacts / "meta.json").read_text())
        self.hmm = joblib.load(artifacts / "hmm.pkl")
        self.scalers = [joblib.load(artifacts / f) for f in ("scaler_pkt.pkl", "scaler_len.pkl")]
        self.K = int(meta["K"])
        self.idle_k = self.K - 1 if meta["idle"] else -1
        self.temp_idle = meta.get("temperature_idle", 1.0)
        self.dist_family = meta.get("dist_family", "student")
        self.net = GlobalStateLenMDN(self.K, hidden=meta["H"], n_mix=meta["M"], use_flow_len=not meta["no_flow_len"])
        self.net.load_state_dict(torch.load(artifacts / "mdn.pth", map_location="cpu", weights_only=True))
        self.net.eval()
        params = sum(p.numel() * p.element_size() for p in self.net.parameters())
        params += sum(b.numel() * b.element_size() for b in self.net.buffers())
        self.model_bytes = int(params + len(pickle.dumps(self.hmm)) + sum(len(pickle.dumps(s)) for s in self.scalers))

    def step(self, x: np.ndarray, states: np.ndarray):
        with torch.no_grad():
            pi, mu, sg, nu = self.net(torch.as_tensor(x, dtype=torch.float32))
            return sample_mixture(pi, mu, sg, nu, torch.as_tensor(states), self.idle_k, self.temp_idle, self.dist_family)


def run_batched(gen, n_flows, n_pkts, len_z, rng):
    """All flows advance together; the HMM step is vectorized over the flows (inverse-CDF sampling)."""
    K, A, pi0 = gen.K, gen.hmm.transmat_, gen.hmm.startprob_
    cdf_A, cdf_0 = A.cumsum(1), pi0.cumsum()
    idx = np.arange(n_flows)
    states = np.searchsorted(cdf_0, rng.random(n_flows))
    x = np.zeros((n_flows, K + 1), dtype=np.float32)
    x[:, K] = len_z
    lat = np.empty(n_pkts)
    for t in range(n_pkts):
        t0 = time.perf_counter()
        x[:, :K] = 0.0
        x[idx, states] = 1.0
        gen.step(x, states)
        states = (cdf_A[states] < rng.random(n_flows)[:, None]).sum(1).clip(0, K - 1)
        lat[t] = time.perf_counter() - t0
    return lat


def run_per_flow(gen, n_flows, n_pkts, len_z, rng):
    """One flow after another, one forward pass per flow."""
    K, A, pi0 = gen.K, gen.hmm.transmat_, gen.hmm.startprob_
    lat = np.empty(n_flows)
    for f in range(n_flows):
        t0 = time.perf_counter()
        states = np.empty(n_pkts, dtype=int)
        states[0] = rng.choice(K, p=pi0)
        for t in range(1, n_pkts):
            states[t] = rng.choice(K, p=A[states[t - 1]])
        x = np.zeros((n_pkts, K + 1), dtype=np.float32)
        x[np.arange(n_pkts), states] = 1.0
        x[:, K] = len_z
        gen.step(x, states)
        lat[f] = time.perf_counter() - t0
    return lat


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--artifacts", type=Path, required=True, help="runs/<dataset>/seed<k>/artifacts/proposed")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--flows", nargs="*", type=int, default=[1, 5, 10, 50, 100, 500, 1000])
    ap.add_argument("--packets-per-flow", type=int, default=200)
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--modes", nargs="*", default=["batched", "per-flow"], choices=["batched", "per-flow"])
    ap.add_argument("--flow-len", type=int, default=1000, help="flow length used for the conditioning input")
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(args.threads)

    gen = Generator(args.artifacts)
    len_z = float(gen.scalers[1].transform([[np.log1p(args.flow_len)]])[0, 0])
    rows = []
    for mode in args.modes:
        fn = run_batched if mode == "batched" else run_per_flow
        for n in args.flows:
            fn(gen, min(n, 2), 5, len_z, np.random.default_rng(args.seed))      # warm-up
            gc.collect()
            base_rss = peak_rss = rss_bytes()
            times, lats = [], []
            for rep in range(args.reps):
                rng = np.random.default_rng(args.seed + rep)
                t0 = time.perf_counter()
                lats.append(fn(gen, n, args.packets_per_flow, len_z, rng))
                times.append(time.perf_counter() - t0)
                peak_rss = max(peak_rss, rss_bytes())
            times, lat_all = np.array(times), np.concatenate(lats)
            n_pkts = n * args.packets_per_flow
            rows.append({
                "mode": mode, "n_flows": n, "packets": n_pkts,
                "time_mean_s": times.mean(), "time_std_s": times.std(ddof=1) if args.reps > 1 else np.nan,
                "throughput_pkt_s": n_pkts / times.mean(),
                "latency_step": "one packet in every flow" if mode == "batched" else "one whole flow",
                "latency_ms_mean": 1e3 * lat_all.mean(),
                "latency_ms_p99": 1e3 * np.percentile(lat_all, 99),
                "rss_delta_kb": (peak_rss - base_rss) / 1024,
                "model_bytes": gen.model_bytes,
                "state_bytes_per_flow": 4 * (gen.K + 1) + 8 + 2 * 4,   # input row, state index, last sample
            })
            print(f"{mode:9s} N={n:5d}  {rows[-1]['throughput_pkt_s']:10.0f} packets/s  "
                  f"step {rows[-1]['latency_ms_mean']:.2f} ms (p99 {rows[-1]['latency_ms_p99']:.2f} ms)")

    df = pd.DataFrame(rows)
    df.to_csv(args.out / "scalability.csv", index=False)
    bat = df[df["mode"] == "batched"]
    if not bat.empty:
        lines = [r"\hline", r"Concurrent flows & Throughput (packets/s) & Latency per step (ms) & "
                 r"99th percentile (ms) & Memory growth (kB) \\ \hline"]
        for _, r in bat.iterrows():
            lines.append(f"{int(r['n_flows'])} & {r['throughput_pkt_s']:,.0f} & {r['latency_ms_mean']:.2f} & "
                         f"{r['latency_ms_p99']:.2f} & {r['rss_delta_kb']:.0f} \\\\")
        lines.append(r"\hline")
        (args.out / "table_scalability.tex").write_text("\n".join(lines))
    print(f"\nmodel size: {gen.model_bytes:,} bytes, independent of the number of flows; wrote {args.out.resolve()}")


if __name__ == "__main__":
    main()
