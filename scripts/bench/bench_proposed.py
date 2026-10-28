#!/usr/bin/env python3
"""
Table 4 benchmark of the proposed generator: 100 flows of 50 packets (5,000 packets) on one CPU
thread, with the HMM, scalers and MDN of a trained run. The flow-length input is fixed at 0.5.

    python scripts/bench/bench_proposed.py --artifacts runs/facebook_video/seed0/artifacts/proposed
    python scripts/bench/bench_proposed.py --artifacts runs/facebook_video/seed0/artifacts/proposed --no-reject
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import timing  # noqa: E402  (first import: hides the GPU and limits torch to one thread)

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--artifacts", type=Path, default=Path("runs/facebook_video/seed0/artifacts/proposed"))
ap.add_argument("--reps", type=int, default=10)
ap.add_argument("--no-reject", action="store_true", help="clip gaps above 3600 s instead of drawing them again")
args = ap.parse_args()

import joblib  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from hmm_mdn import GlobalStateLenMDN, sample_mixture  # noqa: E402

meta = json.loads((args.artifacts / "meta.json").read_text())
device = torch.device("cpu")
hmm_boot = joblib.load(args.artifacts / "hmm.pkl")
scaler_pkt = joblib.load(args.artifacts / "scaler_pkt.pkl")
scaler_len = joblib.load(args.artifacts / "scaler_len.pkl")

K_STATES = int(meta["K"])
IDLE_K = K_STATES - 1 if meta["idle"] else -1
TEMPERATURE_IDLE = meta.get("temperature_idle", 1.0)
DIST_FAMILY = meta.get("dist_family", "student")
USE_FLOW_LEN = not meta["no_flow_len"]

model_mdn = GlobalStateLenMDN(K_STATES, hidden=meta["H"], n_mix=meta["M"], use_flow_len=USE_FLOW_LEN).to(device)
model_mdn.load_state_dict(torch.load(args.artifacts / "mdn.pth", map_location="cpu", weights_only=True))
model_mdn.eval()

N_FLOWS, L, LEN_Z = 100, 50, 0.5
onehot = torch.eye(K_STATES, device=device)
dt_mean = torch.tensor(scaler_pkt.mean_[1], device=device)
dt_scale = torch.tensor(scaler_pkt.scale_[1], device=device)


def sample(pi, mu, sg, nu, st):
    return sample_mixture(pi, mu, sg, nu, st, IDLE_K, TEMPERATURE_IDLE, DIST_FAMILY)


def generate_proposed():
    rng = np.random.default_rng(43)
    A, pi0 = hmm_boot.transmat_, hmm_boot.startprob_
    records = []
    with torch.no_grad():
        for fid in range(N_FLOWS):
            states = np.empty(L, dtype=np.int64)
            states[0] = rng.choice(K_STATES, p=pi0)
            for t in range(1, L):
                states[t] = rng.choice(K_STATES, p=A[states[t - 1]])
            st = torch.tensor(states, device=device)
            x = onehot[st]
            if USE_FLOW_LEN:
                x = torch.cat([x, torch.full((L, 1), LEN_Z, device=device)], dim=1)
            pi, mu, sg, nu = model_mdn(x)
            x_sample = sample(pi, mu, sg, nu, st)
            if not args.no_reject:                      # the same redraw loop as in generation
                dt = torch.expm1(x_sample[:, 1] * dt_scale + dt_mean)
                bad = (dt > 3600.0) | (~torch.isfinite(dt))
                n_redraw = 0
                while bad.any() and n_redraw < 10:
                    new = sample(pi[bad], mu[bad], sg[bad], nu[bad], st[bad])
                    x_sample[bad] = new
                    dt[bad] = torch.expm1(new[:, 1] * dt_scale + dt_mean)
                    bad = (dt > 3600.0) | (~torch.isfinite(dt))
                    n_redraw += 1
            real = scaler_pkt.inverse_transform(x_sample.cpu().numpy())
            payloads = np.clip(np.expm1(real[:, 0]), 0.0, 1500.0)
            dts = np.maximum(np.expm1(real[:, 1]), 1e-6)
            if args.no_reject:
                dts = np.clip(dts, 1e-6, 3600.0)
            records.extend([[fid, round(p), float(t)] for p, t in zip(payloads, dts)])
    return records


name = "Proposed (no reject)" if args.no_reject else "Proposed"
timing.benchmark(name, generate_proposed, reps=args.reps,
                 **timing.footprint(model_mdn, hmm_boot, scaler_pkt, scaler_len),
                 M=meta["M"], H=meta["H"], K=K_STATES)
