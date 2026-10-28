#!/usr/bin/env python3
"""
Table 4 benchmark of TimeGAN: 100 flows of 50 packets in one batch on one CPU thread.

    python scripts/bench/bench_timegan.py --artifacts runs/facebook_video/seed0/artifacts/timegan
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import timing  # noqa: E402  (first import: hides the GPU and limits torch to one thread)

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--artifacts", type=Path, default=Path("runs/facebook_video/seed0/artifacts/timegan"))
ap.add_argument("--reps", type=int, default=10)
args = ap.parse_args()

import joblib  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

device = torch.device("cpu")
FEATURE_DIM, STATIC_DIM, Z_DIM, HIDDEN_DIM, NUM_LAYERS, MAX_SEQ_LEN = 2, 1, 4, 24, 3, 50
N_FLOWS, LEN_Z = 100, 0.5


class TimeGAN(nn.Module):
    """The network of notebooks/timegan_batch.ipynb (only the parts used for generation are called)."""

    def __init__(self, feature_dim, static_dim, hidden_dim, num_layers, z_dim):
        super().__init__()
        self.embedder_rnn = nn.GRU(feature_dim + static_dim, hidden_dim, num_layers, batch_first=True)
        self.embedder_out = nn.Linear(hidden_dim, hidden_dim)
        self.recovery_rnn = nn.GRU(hidden_dim, hidden_dim, num_layers, batch_first=True)
        self.recovery_out = nn.Linear(hidden_dim, feature_dim)
        self.generator_rnn = nn.GRU(z_dim + static_dim, hidden_dim, num_layers, batch_first=True)
        self.generator_out = nn.Linear(hidden_dim, hidden_dim)
        self.supervisor_rnn = nn.GRU(hidden_dim, hidden_dim, num_layers - 1 if num_layers > 1 else 1, batch_first=True)
        self.supervisor_out = nn.Linear(hidden_dim, hidden_dim)
        self.discriminator_rnn = nn.GRU(hidden_dim + static_dim, hidden_dim, num_layers, batch_first=True, bidirectional=True)
        self.discriminator_out = nn.Linear(hidden_dim * 2, 1)

    def recover(self, h):
        h_out, _ = self.recovery_rnn(h)
        return self.recovery_out(h_out)

    def generate(self, z, s):
        s_expanded = s.unsqueeze(1).repeat(1, z.size(1), 1)
        e, _ = self.generator_rnn(torch.cat([z, s_expanded], dim=2))
        return torch.sigmoid(self.generator_out(e))

    def supervise(self, h):
        h_next, _ = self.supervisor_rnn(h)
        return torch.sigmoid(self.supervisor_out(h_next))


scaler_pkt = joblib.load(args.artifacts / "scaler_pkt_timegan.pkl")
model = TimeGAN(FEATURE_DIM, STATIC_DIM, HIDDEN_DIM, NUM_LAYERS, Z_DIM).to(device)
model.load_state_dict(torch.load(args.artifacts / "timegan_weights.pth", map_location="cpu", weights_only=True))
model.eval()


def generate_timegan():
    records = []
    with torch.no_grad():
        z = torch.randn(N_FLOWS, MAX_SEQ_LEN, Z_DIM).to(device)
        s = torch.full((N_FLOWS, 1), LEN_Z, dtype=torch.float32).to(device)
        x_gen = model.recover(model.supervise(model.generate(z, s))).cpu().numpy()   # (100, 50, 2)
    for fid in range(N_FLOWS):
        inverse = scaler_pkt.inverse_transform(x_gen[fid])
        payloads = np.clip(np.expm1(inverse[:, 0]), 0.0, 1500.0)
        dts = np.maximum(np.expm1(inverse[:, 1]), 0.0)
        records.extend([[fid, round(p), float(t)] for p, t in zip(payloads, dts)])
    return records


timing.benchmark("TimeGAN [7]", generate_timegan, reps=args.reps, **timing.footprint(model, scaler_pkt))
