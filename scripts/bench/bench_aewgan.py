#!/usr/bin/env python3
"""
Table 4 benchmark of AE-WGAN: 250 sequences of at most 20 packets in one batch on one CPU thread;
a sequence ends at the first padding value.

    python scripts/bench/bench_aewgan.py --artifacts runs/facebook_video/seed0/artifacts/aewgan
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import timing  # noqa: E402  (first import: hides the GPU and limits torch to one thread)

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--artifacts", type=Path, default=Path("runs/facebook_video/seed0/artifacts/aewgan"))
ap.add_argument("--reps", type=int, default=10)
args = ap.parse_args()

import joblib  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

device = torch.device("cpu")
SEQ_LEN, LATENT_DIM, N_SEQ = 20, 16, 250


class Encoder(nn.Module):
    def __init__(self, input_dim=2, seq_len=20, latent_dim=16):
        super().__init__()
        self.seq_len = seq_len
        self.td_dense = nn.Linear(input_dim, 128)
        self.lstm1 = nn.LSTM(128, 128, batch_first=True)
        self.lstm2 = nn.LSTM(128, 64, batch_first=True)
        self.dense1 = nn.Linear(64, 32)
        self.bn1 = nn.BatchNorm1d(32)
        self.dense2 = nn.Linear(32, latent_dim)
        self.bn2 = nn.BatchNorm1d(latent_dim)
        self.tanh = nn.Tanh()


class Decoder(nn.Module):
    def __init__(self, latent_dim=16, seq_len=20, output_dim=2):
        super().__init__()
        self.seq_len = seq_len
        self.dense1 = nn.Linear(latent_dim, 32)
        self.bn1 = nn.BatchNorm1d(32)
        self.lstm1 = nn.LSTM(32, 64, batch_first=True)
        self.lstm2 = nn.LSTM(64, 128, batch_first=True)
        self.td_dense = nn.Linear(128, output_dim)
        self.tanh = nn.Tanh()

    def forward(self, z):
        x = self.tanh(self.bn1(self.dense1(z)))
        x = x.unsqueeze(1).repeat(1, self.seq_len, 1)
        x, _ = self.lstm1(x)
        x, _ = self.lstm2(x)
        return self.td_dense(x)


class Autoencoder(nn.Module):
    def __init__(self, seq_len=20, latent_dim=16, feature_dim=2):
        super().__init__()
        self.encoder = Encoder(feature_dim, seq_len, latent_dim)
        self.decoder = Decoder(latent_dim, seq_len, feature_dim)


class WGANGenerator(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()

        def block(in_feat, out_feat):
            return [nn.Linear(in_feat, out_feat), nn.BatchNorm1d(out_feat), nn.Tanh()]

        self.model = nn.Sequential(*block(latent_dim, 16), *block(16, 16), *block(16, 16), *block(16, 16),
                                   nn.Linear(16, latent_dim))

    def forward(self, z):
        return self.model(z)


scaler_pkt = joblib.load(args.artifacts / "scaler_pkt_wgan.pkl")
ae = Autoencoder(SEQ_LEN, LATENT_DIM, feature_dim=2).to(device)
ae.load_state_dict(torch.load(args.artifacts / "ae_weights_wgan.pth", map_location="cpu", weights_only=True))
ae.eval()
generator = WGANGenerator(LATENT_DIM).to(device)
generator.load_state_dict(torch.load(args.artifacts / "generator_weights_wgan.pth", map_location="cpu", weights_only=True))
generator.eval()


def generate_wgan():
    records = []
    with torch.no_grad():
        seqs = ae.decoder(generator(torch.randn(N_SEQ, LATENT_DIM).to(device))).cpu().numpy()   # (250, 20, 2)
    for fid in range(N_SEQ):
        inverse = scaler_pkt.inverse_transform(seqs[fid])
        for t in range(SEQ_LEN):
            if seqs[fid, t, 0] < -5.0:                  # padding value: end of the sequence
                break
            payload = np.clip(np.expm1(inverse[t, 0]), 0.0, 1500.0)
            dt = max(np.expm1(inverse[t, 1]), 0.0)
            records.append([fid, round(payload), float(dt)])
    return records


timing.benchmark("AE-WGAN [8]", generate_wgan, reps=args.reps, **timing.footprint(ae, generator, scaler_pkt))
