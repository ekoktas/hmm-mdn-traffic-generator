#!/usr/bin/env python3
"""
Table 4 benchmark of NeCSTGen: 100 flows of 50 packets on one CPU thread (float64, as in training).

    python scripts/bench/bench_necstgen.py --artifacts runs/facebook_video/seed0/artifacts/necstgen
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import timing  # noqa: E402  (first import: hides the GPU and limits torch to one thread)

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--artifacts", type=Path, default=Path("runs/facebook_video/seed0/artifacts/necstgen"))
ap.add_argument("--reps", type=int, default=10)
args = ap.parse_args()

import joblib  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

device = torch.device("cpu")
torch.set_default_dtype(torch.float64)
TIMESTEPS, LATENT_DIM, N_FLOWS, L = 11, 4, 100, 50


class NeCSTGenEncoder(nn.Module):
    def __init__(self, input_dim=2, latent_dim=4):
        super().__init__()
        self.shared = nn.Sequential(nn.Linear(input_dim, 8), nn.LeakyReLU(0.1), nn.Linear(8, 6), nn.LeakyReLU(0.1),
                                    nn.Linear(6, 4), nn.LeakyReLU(0.1))
        self.z_mean = nn.Linear(4, latent_dim)
        self.z_log_var = nn.Linear(4, latent_dim)


class NeCSTGenDecoder(nn.Module):
    def __init__(self, latent_dim=4, output_dim=2):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(latent_dim, 4), nn.LeakyReLU(0.1), nn.Linear(4, 6), nn.LeakyReLU(0.1),
                                 nn.Linear(6, 8), nn.LeakyReLU(0.1), nn.Linear(8, output_dim), nn.Sigmoid())

    def forward(self, z):
        return self.net(z)


class NeCSTGenVAE(nn.Module):
    def __init__(self, input_dim=2, latent_dim=4):
        super().__init__()
        self.encoder = NeCSTGenEncoder(input_dim, latent_dim)
        self.decoder = NeCSTGenDecoder(latent_dim, input_dim)


class NeCSTGenPredictor(nn.Module):
    def __init__(self, input_dim=2, hidden_dim=4, output_dim=2):
        super().__init__()
        self.gru = nn.GRU(input_size=input_dim, hidden_size=hidden_dim, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        gru_out, _ = self.gru(x)
        return self.sigmoid(self.fc(gru_out[:, -1, :]))


bounds = joblib.load(args.artifacts / "necstgen_bounds.pkl")
min_p, max_p, min_t, max_t = bounds["min_p"], bounds["max_p"], bounds["min_t"], bounds["max_t"]
bgm = joblib.load(args.artifacts / "bgm_model.sav")
vae = NeCSTGenVAE(input_dim=2, latent_dim=LATENT_DIM).to(device)
vae.load_state_dict(torch.load(args.artifacts / "vae_weights.pth", map_location="cpu", weights_only=True))
vae.eval()
predictor = NeCSTGenPredictor(input_dim=2, hidden_dim=LATENT_DIM, output_dim=2).to(device)
predictor.load_state_dict(torch.load(args.artifacts / "predictor_weights.pth", map_location="cpu", weights_only=True))
predictor.eval()
perm_rng = np.random.default_rng(0)


def generate_necstgen():
    records = []
    with torch.no_grad():
        for fid in range(N_FLOWS):
            latents, _ = bgm.sample(L)
            latents = latents[perm_rng.permutation(L)]          # sample() returns the draws grouped by component
            features = vae.decoder(torch.tensor(latents, dtype=torch.float64, device=device))
            p_scaled = features[:, 0].cpu().numpy()
            t_scaled = features[:, 1].cpu().numpy()
            if L > TIMESTEPS:                                   # IAT of later packets from the recurrent predictor
                ctx = features[:TIMESTEPS].unsqueeze(0)
                for i in range(TIMESTEPS, L):
                    t_scaled[i] = predictor(ctx)[0, 1].cpu().numpy()
                    ctx = torch.cat([ctx[:, 1:, :], features[i].view(1, 1, 2)], dim=1)
            for i in range(L):
                payload = np.clip(np.expm1(p_scaled[i] * (max_p - min_p) + min_p), 0.0, 1500.0)
                dt = max(np.expm1(t_scaled[i] * (max_t - min_t) + min_t), 0.0)
                records.append([fid, round(float(payload)), float(dt)])
    return records


timing.benchmark("NeCSTGen [9]", generate_necstgen, reps=args.reps, **timing.footprint(vae, predictor, bgm, bounds))
