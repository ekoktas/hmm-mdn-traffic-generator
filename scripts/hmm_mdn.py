"""
The proposed generator: the mixture density network (MDN), its training data and loss, the
mixture sampler and the generation of synthetic flows. The HMM is fitted in
notebooks/proposed_batch.ipynb with hmmlearn (see safe_hmm.py).

Notation as in the article: z = (normalized log payload, normalized log IAT) of a packet,
h = [one-hot state, xi] with xi the normalized log flow length, M mixture components and
H hidden units.
"""
import math
import warnings

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from tqdm import tqdm

NU_FLOOR = 1.001        # the degrees of freedom stay above 1
SIGMA_FLOOR = 1e-3      # lower bound of the component scales
PAYLOAD_MAX = 1500.0    # bytes
IAT_MAX = 3600.0        # seconds; longer generated gaps are drawn again
IAT_MIN = 1e-6          # seconds; smallest generated IAT
MAX_REDRAWS = 10        # redraws of a packet whose gap exceeds IAT_MAX


class GlobalStateLenMDN(nn.Module):
    """MDN that maps h = [one-hot state, xi] to a mixture of M diagonal bivariate components.

    Two tanh hidden layers of width H feed four heads: mixture weights (softmax), locations,
    scales (softplus + SIGMA_FLOOR) and one degree of freedom per component
    (softplus + NU_FLOOR). With use_flow_len=False the input is the one-hot state only.
    """

    def __init__(self, n_states, hidden=128, n_mix=16, out_dim=2, use_flow_len=True):
        super().__init__()
        self.n_mix, self.out_dim = n_mix, out_dim
        in_dim = n_states + (1 if use_flow_len else 0)
        self.net = nn.Sequential(nn.Linear(in_dim, hidden), nn.Tanh(), nn.Linear(hidden, hidden), nn.Tanh())
        self.fc_pi = nn.Linear(hidden, n_mix)
        self.fc_mu = nn.Linear(hidden, n_mix * out_dim)
        self.fc_ls = nn.Linear(hidden, n_mix * out_dim)
        self.fc_lnu = nn.Linear(hidden, n_mix)

    def forward(self, x):
        h = self.net(x)
        pi = torch.softmax(self.fc_pi(h), dim=-1)                          # (B, M)
        mu = self.fc_mu(h).view(-1, self.n_mix, self.out_dim)              # (B, M, 2)
        sigma = F.softplus(self.fc_ls(h)).view_as(mu) + SIGMA_FLOOR        # (B, M, 2)
        nu = F.softplus(self.fc_lnu(h)).view(-1, self.n_mix, 1) + NU_FLOOR  # (B, M, 1)
        return pi, mu, sigma, nu


@torch.jit.script
def nll_student_t(y, pi, mu, sigma, nu):
    """Negative log-likelihood of y (B, 2) under the mixture of diagonal Student-t components."""
    y = y.unsqueeze(1)
    half_log_pi = 0.5 * math.log(math.pi)
    z2 = ((y - mu) / sigma).pow(2)
    log_c = torch.lgamma((nu + 1) * 0.5) - torch.lgamma(nu * 0.5) \
        - 0.5 * torch.log(nu) - half_log_pi - torch.log(sigma)
    log_comp = log_c - 0.5 * (nu + 1) * torch.log1p(z2 / nu)
    log_prob = torch.logsumexp(torch.log(pi) + log_comp.sum(-1), dim=-1)
    return -log_prob


def nll_gaussian(y, pi, mu, sigma):
    """Negative log-likelihood of y (B, 2) under the mixture of diagonal Gaussian components."""
    y = y.unsqueeze(1)
    comp = -0.5 * ((y - mu) / sigma) ** 2 - torch.log(sigma) - 0.5 * math.log(2 * math.pi)
    log_prob = torch.logsumexp(torch.log(pi) + comp.sum(-1), dim=-1)
    return -log_prob


class PosteriorDataset(Dataset):
    """Training samples of the MDN, built from the HMM state posteriors.

    A packet enters once for every state whose posterior exceeds thresh_core (thresh_idle for
    the idle state), with input h = [one-hot state, xi], target z and the posterior as weight.
    The weights of each state are then rescaled to a mean of one, so that a state's total weight
    equals its number of samples, and finally all weights are rescaled to a mean of one.
    """

    def __init__(self, frame, n_states, idle_state=-1, use_flow_len=True, thresh_core=1e-4, thresh_idle=1e-6):
        gamma = frame[[f"state_post_{k}" for k in range(n_states)]].to_numpy(dtype=np.float32)
        mask = gamma > thresh_core
        if idle_state >= 0:
            mask[:, idle_state] = gamma[:, idle_state] > thresh_idle
        rows, states = np.where(mask)
        self.Y = torch.tensor(frame[["plog_z", "dt_z"]].to_numpy(dtype=np.float32)[rows])
        len_z = frame["len_z"].to_numpy(dtype=np.float32)[rows].reshape(-1, 1)
        onehot = np.eye(n_states, dtype=np.float32)[states]
        self.X = torch.tensor(np.hstack([onehot, len_z]) if use_flow_len else onehot)
        weights = gamma[rows, states]
        for k in range(n_states):
            k_mask = states == k
            if k_mask.any():
                weights[k_mask] *= k_mask.sum() / weights[k_mask].sum()
        self.W = torch.tensor(weights)
        self.W = self.W * (len(self.W) / self.W.sum())

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx], self.W[idx]


def sample_mixture(pi, mu, sigma, nu, states, idle_state=-1, temp_idle=1.0, dist_family="student"):
    """One sample per row: a component index drawn from the mixture weights, then a point drawn
    from that component. In the idle state the weights can be raised to the power 1/temp_idle
    (temp_idle = 1 leaves them unchanged)."""
    is_idle = (states == idle_state).unsqueeze(1)
    weights = torch.where(is_idle, pi ** (1.0 / temp_idle), pi)
    weights = weights / weights.sum(dim=-1, keepdim=True)
    comp = torch.multinomial(weights, 1).squeeze(-1)
    idx = comp.view(-1, 1, 1).expand(-1, 1, mu.size(-1))
    loc = torch.gather(mu, 1, idx).squeeze(1)
    scale = torch.gather(sigma, 1, idx).squeeze(1)
    df = torch.gather(nu, 1, comp.view(-1, 1, 1)).squeeze(1)
    if dist_family == "gaussian":
        return torch.distributions.Normal(loc, scale).sample()
    return torch.distributions.StudentT(df=df, loc=loc, scale=scale).sample()


def flow_length_selector(mode, flow_sizes, train_flows, seed):
    """Return a function fid -> number of packets of the generated flow.

    oracle     the length of the real flow fid
    empirical  a length drawn uniformly with replacement from the lengths of the training flows
    """
    train_len = flow_sizes.loc[train_flows].to_numpy()
    rng = np.random.default_rng(seed + 10_000)     # own stream: the state paths do not change

    def flow_len_for(fid):
        if mode == "oracle":
            return int(flow_sizes.loc[fid])
        if mode == "empirical":
            return int(rng.choice(train_len))
        raise ValueError(mode)
    return flow_len_for


def generate_flows(model, transmat, startprob, scaler_pkt, scaler_len, flow_ids, flow_len_for, *,
                   idle_state=-1, temp_idle=1.0, dist_family="student", use_flow_len=True, seed=0,
                   device="cpu"):
    """Generate one synthetic flow per entry of flow_ids.

    For every flow: draw a state path of the HMM, evaluate the MDN for all packets at once,
    sample (payload, IAT) in the normalized space, redraw packets whose IAT exceeds IAT_MAX,
    and undo the normalization and the log transforms. Returns a DataFrame with the columns
    flow_id, payload_length, time_diff.
    """
    n_states = len(startprob)
    rng = np.random.default_rng(seed)
    onehot = torch.eye(n_states, device=device)
    dt_mean = torch.tensor(scaler_pkt.mean_[1], device=device)
    dt_scale = torch.tensor(scaler_pkt.scale_[1], device=device)
    records = []
    model.eval()
    with torch.no_grad(), warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning, message="X does not have valid feature names")
        for fid in tqdm(flow_ids, desc="flows"):
            n_pkts = flow_len_for(fid)
            if n_pkts == 0:
                continue
            len_z = float(scaler_len.transform([[math.log1p(n_pkts)]])[0, 0])

            states = np.empty(n_pkts, dtype=np.int64)
            states[0] = rng.choice(n_states, p=startprob)
            for t in range(1, n_pkts):
                states[t] = rng.choice(n_states, p=transmat[states[t - 1]])
            states_t = torch.tensor(states, device=device)

            x = onehot[states_t]
            if use_flow_len:
                x = torch.cat([x, torch.full((n_pkts, 1), len_z, device=device)], dim=1)
            pi, mu, sigma, nu = model(x)
            sample = sample_mixture(pi, mu, sigma, nu, states_t, idle_state, temp_idle, dist_family)

            dt = torch.expm1(sample[:, 1] * dt_scale + dt_mean)
            bad = (dt > IAT_MAX) | (~torch.isfinite(dt))
            n_redraw = 0
            while bad.any() and n_redraw < MAX_REDRAWS:
                new = sample_mixture(pi[bad], mu[bad], sigma[bad], nu[bad], states_t[bad],
                                     idle_state, temp_idle, dist_family)
                sample[bad] = new
                dt[bad] = torch.expm1(new[:, 1] * dt_scale + dt_mean)
                bad = (dt > IAT_MAX) | (~torch.isfinite(dt))
                n_redraw += 1

            real = scaler_pkt.inverse_transform(sample.cpu().numpy())
            payloads = np.clip(np.expm1(real[:, 0]), 0.0, PAYLOAD_MAX)
            dts = np.maximum(np.expm1(real[:, 1]), IAT_MIN)
            records.extend([fid, round(p), float(t)] for p, t in zip(payloads, dts))
    return pd.DataFrame(records, columns=["flow_id", "payload_length", "time_diff"])
