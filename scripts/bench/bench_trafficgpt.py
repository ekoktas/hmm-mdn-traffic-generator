#!/usr/bin/env python3
"""
Table 4 benchmark of the adapted TrafficGPT: 100 flows of 100 tokens (50 packets) on one CPU
thread, top-k sampling with k = 10. Generation does not stop at the end token, so that every flow
has the same number of tokens; token pairs that are not (payload, IAT) are dropped.

    python scripts/bench/bench_trafficgpt.py --artifacts runs/facebook_video/seed0/artifacts/trafficgpt
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import timing  # noqa: E402  (first import: hides the GPU and limits torch to one thread)

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--artifacts", type=Path, default=Path("runs/facebook_video/seed0/artifacts/trafficgpt"))
ap.add_argument("--reps", type=int, default=10)
args = ap.parse_args()

import dill  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402

device = torch.device("cpu")
N_FLOWS, TOKENS_PER_FLOW, TOP_K = 100, 100, 10


class TrafficGPT(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_heads, num_layers, dropout, max_len):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.position_embedding = nn.Embedding(max_len, embed_dim)
        layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, dim_feedforward=embed_dim * 4,
                                           dropout=dropout, batch_first=True, norm_first=True)
        self.transformer_decoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.ln_f = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, vocab_size, bias=False)
        self.register_buffer("causal_mask", torch.triu(torch.ones(max_len, max_len) * float("-inf"), diagonal=1))

    def forward(self, idx):
        _, t = idx.size()
        pos = torch.arange(0, t, dtype=torch.long, device=idx.device).unsqueeze(0)
        x = self.token_embedding(idx) + self.position_embedding(pos)
        x = self.transformer_decoder(x, mask=self.causal_mask[:t, :t], is_causal=True)
        return self.head(self.ln_f(x))


with open(args.artifacts / "trafficgpt_tokenizer.pkl", "rb") as fh:
    tokenizer = dill.load(fh)
state = torch.load(args.artifacts / "trafficgpt_weights.pth", map_location=device, weights_only=True)
model = TrafficGPT(vocab_size=tokenizer.vocab_size, embed_dim=128, num_heads=4, num_layers=4, dropout=0.1,
                   max_len=state["position_embedding.weight"].shape[0]).to(device)
model.load_state_dict(state)
model.eval()


def generate_trafficgpt():
    records, dropped = [], 0
    with torch.no_grad():
        for fid in range(N_FLOWS):
            ctx = torch.tensor([[tokenizer.special_tokens["SOS"]]], dtype=torch.long, device=device)
            for _ in range(TOKENS_PER_FLOW):
                logits = model(ctx)[:, -1, :]
                v, _ = torch.topk(logits, TOP_K)
                logits[logits < v[:, [-1]]] = -float("Inf")
                ctx = torch.cat((ctx, torch.multinomial(F.softmax(logits, dim=-1), num_samples=1)), dim=1)
            tokens = ctx[0].cpu().numpy()[1:]
            vals_p, vals_t, i = [], [], 0
            while i < len(tokens) - 1:
                val1, type1 = tokenizer.detokenize(tokens[i])
                val2, type2 = tokenizer.detokenize(tokens[i + 1])
                if type1 == "PAYLOAD" and type2 == "TIME":
                    vals_p.append(val1)
                    vals_t.append(val2)
                    i += 2
                else:
                    dropped += 1
                    i += 1
            for p, t in zip(vals_p[:50], vals_t[:50]):
                records.append([fid, round(max(0, min(p, 1500))), float(max(1e-6, t))])
    print(f"   tokens dropped as misaligned: {dropped}")
    return records


timing.benchmark("TrafficGPT (adapted) [12]", generate_trafficgpt, reps=args.reps, **timing.footprint(model, tokenizer))
