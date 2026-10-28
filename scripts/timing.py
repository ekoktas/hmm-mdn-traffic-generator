"""
Timing helper of Table 4. Import it before torch: it hides the GPU and limits the process to one
CPU thread.

    import timing
    def generate_5000():
        ...                                  # generate 5,000 packets and return the records
    timing.benchmark("Model", generate_5000, **timing.footprint(model, scaler))

benchmark() warms up once, times `reps` calls, prints mean and standard deviation, and appends
one JSON line to timing_results.jsonl in the current directory.
"""
from __future__ import annotations

import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")     # must be set before torch is imported
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import json  # noqa: E402
import platform  # noqa: E402
import subprocess  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402


def cpu_model() -> str:
    try:        # LC_ALL=C: lscpu prints localized field names otherwise
        out = subprocess.run(["lscpu"], capture_output=True, text=True, check=False,
                             env={**os.environ, "LC_ALL": "C"}).stdout
        for line in out.splitlines():
            if line.startswith("Model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or "unknown"


def benchmark(name: str, generate_5000, reps: int = 10, n_packets: int = 5000,
              out: str = "timing_results.jsonl", **extra) -> dict:
    """Warm up once, time `reps` calls of generate_5000, print the result and append it to `out`.

    If generate_5000 returns its records (list, array or DataFrame) or an int, the number of valid
    packets it produced is recorded instead of n_packets; some generators drop packets.
    """
    try:
        import torch
        torch.set_num_threads(1)
        torch_version, threads = torch.__version__, torch.get_num_threads()
    except ImportError:
        torch_version, threads = "n/a", 1
    generate_5000()                                          # warm-up
    times, produced = [], []
    for _ in range(reps):
        t0 = time.perf_counter()
        r = generate_5000()
        times.append(time.perf_counter() - t0)
        if isinstance(r, (int, np.integer)):
            produced.append(int(r))
        elif hasattr(r, "__len__"):
            produced.append(len(r))
        else:
            produced.append(n_packets)
    rec = {"model": name, "mean_s": float(np.mean(times)),
           "std_s": float(np.std(times, ddof=1)) if reps > 1 else float("nan"),
           "reps": reps, "valid_packets": int(np.mean(produced)),
           "pkt_per_s": float(np.mean(produced) / np.mean(times)),
           "cpu": cpu_model(), "torch": torch_version, "threads": threads, **extra}
    fp = f" | footprint {rec['footprint_MB']} MB, {rec['nn_params']:,} NN parameters" if "footprint_MB" in rec else ""
    print(f"{name}: {rec['mean_s']:.4f} +- {rec['std_s']:.4f} s for {rec['valid_packets']} packets "
          f"({rec['pkt_per_s']:,.0f} packets/s) | {rec['cpu']} | torch {rec['torch']} | {threads} thread{fp}")
    with Path(out).open("a") as fh:
        fh.write(json.dumps(rec) + "\n")
    return rec


def footprint(*objects) -> dict:
    """Memory footprint of a model: torch modules by the bytes of their parameters and buffers,
    all other objects (HMMs, scalers, mixtures, tokenizers) by their pickled size."""
    import pickle
    n_params, torch_bytes, other_bytes = 0, 0, 0
    for obj in objects:
        if hasattr(obj, "parameters") and hasattr(obj, "buffers"):
            for p in obj.parameters():
                n_params += p.numel()
                torch_bytes += p.numel() * p.element_size()
            for b in obj.buffers():
                torch_bytes += b.numel() * b.element_size()
        else:
            try:
                other_bytes += len(pickle.dumps(obj))
            except Exception:
                import dill
                other_bytes += len(dill.dumps(obj))
    total = torch_bytes + other_bytes
    return {"nn_params": int(n_params), "nn_bytes": int(torch_bytes), "other_bytes": int(other_bytes),
            "footprint_MB": round(total / 2**20, 4)}
