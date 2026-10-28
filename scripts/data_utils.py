"""
Loading and cleaning of the packet traces, the shared train/test split, and output files.
All notebooks use these functions, so that every model is trained and evaluated on the same data.
"""
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from split_rule import constrained_split

COLUMNS = ["flow_id", "payload_length", "time_diff"]
PAYLOAD_MAX = 1500   # bytes: payloads are clipped to the Ethernet MTU
IAT_MAX = 3600       # seconds: packets with a gap of an hour or more to their predecessor are dropped


def load_packets(path, verbose=True):
    """Read a packet trace (one row per packet: flow_id, payload_length, time_diff) and clean it.

    1. stable sort by flow_id: the packets of a flow become contiguous and keep their order
    2. drop rows with non-finite values
    3. clip the payload to [0, PAYLOAD_MAX] bytes and the IAT to values >= 0
    4. drop packets whose IAT is IAT_MAX seconds or more
    5. drop rows with zero payload and zero IAT
    """
    df = pd.read_csv(path, usecols=COLUMNS)
    df = df.sort_values("flow_id", kind="stable").reset_index(drop=True)
    invalid = (~np.isfinite(df["payload_length"])) | (~np.isfinite(df["time_diff"]))
    if invalid.any():
        if verbose:
            print(f"dropping {int(invalid.sum())} rows with non-finite values")
        df = df[~invalid]
    df["payload_length"] = df["payload_length"].clip(lower=0, upper=PAYLOAD_MAX).fillna(0)
    df["time_diff"] = df["time_diff"].clip(lower=0).fillna(0)
    df = df[df["time_diff"] < IAT_MAX]
    zero = (df["payload_length"] == 0) & (df["time_diff"] == 0)
    if zero.any():
        if verbose:
            print(f"dropping {int(zero.sum())} rows with zero payload and zero IAT")
        df = df[~zero]
    return df


def atomic_write_csv(frame, path):
    """Write a CSV through a temporary file, so that a half-written file never looks finished."""
    path = Path(path)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    frame.to_csv(tmp, index=False)
    tmp.replace(path)


def create_or_load_split(split_file, flow_ids, flow_sizes, test_size, seed, min_train_share):
    """Return (train_flows, test_flows, draw).

    If split_file exists, the split is read from it. Otherwise it is drawn with the split rule
    (split_rule.constrained_split) and written to split_file, so that all models of the same
    dataset and seed use the same flows.
    """
    split_file = Path(split_file)
    if split_file.exists():
        s = json.loads(split_file.read_text())
        return np.asarray(s["train"]), np.asarray(s["test"]), s.get("draw", 0)
    train, test, draw, _ = constrained_split(flow_ids, flow_sizes, test_size, seed, min_train_share)
    split_file.parent.mkdir(parents=True, exist_ok=True)
    tmp = split_file.with_name(f".{split_file.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps({"train": train.tolist(), "test": test.tolist(), "test_size": test_size,
                               "seed": seed, "draw": draw, "min_train_packet_share": min_train_share}))
    tmp.replace(split_file)
    return train, test, draw


def read_split(split_file):
    """Train and test flow ids of an existing split file."""
    split_file = Path(split_file)
    if not split_file.exists():
        raise FileNotFoundError(f"{split_file} not found: run the proposed model for this dataset and seed first")
    s = json.loads(split_file.read_text())
    return np.asarray(s["train"]), np.asarray(s["test"])


def load_split_checked(df, out_dir):
    """Split of a baseline run. Reads split.json and checks that the cleaned test flows of `df`
    have exactly the packet counts of the shared test.csv."""
    out = Path(out_dir)
    for f in ("split.json", "test.csv"):
        if not (out / f).exists():
            raise FileNotFoundError(f"{out / f} not found: run the proposed model for this dataset and seed first")
    train_flows, test_flows = read_split(out / "split.json")
    ref = pd.read_csv(out / "test.csv", usecols=["flow_id"]).groupby("flow_id").size().sort_index()
    mine = df[df["flow_id"].isin(test_flows)].groupby("flow_id").size().sort_index()
    if not (ref.index.equals(mine.index) and (ref.to_numpy() == mine.to_numpy()).all()):
        raise RuntimeError("the cleaned test flows differ from test.csv: the preprocessing is not the same")
    print(f"split: {len(train_flows)} train / {len(test_flows)} test flows (shared with the proposed model)")
    return train_flows, test_flows


def postprocess_synthetic(synth_df):
    """Post-processing of the baseline outputs: payload clipped to [0, PAYLOAD_MAX] bytes and
    rounded to whole bytes, IAT clipped to [0, IAT_MAX] seconds."""
    out = synth_df[COLUMNS].copy()
    out["payload_length"] = out["payload_length"].clip(0, PAYLOAD_MAX).round()
    out["time_diff"] = out["time_diff"].clip(0, IAT_MAX)
    return out
