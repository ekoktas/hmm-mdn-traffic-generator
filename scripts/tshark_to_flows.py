#!/usr/bin/env python3
"""
Turn a tshark field export of a pcap file into the packet format of the pipeline:
flow_id, payload_length, time_diff, direction.

The export must contain the fields frame.time_epoch, ip.len, ip.hdr_len, ipv6.plen, tcp.srcport,
tcp.dstport, udp.srcport, udp.dstport, ip.proto, ipv6.nxt, ip.src, ip.dst, ipv6.src and ipv6.dst, with
one value per field (tshark -E occurrence=f; the full command is in data/README.md).

  flow            bidirectional 5-tuple: the sorted (address, port) end points and the IP protocol;
                  flow ids are numbered in the order in which the flows first appear
  payload_length  IP payload: ip.len - ip.hdr_len for IPv4, ipv6.plen for IPv6
  time_diff       gap to the previous packet of the same flow in seconds; 0 for the first packet
  direction       1 if the packet goes the same way as the first packet of its flow, else 0

Packets without an IPv4 or IPv6 source address are dropped.

    python scripts/tshark_to_flows.py data/raw/facebook_video1a_raw.csv data/facebook_video.csv
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

NUM_COLS = ["frame.time_epoch", "ip.len", "ip.hdr_len", "ipv6.plen", "tcp.srcport", "tcp.dstport",
            "udp.srcport", "udp.dstport", "ip.proto", "ipv6.nxt"]
ADDR_COLS = ["ip.src", "ip.dst", "ipv6.src", "ipv6.dst"]


def pick_port(row, *cols):
    for c in cols:
        v = row[c]
        if not np.isnan(v):
            return int(v)
    return 0


def ip_payload(r):
    if not np.isnan(r["ip.len"]) and not np.isnan(r["ip.hdr_len"]):
        return int(r["ip.len"] - r["ip.hdr_len"])
    if not np.isnan(r["ipv6.plen"]):
        return int(r["ipv6.plen"])
    return 0


def flow_key(r):
    sa = r["ip.src"] if pd.notna(r["ip.src"]) else r["ipv6.src"]
    da = r["ip.dst"] if pd.notna(r["ip.dst"]) else r["ipv6.dst"]
    if not np.isnan(r["ip.proto"]):
        proto = int(r["ip.proto"])
    elif not np.isnan(r["ipv6.nxt"]):
        proto = int(r["ipv6.nxt"])
    else:
        proto = 17
    sp = pick_port(r, "tcp.srcport", "udp.srcport")
    dp = pick_port(r, "tcp.dstport", "udp.dstport")
    return (*sorted(((sa, sp), (da, dp))), proto)


def convert(csv_in: Path, csv_out: Path):
    df = pd.read_csv(csv_in, dtype=str)
    for c in NUM_COLS:
        if c in df:
            # with -E occurrence=f every field holds a single value; commas (thousands separators) are removed
            df[c] = df[c].str.replace(",", "", regex=False).astype(float)
        else:
            df[c] = np.nan
    for c in ADDR_COLS:
        df[c] = df[c].replace("", np.nan) if c in df else np.nan

    flow_map, initiators, records = {}, {}, []
    df = df.dropna(subset=["ip.src", "ipv6.src"], how="all")
    for row in df.to_dict(orient="records"):
        key = flow_key(row)
        src = row["ip.src"] if pd.notna(row["ip.src"]) else row["ipv6.src"]
        if key not in flow_map:
            flow_map[key] = len(flow_map) + 1
            initiators[key] = src
        records.append((flow_map[key], ip_payload(row), row["frame.time_epoch"], 1 if src == initiators[key] else 0))

    flows = pd.DataFrame(records, columns=["flow_id", "payload_length", "ts", "direction"])
    flows.sort_values(["flow_id", "ts"], inplace=True)
    flows["time_diff"] = flows.groupby("flow_id")["ts"].diff().fillna(0)
    csv_out.parent.mkdir(parents=True, exist_ok=True)
    flows[["flow_id", "payload_length", "time_diff", "direction"]].to_csv(csv_out, index=False)
    print(f"wrote {csv_out} ({flows.flow_id.nunique()} flows, {len(flows)} packets)")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("csv_in", type=Path, help="tshark field export (CSV with a header row)")
    ap.add_argument("csv_out", type=Path)
    args = ap.parse_args()
    convert(args.csv_in, args.csv_out)


if __name__ == "__main__":
    main()
