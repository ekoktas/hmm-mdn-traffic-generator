#!/usr/bin/env python3
"""
Convert a CSV of the 5G traffic dataset of Choi et al. (Wireshark export with the columns
No., Time, Source, Destination, Protocol, Length, Info) into the packet format of the other
traces: flow_id, payload_length, time_diff, direction.

  payload_length  IP payload: IP total length minus the IP header (20 B for IPv4, 40 B for IPv6).
                  The "Length" column of the export is the IP total length; the script checks
                  this on pure TCP ACKs (40 or 52 B).
  flow            bidirectional 5-tuple: the sorted (address, port) end points and the transport
                  family
  time_diff       gap to the previous packet of the same flow; 0 for the first packet
  direction       1 if the packet goes the same way as the first packet of its flow, else 0

The export shows ports in "Info" only for plain TCP and UDP rows, not for TLS or QUIC records.
A row without ports is assigned to a connection by the first rule that applies:
  1. the next row with ports in the reverse direction between the same hosts within
     --ack-window seconds, i.e. the acknowledgment that answers it
  2. the previous row with ports in the same direction
  3. the next row with ports in the same direction, or the previous one in the reverse direction
  4. port 0 on both sides, i.e. one flow per host pair (e.g. DNS shown without ports)
The share of rows assigned by each rule is printed. --flow-key hostpair gives host-pair flows.

    python scripts/convert_5g_csv.py data/raw/Netflix_1.csv data/5g_netflix.csv --max-hours 4
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd

TCP_LIKE = ("TCP", "TLS", "SSL", "HTTP", "OCSP", "WEBSOCKET")
UDP_LIKE = ("UDP", "QUIC", "GQUIC", "DNS", "MDNS", "SSDP", "NTP", "DTLS", "RTP", "RTCP", "STUN",
            "RAKNET", "DHCP", "LLMNR", "NBNS", "CLASSIC-STUN")
IPV4 = re.compile(r"^\d{1,3}(\.\d{1,3}){3}$")
MAC = re.compile(r"^([0-9a-fA-F]{2}:){5}[0-9a-fA-F]{2}$")
PORTS = r"(\d{1,5})\s+>\s+(\d{1,5})"


def family(proto: str) -> str:
    p = str(proto).upper()
    if p.startswith(TCP_LIKE):
        return "tcp"
    if p.startswith(UDP_LIKE):
        return "udp"
    return p.lower()


def read(path: Path, max_hours: float | None, max_packets: int | None) -> pd.DataFrame:
    cols = ["Time", "Source", "Destination", "Protocol", "Length", "Info"]
    parts, t0, n = [], None, 0
    for chunk in pd.read_csv(path, usecols=cols, dtype=str, chunksize=1_000_000):
        t = pd.to_datetime(chunk["Time"], format="%Y-%m-%d %H:%M:%S.%f")
        if t0 is None:
            t0 = t.iloc[0]
        chunk = chunk.assign(ts=(t - t0).dt.total_seconds())
        stop = False
        if max_hours is not None and (chunk["ts"] > max_hours * 3600).any():
            chunk, stop = chunk[chunk["ts"] <= max_hours * 3600], True
        if max_packets is not None and n + len(chunk) >= max_packets:
            chunk, stop = chunk.iloc[: max_packets - n], True
        parts.append(chunk)
        n += len(chunk)
        if stop:
            break
    return pd.concat(parts, ignore_index=True)


def check_length_is_ip_total(df: pd.DataFrame) -> str:
    info = df["Info"].fillna("")
    acks = df[(df["Protocol"] == "TCP") & info.str.contains(r"\[ACK\] ", regex=True)
              & info.str.contains("Len=0", regex=False) & ~info.str.contains("SYN|FIN|RST", regex=True)]
    if acks.empty:
        return "no pure TCP ACKs found; cannot verify"
    vc = acks["Length"].astype(int).value_counts(normalize=True)
    ok = vc.reindex([40, 52, 60, 72]).fillna(0).sum()
    return (f"pure TCP ACKs: {dict((int(k), round(float(v), 3)) for k, v in vc.head(3).items())} -> "
            f"{ok:.1%} consistent with 'Length = IP total length'")


def recover_ports(df: pd.DataFrame, ack_window: float) -> pd.Series:
    """Fill sport/dport of the rows without ports (rules in the module docstring); return the rule per row."""
    p = df["Info"].fillna("").str.extract(PORTS).astype(float)
    ok = (p[0] <= 65535) & (p[1] <= 65535)
    df["sport"], df["dport"] = p[0].where(ok), p[1].where(ok)
    df["dk"] = df["Source"] + ">" + df["Destination"] + "|" + df["fam"]
    df["rk"] = df["Destination"] + ">" + df["Source"] + "|" + df["fam"]
    how = pd.Series(np.where(df["sport"].notna(), "own ports", ""), index=df.index)
    P = df.loc[df["sport"].notna(), ["ts", "dk", "sport", "dport"]].sort_values("ts")

    def fill(rule, left_by, direction, tol, swap):
        todo = df.index[df["sport"].isna()]
        if len(todo) == 0 or P.empty:
            return
        Q = df.loc[todo, ["ts", left_by]].reset_index().sort_values("ts")
        R = P.rename(columns={"dk": left_by}) if left_by != "dk" else P
        m = pd.merge_asof(Q, R, on="ts", by=left_by, direction=direction, tolerance=tol).set_index("index")
        hit = m["sport"].notna()
        idx = m.index[hit]
        a, b = (m.loc[hit, "dport"], m.loc[hit, "sport"]) if swap else (m.loc[hit, "sport"], m.loc[hit, "dport"])
        df.loc[idx, "sport"], df.loc[idx, "dport"] = a.to_numpy(), b.to_numpy()
        how.loc[idx] = rule

    fill("next reverse ACK", "rk", "forward", ack_window, swap=True)
    fill("previous same direction", "dk", "backward", None, swap=False)
    fill("next same direction", "dk", "forward", None, swap=False)
    fill("previous reverse", "rk", "backward", None, swap=True)
    rest = df["sport"].isna()
    df.loc[rest, ["sport", "dport"]] = 0.0
    how.loc[rest] = "host pair (no ports)"
    return how


def convert(df: pd.DataFrame, flow_key: str, idle_timeout: float | None, ack_window: float):
    src, dst = df["Source"].astype(str), df["Destination"].astype(str)
    is_v4 = src.str.match(IPV4) & dst.str.match(IPV4)
    is_v6 = src.str.contains(":") & dst.str.contains(":") & ~src.str.match(MAC) & ~dst.str.match(MAC)
    df = df[is_v4 | is_v6].copy()
    v4 = df["Source"].str.match(IPV4)
    df["payload_length"] = (df["Length"].astype(int) - np.where(v4, 20, 40)).clip(lower=0)
    df["fam"] = df["Protocol"].map(family)

    if flow_key == "conn":
        how = recover_ports(df, ack_window)
        a = df["Source"] + ":" + df["sport"].astype(int).astype(str)
        b = df["Destination"] + ":" + df["dport"].astype(int).astype(str)
    else:
        how = pd.Series("host pair", index=df.index)
        a, b = df["Source"], df["Destination"]
    lo, hi = np.where(a <= b, a, b), np.where(a <= b, b, a)
    df["pair"] = pd.Series(lo, index=df.index) + "|" + pd.Series(hi, index=df.index) + "|" + df["fam"]

    df = df.sort_values(["pair", "ts"], kind="stable")
    if idle_timeout is not None:
        gap = df.groupby("pair")["ts"].diff()
        df["segment"] = (gap > idle_timeout).groupby(df["pair"]).cumsum()
    else:
        df["segment"] = 0
    key = df["pair"] + "#" + df["segment"].astype(str)
    first_ts = df.groupby(key)["ts"].transform("min")
    order = pd.DataFrame({"key": key, "t": first_ts}).drop_duplicates("key").sort_values("t", kind="stable")
    ids = pd.Series(np.arange(1, len(order) + 1), index=order["key"].to_numpy())
    df["flow_id"] = key.map(ids).to_numpy()
    initiator = df.groupby("flow_id")["Source"].transform("first")
    df["direction"] = (df["Source"] == initiator).astype(int)
    df = df.sort_values(["flow_id", "ts"], kind="stable")
    df["time_diff"] = df.groupby("flow_id")["ts"].diff().fillna(0.0)
    return df, how


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("csv_in", type=Path)
    ap.add_argument("csv_out", type=Path)
    ap.add_argument("--max-hours", type=float, default=None, help="use only the first H hours of the capture")
    ap.add_argument("--max-packets", type=int, default=None, help="use only the first N packets")
    ap.add_argument("--flow-key", choices=["conn", "hostpair"], default="conn",
                    help="conn: 5-tuple with recovered ports (default); hostpair: address pair only")
    ap.add_argument("--ack-window", type=float, default=2.0, help="seconds to look ahead for the answering ACK")
    ap.add_argument("--idle-timeout", type=float, default=None,
                    help="split a flow after this many idle seconds (default: none for conn, 3600 for hostpair)")
    args = ap.parse_args()
    idle = args.idle_timeout if args.idle_timeout is not None else (None if args.flow_key == "conn" else 3600.0)

    raw = read(args.csv_in, args.max_hours, args.max_packets)
    print(f"read {len(raw):,} packets covering {raw['ts'].max() / 3600:.2f} h")
    print(check_length_is_ip_total(raw))
    full, how = convert(raw, args.flow_key, idle, args.ack_window)
    out = full[["flow_id", "payload_length", "time_diff", "direction"]].reset_index(drop=True)
    args.csv_out.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.csv_out, index=False)

    print("connection assignment: " + ", ".join(f"{k} {v:.1%}" for k, v in how.value_counts(normalize=True).items()))
    sizes = out.groupby("flow_id").size().sort_values(ascending=False)
    print(f"wrote {args.csv_out}: {out['flow_id'].nunique()} flows, {len(out):,} packets, "
          f"{len(out) / out['flow_id'].nunique():.1f} packets per flow, "
          f"{(sizes <= 10).mean():.1%} of flows with <= 10 packets")
    print("largest flows hold " + ", ".join(f"{s / len(out):.1%}" for s in sizes.head(3)) + " of the packets")


if __name__ == "__main__":
    main()
