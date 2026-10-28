# Data

The pipeline reads one CSV file per trace, with one row per packet and the columns

| Column | Meaning |
|---|---|
| `flow_id` | id of the flow |
| `payload_length` | IP payload in bytes: IP total length minus the IP header, so the transport header is included |
| `time_diff` | time since the previous packet of the same flow in seconds, 0 for the first packet |

A flow is a bidirectional 5-tuple (the two end points with their ports, and the transport protocol). The packets of a flow must be in temporal order; flows may be interleaved, since `scripts/data_utils.py` sorts the rows by `flow_id` with a stable sort. Further columns are ignored.

## Files and sources

| File | Trace | Source |
|---|---|---|
| `google_home_udp.csv` | UDP traffic of a Google Home device | `data/raw/df_raw_UDP_GOOGLE_HOME.csv` of the NeCSTGen repository, https://github.com/fmeslet/NeCSTGen (MIT license) |
| `facebook_audio.csv` | `facebook_audio2a.pcap` | ISCX VPN-nonVPN dataset (Draper-Gil et al., ICISSP 2016), https://www.unb.ca/cic/datasets/vpn.html |
| `facebook_video.csv` | `facebook_video1a.pcap` | ISCX VPN-nonVPN dataset |
| `ftps.csv` | `ftps_down_1a.pcap` | ISCX VPN-nonVPN dataset |
| `5g_netflix.csv` | `Netflix_1.csv`, first 4 hours | 5G Traffic Datasets of Choi et al. (IEEE Communications Magazine, 2023), https://ieee-dataport.org/documents/5g-traffic-datasets |

## Preparation

Run the commands from the repository root, with the downloaded files in `data/raw/`.

**Google Home.** The file of the NeCSTGen repository already contains the three columns, among others, and is used as it is:

```bash
curl -L -o data/google_home_udp.csv https://raw.githubusercontent.com/fmeslet/NeCSTGen/main/data/raw/df_raw_UDP_GOOGLE_HOME.csv
```

**ISCX traces.** Export the packet fields with tshark, keeping only the first occurrence of every field, and build the flows. The traces of the article were exported with the tshark of the `nicolaka/netshoot` container:

```bash
docker run --rm -v "$(pwd)/data/raw:/data" nicolaka/netshoot \
  tshark -r /data/facebook_video1a.pcap -T fields -E header=y -E separator=, -E quote=d -E occurrence=f \
  -e frame.time_epoch -e ip.len -e ip.hdr_len -e ipv6.plen -e tcp.srcport -e tcp.dstport \
  -e udp.srcport -e udp.dstport -e ip.proto -e ipv6.nxt -e ip.src -e ip.dst -e ipv6.src -e ipv6.dst \
  > data/raw/facebook_video1a_raw.csv
python scripts/tshark_to_flows.py data/raw/facebook_video1a_raw.csv data/facebook_video.csv
```

A local Wireshark installation runs the same `tshark` command without the `docker run` prefix. `facebook_audio2a.pcap` becomes `facebook_audio.csv` and `ftps_down_1a.pcap` becomes `ftps.csv` in the same way. Packets without an IP layer are dropped.

**5G trace.** The trace is a Wireshark CSV export in which TLS records show no ports. `scripts/convert_5g_csv.py` recovers the connections and keeps the first 4 hours:

```bash
python scripts/convert_5g_csv.py data/raw/Netflix_1.csv data/5g_netflix.csv --max-hours 4
```

## Check

`python scripts/check_data.py` prints the statistics of Table 2 for every file. The files of the article give

| File | Flows | Rows (packets) |
|---|---|---|
| `google_home_udp.csv` | 588 | 10,000 |
| `facebook_audio.csv` | 250 | 91,737 |
| `facebook_video.csv` | 173 | 156,982 |
| `ftps.csv` | 585 | 5,743,493 |
| `5g_netflix.csv` | 115 | 677,845 |

The cleaning in `data_utils.load_packets` then drops, among others, the rows with zero payload and zero IAT; Table 2 of the article lists 5,743,492 packets for FTPS after this step.
