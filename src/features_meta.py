"""Feature metadata and category mappings for KDD Cup 1999 and CIC-IDS-2017."""

from __future__ import annotations

# KDD Cup 1999 — 41 connection-level features (Stolfo et al.)
KDD_FEATURE_GROUPS: dict[str, list[str]] = {
    "basic_connection": [
        "duration",
        "protocol_type",
        "service",
        "flag",
        "src_bytes",
        "dst_bytes",
        "land",
        "wrong_fragment",
        "urgent",
    ],
    "content": [
        "hot",
        "num_failed_logins",
        "logged_in",
        "num_compromised",
        "root_shell",
        "su_attempted",
        "num_root",
        "num_file_creations",
        "num_shells",
        "num_access_files",
        "num_outbound_cmds",
        "is_host_login",
        "is_guest_login",
    ],
    "time_traffic_statistics": [
        "count",
        "srv_count",
        "serror_rate",
        "srv_serror_rate",
        "rerror_rate",
        "srv_rerror_rate",
        "same_srv_rate",
        "diff_srv_rate",
        "srv_diff_host_rate",
    ],
    "host_traffic_statistics": [
        "dst_host_count",
        "dst_host_srv_count",
        "dst_host_same_srv_rate",
        "dst_host_diff_srv_rate",
        "dst_host_same_src_port_rate",
        "dst_host_srv_diff_host_rate",
        "dst_host_serror_rate",
        "dst_host_srv_serror_rate",
        "dst_host_rerror_rate",
        "dst_host_srv_rerror_rate",
    ],
}

KDD_ATTACK_TO_CATEGORY: dict[str, str] = {
    "normal.": "normal",
    "normal": "normal",
    "back": "DoS",
    "land": "DoS",
    "neptune": "DoS",
    "pod": "DoS",
    "smurf": "DoS",
    "teardrop": "DoS",
    "mailbomb": "DoS",
    "apache2": "DoS",
    "processtable": "DoS",
    "udpstorm": "DoS",
    "ipsweep": "Probe",
    "mscan": "Probe",
    "nmap": "Probe",
    "portsweep": "Probe",
    "saint": "Probe",
    "satan": "Probe",
    "ftp_write": "R2L",
    "guess_passwd": "R2L",
    "imap": "R2L",
    "multihop": "R2L",
    "named": "R2L",
    "phf": "R2L",
    "sendmail": "R2L",
    "snmpgetattack": "R2L",
    "snmpguess": "R2L",
    "spy": "R2L",
    "warezclient": "R2L",
    "warezmaster": "R2L",
    "worm": "R2L",
    "xlock": "R2L",
    "xsnoop": "R2L",
    "xterm": "R2L",
    "buffer_overflow": "U2R",
    "loadmodule": "U2R",
    "perl": "U2R",
    "ps": "U2R",
    "rootkit": "U2R",
    "sqlattack": "U2R",
    "httptunnel": "U2R",
}

KDD_CATEGORIES = ["normal", "DoS", "Probe", "R2L", "U2R"]

# CIC-IDS-2017 — 78 flow features + Label (Sharafaldin et al.)
CIC_FEATURE_GROUPS: dict[str, list[str]] = {
    "flow_identification": [
        " Destination Port",
        " Flow Duration",
        " Total Fwd Packets",
        " Total Backward Packets",
        "Total Length of Fwd Packets",
        " Total Length of Bwd Packets",
        " Fwd Packet Length Max",
        " Fwd Packet Length Min",
        " Fwd Packet Length Mean",
        " Fwd Packet Length Std",
        "Bwd Packet Length Max",
        " Bwd Packet Length Min",
        " Bwd Packet Length Mean",
        " Bwd Packet Length Std",
    ],
    "packet_timing_iat": [
        " Flow IAT Mean",
        " Flow IAT Std",
        " Flow IAT Max",
        " Flow IAT Min",
        "Fwd IAT Total",
        " Fwd IAT Mean",
        " Fwd IAT Std",
        " Fwd IAT Max",
        " Fwd IAT Min",
        "Bwd IAT Total",
        " Bwd IAT Mean",
        " Bwd IAT Std",
        " Bwd IAT Max",
        " Bwd IAT Min",
    ],
    "flags_window": [
        " Fwd PSH Flags",
        " Bwd PSH Flags",
        " Fwd URG Flags",
        " Bwd URG Flags",
        " Fwd Header Length",
        " Bwd Header Length",
        " Fwd Packets/s",
        " Bwd Packets/s",
        " Min Packet Length",
        " Max Packet Length",
        " Packet Length Mean",
        " Packet Length Std",
        " Packet Length Variance",
        "FIN Flag Count",
        " SYN Flag Count",
        " RST Flag Count",
        " PSH Flag Count",
        " ACK Flag Count",
        " URG Flag Count",
        " CWE Flag Count",
        " ECE Flag Count",
        " Down/Up Ratio",
        " Average Packet Size",
        " Avg Fwd Segment Size",
        " Avg Bwd Segment Size",
        " Fwd Header Length.1",
        "Init_Win_bytes_forward",
        " Init_Win_bytes_backward",
        " act_data_pkt_fwd",
        " min_seg_size_forward",
        "Active Mean",
        " Active Std",
        " Active Max",
        " Active Min",
        "Idle Mean",
        " Idle Std",
        " Idle Max",
        " Idle Min",
    ],
    "bulk_subflow": [
        " Fwd Header Length",
        " Subflow Fwd Packets",
        " Subflow Fwd Bytes",
        " Subflow Bwd Packets",
        " Subflow Bwd Bytes",
        " Fwd Avg Bytes/Bulk",
        " Fwd Avg Packets/Bulk",
        " Fwd Avg Bulk Rate",
        " Bwd Avg Bytes/Bulk",
        " Bwd Avg Packets/Bulk",
        " Bwd Avg Bulk Rate",
    ],
}

# Era-aware hierarchical layers (innovation module)
ERA_AWARE_LAYERS_KDD: dict[str, list[str]] = {
    "connection": ["duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes"],
    "statistics": [
        "count", "srv_count", "serror_rate", "srv_serror_rate",
        "same_srv_rate", "diff_srv_rate", "dst_host_count", "dst_host_srv_count",
    ],
    "content": ["hot", "num_failed_logins", "logged_in", "root_shell", "num_compromised"],
    "timing": [
        "dst_host_same_srv_rate", "dst_host_serror_rate", "dst_host_rerror_rate",
    ],
}

ERA_AWARE_LAYERS_CIC: dict[str, list[str]] = {
    "connection": [" Destination Port", " Flow Duration", " Total Fwd Packets", " Total Backward Packets"],
    "statistics": [
        " Total Length of Fwd Packets", " Total Length of Bwd Packets",
        " Fwd Packets/s", " Bwd Packets/s", " Packet Length Mean",
    ],
    "content": [
        "FIN Flag Count", " SYN Flag Count", " RST Flag Count", " ACK Flag Count",
        " Fwd PSH Flags", " Bwd PSH Flags",
    ],
    "timing": [
        " Flow IAT Mean", " Flow IAT Std", " Flow IAT Max", " Flow IAT Min",
        " Active Mean", " Idle Mean",
    ],
}

# Cross-dataset semantic alignment (transfer experiment)
CROSS_DATASET_ALIGNED: list[tuple[str, str]] = [
    ("duration", " Flow Duration"),
    ("src_bytes", " Total Length of Fwd Packets"),
    ("dst_bytes", " Total Length of Bwd Packets"),
    ("count", " Total Fwd Packets"),
    ("srv_count", " Total Backward Packets"),
]


def kdd_feature_table_markdown() -> str:
    """Generate markdown table of KDD feature groups."""
    lines = ["| 分组 | 特征数 | 代表特征 |", "|------|--------|----------|"]
    for group, feats in KDD_FEATURE_GROUPS.items():
        lines.append(f"| {group} | {len(feats)} | `{', '.join(feats[:4])}...` |")
    return "\n".join(lines)


def cic_feature_table_markdown() -> str:
    """Generate markdown table of CIC feature groups."""
    lines = ["| 分组 | 特征数 | 代表特征 |", "|------|--------|----------|"]
    for group, feats in CIC_FEATURE_GROUPS.items():
        sample = feats[0].strip() if feats else ""
        lines.append(f"| {group} | {len(feats)} | `{sample}` 等 |")
    return "\n".join(lines)
