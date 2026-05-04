"""NSL-KDD dataset loader and preprocessor."""

import os
import urllib.request
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")

# NSL-KDD column names
COLUMNS = [
    "duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes",
    "land", "wrong_fragment", "urgent", "hot", "num_failed_logins", "logged_in",
    "num_compromised", "root_shell", "su_attempted", "num_root",
    "num_file_creations", "num_shells", "num_access_files", "num_outbound_cmds",
    "is_host_login", "is_guest_login", "count", "srv_count", "serror_rate",
    "srv_serror_rate", "rerror_rate", "srv_rerror_rate", "same_srv_rate",
    "diff_srv_rate", "srv_diff_host_rate", "dst_host_count", "dst_host_srv_count",
    "dst_host_same_srv_rate", "dst_host_diff_srv_rate",
    "dst_host_same_src_port_rate", "dst_host_srv_diff_host_rate",
    "dst_host_serror_rate", "dst_host_srv_serror_rate", "dst_host_rerror_rate",
    "dst_host_srv_rerror_rate", "label", "difficulty"
]

CATEGORICAL_COLS = ["protocol_type", "service", "flag"]

# Attack type to category mapping
ATTACK_MAP = {
    "normal": "normal",
    # DoS
    "back": "DoS", "land": "DoS", "neptune": "DoS", "pod": "DoS",
    "smurf": "DoS", "teardrop": "DoS", "apache2": "DoS", "udpstorm": "DoS",
    "processtable": "DoS", "mailbomb": "DoS",
    # Probe
    "satan": "Probe", "ipsweep": "Probe", "nmap": "Probe", "portsweep": "Probe",
    "mscan": "Probe", "saint": "Probe",
    # R2L
    "guess_passwd": "R2L", "ftp_write": "R2L", "imap": "R2L", "phf": "R2L",
    "multihop": "R2L", "warezmaster": "R2L", "warezclient": "R2L", "spy": "R2L",
    "xlock": "R2L", "xsnoop": "R2L", "snmpguess": "R2L", "snmpgetattack": "R2L",
    "httptunnel": "R2L", "sendmail": "R2L", "named": "R2L", "worm": "R2L",
    # U2R
    "buffer_overflow": "U2R", "loadmodule": "U2R", "rootkit": "U2R", "perl": "U2R",
    "sqlattack": "U2R", "xterm": "U2R", "ps": "U2R",
}

CATEGORY_LABELS = ["normal", "DoS", "Probe", "R2L", "U2R"]


def download_nslkdd():
    """Download NSL-KDD dataset if not present."""
    os.makedirs(DATA_DIR, exist_ok=True)
    base_url = "https://raw.githubusercontent.com/defcom17/NSL_KDD/master/"
    files = {
        "KDDTrain+.txt": base_url + "KDDTrain+.txt",
        "KDDTest+.txt": base_url + "KDDTest+.txt",
    }
    for fname, url in files.items():
        fpath = os.path.join(DATA_DIR, fname)
        if not os.path.exists(fpath):
            print(f"Downloading {fname}...")
            urllib.request.urlretrieve(url, fpath)
            print(f"  Saved to {fpath}")
        else:
            print(f"  {fname} already exists")
    return files


def load_nslkdd(binary=False):
    """Load and preprocess NSL-KDD dataset.

    Args:
        binary: If True, use binary labels (normal vs attack).

    Returns:
        X_train, X_test, y_train, y_test, feature_names, label_names, scaler, raw_test_df
    """
    download_nslkdd()

    train_path = os.path.join(DATA_DIR, "KDDTrain+.txt")
    test_path = os.path.join(DATA_DIR, "KDDTest+.txt")

    df_train = pd.read_csv(train_path, header=None, names=COLUMNS)
    df_test = pd.read_csv(test_path, header=None, names=COLUMNS)

    # Map labels to categories
    df_train["category"] = df_train["label"].map(ATTACK_MAP).fillna("unknown")
    df_test["category"] = df_test["label"].map(ATTACK_MAP).fillna("unknown")

    # Remove unknown categories
    df_train = df_train[df_train["category"] != "unknown"]
    df_test = df_test[df_test["category"] != "unknown"]

    if binary:
        df_train["category"] = df_train["category"].apply(
            lambda x: "normal" if x == "normal" else "attack"
        )
        df_test["category"] = df_test["category"].apply(
            lambda x: "normal" if x == "normal" else "attack"
        )
        label_names = ["normal", "attack"]
    else:
        label_names = CATEGORY_LABELS

    # Encode categorical features
    label_encoders = {}
    for col in CATEGORICAL_COLS:
        le = LabelEncoder()
        combined = pd.concat([df_train[col], df_test[col]])
        le.fit(combined)
        df_train[col] = le.transform(df_train[col])
        df_test[col] = le.transform(df_test[col])
        label_encoders[col] = le

    # Encode target with fixed mapping (avoid LabelEncoder alphabetical sort)
    label_to_idx = {name: i for i, name in enumerate(label_names)}
    y_train = df_train["category"].map(label_to_idx).values
    y_test = df_test["category"].map(label_to_idx).values

    # Features
    feature_cols = [c for c in COLUMNS if c not in ["label", "difficulty"]]
    X_train = df_train[feature_cols].values.astype(np.float32)
    X_test = df_test[feature_cols].values.astype(np.float32)

    # Save raw test df for LLM text conversion (before scaling)
    raw_test_df = df_test.copy()

    # Normalize
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    print(f"Train: {X_train.shape}, Test: {X_test.shape}")
    print(f"Train label distribution: {dict(zip(*np.unique(y_train, return_counts=True)))}")
    print(f"Test label distribution: {dict(zip(*np.unique(y_test, return_counts=True)))}")

    return X_train, X_test, y_train, y_test, feature_cols, label_names, scaler, raw_test_df


def get_text_representation(row, feature_cols):
    """Convert a network record to natural language text for LLM classification."""
    parts = []
    parts.append(f"Duration: {row['duration']}s")
    parts.append(f"Protocol: {row['protocol_type']}")
    parts.append(f"Service: {row['service']}")
    parts.append(f"Flag: {row['flag']}")
    parts.append(f"Source bytes: {row['src_bytes']}")
    parts.append(f"Destination bytes: {row['dst_bytes']}")
    parts.append(f"Land: {row['land']}")
    parts.append(f"Wrong fragments: {row['wrong_fragment']}")
    parts.append(f"Urgent packets: {row['urgent']}")
    parts.append(f"Hot indicators: {row['hot']}")
    parts.append(f"Failed logins: {row['num_failed_logins']}")
    parts.append(f"Logged in: {'yes' if row['logged_in'] else 'no'}")
    parts.append(f"Compromised conditions: {row['num_compromised']}")
    parts.append(f"Root shell: {'yes' if row['root_shell'] else 'no'}")
    parts.append(f"Su attempted: {row['su_attempted']}")
    parts.append(f"Num root accesses: {row['num_root']}")
    parts.append(f"File creations: {row['num_file_creations']}")
    parts.append(f"Connections to same host (last 2s): {row['count']}")
    parts.append(f"Connections to same service (last 2s): {row['srv_count']}")
    parts.append(f"SYN error rate: {row['serror_rate']:.2f}")
    parts.append(f"Same service rate: {row['same_srv_rate']:.2f}")
    parts.append(f"Dst host count: {row['dst_host_count']}")
    parts.append(f"Dst host same srv rate: {row['dst_host_same_srv_rate']:.2f}")
    parts.append(f"Dst host serror rate: {row['dst_host_serror_rate']:.2f}")
    return "; ".join(parts)


if __name__ == "__main__":
    X_train, X_test, y_train, y_test, feat, labels, scaler, raw = load_nslkdd()
    print(f"\nFeatures: {len(feat)}")
    print(f"Labels: {labels}")
