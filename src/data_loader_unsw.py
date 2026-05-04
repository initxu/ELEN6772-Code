"""UNSW-NB15 dataset loader and preprocessor."""

import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")

CATEGORICAL_COLS = ["proto", "service", "state"]

CATEGORY_LABELS = [
    "Normal", "Generic", "Exploits", "Fuzzers", "DoS",
    "Reconnaissance", "Analysis", "Backdoor", "Shellcode", "Worms"
]

# Features to drop (not useful for classification)
DROP_COLS = ["id", "attack_cat", "label"]


def load_unsw_nb15():
    """Load and preprocess UNSW-NB15 dataset.

    Returns:
        X_train, X_test, y_train, y_test, feature_names, label_names, scaler, raw_test_df
    """
    train_path = os.path.join(DATA_DIR, "UNSW_NB15_training-set.csv")
    test_path = os.path.join(DATA_DIR, "UNSW_NB15_testing-set.csv")

    df_train = pd.read_csv(train_path)
    df_test = pd.read_csv(test_path)

    # Clean attack_cat: strip whitespace, fix empty strings
    df_train["attack_cat"] = df_train["attack_cat"].fillna("Normal").str.strip()
    df_test["attack_cat"] = df_test["attack_cat"].fillna("Normal").str.strip()

    # Fix empty attack_cat entries
    df_train.loc[df_train["attack_cat"] == "", "attack_cat"] = "Normal"
    df_test.loc[df_test["attack_cat"] == "", "attack_cat"] = "Normal"

    # Map " Backdoors" to "Backdoor" etc.
    name_fixes = {"Backdoors": "Backdoor", " Fuzzers": "Fuzzers", " Reconnaissance": "Reconnaissance"}
    df_train["attack_cat"] = df_train["attack_cat"].replace(name_fixes)
    df_test["attack_cat"] = df_test["attack_cat"].replace(name_fixes)

    label_names = CATEGORY_LABELS

    # Encode categorical features
    for col in CATEGORICAL_COLS:
        le = LabelEncoder()
        combined = pd.concat([df_train[col].astype(str), df_test[col].astype(str)])
        le.fit(combined)
        df_train[col] = le.transform(df_train[col].astype(str))
        df_test[col] = le.transform(df_test[col].astype(str))

    # Encode target with fixed mapping
    label_to_idx = {name: i for i, name in enumerate(label_names)}
    y_train = df_train["attack_cat"].map(label_to_idx).values
    y_test = df_test["attack_cat"].map(label_to_idx).values

    # Handle any unmapped labels
    mask_train = ~np.isnan(y_train.astype(float))
    mask_test = ~np.isnan(y_test.astype(float))
    df_train = df_train[mask_train]
    df_test = df_test[mask_test]
    y_train = y_train[mask_train].astype(int)
    y_test = y_test[mask_test].astype(int)

    # Save raw test df for LLM
    raw_test_df = df_test.copy()

    # Features
    feature_cols = [c for c in df_train.columns if c not in DROP_COLS]
    X_train = df_train[feature_cols].values.astype(np.float32)
    X_test = df_test[feature_cols].values.astype(np.float32)

    # Handle NaN/inf
    X_train = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)
    X_test = np.nan_to_num(X_test, nan=0.0, posinf=0.0, neginf=0.0)

    # Normalize
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    print(f"UNSW-NB15 Train: {X_train.shape}, Test: {X_test.shape}")
    print(f"Train label distribution: {dict(zip(*np.unique(y_train, return_counts=True)))}")
    print(f"Test label distribution: {dict(zip(*np.unique(y_test, return_counts=True)))}")

    return X_train, X_test, y_train, y_test, feature_cols, label_names, scaler, raw_test_df


def format_unsw_record(row):
    """Convert an UNSW-NB15 record to text for LLM classification."""
    parts = [
        f"duration={row['dur']}",
        f"protocol={row['proto']}",
        f"service={row['service']}",
        f"state={row['state']}",
        f"src_packets={row['spkts']}",
        f"dst_packets={row['dpkts']}",
        f"src_bytes={row['sbytes']}",
        f"dst_bytes={row['dbytes']}",
        f"rate={row['rate']}",
        f"src_ttl={row['sttl']}",
        f"dst_ttl={row['dttl']}",
        f"src_load={row['sload']}",
        f"dst_load={row['dload']}",
        f"src_loss={row['sloss']}",
        f"dst_loss={row['dloss']}",
        f"src_interpacket_time={row['sinpkt']}",
        f"dst_interpacket_time={row['dinpkt']}",
        f"src_jitter={row['sjit']}",
        f"dst_jitter={row['djit']}",
        f"tcp_rtt={row['tcprtt']}",
        f"syn_ack_time={row['synack']}",
        f"src_mean_pkt_size={row['smean']}",
        f"dst_mean_pkt_size={row['dmean']}",
        f"ct_srv_src={row['ct_srv_src']}",
        f"ct_state_ttl={row['ct_state_ttl']}",
        f"ct_dst_ltm={row['ct_dst_ltm']}",
        f"ct_src_dport_ltm={row['ct_src_dport_ltm']}",
        f"ct_dst_sport_ltm={row['ct_dst_sport_ltm']}",
        f"is_ftp_login={row['is_ftp_login']}",
        f"is_sm_ips_ports={row['is_sm_ips_ports']}",
    ]
    return ", ".join(parts)


if __name__ == "__main__":
    X_train, X_test, y_train, y_test, feat, labels, scaler, raw = load_unsw_nb15()
    print(f"\nFeatures: {len(feat)}")
    print(f"Labels: {labels}")
