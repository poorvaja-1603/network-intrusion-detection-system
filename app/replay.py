"""Replay KDDTest+ records through the running API."""
import argparse
import time
from pathlib import Path

import pandas as pd
import requests

from src.data import ALL_COLUMNS, CATEGORICAL_COLUMNS, FEATURE_COLUMNS


def load_flows(path: Path, attack_type: str | None):
    frame = pd.read_csv(path, header=None, names=ALL_COLUMNS)
    frame["attack_type"] = frame["attack_type"].astype(str).str.strip().str.lower()
    if attack_type:
        requested = attack_type.strip().lower()
        frame = frame.loc[frame["attack_type"] == requested]
        if frame.empty:
            raise ValueError(f"No rows found for attack type {requested!r}")
    for column in CATEGORICAL_COLUMNS:
        frame[column] = frame[column].astype(str).str.strip()
    return frame[FEATURE_COLUMNS]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/KDDTest+.txt"))
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--rate", type=float, default=4.0, help="Batches sent per second; 0 disables pacing")
    parser.add_argument("--attack-type", help="Only replay this raw attack name, e.g. neptune or normal")
    parser.add_argument("--limit", type=int, help="Maximum number of flows to replay")
    args = parser.parse_args()
    if args.batch_size < 1 or args.rate < 0:
        parser.error("--batch-size must be positive and --rate cannot be negative")
    flows = load_flows(args.data, args.attack_type)
    if args.limit:
        flows = flows.iloc[:args.limit]
    sent = 0
    for offset in range(0, len(flows), args.batch_size):
        batch = flows.iloc[offset:offset + args.batch_size].to_dict(orient="records")
        response = requests.post(f"{args.url.rstrip('/')}/predict", json={"flows": batch}, timeout=60)
        response.raise_for_status()
        result = response.json()
        sent += len(batch)
        print(f"Sent {sent}/{len(flows)} flows | API latency {result['latency_ms']:.2f} ms")
        if args.rate:
            time.sleep(1.0 / args.rate)
    print(f"Replay complete: {sent} flows")


if __name__ == "__main__":
    main()

