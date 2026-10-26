"""Replay KDDTest+ records through the running API with customizable speed and volume."""
import argparse
import time
from pathlib import Path
from typing import Callable, Any

import pandas as pd
import requests

from src.data import ALL_COLUMNS, CATEGORICAL_COLUMNS, FEATURE_COLUMNS
from src.attack_map import ATTACK_MAP, TEST_ONLY_ATTACKS

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = ROOT / "data" / "KDDTest+.txt"


def load_flows(
    path: Path = DEFAULT_DATA,
    attack_category: str | None = None,
    attack_type: str | None = None,
    zero_day_only: bool = False,
    attacks_only: bool = False,
    normal_only: bool = False,
) -> pd.DataFrame:
    """Load and filter KDDTest+ flows based on specified criteria."""
    if not path.is_file():
        raise FileNotFoundError(f"Dataset not found at {path}")

    frame = pd.read_csv(path, header=None, names=ALL_COLUMNS)
    frame["attack_type"] = frame["attack_type"].astype(str).str.strip().str.lower()
    frame["category"] = frame["attack_type"].map(lambda a: ATTACK_MAP.get(a, "unknown"))

    # Apply filters
    if zero_day_only:
        novel_names = set(TEST_ONLY_ATTACKS.keys())
        frame = frame.loc[frame["attack_type"].isin(novel_names)]
    elif normal_only:
        frame = frame.loc[frame["category"] == "normal"]
    elif attacks_only:
        frame = frame.loc[frame["category"] != "normal"]
    elif attack_type:
        requested = attack_type.strip().lower()
        frame = frame.loc[frame["attack_type"] == requested]
    elif attack_category and attack_category.lower() != "all":
        requested_cat = attack_category.strip().lower()
        frame = frame.loc[frame["category"] == requested_cat]

    if frame.empty:
        raise ValueError("No flows found matching the specified filter criteria.")

    for column in CATEGORICAL_COLUMNS:
        frame[column] = frame[column].astype(str).str.strip()

    return frame


def stream_replay(
    url: str = "http://127.0.0.1:8000",
    num_flows: int = 200,
    duration_seconds: float = 10.0,
    attack_category: str | None = None,
    attack_type: str | None = None,
    zero_day_only: bool = False,
    attacks_only: bool = False,
    normal_only: bool = False,
    data_path: Path = DEFAULT_DATA,
    progress_callback: Callable[[int, int, float, dict], None] | None = None,
    batch_size: int = 25,
) -> dict[str, Any]:
    """
    Stream a customized number of flows across a specified duration.
    
    Parameters
    ----------
    url : str
        Target API base URL.
    num_flows : int
        Total number of flows to simulate.
    duration_seconds : float
        Target simulation duration in seconds (0 = instant).
    progress_callback : callable, optional
        Callback invoked as (sent_count, total_flows, elapsed_time, last_response).
    """
    flows_df = load_flows(
        path=data_path,
        attack_category=attack_category,
        attack_type=attack_type,
        zero_day_only=zero_day_only,
        attacks_only=attacks_only,
        normal_only=normal_only,
    )

    # If requested more flows than filtered rows, sample with replacement or shuffle
    if num_flows <= len(flows_df):
        selected = flows_df.sample(n=num_flows, random_state=42).copy()
    else:
        selected = flows_df.sample(n=num_flows, replace=True, random_state=42).copy()

    total_to_send = len(selected)
    batch_size = max(1, min(batch_size, total_to_send))
    num_batches = (total_to_send + batch_size - 1) // batch_size
    delay_per_batch = (duration_seconds / num_batches) if (num_batches > 0 and duration_seconds > 0) else 0.0

    sent = 0
    total_latency_ms = 0.0
    start_time = time.perf_counter()
    last_res = {}

    for offset in range(0, total_to_send, batch_size):
        batch = selected.iloc[offset : offset + batch_size][FEATURE_COLUMNS].to_dict(orient="records")
        res = requests.post(f"{url.rstrip('/')}/predict", json={"flows": batch}, timeout=30)
        res.raise_for_status()
        last_res = res.json()
        total_latency_ms += last_res.get("latency_ms", 0.0)
        sent += len(batch)

        elapsed = time.perf_counter() - start_time
        if progress_callback:
            progress_callback(sent, total_to_send, elapsed, last_res)

        if delay_per_batch > 0 and sent < total_to_send:
            time.sleep(delay_per_batch)

    total_elapsed = time.perf_counter() - start_time
    return {
        "flows_sent": sent,
        "duration_seconds": round(total_elapsed, 2),
        "avg_latency_ms": round(total_latency_ms / max(1, num_batches), 2),
        "flows_per_sec": round(sent / max(0.01, total_elapsed), 1),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--limit", type=int, default=200, help="Number of flows to send")
    parser.add_argument("--duration", type=float, default=10.0, help="Duration in seconds")
    parser.add_argument("--category", help="dos, probe, r2l, u2r, or normal")
    parser.add_argument("--attack-type", help="Specific raw attack (e.g. neptune, satan)")
    parser.add_argument("--zero-day", action="store_true", help="Send only novel zero-day attacks")
    parser.add_argument("--attacks-only", action="store_true", help="Send only attack flows")
    args = parser.parse_args()

    print(f"Starting replay: {args.limit} flows over {args.duration}s to {args.url} ...")
    stats = stream_replay(
        url=args.url,
        num_flows=args.limit,
        duration_seconds=args.duration,
        attack_category=args.category,
        attack_type=args.attack_type,
        zero_day_only=args.zero_day,
        attacks_only=args.attacks_only,
        data_path=args.data,
        progress_callback=lambda s, t, el, r: print(f"  Sent {s}/{t} ({el:.1f}s) | Latency: {r.get('latency_ms', 0):.2f}ms"),
    )
    print("\nReplay finished successfully:")
    print(stats)


if __name__ == "__main__":
    main()
