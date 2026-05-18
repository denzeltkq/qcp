"""
Event dataclass and event stream loader.

load_events() merges quotes_l1_5s.parquet and candles_1m.parquet into a single
chronological list of Event objects.  Within a shared timestamp the ordering is:

  priority 0 — future quotes   (forward price known before options are priced)
  priority 1 — option quotes
  priority 2 — candle bars
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))


@dataclass
class Event:
    timestamp:  pd.Timestamp
    event_type: str   # 'quote' | 'candle'
    instrument: str
    payload:    dict


def _to_utc(df: pd.DataFrame) -> None:
    """Normalise the 'time' column to UTC in-place."""
    if pd.api.types.is_integer_dtype(df["time"]):
        df["time"] = pd.to_datetime(df["time"], unit="ns", utc=True)
    elif df["time"].dt.tz is None:
        df["time"] = df["time"].dt.tz_localize("UTC")
    else:
        df["time"] = df["time"].dt.tz_convert("UTC")


def _quote_priority(instrument: str) -> int:
    return 0 if instrument.endswith("-future") else 1


def load_events(data_path: Path) -> list[Event]:
    """
    Load and merge quote + candle events into a single sorted event stream.

    Ordering within a shared timestamp:
      future quotes (0) → option quotes (1) → candles (2)
    """
    quotes  = pd.read_parquet(data_path / "quotes_l1_5s.parquet")
    candles = pd.read_parquet(data_path / "candles_1m.parquet")

    _to_utc(quotes)
    _to_utc(candles)

    quote_payload_cols  = [c for c in quotes.columns  if c not in ("time", "instrument")]
    candle_payload_cols = [c for c in candles.columns if c not in ("time", "instrument")]

    records: list[tuple] = []  # (timestamp, priority, event_type, instrument, payload)

    for row in quotes.itertuples(index=False):
        payload = {c: getattr(row, c) for c in quote_payload_cols}
        records.append((row.time, _quote_priority(row.instrument),
                         "quote", row.instrument, payload))

    for row in candles.itertuples(index=False):
        payload = {c: getattr(row, c) for c in candle_payload_cols}
        records.append((row.time, 2, "candle", row.instrument, payload))

    records.sort(key=lambda r: (r[0], r[1]))  # stable: preserves within-priority order

    events = [
        Event(timestamp=ts, event_type=etype, instrument=instr, payload=payload)
        for ts, _, etype, instr, payload in records
    ]

    # Sanity print: first 20 events + monotonicity check
    print(f"\nLoaded {len(events):,} events total")
    print(f"  quotes: {len(quotes):,}   candles: {len(candles):,}")
    print("\nFirst 20 events:")
    for i, e in enumerate(events[:20]):
        print(f"  {i:2d}  {e.timestamp}  {e.event_type:<6}  {e.instrument}")

    # Confirm non-decreasing timestamps
    bad = sum(
        1 for a, b in zip(events, events[1:]) if b.timestamp < a.timestamp
    )
    if bad:
        print(f"\n  WARNING: {bad} timestamp inversions detected — check sort")
    else:
        print(f"\n  OK: timestamps are non-decreasing across all {len(events):,} events")

    return events


if __name__ == "__main__":
    events = load_events(Path(__file__).parent.parent / "takehome_data")
    print(f"\nDone. First event: {events[0]}  Last: {events[-1]}")
