"""
MarketState — per-instrument price, IV, and Greeks, updated tick-by-tick.

One source of time: every method that touches state takes the current event's
timestamp as the authoritative clock.  No wall-clock calls anywhere.

Currency conventions (inherited from Phase 1):
  - Option mids are BTC-denominated in the raw data; converted to USD here via
    mid_usd = mid_btc * F before the IV solve.
  - F (self._F) is the 30JAN26 forward mid in USD.
  - Greeks from black76_greeks are in USD per unit.
  - PERPETUAL is tracked (BBO) but does not feed the pricer.
"""

from __future__ import annotations

import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from pricing.black76  import black76_greeks
from pricing.iv_solver import implied_vol
from engine.events    import Event

_EXPIRY  = pd.Timestamp("2026-01-30 08:00:00", tz="UTC")
_FORWARD = "deribit-BTC-30JAN26-future"
_OPT_RE  = re.compile(r"-(\d+)-([CP])-option$")
_STALE_S = 900  # 15 minutes


@dataclass
class InstrumentState:
    bid:             float = np.nan
    ask:             float = np.nan
    mid:             float = np.nan       # BTC for options; USD for futures
    iv:              float = np.nan
    greeks:          dict | None = None   # delta, gamma, vega, theta (USD)
    last_candle:     dict | None = None
    last_quote_time: pd.Timestamp | None = None
    is_stale:        bool = True          # True until first real quote arrives


class MarketState:
    """
    Maintains live per-instrument state updated event-by-event.

    Parameters
    ----------
    expiry            : option expiry timestamp (default 2026-01-30 08:00 UTC)
    iv_cache_dir      : directory containing iv_surface_calls/puts.parquet
    stale_threshold_s : seconds since last real quote before is_stale → True
    """

    def __init__(
        self,
        expiry:            pd.Timestamp = _EXPIRY,
        iv_cache_dir:      Path        = Path(__file__).parent.parent / "pricing" / "cache",
        stale_threshold_s: int         = _STALE_S,
    ) -> None:
        self._expiry   = expiry
        self._stale_s  = stale_threshold_s
        self._states:  dict[str, InstrumentState] = {}
        self._F:       float = np.nan   # 30JAN26 forward mid in USD

        self._iv_cache_c, self._iv_cache_p = self._load_iv_cache(iv_cache_dir)

    # ── Public API ────────────────────────────────────────────────────────────

    def update(self, event: Event) -> None:
        """
        Process one event.  Call order within update():
          1. Update stale flags (uses event.timestamp as 'now').
          2. Apply the event to the relevant instrument state.
        """
        self._update_stale_flags(event.timestamp)

        if event.event_type == "quote":
            self._on_quote(event)
        elif event.event_type == "candle":
            self._on_candle(event)

    @property
    def forward_price(self) -> float:
        """30JAN26 forward mid price in USD; NaN before first forward quote."""
        return self._F

    def get_instrument(self, instrument: str) -> InstrumentState:
        """Return current state for an instrument (default-constructed if unseen)."""
        return self._states.get(instrument, InstrumentState())

    def get_iv_history(
        self,
        instrument:       str,
        current_ts:       pd.Timestamp,
        lookback_minutes: int,
    ) -> pd.Series:
        """
        Return the IV time series for one option instrument, covering the window
        [current_ts - lookback_minutes, current_ts].

        No look-ahead: `.loc[:current_ts]` is an inclusive upper-bound slice on
        the sorted DatetimeIndex — rows strictly after current_ts are never
        accessible regardless of lookback size.

        Returns an empty Series if the instrument has not yet been observed or
        is not in the IV cache.
        """
        try:
            K, flag = self._parse_option_meta(instrument)
        except AssertionError:
            return pd.Series(dtype=float)

        cache = self._iv_cache_c if flag == "c" else self._iv_cache_p
        if K not in cache.columns:
            return pd.Series(dtype=float)

        hist   = cache.loc[:current_ts, K]           # Series, index ≤ current_ts
        cutoff = current_ts - pd.Timedelta(minutes=lookback_minutes)
        return hist.loc[cutoff:]

    def portfolio_greeks(self, positions: dict[str, float]) -> dict:
        """
        Aggregate Greeks across all positions.

        positions : {instrument: signed_qty}
          - Futures/perp: delta = signed_qty × 1.0 (1 contract = 1 USD delta);
            gamma = vega = theta = 0.
          - Options: Greeks from black76_greeks (USD), scaled by qty.

        Returns dict with keys: delta, gamma, vega, theta.
        """
        total = {"delta": 0.0, "gamma": 0.0, "vega": 0.0, "theta": 0.0}
        for instr, qty in positions.items():
            state = self._states.get(instr)
            if state is None:
                continue
            if instr.endswith("-option") and state.greeks is not None:
                for k in total:
                    total[k] += qty * state.greeks[k]
            elif instr.endswith("-future"):
                total["delta"] += qty  # gamma/vega/theta stay 0
        return total

    # ── Private helpers ───────────────────────────────────────────────────────

    def _on_quote(self, event: Event) -> None:
        p     = event.payload
        instr = event.instrument
        state = self._states.setdefault(instr, InstrumentState())

        state.bid             = p["bid_price"]
        state.ask             = p["ask_price"]
        state.mid             = (p["bid_price"] + p["ask_price"]) / 2
        state.last_quote_time = event.timestamp
        state.is_stale        = False

        if instr == _FORWARD:
            self._F = state.mid   # mid is USD for futures
            return

        if not instr.endswith("-option"):
            return  # PERPETUAL or unknown: BBO tracked, no pricer

        # ── Option: solve IV + Greeks ─────────────────────────────────────────
        F = self._F
        if not (np.isfinite(F) and F > 0):
            state.iv     = np.nan
            state.greeks = None
            return

        K, flag  = self._parse_option_meta(instr)
        mid_usd  = state.mid * F   # BTC mid × F_usd = USD price
        T        = (self._expiry - event.timestamp).total_seconds() / (365.25 * 86400)

        state.iv = implied_vol(mid_usd, F, K, T, flag)

        if not np.isnan(state.iv):
            state.greeks = {
                k: float(v)
                for k, v in black76_greeks(F, K, T, state.iv, flag).items()
            }
        else:
            state.greeks = None

    def _on_candle(self, event: Event) -> None:
        state = self._states.setdefault(event.instrument, InstrumentState())
        state.last_candle = event.payload
        # Candles do not update IV or Greeks.

    def _update_stale_flags(self, current_ts: pd.Timestamp) -> None:
        for state in self._states.values():
            if state.last_quote_time is None:
                state.is_stale = True
            else:
                gap_s = (current_ts - state.last_quote_time).total_seconds()
                if gap_s > self._stale_s:
                    state.is_stale = True

    def _parse_option_meta(self, instrument: str) -> tuple[int, str]:
        """Return (strike_int, flag_str) e.g. (90000, 'c')."""
        m = _OPT_RE.search(instrument)
        assert m, f"Not an option instrument: {instrument}"
        return int(m.group(1)), m.group(2).lower()

    @staticmethod
    def _load_iv_cache(cache_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Load both IV surface parquets once at construction.
        Parquet stores column names as strings; rename back to int for lookup.
        """
        def _load(path: Path) -> pd.DataFrame:
            df = pd.read_parquet(path)
            iv_cols = [c for c in df.columns if not str(c).startswith("stale")]
            df.rename(columns={c: int(c) for c in iv_cols}, inplace=True)
            return df

        c_path = cache_dir / "iv_surface_calls.parquet"
        p_path = cache_dir / "iv_surface_puts.parquet"
        iv_c   = _load(c_path) if c_path.exists() else pd.DataFrame()
        iv_p   = _load(p_path) if p_path.exists() else pd.DataFrame()
        return iv_c, iv_p


if __name__ == "__main__":
    from pathlib import Path
    from engine.events import load_events

    DATA   = Path(__file__).parent.parent / "takehome_data"
    events = load_events(DATA)

    ms = MarketState()
    N  = 50_000
    print(f"\nProcessing first {N:,} events...")
    for e in events[:N]:
        ms.update(e)

    snap_ts = events[N - 1].timestamp
    print(f"\n{'─'*75}")
    print(f"  MarketState snapshot at {snap_ts}")
    print(f"{'─'*75}")
    print(f"  {'Instrument':<45}  {'mid':>10}  {'IV':>7}  {'delta':>6}  {'stale':>5}")
    print(f"  {'─'*45}  {'─'*10}  {'─'*7}  {'─'*6}  {'─'*5}")

    for instr in sorted(ms._states.keys()):
        state = ms._states[instr]
        mid_s   = f"{state.mid:>10.2f}"
        iv_s    = f"{state.iv:.2%}"              if np.isfinite(state.iv)    else "    NaN"
        delta_s = f"{state.greeks['delta']:+.3f}" if state.greeks is not None else "   None"
        stale_s = "STALE" if state.is_stale else "ok"
        print(f"  {instr:<45}  {mid_s}  {iv_s:>7}  {delta_s:>6}  {stale_s:>5}")

    print(f"{'─'*75}")
    print(f"  Forward (F): {ms._F:.2f} USD")
    print()

    # IV history look-ahead check
    instr = "deribit-BTC-30JAN26-90000-C-option"
    hist  = ms.get_iv_history(instr, snap_ts, lookback_minutes=60)
    print(f"  get_iv_history(90K-C, t, 60 min): {len(hist)} rows, "
          f"max_idx={hist.index.max()}, future_rows={(hist.index > snap_ts).sum()}")
