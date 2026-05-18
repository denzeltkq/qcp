"""
Strategy layer for the BTC options backtester.

All strategies implement Strategy.on_event(event, state) → list[Order].
They are read-only consumers of MarketState — they never call state.update()
and never access parquet files directly.

Position tracking: self._positions is updated immediately when an order is
emitted, assuming fills (no fill simulator in Phase 3).

Currency convention (inherited from Phase 1/2):
  - Option mids are BTC-denominated; limit_price is always in USD.
  - limit_price for options = mid_btc × F_usd.
  - Forward mid is already USD.
"""

from __future__ import annotations

import sys
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from engine.events       import Event
from engine.market_state import MarketState

_FORWARD = "deribit-BTC-30JAN26-future"
_STRIKES = [70_000, 80_000, 90_000, 100_000, 110_000]


# ── Order ─────────────────────────────────────────────────────────────────────

@dataclass
class Order:
    instrument:  str
    side:        str           # 'buy' | 'sell'
    qty:         float         # always positive; side encodes direction
    limit_price: float         # USD, mid at time of order
    timestamp:   pd.Timestamp


# ── Strategy ABC ──────────────────────────────────────────────────────────────

class Strategy(ABC):
    def __init__(self) -> None:
        self._positions: dict[str, float] = {}  # instrument → signed qty

    @abstractmethod
    def on_event(self, event: Event, state: MarketState) -> list[Order]: ...

    def get_positions(self) -> dict[str, float]:
        """Return a snapshot copy of current positions."""
        return dict(self._positions)

    def _apply(self, order: Order) -> None:
        """Update internal position immediately on emit (assume fill)."""
        sign = 1.0 if order.side == "buy" else -1.0
        self._positions[order.instrument] = (
            self._positions.get(order.instrument, 0.0) + sign * order.qty
        )


# ── CoveredCall ───────────────────────────────────────────────────────────────

class CoveredCall(Strategy):
    """
    Buy 1 unit of the forward + sell 1 unit of the nearest ATM call.
    Entry fires once at the first event where the forward price F is valid.
    ATM strike is determined at entry and never recalculated.
    No re-entry, no stop-loss — hold to end of replay.
    """

    def __init__(self) -> None:
        super().__init__()
        self._entered:         bool      = False
        self._call_instrument: str | None = None

    def on_event(self, event: Event, state: MarketState) -> list[Order]:
        if self._entered:
            return []
        if event.event_type != "quote":
            return []

        fwd_state = state.get_instrument(_FORWARD)
        F = fwd_state.mid
        if not (np.isfinite(F) and F > 0):
            return []

        atm_strike = min(_STRIKES, key=lambda k: abs(k - F))
        call_instr = f"deribit-BTC-30JAN26-{atm_strike}-C-option"
        call_state = state.get_instrument(call_instr)

        if not (np.isfinite(call_state.mid) and call_state.mid > 0):
            return []  # wait until the ATM call has a valid BBO

        fwd_limit  = fwd_state.mid          # already USD
        call_limit = call_state.mid * F     # BTC × F_usd → USD

        buy_fwd   = Order(_FORWARD,    "buy",  1.0, fwd_limit,  event.timestamp)
        sell_call = Order(call_instr,  "sell", 1.0, call_limit, event.timestamp)

        self._apply(buy_fwd)
        self._apply(sell_call)
        self._entered         = True
        self._call_instrument = call_instr

        return [buy_fwd, sell_call]


# ── DeltaHedgedShortVol ───────────────────────────────────────────────────────

class DeltaHedgedShortVol(Strategy):
    """
    Sell 1 unit of option_instrument, then rehedge delta via the forward
    whenever |portfolio delta| > delta_threshold.

    Portfolio delta:
      port_delta = greeks['delta'] × opt_qty   +   fwd_qty × 1.0
    Forward delta = 1.0 per contract (no gamma/vega).

    Rehedge only fires on option quote events with valid, non-stale Greeks.
    Candle events and forward quote events are ignored.
    """

    def __init__(
        self,
        option_instrument: str   = "deribit-BTC-30JAN26-90000-C-option",
        delta_threshold:   float = 0.05,
    ) -> None:
        super().__init__()
        self._option  = option_instrument
        self._thresh  = delta_threshold
        self._entered = False

    def on_event(self, event: Event, state: MarketState) -> list[Order]:
        if event.event_type != "quote":
            return []
        if event.instrument != self._option:
            return []

        opt_state = state.get_instrument(self._option)
        if (opt_state.greeks is None
                or np.isnan(opt_state.iv)
                or opt_state.is_stale):
            return []

        if not self._entered:
            return self._enter(event, state, opt_state)

        return self._maybe_rehedge(event, state, opt_state)

    # ── private helpers ───────────────────────────────────────────────────────

    def _enter(
        self,
        event:     Event,
        state:     MarketState,
        opt_state, # InstrumentState
    ) -> list[Order]:
        fwd_state = state.get_instrument(_FORWARD)
        F = fwd_state.mid
        if not (np.isfinite(F) and F > 0):
            return []  # can't price or hedge without a valid forward

        opt_limit = opt_state.mid * F   # BTC → USD
        entry     = Order(self._option, "sell", 1.0, opt_limit, event.timestamp)
        self._apply(entry)              # _positions = {option: -1.0}
        self._entered = True

        # Portfolio delta after selling 1 option, before any forward position:
        opt_qty    = self._positions[self._option]       # -1.0
        port_delta = opt_state.greeks["delta"] * opt_qty # e.g. 0.50 × (-1) = -0.50

        # Always hedge to zero at entry (no threshold check here)
        hedge_qty = -port_delta                          # e.g. +0.50
        side      = "buy" if hedge_qty > 0 else "sell"
        hedge     = Order(_FORWARD, side, abs(hedge_qty), fwd_state.mid, event.timestamp)
        self._apply(hedge)

        return [entry, hedge]

    def _maybe_rehedge(
        self,
        event:     Event,
        state:     MarketState,
        opt_state, # InstrumentState
    ) -> list[Order]:
        opt_qty    = self._positions.get(self._option, 0.0)
        fwd_qty    = self._positions.get(_FORWARD, 0.0)
        port_delta = opt_state.greeks["delta"] * opt_qty + fwd_qty * 1.0

        if abs(port_delta) <= self._thresh:
            return []

        fwd_state = state.get_instrument(_FORWARD)
        if not (np.isfinite(fwd_state.mid) and fwd_state.mid > 0):
            return []  # can't price hedge order

        hedge_qty = -port_delta
        side      = "buy" if hedge_qty > 0 else "sell"
        order     = Order(_FORWARD, side, abs(hedge_qty), fwd_state.mid, event.timestamp)
        self._apply(order)
        return [order]
