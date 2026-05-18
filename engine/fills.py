"""
Fill simulator — converts Orders into Fills at real bid/ask prices.

Fill price convention:
  buy  → fills at ask price
  sell → fills at bid price

Options: bid/ask are BTC-denominated in MarketState; multiplied by the
30JAN26 forward mid (F) to get USD.  If F is unavailable (NaN) or the raw
bid/ask is NaN, falls back to order.limit_price (which was already set to
mid_btc × F at order creation time).

Always returns a Fill — never rejects an order.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from engine.events     import Event        # noqa: F401  (kept for type completeness)
from engine.strategies import Order

_FORWARD = "deribit-BTC-30JAN26-future"


# ── Fill ──────────────────────────────────────────────────────────────────────

@dataclass
class Fill:
    instrument:  str
    side:        str            # 'buy' | 'sell'
    qty:         float          # always positive
    price:       float          # USD — actual fill price
    timestamp:   pd.Timestamp
    order:       Order          # reference to originating order


# ── FillSimulator ─────────────────────────────────────────────────────────────

class FillSimulator:
    """
    Stateless simulator: fills each Order at the current top-of-book bid/ask.
    """

    def simulate(self, order: Order, state) -> Fill:
        """
        state : MarketState (typed loosely to avoid circular import)
        """
        inst_state = state.get_instrument(order.instrument)
        is_option  = order.instrument.endswith("-option")

        if is_option:
            F = state.get_instrument(_FORWARD).mid
            if np.isfinite(F) and F > 0:
                raw_bid = inst_state.bid * F
                raw_ask = inst_state.ask * F
            else:
                # F unavailable — both sides fall back; handled below
                raw_bid = np.nan
                raw_ask = np.nan
        else:
            raw_bid = inst_state.bid   # already USD
            raw_ask = inst_state.ask

        if order.side == "buy":
            price = raw_ask if np.isfinite(raw_ask) else order.limit_price
        else:
            price = raw_bid if np.isfinite(raw_bid) else order.limit_price

        return Fill(
            instrument = order.instrument,
            side       = order.side,
            qty        = order.qty,
            price      = price,
            timestamp  = order.timestamp,
            order      = order,
        )
