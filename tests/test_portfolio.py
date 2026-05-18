"""
Unit tests for engine/fills.py and engine/portfolio.py.

Uses lightweight stubs for MarketState and InstrumentState — no parquet files,
no real IV solver, no event loop.  Each test proves one specific contract.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from engine.market_state import InstrumentState
from engine.strategies   import Order
from engine.fills        import Fill, FillSimulator
from engine.portfolio    import Portfolio, PositionRecord, compute_attribution

# ── Constants ─────────────────────────────────────────────────────────────────

_TS  = pd.Timestamp("2025-12-15 10:00:00", tz="UTC")
_FWD = "deribit-BTC-30JAN26-future"
_OPT = "deribit-BTC-30JAN26-90000-C-option"


# ── Stubs ─────────────────────────────────────────────────────────────────────

class _FakeMarketState:
    def __init__(self, states: dict[str, InstrumentState]) -> None:
        self._states = states

    def get_instrument(self, instrument: str) -> InstrumentState:
        return self._states.get(instrument, InstrumentState())


def _fwd_state(mid: float = 88_000.0, bid: float = 87_950.0, ask: float = 88_050.0) -> InstrumentState:
    return InstrumentState(bid=bid, ask=ask, mid=mid, iv=np.nan, greeks=None,
                           is_stale=False, last_quote_time=_TS)


def _opt_state(
    mid_btc: float = 0.05,
    bid_btc: float = 0.045,
    ask_btc: float = 0.055,
    iv:      float = 0.43,
    delta:   float = 0.50,
    is_stale: bool = False,
) -> InstrumentState:
    return InstrumentState(
        bid=bid_btc, ask=ask_btc, mid=mid_btc,
        iv=iv,
        greeks={"delta": delta, "gamma": 1e-5, "vega": 12_000.0, "theta": -50.0},
        is_stale=is_stale,
        last_quote_time=_TS,
    )


def _order(instrument: str = _FWD, side: str = "buy", qty: float = 1.0,
           limit_price: float = 88_000.0) -> Order:
    return Order(instrument=instrument, side=side, qty=qty,
                 limit_price=limit_price, timestamp=_TS)


def _fill(instrument: str = _FWD, side: str = "buy", qty: float = 1.0,
          price: float = 88_000.0) -> Fill:
    return Fill(instrument=instrument, side=side, qty=qty,
                price=price, timestamp=_TS, order=_order(instrument, side, qty, price))


# ═══════════════════════════════════════════════════════════════════════════════
# Portfolio — apply_fill
# ═══════════════════════════════════════════════════════════════════════════════

class TestApplyFill:

    def test_new_long_position(self):
        """buy 1 fwd@88K → qty=1, avg=88K, cash=-88K, realized=0."""
        p = Portfolio()
        p.apply_fill(_fill(_FWD, "buy", 1.0, 88_000.0))
        rec = p._positions[_FWD]
        assert rec.qty             == pytest.approx(1.0)
        assert rec.avg_entry_price == pytest.approx(88_000.0)
        assert rec.realized_pnl    == pytest.approx(0.0)
        assert p._cash             == pytest.approx(-88_000.0)

    def test_wac_averaging(self):
        """buy 1@88K + buy 1@90K → avg=89K, qty=2, cash=-178K."""
        p = Portfolio()
        p.apply_fill(_fill(_FWD, "buy", 1.0, 88_000.0))
        p.apply_fill(_fill(_FWD, "buy", 1.0, 90_000.0))
        rec = p._positions[_FWD]
        assert rec.qty             == pytest.approx(2.0)
        assert rec.avg_entry_price == pytest.approx(89_000.0)
        assert p._cash             == pytest.approx(-178_000.0)

    def test_partial_close_books_realized(self):
        """buy 2@88K, sell 1@90K → realized=2K, qty=1, avg unchanged."""
        p = Portfolio()
        p.apply_fill(_fill(_FWD, "buy",  2.0, 88_000.0))
        p.apply_fill(_fill(_FWD, "sell", 1.0, 90_000.0))
        rec = p._positions[_FWD]
        assert rec.qty             == pytest.approx(1.0)
        assert rec.avg_entry_price == pytest.approx(88_000.0)   # WAC unchanged
        assert rec.realized_pnl    == pytest.approx(2_000.0)
        assert p._cash             == pytest.approx(-88_000.0 * 2 + 90_000.0)

    def test_full_close_clears_unrealized(self):
        """buy 2@88K, sell 2@91K → realized=6K, qty=0, unrealized=0."""
        p = Portfolio()
        p.apply_fill(_fill(_FWD, "buy",  2.0, 88_000.0))
        p.apply_fill(_fill(_FWD, "sell", 2.0, 91_000.0))
        rec = p._positions[_FWD]
        assert rec.qty             == pytest.approx(0.0)
        assert rec.realized_pnl    == pytest.approx(6_000.0)
        assert rec.unrealized_pnl  == pytest.approx(0.0)

    def test_partial_then_full_close(self):
        """Sequential close: 2@88K → sell 1@90K → sell 1@91K → realized=5K total."""
        p = Portfolio()
        p.apply_fill(_fill(_FWD, "buy",  2.0, 88_000.0))
        p.apply_fill(_fill(_FWD, "sell", 1.0, 90_000.0))
        p.apply_fill(_fill(_FWD, "sell", 1.0, 91_000.0))
        rec = p._positions[_FWD]
        assert rec.qty          == pytest.approx(0.0)
        assert rec.realized_pnl == pytest.approx(5_000.0)   # 2K + 3K

    def test_short_position_close(self):
        """sell 1@88K, buy 1@85K → realized=+3K (short profit)."""
        p = Portfolio()
        p.apply_fill(_fill(_FWD, "sell", 1.0, 88_000.0))
        p.apply_fill(_fill(_FWD, "buy",  1.0, 85_000.0))
        rec = p._positions[_FWD]
        assert rec.qty          == pytest.approx(0.0)
        assert rec.realized_pnl == pytest.approx(3_000.0)

    def test_total_pnl_sums_realized_and_unrealized(self):
        """total_pnl() = realized + unrealized across all positions."""
        p = Portfolio()
        p.apply_fill(_fill(_FWD, "buy",  2.0, 88_000.0))
        p.apply_fill(_fill(_FWD, "sell", 1.0, 90_000.0))
        # Manually inject unrealized to simulate a mark_to_market call
        p._positions[_FWD].unrealized_pnl = 500.0
        assert p.total_pnl() == pytest.approx(2_000.0 + 500.0)


# ═══════════════════════════════════════════════════════════════════════════════
# Portfolio — mark_to_market
# ═══════════════════════════════════════════════════════════════════════════════

class TestMarkToMarket:

    def test_mark_to_market_future(self):
        """Long 1 forward @ 88K entry; mark at 90K → unrealized = 2K."""
        p = Portfolio()
        p.apply_fill(_fill(_FWD, "buy", 1.0, 88_000.0))
        ms = _FakeMarketState({_FWD: _fwd_state(mid=90_000.0)})
        p.mark_to_market(ms, _TS)
        rec = p._positions[_FWD]
        assert rec.unrealized_pnl == pytest.approx(2_000.0)
        assert rec.last_mark      == pytest.approx(90_000.0)

    def test_mark_to_market_stale_option_skips(self):
        """
        First mark with valid IV sets unrealized.
        Second mark with is_stale=True must leave unrealized unchanged.
        """
        p = Portfolio()
        p.apply_fill(_fill(_OPT, "sell", 1.0, 4_000.0))  # short 1 option @ $4K

        # Pass 1: valid IV — unrealized updates
        ms_valid = _FakeMarketState({
            _FWD: _fwd_state(mid=88_000.0),
            _OPT: _opt_state(iv=0.43, is_stale=False),
        })
        p.mark_to_market(ms_valid, _TS)
        unrealized_after_first_mark = p._positions[_OPT].unrealized_pnl
        assert np.isfinite(unrealized_after_first_mark)

        # Pass 2: stale → unrealized must NOT change
        ms_stale = _FakeMarketState({
            _FWD: _fwd_state(mid=88_000.0),
            _OPT: _opt_state(iv=0.43, is_stale=True),
        })
        p.mark_to_market(ms_stale, _TS + pd.Timedelta("5m"))
        assert p._positions[_OPT].unrealized_pnl == pytest.approx(unrealized_after_first_mark)


# ═══════════════════════════════════════════════════════════════════════════════
# FillSimulator
# ═══════════════════════════════════════════════════════════════════════════════

class TestFillSimulator:

    def test_buy_future_at_ask(self):
        """Buy order on forward fills at ask price (USD, no conversion)."""
        sim   = FillSimulator()
        order = _order(_FWD, "buy", 1.0, 88_000.0)
        ms    = _FakeMarketState({_FWD: _fwd_state(bid=87_950.0, ask=88_050.0)})
        fill  = sim.simulate(order, ms)
        assert fill.price == pytest.approx(88_050.0)
        assert fill.side  == "buy"
        assert fill.qty   == pytest.approx(1.0)

    def test_sell_future_at_bid(self):
        """Sell order on forward fills at bid price."""
        sim   = FillSimulator()
        order = _order(_FWD, "sell", 1.0, 88_000.0)
        ms    = _FakeMarketState({_FWD: _fwd_state(bid=87_950.0, ask=88_050.0)})
        fill  = sim.simulate(order, ms)
        assert fill.price == pytest.approx(87_950.0)

    def test_option_buy_converts_btc_to_usd(self):
        """Option buy: fill.price = ask_btc × F_mid."""
        sim      = FillSimulator()
        # ask_btc=0.06, F=88K → expected fill = 5280.0 USD
        limit_px = 0.05 * 87_000.0   # deliberately different from ask to prove correct source
        order    = _order(_OPT, "buy", 1.0, limit_px)
        ms       = _FakeMarketState({
            _FWD: _fwd_state(mid=88_000.0),
            _OPT: _opt_state(bid_btc=0.054, ask_btc=0.060),
        })
        fill = sim.simulate(order, ms)
        assert fill.price == pytest.approx(0.060 * 88_000.0)

    def test_option_sell_converts_btc_to_usd(self):
        """Option sell: fill.price = bid_btc × F_mid."""
        sim   = FillSimulator()
        order = _order(_OPT, "sell", 1.0, 0.05 * 87_000.0)
        ms    = _FakeMarketState({
            _FWD: _fwd_state(mid=88_000.0),
            _OPT: _opt_state(bid_btc=0.054, ask_btc=0.060),
        })
        fill = sim.simulate(order, ms)
        assert fill.price == pytest.approx(0.054 * 88_000.0)

    def test_nan_bid_ask_fallback(self):
        """NaN bid/ask → fill.price falls back to order.limit_price."""
        sim   = FillSimulator()
        order = _order(_FWD, "buy", 1.0, 88_000.0)
        ms    = _FakeMarketState({_FWD: InstrumentState()})  # all NaN by default
        fill  = sim.simulate(order, ms)
        assert fill.price == pytest.approx(88_000.0)

    def test_nan_forward_option_fallback(self):
        """Option with NaN forward → can't convert BTC→USD → falls back to limit_price."""
        sim      = FillSimulator()
        limit_px = 4_350.0
        order    = _order(_OPT, "buy", 1.0, limit_px)
        ms       = _FakeMarketState({
            _FWD: InstrumentState(),                              # NaN mid
            _OPT: _opt_state(bid_btc=0.054, ask_btc=0.060),
        })
        fill = sim.simulate(order, ms)
        assert fill.price == pytest.approx(limit_px)

    def test_fill_references_originating_order(self):
        """Fill.order must be the same object as the submitted Order."""
        sim   = FillSimulator()
        order = _order(_FWD, "buy", 1.0, 88_000.0)
        ms    = _FakeMarketState({_FWD: _fwd_state()})
        fill  = sim.simulate(order, ms)
        assert fill.order is order


# ═══════════════════════════════════════════════════════════════════════════════
# compute_attribution
# ═══════════════════════════════════════════════════════════════════════════════

class TestPnlAttribution:

    def test_pnl_attribution_math(self):
        """One interval: verify delta, vega, theta, residual against hand-computed values."""
        T0 = pd.Timestamp("2025-12-15 10:00:00", tz="UTC")
        T1 = T0 + pd.Timedelta("1h")

        eq_df = pd.DataFrame([
            {"timestamp": T0, "total_pnl": 0.0,   "F": 88_000.0, "iv_atm": 0.42,
             "realized_pnl": 0, "unrealized_pnl": 0, "cash": 0, "nav": 0},
            {"timestamp": T1, "total_pnl": 500.0,  "F": 89_000.0, "iv_atm": 0.40,
             "realized_pnl": 0, "unrealized_pnl": 500, "cash": 0, "nav": 500},
        ])
        greek_df = pd.DataFrame([
            {"timestamp": T0, "delta": 0.5, "gamma": 0, "vega": 12_000.0, "theta": -50.0},
            {"timestamp": T1, "delta": 0.5, "gamma": 0, "vega": 12_000.0, "theta": -50.0},
        ])

        attr = compute_attribution(eq_df, greek_df)
        assert len(attr) == 1  # row 0 (no prior interval) is dropped

        dt = 3600 / (365.25 * 86400)   # 1 hour in fractional years ≈ 1.141e-4

        assert attr.iloc[0]["delta_pnl"] == pytest.approx(0.5    * 1_000,    abs=0.01)  # +500.00
        assert attr.iloc[0]["vega_pnl"]  == pytest.approx(12_000 * (-0.02),  abs=0.01)  # -240.00
        assert attr.iloc[0]["theta_pnl"] == pytest.approx(-50.0  * dt,       abs=0.01)  # ≈ -0.00571

        expected_residual = 500.0 - (500.0 + (-240.0) + (-50.0 * dt))   # ≈ +240.006
        assert attr.iloc[0]["residual"]         == pytest.approx(expected_residual, abs=0.01)
        assert attr.iloc[0]["total_pnl_change"] == pytest.approx(500.0, abs=0.01)
