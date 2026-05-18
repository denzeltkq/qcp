"""
Unit tests for engine/strategies.py.

Uses a _FakeMarketState stub that wraps a dict of InstrumentState objects.
No parquet files are loaded; no pricing calls are made; no real event loop runs.
Each test proves one specific behavioural contract of CoveredCall or
DeltaHedgedShortVol.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from engine.events       import Event
from engine.market_state import InstrumentState
from engine.strategies   import CoveredCall, DeltaHedgedShortVol, Order

# ── Constants ─────────────────────────────────────────────────────────────────

_TS  = pd.Timestamp("2025-12-15 00:00:05", tz="UTC")
_TS2 = _TS + pd.Timedelta("5s")
_FWD = "deribit-BTC-30JAN26-future"
_OPT = "deribit-BTC-30JAN26-90000-C-option"


# ── Fake MarketState stub ─────────────────────────────────────────────────────

class _FakeMarketState:
    """
    Read-only stub that returns controllable InstrumentState values.
    Never touches parquet files or the real IV solver.
    """
    def __init__(self, states: dict[str, InstrumentState]) -> None:
        self._states = states

    def get_instrument(self, instrument: str) -> InstrumentState:
        return self._states.get(instrument, InstrumentState())

    def get_iv_history(self, instrument, current_ts, lookback_minutes):
        return pd.Series(dtype=float)

    def portfolio_greeks(self, positions):
        return {"delta": 0.0, "gamma": 0.0, "vega": 0.0, "theta": 0.0}


# ── State builder helpers ─────────────────────────────────────────────────────

def _fwd_state(mid: float = 87_000.0) -> InstrumentState:
    return InstrumentState(
        bid=mid - 50, ask=mid + 50, mid=mid,
        iv=np.nan, greeks=None,
        is_stale=False, last_quote_time=_TS,
    )


def _opt_state(
    mid_btc:  float = 0.05,
    iv:       float = 0.43,
    delta:    float = 0.50,
    is_stale: bool  = False,
) -> InstrumentState:
    return InstrumentState(
        bid=mid_btc * 0.9, ask=mid_btc * 1.1, mid=mid_btc,
        iv=iv,
        greeks={"delta": delta, "gamma": 1e-5, "vega": 12_000.0, "theta": -50.0},
        is_stale=is_stale,
        last_quote_time=_TS,
    )


def _make_ms(
    fwd_mid:  float = 87_000.0,
    opt_mid:  float = 0.05,
    delta:    float = 0.50,
    is_stale: bool  = False,
    opt_instr: str  = _OPT,
) -> _FakeMarketState:
    return _FakeMarketState({
        _FWD:      _fwd_state(fwd_mid),
        opt_instr: _opt_state(opt_mid, delta=delta, is_stale=is_stale),
    })


def _quote(instrument: str = _FWD, ts: pd.Timestamp = _TS) -> Event:
    return Event(
        timestamp=ts, event_type="quote", instrument=instrument,
        payload={"bid_price": 0.0, "ask_price": 0.0, "bid_size": 0.0, "ask_size": 0.0},
    )


def _candle(ts: pd.Timestamp = _TS) -> Event:
    return Event(timestamp=ts, event_type="candle", instrument=_FWD, payload={})


# ═══════════════════════════════════════════════════════════════════════════════
# CoveredCall tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestCoveredCall:

    def test_entry_emits_two_orders(self):
        """First quote event with valid F → exactly 2 orders (buy fwd, sell call)."""
        cc  = CoveredCall()
        ms  = _make_ms(fwd_mid=87_000.0, opt_mid=0.05)
        orders = cc.on_event(_quote(_FWD), ms)

        assert len(orders) == 2
        buy = orders[0]
        sell = orders[1]

        assert buy.instrument  == _FWD
        assert buy.side        == "buy"
        assert buy.qty         == 1.0
        assert buy.limit_price == pytest.approx(87_000.0)

        # ATM: F=87K, closest strike = 90K
        assert sell.instrument == _OPT
        assert sell.side       == "sell"
        assert sell.qty        == 1.0
        assert sell.limit_price == pytest.approx(0.05 * 87_000.0)

    def test_no_reentry_after_first_event(self):
        """All subsequent events → empty list."""
        cc = CoveredCall()
        ms = _make_ms()
        cc.on_event(_quote(_FWD), ms)        # enter
        assert cc.on_event(_quote(_FWD), ms) == []
        assert cc.on_event(_quote(_OPT), ms) == []

    def test_candle_event_ignored(self):
        """Candle events do not trigger entry."""
        cc = CoveredCall()
        ms = _make_ms()
        assert cc.on_event(_candle(), ms) == []
        assert not cc._entered

    def test_nan_forward_mid_skipped(self):
        """Strategy waits until F is finite and positive."""
        cc     = CoveredCall()
        ms_nan = _make_ms(fwd_mid=float("nan"))
        assert cc.on_event(_quote(_FWD), ms_nan) == []
        assert not cc._entered

    def test_zero_forward_mid_skipped(self):
        cc = CoveredCall()
        ms = _make_ms(fwd_mid=0.0)
        assert cc.on_event(_quote(_FWD), ms) == []
        assert not cc._entered

    def test_positions_correct_after_entry(self):
        """Internal positions are +1 forward, -1 call."""
        cc = CoveredCall()
        ms = _make_ms()
        cc.on_event(_quote(_FWD), ms)
        pos = cc.get_positions()
        assert pos[_FWD] == pytest.approx(+1.0)
        assert pos[_OPT] == pytest.approx(-1.0)

    def test_atm_strike_selection(self):
        """ATM strike = closest in [70K,80K,90K,100K,110K] to F at entry."""
        # F = 97_000: distances are 90K→7K, 100K→3K — unambiguously 100K
        cc = CoveredCall()
        ms = _make_ms(fwd_mid=97_000.0)
        ms._states["deribit-BTC-30JAN26-100000-C-option"] = _opt_state(mid_btc=0.03)
        orders = cc.on_event(_quote(_FWD), ms)
        assert len(orders) == 2
        assert orders[1].instrument == "deribit-BTC-30JAN26-100000-C-option"

    def test_get_positions_returns_copy(self):
        """Mutating the returned dict must not affect internal state."""
        cc = CoveredCall()
        cc.on_event(_quote(_FWD), _make_ms())
        pos = cc.get_positions()
        pos[_FWD] = 999.0
        assert cc._positions[_FWD] == pytest.approx(1.0)


# ═══════════════════════════════════════════════════════════════════════════════
# DeltaHedgedShortVol tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestDeltaHedgedShortVol:

    def _entered_strategy(self, delta: float = 0.50) -> tuple[DeltaHedgedShortVol, _FakeMarketState]:
        """Return a strategy that has already entered with the given delta."""
        dhsv = DeltaHedgedShortVol()
        ms   = _make_ms(delta=delta)
        dhsv.on_event(_quote(_OPT), ms)
        return dhsv, ms

    def test_entry_emits_sell_plus_hedge(self):
        """First valid option quote → sell 1 option + buy hedge_qty forward."""
        dhsv   = DeltaHedgedShortVol()
        ms     = _make_ms(delta=0.50)
        orders = dhsv.on_event(_quote(_OPT), ms)

        assert len(orders) == 2
        sell  = orders[0]
        hedge = orders[1]

        assert sell.instrument  == _OPT
        assert sell.side        == "sell"
        assert sell.qty         == 1.0
        assert sell.limit_price == pytest.approx(0.05 * 87_000.0)

        assert hedge.instrument == _FWD
        assert hedge.side       == "buy"
        assert hedge.qty        == pytest.approx(0.50)   # -(-0.50) = +0.50

    def test_portfolio_delta_near_zero_after_entry(self):
        """After entry the net delta should be within floating-point noise of 0."""
        dhsv, ms = self._entered_strategy(delta=0.50)
        pos      = dhsv.get_positions()
        opt_qty  = pos.get(_OPT, 0.0)
        fwd_qty  = pos.get(_FWD, 0.0)
        port_delta = 0.50 * opt_qty + fwd_qty * 1.0
        assert abs(port_delta) < 1e-10

    def test_no_rehedge_within_threshold(self):
        """If delta hasn't drifted past threshold, no orders are emitted."""
        dhsv, ms = self._entered_strategy(delta=0.50)
        # delta still 0.50 → port_delta = 0.50*(-1) + 0.50 = 0.0 < 0.05
        orders = dhsv.on_event(_quote(_OPT, ts=_TS2), ms)
        assert orders == []

    def test_rehedge_when_delta_drifts_above_threshold(self):
        """delta drifts to 0.56 → |port_delta| = 0.06 > 0.05 → 1 hedge order."""
        dhsv, _ = self._entered_strategy(delta=0.50)
        ms2     = _make_ms(delta=0.56)
        orders  = dhsv.on_event(_quote(_OPT, ts=_TS2), ms2)

        assert len(orders) == 1
        o = orders[0]
        assert o.instrument == _FWD
        assert o.side       == "buy"      # port_delta = -0.06 → hedge +0.06
        assert o.qty        == pytest.approx(0.06, abs=1e-9)

    def test_positions_accumulate_across_rehedges(self):
        """Forward position grows correctly after each rehedge."""
        dhsv, _ = self._entered_strategy(delta=0.50)  # forward = +0.50
        dhsv.on_event(_quote(_OPT, ts=_TS2), _make_ms(delta=0.56))  # buy 0.06 more
        pos = dhsv.get_positions()
        assert pos[_FWD] == pytest.approx(0.56, abs=1e-9)

    def test_stale_greeks_skipped(self):
        """Stale InstrumentState → no orders at all, even at first event."""
        dhsv     = DeltaHedgedShortVol()
        ms_stale = _make_ms(is_stale=True)
        assert dhsv.on_event(_quote(_OPT), ms_stale) == []
        assert not dhsv._entered

    def test_none_greeks_skipped(self):
        """Greeks = None (IV solve failed) → no orders."""
        dhsv = DeltaHedgedShortVol()
        ms   = _FakeMarketState({
            _FWD: _fwd_state(),
            _OPT: InstrumentState(mid=0.05, iv=np.nan, greeks=None, is_stale=False),
        })
        assert dhsv.on_event(_quote(_OPT), ms) == []

    def test_forward_quote_event_ignored(self):
        """Forward quote events do not trigger entry or rehedge."""
        dhsv = DeltaHedgedShortVol()
        ms   = _make_ms()
        assert dhsv.on_event(_quote(_FWD), ms) == []
        assert not dhsv._entered

    def test_candle_event_ignored(self):
        dhsv = DeltaHedgedShortVol()
        assert dhsv.on_event(_candle(), _make_ms()) == []
        assert not dhsv._entered

    def test_nan_forward_at_entry_defers(self):
        """If forward mid is NaN when option first quotes, entry is deferred."""
        dhsv = DeltaHedgedShortVol()
        ms   = _FakeMarketState({
            _FWD: _fwd_state(mid=float("nan")),
            _OPT: _opt_state(),
        })
        assert dhsv.on_event(_quote(_OPT), ms) == []
        assert not dhsv._entered

    def test_nan_forward_at_rehedge_skips(self):
        """Forward mid becomes NaN after entry → hedge order skipped that tick."""
        dhsv, _ = self._entered_strategy(delta=0.50)
        # delta drifts to 0.90 (would normally trigger rehedge)
        ms_no_fwd = _FakeMarketState({
            _FWD: _fwd_state(mid=float("nan")),
            _OPT: _opt_state(delta=0.90),
        })
        orders = dhsv.on_event(_quote(_OPT, ts=_TS2), ms_no_fwd)
        assert orders == []
