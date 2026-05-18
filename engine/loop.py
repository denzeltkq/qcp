"""
Event loop — wires events into MarketState, strategies, portfolios, and fills.

Usage:
    from engine.events       import load_events
    from engine.market_state import MarketState
    from engine.strategies   import CoveredCall, DeltaHedgedShortVol
    from engine.fills        import FillSimulator
    from engine.portfolio    import Portfolio
    from engine.loop         import run_event_loop
    from pathlib             import Path

    DATA    = Path("takehome_data")
    events  = load_events(DATA)
    ms      = MarketState()
    strats  = [CoveredCall(), DeltaHedgedShortVol()]
    ports   = [Portfolio(), Portfolio()]
    sim     = FillSimulator()
    result  = run_event_loop(events, ms, strategies=strats,
                             portfolios=ports, fill_sim=sim)
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing  import Sequence

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from engine.events       import Event
from engine.market_state import MarketState

_ATM = "deribit-BTC-30JAN26-90000-C-option"


def run_event_loop(
    events:               list[Event],
    market_state:         MarketState,
    strategies:           Sequence = (),
    portfolios:           Sequence = (),        # one Portfolio per strategy (positional match)
    fill_sim                        = None,      # FillSimulator | None
    verbose:              bool     = True,
    event_hook                      = None,      # callable(event_count, event) | None
    event_hook_interval:  int      = 1_000,
) -> dict:
    """
    Replay all events in chronological order.

    For each event:
      1. market_state.update(event)
      2. strategy.on_event(event, market_state) → orders
      3. fill_sim.simulate(order, market_state) → fill  (if fill_sim provided)
      4. portfolio.apply_fill(fill)                      (if fill_sim provided)
      5. portfolio.sample(market_state, ts)              (every 100 ATM quote ticks)
      6. event_hook(event_count, event)                  (every event_hook_interval events)

    Returns a dict keyed by strategy class name:
        {name: {"orders": list[Order], "fills": list[Fill]}}
    """
    result: dict = {
        s.__class__.__name__: {"orders": [], "fills": []}
        for s in strategies
    }

    atm_tick_count: int = 0
    total               = len(events)

    if verbose:
        print(f"\nStarting event loop: {total:,} events  |  strategies: {len(strategies)}")
        print(f"{'Timestamp':^32}  {'IV':^8}  {'δ':^7}  {'ν (vega)':^10}  {'':5}")
        print("-" * 70)

    for idx, event in enumerate(events, 1):
        market_state.update(event)

        for i, strategy in enumerate(strategies):
            orders = strategy.on_event(event, market_state)
            name   = strategy.__class__.__name__
            result[name]["orders"].extend(orders)

            if fill_sim is not None and i < len(portfolios):
                for order in orders:
                    assert order.timestamp <= event.timestamp, (
                        f"Look-ahead detected: order at {order.timestamp} "
                        f"exceeds current event at {event.timestamp}"
                    )
                    fill = fill_sim.simulate(order, market_state)
                    portfolios[i].apply_fill(fill)
                    result[name]["fills"].append(fill)

        if event.event_type == "quote" and event.instrument == _ATM:
            atm_tick_count += 1

            for portfolio in portfolios:
                if atm_tick_count % 100 == 0:
                    portfolio.sample(market_state, event.timestamp)

            if verbose and atm_tick_count % 100 == 0:
                state = market_state.get_instrument(_ATM)

                iv_s    = f"{state.iv:.2%}"               if np.isfinite(state.iv)    else "    NaN "
                delta_s = f"{state.greeks['delta']:+.3f}"  if state.greeks is not None else "   None "
                vega_s  = f"{state.greeks['vega']:>10.1f}" if state.greeks is not None else "      None"
                stale_s = "STALE" if state.is_stale else ""

                print(f"{str(event.timestamp):^32}  {iv_s:^8}  {delta_s:^7}  {vega_s:^10}  {stale_s:5}")

        if event_hook is not None and idx % event_hook_interval == 0:
            event_hook(idx, event)

    if verbose:
        total_orders = sum(len(v["orders"]) for v in result.values())
        total_fills  = sum(len(v["fills"])  for v in result.values())
        print("-" * 70)
        print(f"Event loop complete.  ATM ticks: {atm_tick_count:,}  |  "
              f"Orders: {total_orders}  |  Fills: {total_fills}")

    return result


if __name__ == "__main__":
    from engine.events       import load_events
    from engine.market_state import MarketState
    from engine.strategies   import CoveredCall, DeltaHedgedShortVol
    from engine.fills        import FillSimulator
    from engine.portfolio    import Portfolio

    DATA    = Path(__file__).parent.parent / "takehome_data"
    events  = load_events(DATA)
    ms      = MarketState()
    strats  = [CoveredCall(), DeltaHedgedShortVol()]
    ports   = [Portfolio(), Portfolio()]
    sim     = FillSimulator()
    result  = run_event_loop(events, ms, strategies=strats, portfolios=ports, fill_sim=sim)

    out = Path(__file__).parent.parent / "reports"
    for strat, port in zip(strats, ports):
        port.to_reports(out, strat.__class__.__name__)
        print(f"\n{'─'*50}")
        print((out / strat.__class__.__name__ / "summary.txt").read_text())

    print(f"Reports saved to {out}/")
