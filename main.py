"""
Backtesting entry point — loads config.yaml and runs the full pipeline.

risk_free_rate is read from config but not forwarded to the pricer.
black76.py hardcodes r=0 (known simplification; extend black76_price
signature if a non-zero rate is required in future).
"""

from __future__ import annotations

import sys
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).parent))

from engine.events       import load_events
from engine.fills        import FillSimulator
from engine.loop         import run_event_loop
from engine.market_state import MarketState
from engine.portfolio    import Portfolio
from engine.strategies   import CoveredCall, DeltaHedgedShortVol


def main() -> None:
    # ── 1. Config ─────────────────────────────────────────────────────────────
    try:
        cfg = yaml.safe_load(open("config.yaml"))
    except FileNotFoundError:
        print("ERROR: config.yaml not found. Run from the project root.")
        sys.exit(1)
    except yaml.YAMLError as exc:
        print(f"ERROR: config.yaml is malformed:\n  {exc}")
        sys.exit(1)

    # ── 2. Parameters ─────────────────────────────────────────────────────────
    data_path  = Path(cfg["backtest"]["data_path"])
    output_dir = Path(cfg["backtest"]["output_dir"])
    start_date = pd.Timestamp(cfg["backtest"]["start"], tz="UTC")
    end_date   = (pd.Timestamp(cfg["backtest"]["end"], tz="UTC")
                  + pd.Timedelta("1D") - pd.Timedelta("1ns"))
    # risk_free_rate = cfg["risk_free_rate"]
    # Not forwarded — black76.py hardcodes r=0 (known simplification).

    # ── 3. Events ─────────────────────────────────────────────────────────────
    all_events = load_events(data_path)
    events     = [e for e in all_events if start_date <= e.timestamp <= end_date]
    print(f"Loaded {len(events):,} events ({start_date.date()} → {end_date.date()})")

    if not events:
        print(f"WARNING: No events in range {start_date.date()} → {end_date.date()}. Exiting.")
        sys.exit(0)

    # ── 4. Core components ────────────────────────────────────────────────────
    ms       = MarketState()
    fill_sim = FillSimulator()

    # ── 5. Strategies and portfolios ──────────────────────────────────────────
    strategies: list = []
    portfolios: list = []

    if cfg["strategies"]["covered_call"]["enabled"]:
        strategies.append(CoveredCall())
        portfolios.append(Portfolio())

    if cfg["strategies"]["delta_hedged_short_vol"]["enabled"]:
        dhsv_cfg = cfg["strategies"]["delta_hedged_short_vol"]
        strategies.append(DeltaHedgedShortVol(
            option_instrument = dhsv_cfg["option_instrument"],
            delta_threshold   = dhsv_cfg["delta_threshold"],
        ))
        portfolios.append(Portfolio())

    print(f"Strategies: {[s.__class__.__name__ for s in strategies]}")

    if not strategies:
        print("WARNING: No strategies enabled in config. Nothing to run.")
        sys.exit(0)

    # ── 6. Event loop ─────────────────────────────────────────────────────────
    def _heartbeat(event_count: int, event) -> None:
        F = ms.forward_price
        F_str = f"{F:>10,.0f}" if np.isfinite(F) else "       NaN"
        line  = f"[{event_count:>7,}]  {event.timestamp}  F={F_str}  "
        for strat, port in zip(strategies, portfolios):
            line += (f"{strat.__class__.__name__}: "
                     f"PnL=${port.total_pnl():>8,.0f}  "
                     f"pos={port.get_positions()}  ")
        print(line)

    result = run_event_loop(
        events,
        ms,
        strategies          = strategies,
        portfolios          = portfolios,
        fill_sim            = fill_sim,
        verbose             = False,
        event_hook          = _heartbeat,
        event_hook_interval = 1_000,
    )

    total_order_count = sum(len(v["orders"]) for v in result.values())

    # ── 7. Final sample (guard against duplicate timestamp) ───────────────────
    last_ts = events[-1].timestamp
    for port in portfolios:
        if not port._equity_log or port._equity_log[-1]["timestamp"] != last_ts:
            port.sample(ms, last_ts)

    # ── 8. Reports ────────────────────────────────────────────────────────────
    for strat, port in zip(strategies, portfolios):
        port.to_reports(output_dir, strat.__class__.__name__)

    # ── 9. Console summary ────────────────────────────────────────────────────
    for strat, port in zip(strategies, portfolios):
        name       = strat.__class__.__name__
        positions  = port.get_positions()
        realized   = sum(r.realized_pnl   for r in positions.values())
        unrealized = sum(r.unrealized_pnl for r in positions.values())
        cash       = port._cash
        num_trades = len(port._trade_log)
        eq_df      = pd.DataFrame(port._equity_log)
        max_dd     = Portfolio._max_drawdown(eq_df)
        report_path = output_dir / name

        print(f"\nStrategy: {name}")
        print("─" * 37)
        print(f"Total PnL:    ${realized + unrealized:>12,.2f}")
        print(f"Realized:     ${realized:>12,.2f}")
        print(f"Unrealized:   ${unrealized:>12,.2f}")
        print(f"Cash:         ${cash:>12,.2f}")
        print(f"Num trades:   {num_trades}")
        print(f"Max drawdown: ${max_dd:>12,.2f}")
        print(f"Reports saved to: {report_path}/")

    # ── 10. Look-ahead confirmation ───────────────────────────────────────────
    print(f"\nLook-ahead check: PASSED — no order timestamp exceeded its event "
          f"timestamp across {total_order_count} orders.")


if __name__ == "__main__":
    try:
        main()
    except AssertionError:
        raise   # look-ahead violation — show full traceback as evidence
    except Exception:
        traceback.print_exc()
        sys.exit(1)
