"""
Portfolio — per-strategy position accounting, mark-to-model valuation, and reporting.

Position tracking uses weighted average cost (WAC) basis throughout.
All PnL and prices are in USD.

mark_to_market() uses black76_price for options (requires valid non-stale IV)
and the raw forward mid for futures.  Stale IV → unrealized_pnl frozen at
last valid mark.

equity_log fields: timestamp, total_pnl, realized_pnl, unrealized_pnl, cash, nav, F, iv_atm

to_reports() writes equity_curve.csv, greek_log.csv, trade_log.csv,
summary.txt (with PnL attribution block), equity_curve.png,
greeks_over_time.png, pnl_attribution.csv, and pnl_attribution.png to
output_dir / strategy_name /.
"""

from __future__ import annotations

import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib
matplotlib.use("Agg")   # non-interactive backend; safe in all environments
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from engine.fills    import Fill
from pricing.black76 import black76_price

_FORWARD = "deribit-BTC-30JAN26-future"
_ATM     = "deribit-BTC-30JAN26-90000-C-option"
_EXPIRY  = pd.Timestamp("2026-01-30 08:00:00", tz="UTC")
_OPT_RE  = re.compile(r"-(\d+)-([CP])-option$")


def _parse_option_meta(instrument: str) -> tuple[int, str]:
    """Return (strike_int, flag_lower) e.g. (90000, 'c')."""
    m = _OPT_RE.search(instrument)
    assert m, f"Not an option instrument: {instrument}"
    return int(m.group(1)), m.group(2).lower()


def compute_attribution(eq_df: pd.DataFrame, greek_df: pd.DataFrame) -> pd.DataFrame:
    """
    First-order PnL attribution via Taylor expansion over consecutive sample intervals.

    Returns a DataFrame indexed at t+1 (end of each interval) with columns:
        timestamp, delta_pnl, vega_pnl, theta_pnl, residual, total_pnl_change

    Intervals where F or iv_atm is NaN at either endpoint have attribution columns
    set to NaN (never interpolated). total_pnl_change is always populated.
    Returns empty DataFrame if inputs are empty or missing required columns.
    """
    required_eq  = {"timestamp", "total_pnl", "F", "iv_atm"}
    required_grk = {"timestamp", "delta", "vega", "theta"}
    if eq_df.empty or greek_df.empty:
        return pd.DataFrame()
    if not required_eq.issubset(eq_df.columns) or not required_grk.issubset(greek_df.columns):
        return pd.DataFrame()

    df = (
        eq_df[["timestamp", "total_pnl", "F", "iv_atm"]]
        .merge(greek_df[["timestamp", "delta", "vega", "theta"]], on="timestamp", how="inner")
        .sort_values("timestamp")
        .reset_index(drop=True)
    )

    # Explicit end-minus-start for all changes; shift(0) = identity but documents "end of interval"
    dF     = df["F"].shift(0)         - df["F"].shift(1)
    dsigma = df["iv_atm"].shift(0)    - df["iv_atm"].shift(1)
    dt     = (df["timestamp"].shift(0) - df["timestamp"].shift(1)).dt.total_seconds() / (365.25 * 86400)
    dpnl   = df["total_pnl"].shift(0) - df["total_pnl"].shift(1)

    # Greeks at start of interval (shift(1) = t-1 = start)
    attr = pd.DataFrame({
        "timestamp":        df["timestamp"],
        "delta_pnl":        df["delta"].shift(1) * dF,      # Greek at t-1 (start)
        "vega_pnl":         df["vega"].shift(1)  * dsigma,
        "theta_pnl":        df["theta"].shift(1) * dt,
        "total_pnl_change": dpnl,
    })
    attr["residual"] = (
        attr["total_pnl_change"]
        - (attr["delta_pnl"] + attr["vega_pnl"] + attr["theta_pnl"])
    )

    # Mask intervals where F or iv_atm is NaN at either endpoint
    bad = (
        df["F"].isna()      | df["F"].shift(1).isna()      |
        df["iv_atm"].isna() | df["iv_atm"].shift(1).isna()
    )
    attr.loc[bad, ["delta_pnl", "vega_pnl", "theta_pnl", "residual"]] = np.nan

    return attr.iloc[1:].reset_index(drop=True)  # drop row 0 (no prior interval)


# ── PositionRecord ────────────────────────────────────────────────────────────

@dataclass
class PositionRecord:
    qty:             float          # signed; 0.0 when flat
    avg_entry_price: float          # USD WAC (undefined but retained when flat)
    realized_pnl:    float = 0.0   # USD cumulative booked gains
    unrealized_pnl:  float = 0.0   # USD from last mark_to_market call
    last_mark:       float = np.nan # last valid mark price (USD)


# ── Portfolio ─────────────────────────────────────────────────────────────────

class Portfolio:
    """
    Tracks fills → WAC positions → mark-to-model PnL → sampled equity/greek logs.

    apply_fill() — call immediately after each simulated fill.
    sample()     — call at chosen intervals (e.g. every 100 ATM ticks); internally
                   calls mark_to_market() then records equity + greek snapshots.
    to_reports() — call at end of replay to write CSV / PNG outputs.
    """

    def __init__(self) -> None:
        self._positions:  dict[str, PositionRecord] = {}
        self._cash:       float = 0.0
        self._trade_log:  list[dict] = []
        self._equity_log: list[dict] = []
        self._greek_log:  list[dict] = []

    # ── Public API ────────────────────────────────────────────────────────────

    def apply_fill(self, fill: Fill) -> None:
        sign       = 1.0 if fill.side == "buy" else -1.0
        qty_change = sign * fill.qty
        realized_this_fill = 0.0

        if fill.instrument not in self._positions or self._positions[fill.instrument].qty == 0.0:
            # Case 1 — new position (or re-opening from flat)
            existing_realized = (
                self._positions[fill.instrument].realized_pnl
                if fill.instrument in self._positions else 0.0
            )
            self._positions[fill.instrument] = PositionRecord(
                qty             = qty_change,
                avg_entry_price = fill.price,
                realized_pnl    = existing_realized,
                unrealized_pnl  = 0.0,
                last_mark       = fill.price,
            )

        else:
            rec      = self._positions[fill.instrument]
            old_qty  = rec.qty
            old_avg  = rec.avg_entry_price
            new_qty  = old_qty + qty_change
            same_dir = (old_qty > 0 and qty_change > 0) or (old_qty < 0 and qty_change < 0)

            if same_dir:
                # Case 2 — adding to existing position (WAC update)
                rec.avg_entry_price = (old_qty * old_avg + qty_change * fill.price) / new_qty
                rec.qty = new_qty

            else:
                # Case 3 — reducing existing position
                closed_qty     = min(abs(qty_change), abs(old_qty))
                sign_old       = 1.0 if old_qty > 0 else -1.0
                realized_gain  = closed_qty * (fill.price - old_avg) * sign_old
                rec.realized_pnl += realized_gain
                realized_this_fill = realized_gain

                if abs(new_qty) < 1e-12:
                    # Fully closed: transfer remaining unrealized → realized
                    rec.realized_pnl  += rec.unrealized_pnl
                    rec.unrealized_pnl = 0.0
                    rec.qty            = 0.0
                else:
                    # Partially closed: WAC unchanged on remaining lot
                    rec.qty = new_qty

        # Cash: buy decreases, sell increases
        self._cash -= fill.price * fill.qty * sign

        self._trade_log.append({
            "timestamp":    fill.timestamp,
            "instrument":   fill.instrument,
            "side":         fill.side,
            "qty":          fill.qty,
            "price":        fill.price,
            "realized_pnl": realized_this_fill,
        })

    def mark_to_market(self, state, current_ts: pd.Timestamp) -> None:
        """
        Update unrealized_pnl for all non-flat positions.
        state : MarketState
        """
        F = state.get_instrument(_FORWARD).mid

        for instr, rec in self._positions.items():
            if rec.qty == 0.0:
                continue

            if instr.endswith("-future") or instr.endswith("-perpetual"):
                mark = state.get_instrument(instr).mid      # USD
                if np.isfinite(mark):
                    rec.last_mark      = mark
                    rec.unrealized_pnl = rec.qty * (mark - rec.avg_entry_price)

            elif instr.endswith("-option"):
                inst_state = state.get_instrument(instr)
                if (np.isfinite(inst_state.iv)
                        and not inst_state.is_stale
                        and np.isfinite(F) and F > 0):
                    K, flag = _parse_option_meta(instr)
                    T = (_EXPIRY - current_ts).total_seconds() / (365.25 * 86400)
                    if T > 0:
                        mark_usd = float(black76_price(F, K, T, inst_state.iv, flag))
                        if np.isfinite(mark_usd):
                            rec.last_mark      = mark_usd
                            rec.unrealized_pnl = rec.qty * (mark_usd - rec.avg_entry_price)
                # else: stale / invalid IV → freeze unrealized at last valid value

    def sample(self, state, timestamp: pd.Timestamp) -> None:
        """Mark to market, then record equity and greek snapshots."""
        self.mark_to_market(state, timestamp)

        realized   = sum(r.realized_pnl   for r in self._positions.values())
        unrealized = sum(r.unrealized_pnl for r in self._positions.values())
        nav        = self._cash + unrealized

        self._equity_log.append({
            "timestamp":      timestamp,
            "total_pnl":      realized + unrealized,
            "realized_pnl":   realized,
            "unrealized_pnl": unrealized,
            "cash":           self._cash,
            "nav":            nav,
            "F":              state.forward_price,
            "iv_atm":         state.get_instrument(_ATM).iv,
        })

        port = {"delta": 0.0, "gamma": 0.0, "vega": 0.0, "theta": 0.0}
        for instr, rec in self._positions.items():
            if rec.qty == 0.0:
                continue
            if instr.endswith("-option"):
                g = state.get_instrument(instr).greeks
                if g is not None:
                    for k in port:
                        port[k] += rec.qty * g[k]
            elif instr.endswith("-future"):
                port["delta"] += rec.qty  # forward delta = 1 per unit
        self._greek_log.append({"timestamp": timestamp, **port})

    def get_positions(self) -> dict[str, PositionRecord]:
        return dict(self._positions)

    def total_pnl(self) -> float:
        return sum(r.realized_pnl + r.unrealized_pnl for r in self._positions.values())

    def to_reports(self, output_dir: Path, strategy_name: str) -> None:
        out = output_dir / strategy_name
        out.mkdir(parents=True, exist_ok=True)

        # ── DataFrames ────────────────────────────────────────────────────────
        eq_df   = pd.DataFrame(self._equity_log)
        grk_df  = pd.DataFrame(self._greek_log)
        trd_df  = pd.DataFrame(self._trade_log)

        if not eq_df.empty:
            eq_df.to_csv(out / "equity_curve.csv", index=False)
        if not grk_df.empty:
            grk_df.to_csv(out / "greek_log.csv", index=False)
        if not trd_df.empty:
            trd_df.to_csv(out / "trade_log.csv", index=False)

        attr_df = pd.DataFrame()
        if "F" in eq_df.columns:
            attr_df = compute_attribution(eq_df, grk_df)
            if not attr_df.empty:
                attr_df.to_csv(out / "pnl_attribution.csv", index=False)

        # ── Summary stats ─────────────────────────────────────────────────────
        realized   = sum(r.realized_pnl   for r in self._positions.values())
        unrealized = sum(r.unrealized_pnl for r in self._positions.values())
        cash_final = self._cash
        nav_final  = cash_final + unrealized

        num_trades = len(self._trade_log)

        closing = [r["realized_pnl"] for r in self._trade_log if r["realized_pnl"] != 0.0]
        hit_rate = (sum(1 for p in closing if p > 0) / len(closing)) if closing else float("nan")

        max_dd = self._max_drawdown(eq_df)
        sharpe, n_days = self._sharpe(eq_df)

        summary = (
            f"Strategy:       {strategy_name}\n"
            f"Total PnL:      ${realized + unrealized:>12,.2f}\n"
            f"  Realized:     ${realized:>12,.2f}\n"
            f"  Unrealized:   ${unrealized:>12,.2f}\n"
            f"Cash:           ${cash_final:>12,.2f}\n"
            f"Final NAV:      ${nav_final:>12,.2f}\n"
            f"Num Trades:     {num_trades}\n"
            f"Hit Rate:       {hit_rate:.1%}   (closing trades with PnL > 0)\n"
            f"Max Drawdown:   ${max_dd:>12,.2f}\n"
            f"Sharpe (ann.):  {sharpe if np.isfinite(sharpe) else 'N/A':>8}     "
            f"({n_days} daily samples)\n"
        )
        attr_block = ""
        if not attr_df.empty:
            delta_total = attr_df["delta_pnl"].sum()
            vega_total  = attr_df["vega_pnl"].sum()
            theta_total = attr_df["theta_pnl"].sum()
            res_total   = attr_df["residual"].sum()
            total_pnl_val = realized + unrealized
            if total_pnl_val != 0 and np.isfinite(total_pnl_val):
                explained_pct = (delta_total + vega_total + theta_total) / total_pnl_val
                pct_str = f"{explained_pct:.1%}"
            else:
                pct_str = "N/A"
            attr_block = (
                f"\nPnL Attribution (cumulative):\n"
                f"  Delta PnL:   ${delta_total:>12,.2f}\n"
                f"  Vega PnL:    ${vega_total:>12,.2f}\n"
                f"  Theta PnL:   ${theta_total:>12,.2f}\n"
                f"  Residual:    ${res_total:>12,.2f}\n"
                f"  {'─'*37}\n"
                f"  Explained:   {pct_str} of total PnL\n"
            )
        # ── Stress test / scenario analysis ──────────────────────────────────
        stress_df    = pd.DataFrame()
        stress_block = ""

        if self._trade_log:
            entry_ts  = self._trade_log[0]["timestamp"]
            fwd_entry = next(
                (t for t in self._trade_log
                 if t["timestamp"] == entry_ts and t["instrument"] == _FORWARD),
                None,
            )
            entry_F = (
                fwd_entry["price"] if fwd_entry is not None
                else self._equity_log[0]["F"] if self._equity_log else np.nan
            )
            entry_IV = next(
                (e["iv_atm"] for e in self._equity_log
                 if np.isfinite(e.get("iv_atm", np.nan))),
                np.nan,
            )
            T_entry = (_EXPIRY - entry_ts).total_seconds() / (365.25 * 86400)

            if np.isfinite(entry_F) and np.isfinite(entry_IV) and T_entry > 0:
                entry_positions: dict[str, float] = {}
                for t in self._trade_log:
                    if t["timestamp"] == entry_ts:
                        sign = 1.0 if t["side"] == "buy" else -1.0
                        entry_positions[t["instrument"]] = (
                            entry_positions.get(t["instrument"], 0.0) + sign * t["qty"]
                        )

                actual_pnl_stress = realized + unrealized
                f_shocks  = [-0.20, -0.10, -0.05, 0.00, +0.05, +0.10, +0.20]
                iv_shocks = [-0.10, -0.05,  0.00, +0.05, +0.10]
                rows = []
                for f_shock in f_shocks:
                    for iv_shock in iv_shocks:
                        F_s  = entry_F * (1.0 + f_shock)
                        IV_s = max(entry_IV + iv_shock, 0.001)
                        option_delta  = 0.0
                        forward_delta = 0.0
                        for instr, qty in entry_positions.items():
                            if instr.endswith("-future") or instr.endswith("-perpetual"):
                                forward_delta += qty * (F_s - entry_F)
                            elif instr.endswith("-option"):
                                K, flag  = _parse_option_meta(instr)
                                base_px  = float(black76_price(entry_F, K, T_entry, entry_IV, flag))
                                shock_px = float(black76_price(F_s,     K, T_entry, IV_s,     flag))
                                option_delta += qty * (shock_px - base_px)
                        rows.append({
                            "f_shock_pct":     f_shock,
                            "iv_shock":        iv_shock,
                            "F_scenario":      F_s,
                            "IV_scenario":     IV_s,
                            "option_delta":    option_delta,
                            "forward_delta":   forward_delta,
                            "portfolio_delta": option_delta + forward_delta,
                            "actual_pnl":      actual_pnl_stress,
                        })
                stress_df = pd.DataFrame(rows)
                stress_df.to_csv(out / "stress_test.csv", index=False)

                # ── Stress block for summary.txt ──────────────────────────────
                min_s = stress_df["portfolio_delta"].min()
                max_s = stress_df["portfolio_delta"].max()
                inside = min_s <= actual_pnl_stress <= max_s
                T_days = T_entry * 365.25

                lines = [
                    f"\nScenario Analysis (entry positions, instantaneous shock):\n"
                    f"Entry F:  ${entry_F:>10,.2f}    "
                    f"Entry IV: {entry_IV:.1%}    T: {T_days:.1f} days\n",
                    f"{'F shock':>8}  {'IV shock':>8}  {'Portfolio ΔValue':>18}  {'vs Actual PnL':>14}\n",
                    "─" * 56 + "\n",
                ]
                for _, row in stress_df.iterrows():
                    f_str  = f"{int(row['f_shock_pct'] * 100):+d}%"
                    iv_str = f"{int(round(row['iv_shock'] * 100)):+d}pts"
                    pd_str = f"${row['portfolio_delta']:>12,.2f}"
                    if row["f_shock_pct"] == 0.0 and row["iv_shock"] == 0.0:
                        vs = "base"
                    elif row["portfolio_delta"] > actual_pnl_stress:
                        vs = "above"
                    else:
                        vs = "below"
                    lines.append(f"{f_str:>8}  {iv_str:>8}  {pd_str:>18}  {vs:>14}\n")
                lines += [
                    "\n",
                    f"Actual Dec 31 PnL: ${actual_pnl_stress:>12,.2f}\n",
                    f"Scenario range:    ${min_s:>12,.2f} to ${max_s:>12,.2f}\n",
                    f"Outcome:           {'INSIDE' if inside else 'OUTSIDE'} scenario envelope\n",
                ]
                stress_block = "".join(lines)

        (out / "summary.txt").write_text(summary + attr_block + stress_block)

        # ── Plots ─────────────────────────────────────────────────────────────
        if not eq_df.empty:
            fig, ax = plt.subplots(figsize=(12, 4))
            ax.plot(eq_df["timestamp"], eq_df["total_pnl"])
            ax.set_xlabel("Date")
            ax.set_ylabel("PnL (USD)")
            ax.set_title(f"{strategy_name} — Equity Curve")
            ax.grid(True)
            fig.tight_layout()
            fig.savefig(out / "equity_curve.png", dpi=150)
            plt.close(fig)

        if not grk_df.empty:
            fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(12, 6))
            ax1.plot(grk_df["timestamp"], grk_df["delta"])
            ax1.set_ylabel("Delta")
            ax1.set_title(f"{strategy_name} — Greeks Over Time")
            ax1.grid(True)
            ax2.plot(grk_df["timestamp"], grk_df["vega"])
            ax2.set_ylabel("Vega (USD)")
            ax2.set_xlabel("Date")
            ax2.grid(True)
            fig.tight_layout()
            fig.savefig(out / "greeks_over_time.png", dpi=150)
            plt.close(fig)

        if not attr_df.empty:
            attr_valid = attr_df.dropna(subset=["delta_pnl"])

            fig, ax1 = plt.subplots(figsize=(14, 5))
            ax2 = ax1.twinx()

            components = ["delta_pnl",  "vega_pnl",    "theta_pnl",   "residual"]
            colors     = ["steelblue",  "darkorange",  "forestgreen", "gray"]
            labels     = ["Delta",      "Vega",        "Theta",       "Residual"]

            if len(attr_valid) > 1:
                med_s = attr_valid["timestamp"].diff().dt.total_seconds().dropna().median()
                bar_w = pd.Timedelta(seconds=float(med_s) * 0.8)
            else:
                bar_w = pd.Timedelta(hours=1)

            bottoms_pos = np.zeros(len(attr_valid))
            bottoms_neg = np.zeros(len(attr_valid))

            for col, color, label in zip(components, colors, labels):
                vals = attr_valid[col].fillna(0).values
                pos  = np.where(vals > 0, vals, 0.0)
                neg  = np.where(vals < 0, vals, 0.0)
                ax1.bar(attr_valid["timestamp"], pos, bottom=bottoms_pos,
                        color=color, label=label, alpha=0.8, width=bar_w)
                ax1.bar(attr_valid["timestamp"], neg, bottom=bottoms_neg,
                        color=color, alpha=0.8, width=bar_w)
                bottoms_pos += pos
                bottoms_neg += neg

            ax2.plot(eq_df["timestamp"], eq_df["total_pnl"],
                     color="black", linewidth=1.5, label="Cumulative PnL", zorder=5)
            ax2.set_ylabel("Cumulative PnL (USD)")

            ax1.set_xlabel("Date")
            ax1.set_ylabel("PnL Contribution per Interval (USD)")
            ax1.set_title(f"{strategy_name} — PnL Attribution")
            ax1.legend(loc="upper left")
            ax1.grid(True, alpha=0.3)
            fig.tight_layout()
            fig.savefig(out / "pnl_attribution.png", dpi=150)
            plt.close(fig)

        if not stress_df.empty:
            from matplotlib.colors import TwoSlopeNorm

            pivot = stress_df.pivot(
                index="iv_shock", columns="f_shock_pct", values="portfolio_delta"
            ).sort_index(ascending=False)

            vmin = pivot.values.min()
            vmax = pivot.values.max()
            norm = (TwoSlopeNorm(vmin=vmin, vcenter=0.0, vmax=vmax)
                    if vmin < 0 < vmax else None)

            fig, ax = plt.subplots(figsize=(11, 5))
            im = ax.imshow(pivot.values, cmap="RdYlGn", norm=norm, aspect="auto")

            # Annotate cells
            for i in range(pivot.shape[0]):
                for j in range(pivot.shape[1]):
                    v = pivot.values[i, j]
                    ax.text(j, i, f"${v:,.0f}", ha="center", va="center",
                            fontsize=7.5, color="black")

            # Mark base scenario cell
            base_row = list(pivot.index).index(0.0)
            base_col = list(pivot.columns).index(0.0)
            rect = plt.Rectangle(
                (base_col - 0.5, base_row - 0.5), 1, 1,
                linewidth=2.5, edgecolor="black", facecolor="none",
            )
            ax.add_patch(rect)

            ax.set_xticks(range(len(pivot.columns)))
            ax.set_xticklabels([f"{int(f * 100):+d}%" for f in pivot.columns])
            ax.set_yticks(range(len(pivot.index)))
            ax.set_yticklabels([f"{int(round(iv * 100)):+d}pts" for iv in pivot.index])
            ax.set_xlabel("F shock")
            ax.set_ylabel("IV shock")
            ax.set_title(f"{strategy_name} — Entry Scenario Analysis")

            cbar = fig.colorbar(im, ax=ax)
            cbar.set_label("Portfolio ΔValue (USD)")
            # Mark actual PnL on colorbar
            if vmin <= actual_pnl_stress <= vmax:
                cbar.ax.axhline(y=actual_pnl_stress, color="black",
                                linewidth=2, linestyle="--")

            fig.tight_layout()
            fig.savefig(out / "stress_test.png", dpi=150)
            plt.close(fig)

    # ── Private helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _max_drawdown(eq_df: pd.DataFrame) -> float:
        if eq_df.empty or "total_pnl" not in eq_df.columns:
            return 0.0
        pnl    = eq_df["total_pnl"].values
        peak   = np.maximum.accumulate(pnl)
        dd     = peak - pnl
        return float(np.max(dd)) if len(dd) else 0.0

    @staticmethod
    def _sharpe(eq_df: pd.DataFrame) -> tuple[float, int]:
        if eq_df.empty or "nav" not in eq_df.columns:
            return float("nan"), 0
        df  = eq_df.set_index("timestamp")["nav"]
        daily = df.resample("1D").last().dropna()
        ret   = daily.diff().dropna()
        n     = len(ret)
        if n < 5:
            return float("nan"), n
        std = ret.std()
        if std == 0:
            return float("nan"), n
        return float(ret.mean() / std * np.sqrt(252)), n
