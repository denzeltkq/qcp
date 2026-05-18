"""
Implied volatility solver for Black-76.

implied_vol()    — single-solve wrapper (returns NaN on failure, never raises)
solve_iv_for_quotes() — batch solve over original (non-filled) quote rows only.
                        Returns a long-format DataFrame of (time, strike, flag, iv).
"""

import numpy as np
import pandas as pd
import warnings
from scipy.optimize import brentq

from pricing.black76 import black76_price

_SIGMA_LO = 1e-4
_SIGMA_HI = 20.0   # 2000% vol ceiling — no real BTC option will hit this


def implied_vol(
    market_price_usd: float,
    F: float,
    K: float,
    T: float,
    flag: str,
    sigma_lo: float = _SIGMA_LO,
    sigma_hi: float = _SIGMA_HI,
) -> float:
    """
    Invert Black-76 to find the implied volatility.

    Parameters
    ----------
    market_price_usd : observed mid price converted to USD
    F    : forward price in USD
    K    : strike in USD
    T    : time to expiry in years
    flag : 'c' for call, 'p' for put

    Returns
    -------
    Implied vol (annualised) or np.nan on any failure.
    """
    if T <= 0 or market_price_usd <= 0 or F <= 0 or K <= 0:
        return np.nan

    # Price must be above intrinsic for a real vol solution to exist
    if flag == "c":
        intrinsic = max(F - K, 0.0)
    else:
        intrinsic = max(K - F, 0.0)

    if market_price_usd <= intrinsic:
        return np.nan

    def objective(sigma):
        return float(black76_price(F, K, T, sigma, flag)) - market_price_usd

    try:
        f_lo = objective(sigma_lo)
        f_hi = objective(sigma_hi)
    except Exception:
        return np.nan

    # Bracket must straddle zero
    if f_lo * f_hi > 0:
        return np.nan

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            iv = brentq(objective, sigma_lo, sigma_hi, xtol=1e-7, rtol=1e-7, maxiter=100)
        return float(iv)
    except Exception:
        return np.nan


def solve_iv_for_quotes(
    real_quotes: pd.DataFrame,
    forward_at_quote: pd.Series,
    expiry: pd.Timestamp,
) -> pd.DataFrame:
    """
    Solve IV for each original quote event (not forward-filled rows).

    Parameters
    ----------
    real_quotes       : DataFrame with columns [time, instrument, mid_usd].
                        Must contain ONLY original (non-filled) quote rows.
    forward_at_quote  : Series indexed by timestamp → forward USD price,
                        aligned to the same timestamps as real_quotes.
    expiry            : option expiry timestamp (UTC)

    Returns
    -------
    DataFrame with columns [time, strike, flag, iv].
    NaN iv rows are included so the caller sees total vs failed counts.
    """
    strikes = [70_000, 80_000, 90_000, 100_000, 110_000]

    options = real_quotes[real_quotes["instrument"].str.endswith("-option")].copy()
    options["strike"] = options["instrument"].str.extract(r"-(\d+)-[CP]-option")[0].astype(int)
    options["flag"]   = options["instrument"].str.extract(r"-([CP])-option")[0].str.lower()
    options = options[options["strike"].isin(strikes)].copy()

    n = len(options)
    print(f"  Solving IV for {n:,} original quote events...")

    ivs = np.empty(n, dtype=float)
    ivs[:] = np.nan

    for i, (_, row) in enumerate(options.iterrows()):
        ts  = row["time"]
        K   = int(row["strike"])
        flg = row["flag"]
        mkt = row["mid_usd"]

        F = forward_at_quote.get(ts, np.nan)
        if np.isnan(F) or F <= 0:
            continue

        T = (expiry - ts).total_seconds() / (365.25 * 86400)
        ivs[i] = implied_vol(mkt, F, K, T, flg)

        if (i + 1) % 100_000 == 0:
            print(f"    ... {i+1:,} / {n:,}")

    options = options.copy()
    options["iv"] = ivs

    # Print per-instrument failure table
    print(f"\n  {'Instrument':<45} {'total':>8} {'failed':>8} {'fail%':>8} {'status':>8}")
    print(f"  {'-'*45} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")
    for instr in sorted(options["instrument"].unique()):
        sub = options[options["instrument"] == instr]
        total  = len(sub)
        failed = sub["iv"].isna().sum()
        pct    = failed / total * 100 if total > 0 else 0
        status = "WARNING" if pct > 5 else "ok"
        print(f"  {instr:<45} {total:>8,} {failed:>8,} {pct:>7.2f}% {status:>8}")

    return options[["time", "instrument", "strike", "flag", "iv"]]
