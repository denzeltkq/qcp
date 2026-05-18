"""
Black-76 pricer for European options on a forward.

All inputs may be scalar or numpy arrays (broadcasting applies).
r = 0 throughout — carry is already embedded in the forward price F.
Prices and Greeks are in the same currency as F and K (USD).
"""

import numpy as np
from scipy.stats import norm

_N = norm.cdf   # standard normal CDF
_n = norm.pdf   # standard normal PDF


def _d1d2(F, K, T, sigma):
    """Compute d1 and d2. T and sigma must be > 0."""
    sqrtT = np.sqrt(T)
    d1 = (np.log(F / K) + 0.5 * sigma**2 * T) / (sigma * sqrtT)
    d2 = d1 - sigma * sqrtT
    return d1, d2


def black76_price(F, K, T, sigma, flag: str) -> float | np.ndarray:
    """
    Black-76 theoretical price (r=0).

    Parameters
    ----------
    F     : forward price (USD)
    K     : strike price (USD)
    T     : time to expiry in years (> 0)
    sigma : annualised volatility (> 0)
    flag  : 'c' for call, 'p' for put

    Returns
    -------
    Option price in USD. At T <= 0, returns intrinsic value.
    """
    F, K, T, sigma = map(np.asarray, (F, K, T, sigma))
    if np.any(F <= 0) or np.any(K <= 0):
        raise ValueError("F and K must be positive")
    if np.any(sigma <= 0):
        raise ValueError("sigma must be positive")

    flag = flag.lower()
    if flag not in ("c", "p"):
        raise ValueError("flag must be 'c' or 'p'")

    # Handle T <= 0: intrinsic only
    expired = T <= 0
    T_safe = np.where(expired, 1.0, T)       # dummy to avoid division by zero
    sigma_safe = np.where(expired, 1.0, sigma)

    d1, d2 = _d1d2(F, K, T_safe, sigma_safe)

    if flag == "c":
        price = F * _N(d1) - K * _N(d2)
        intrinsic = np.maximum(F - K, 0.0)
    else:
        price = K * _N(-d2) - F * _N(-d1)
        intrinsic = np.maximum(K - F, 0.0)

    return np.where(expired, intrinsic, price)


def black76_greeks(F, K, T, sigma, flag: str) -> dict:
    """
    Black-76 Greeks (r=0).

    Returns
    -------
    dict with keys: delta, gamma, vega, theta
    - delta : ∂price/∂F  (call: [0,1], put: [-1,0])
    - gamma : ∂²price/∂F²  (always >= 0)
    - vega  : ∂price/∂sigma, in USD per unit of vol
    - theta : ∂price/∂t, annualised (negative for long option);
              divide by 365 for per-calendar-day decay
    At T <= 0, all Greeks are 0.
    """
    F, K, T, sigma = map(np.asarray, (F, K, T, sigma))
    if np.any(F <= 0) or np.any(K <= 0):
        raise ValueError("F and K must be positive")
    if np.any(sigma <= 0):
        raise ValueError("sigma must be positive")

    flag = flag.lower()
    if flag not in ("c", "p"):
        raise ValueError("flag must be 'c' or 'p'")

    expired = T <= 0
    T_safe = np.where(expired, 1.0, T)
    sigma_safe = np.where(expired, 1.0, sigma)
    sqrtT = np.sqrt(T_safe)

    d1, d2 = _d1d2(F, K, T_safe, sigma_safe)
    nd1 = _n(d1)

    if flag == "c":
        delta = _N(d1)
    else:
        delta = -_N(-d1)

    gamma = nd1 / (F * sigma_safe * sqrtT)
    vega  = F * nd1 * sqrtT
    # Theta: annualised rate of time decay (negative for long option)
    theta = -(F * sigma_safe * nd1) / (2 * sqrtT)

    zero = np.zeros_like(delta)
    return {
        "delta": np.where(expired, zero, delta),
        "gamma": np.where(expired, zero, gamma),
        "vega":  np.where(expired, zero, vega),
        "theta": np.where(expired, zero, theta),
    }
