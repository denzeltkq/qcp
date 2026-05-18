"""
Validation suite for the Black-76 pricer and IV solver.

Run as:  python -m pricing.validation

All checks are self-contained and use no real market data.
"""

import sys
import numpy as np
from pathlib import Path

# Allow running from repo root or pricing/
sys.path.insert(0, str(Path(__file__).parent.parent))

from pricing.black76 import black76_price, black76_greeks

PASS_COUNT = 0
FAIL_COUNT = 0


def _ok(msg: str):
    global PASS_COUNT
    PASS_COUNT += 1
    print(f"  PASS  {msg}")


def _fail(msg: str):
    global FAIL_COUNT
    FAIL_COUNT += 1
    print(f"  FAIL  {msg}", file=sys.stderr)


# ─── Reference constants ──────────────────────────────────────────────────────
F_REF    = 88_000.0          # USD, representative BTC forward
T_REF    = 48 / 365.25       # ~48 days to expiry (mid-period)
SIGMA_REF = 0.60              # 60% vol — typical for BTC
STRIKES  = [70_000, 80_000, 90_000, 100_000, 110_000]


# ─── Check 1: Put-Call Parity on the pricer ───────────────────────────────────
print("\n[Check 1] Put-Call Parity  (C - P = F - K, r=0)")
for K in STRIKES:
    C = float(black76_price(F_REF, K, T_REF, SIGMA_REF, "c"))
    P = float(black76_price(F_REF, K, T_REF, SIGMA_REF, "p"))
    lhs = C - P
    rhs = F_REF - K
    err = abs(lhs - rhs)
    if err < 1e-8:
        _ok(f"K={K//1000}K: C-P={lhs:.4f}, F-K={rhs:.4f}, err={err:.2e}")
    else:
        _fail(f"K={K//1000}K: C-P={lhs:.6f} != F-K={rhs:.6f}, err={err:.2e}")


# ─── Check 2: IV Round-Trip ───────────────────────────────────────────────────
# Import iv_solver here so Check 1 (pure math) runs even if scipy is missing
print("\n[Check 2] IV Round-Trip  (price → IV → price back, err < 0.01%)")
try:
    from pricing.iv_solver import implied_vol

    test_sigmas = [0.30, 0.50, 0.80, 1.20]
    for sigma_test in test_sigmas:
        for K in STRIKES:
            for flag in ("c", "p"):
                price = float(black76_price(F_REF, K, T_REF, sigma_test, flag))
                iv_back = implied_vol(price, F_REF, K, T_REF, flag)
                if np.isnan(iv_back):
                    _fail(f"σ={sigma_test} K={K//1000}K {flag}: IV solve returned NaN")
                else:
                    rel_err = abs(iv_back - sigma_test) / sigma_test
                    if rel_err < 1e-4:
                        _ok(f"σ={sigma_test:.2f} K={K//1000}K {flag}: iv_back={iv_back:.6f}, rel_err={rel_err:.2e}")
                    else:
                        _fail(f"σ={sigma_test:.2f} K={K//1000}K {flag}: iv_back={iv_back:.6f}, rel_err={rel_err:.2e} (threshold 1e-4)")
except ImportError:
    print("  SKIP  iv_solver not yet available")


# ─── Check 3: Greek Sanity Bounds ─────────────────────────────────────────────
print("\n[Check 3] Greek Sanity Bounds")
test_grid = [
    (F_REF, K, T_REF, SIGMA_REF)
    for K in STRIKES
]
test_grid += [
    # Near-ATM
    (88_000, 88_000, T_REF, 0.50),
    # Short time to expiry
    (88_000, 90_000, 5 / 365.25, 0.60),
    # High vol
    (88_000, 70_000, T_REF, 1.50),
]

all_greek_ok = True
for F, K, T, sigma in test_grid:
    for flag in ("c", "p"):
        g = black76_greeks(F, K, T, sigma, flag)
        d = float(g["delta"])
        gm = float(g["gamma"])
        ve = float(g["vega"])
        th = float(g["theta"])

        checks = []
        if flag == "c":
            if not (0 < d < 1):
                checks.append(f"delta_call={d:.4f} not in (0,1)")
        else:
            if not (-1 < d < 0):
                checks.append(f"delta_put={d:.4f} not in (-1,0)")
        if gm < 0:
            checks.append(f"gamma={gm:.6f} < 0")
        if ve < 0:
            checks.append(f"vega={ve:.4f} < 0")
        if th > 0:
            checks.append(f"theta={th:.4f} > 0")

        if checks:
            _fail(f"F={F} K={K} T={T:.3f} σ={sigma} {flag}: {'; '.join(checks)}")
            all_greek_ok = False
        else:
            _ok(f"F={F} K={K//1000}K {flag}: δ={d:.3f} γ={gm:.6f} ν={ve:.1f} θ={th:.1f}")

# ATM delta check
for flag, expected, tol in [("c", 0.5, 0.10), ("p", -0.5, 0.10)]:
    g = black76_greeks(88_000, 88_000, T_REF, SIGMA_REF, flag)
    d = float(g["delta"])
    if abs(d - expected) < tol:
        _ok(f"ATM {flag} delta={d:.3f} ≈ {expected} (tol {tol})")
    else:
        _fail(f"ATM {flag} delta={d:.3f} far from {expected}")


# ─── Check 4: Vol Smile Direction (requires iv_solver + real surface cache) ───
print("\n[Check 4] Vol Smile Direction (from cached surface)")
try:
    import pandas as pd
    from pathlib import Path

    cache = Path(__file__).parent / "cache"
    c_file = cache / "iv_surface_calls.parquet"
    p_file = cache / "iv_surface_puts.parquet"

    if not c_file.exists():
        print("  SKIP  IV surface cache not built yet — run iv_surface.py first")
    else:
        surf_c = pd.read_parquet(c_file)
        surf_p = pd.read_parquet(p_file)

        for surf, label in [(surf_c, "calls"), (surf_p, "puts")]:
            iv_cols = [c for c in surf.columns if not str(c).startswith("stale")]
            iv_cols_int = [int(c) for c in iv_cols]
            medians = surf[iv_cols].rename(columns=dict(zip(iv_cols, iv_cols_int))).median()
            atm_k = 90_000   # closest to F ≈ 88K
            atm_iv = medians.get(atm_k, np.nan)
            wing_lo = medians.get(70_000, np.nan)
            wing_hi = medians.get(110_000, np.nan)

            if np.isnan(atm_iv) or np.isnan(wing_lo) or np.isnan(wing_hi):
                print(f"  NOTE  {label}: some medians are NaN — surface may be sparse")
            elif wing_lo > atm_iv and wing_hi > atm_iv:
                _ok(f"{label} smile: 70K IV={wing_lo:.2%}, ATM IV={atm_iv:.2%}, 110K IV={wing_hi:.2%}")
            else:
                print(f"  NOTE  {label} smile is flat or inverted: 70K={wing_lo:.2%}, ATM={atm_iv:.2%}, 110K={wing_hi:.2%}")

except ImportError:
    print("  SKIP  pandas not available")


# ─── Check 5: NaN IV Count Table (from cached surface) ────────────────────────
print("\n[Check 5] NaN IV Count per Instrument")
try:
    import pandas as pd
    from pathlib import Path

    cache = Path(__file__).parent / "cache"
    for fname, label in [("iv_surface_calls.parquet", "calls"), ("iv_surface_puts.parquet", "puts")]:
        fpath = cache / fname
        if not fpath.exists():
            print(f"  SKIP  {fname} not built yet")
            continue
        surf = pd.read_parquet(fpath)
        total = len(surf)
        iv_cols = [c for c in surf.columns if not str(c).startswith("stale")]
        print(f"\n  {label.upper()} IV surface ({total} timestamps):")
        print(f"  {'Strike':>8}  {'NaN count':>10}  {'NaN %':>8}  {'Status':>8}")
        any_warn = False
        for col in sorted(iv_cols, key=int):
            nan_count = surf[col].isna().sum()
            nan_pct = nan_count / total * 100
            status = "WARNING" if nan_pct > 5 else "ok"
            if status == "WARNING":
                any_warn = True
            print(f"  {int(col)//1000:>5}K  {nan_count:>10,}  {nan_pct:>7.2f}%  {status:>8}")
        if not any_warn:
            _ok(f"{label}: all strikes <5% NaN")
        else:
            print(f"  NOTE  {label}: some strikes exceed 5% NaN threshold")

except ImportError:
    print("  SKIP  pandas not available")


# ─── Summary ──────────────────────────────────────────────────────────────────
print(f"\n{'='*50}")
print(f"  PASSED: {PASS_COUNT}  |  FAILED: {FAIL_COUNT}")
print(f"{'='*50}\n")
if FAIL_COUNT > 0:
    sys.exit(1)

import pandas as pd
from pathlib import Path

CACHE = Path("pricing/cache")

iv_c = pd.read_parquet(CACHE / "iv_surface_calls.parquet")
iv_p = pd.read_parquet(CACHE / "iv_surface_puts.parquet")
print(iv_c)
print(iv_p)
