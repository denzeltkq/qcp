"""
Phase 0: Data Quality Exploration
Deribit BTC options dataset, Dec 15-31 2025 UTC.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # non-interactive; swap to "TkAgg" / "MacOSX" for live plots
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path

DATA = Path(__file__).parent.parent / "takehome_data"
PLOTS = Path(__file__).parent / "plots"
PLOTS.mkdir(exist_ok=True)

OBS: list[str] = []

def obs(step: int, tag: str, msg: str):
    line = f"[STEP {step}] [{tag}] {msg}"
    OBS.append(line)
    print(line)

EXPECTED_INSTRUMENTS = [
    "deribit-BTC-PERPETUAL-future",
    "deribit-BTC-30JAN26-future",
    "deribit-BTC-70000-20260130-C-option",
    "deribit-BTC-70000-20260130-P-option",
    "deribit-BTC-80000-20260130-C-option",
    "deribit-BTC-80000-20260130-P-option",
    "deribit-BTC-90000-20260130-C-option",
    "deribit-BTC-90000-20260130-P-option",
    "deribit-BTC-100000-20260130-C-option",
    "deribit-BTC-100000-20260130-P-option",
    "deribit-BTC-110000-20260130-C-option",
    "deribit-BTC-110000-20260130-P-option",
]

FORWARD = "deribit-BTC-30JAN26-future"
EXPIRY = pd.Timestamp("2026-01-30", tz="UTC")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 — Load all 3 parquets
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*70)
print("STEP 1 — Load parquets, shape / dtypes / head")
print("="*70)

quotes   = pd.read_parquet(DATA / "quotes_l1_5s.parquet")
candles  = pd.read_parquet(DATA / "candles_1m.parquet")
instrs   = pd.read_parquet(DATA / "instruments.parquet")

for name, df in [("quotes_l1_5s", quotes), ("candles_1m", candles), ("instruments", instrs)]:
    print(f"\n--- {name} ---")
    print(f"  shape : {df.shape}")
    print(f"  dtypes:\n{df.dtypes.to_string()}")
    print(f"  head:\n{df.head(3).to_string()}")

for name, df in [("quotes_l1_5s", quotes), ("candles_1m", candles)]:
    if df.shape[0] == 0:
        obs(1, "BLOCKER", f"{name} has 0 rows")
    else:
        obs(1, "GOOD", f"{name}: {df.shape[0]:,} rows × {df.shape[1]} cols")

for name, df in [("quotes_l1_5s", quotes), ("candles_1m", candles)]:
    if "time" in df.columns:
        if pd.api.types.is_integer_dtype(df["time"]):
            obs(1, "NOTE", f"{name}.time is raw integer — will need explicit datetime parse")
        else:
            obs(1, "GOOD", f"{name}.time dtype: {df['time'].dtype}")

# Normalise time columns to UTC datetime
def to_utc(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "time" not in df.columns:
        return df
    if pd.api.types.is_integer_dtype(df["time"]):
        df["time"] = pd.to_datetime(df["time"], unit="ns", utc=True)
    elif df["time"].dt.tz is None:
        df["time"] = df["time"].dt.tz_localize("UTC")
    else:
        df["time"] = df["time"].dt.tz_convert("UTC")
    return df

quotes  = to_utc(quotes)
candles = to_utc(candles)

# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 — Time range and granularity
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*70)
print("STEP 2 — Time range and granularity")
print("="*70)

for name, df, expected_s in [("quotes_l1_5s", quotes, 5), ("candles_1m", candles, 60)]:
    t = df["time"]
    print(f"\n--- {name} ---")
    print(f"  min : {t.min()}")
    print(f"  max : {t.max()}")

    diffs = df.sort_values("time").groupby("instrument")["time"].diff().dt.total_seconds().dropna()
    median_diff = diffs.median()
    max_diff    = diffs.max()
    print(f"  median consecutive diff : {median_diff:.0f}s  (expected {expected_s}s)")
    print(f"  max consecutive diff    : {max_diff:.0f}s  ({max_diff/3600:.2f} h)")

    if abs(median_diff - expected_s) > expected_s * 0.1:
        obs(2, "BLOCKER", f"{name} median diff={median_diff:.0f}s, expected {expected_s}s")
    else:
        obs(2, "GOOD", f"{name} median diff={median_diff:.0f}s ✓")

    if t.min() > pd.Timestamp("2025-12-15 01:00", tz="UTC"):
        obs(2, "NOTE", f"{name} starts late: {t.min()}")
    if t.max() < pd.Timestamp("2025-12-31 22:00", tz="UTC"):
        obs(2, "NOTE", f"{name} ends early: {t.max()}")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 3 — Instrument list
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*70)
print("STEP 3 — Instrument list")
print("="*70)

instr_symbols  = set(instrs["symbol"].unique()) if "symbol" in instrs.columns else set(instrs.index)
quotes_instrs  = set(quotes["instrument"].unique())
candles_instrs = set(candles["instrument"].unique())

print(f"\ninstruments.parquet symbols ({len(instr_symbols)}):")
for s in sorted(instr_symbols): print(f"  {s}")

print(f"\nquotes instruments  ({len(quotes_instrs)}):")
for s in sorted(quotes_instrs): print(f"  {s}")

print(f"\ncandles instruments ({len(candles_instrs)}):")
for s in sorted(candles_instrs): print(f"  {s}")

# Check all 12 expected exist — try both with and without 'deribit-' prefix
all_known = quotes_instrs | candles_instrs | instr_symbols
for sym in EXPECTED_INSTRUMENTS:
    bare = sym.replace("deribit-", "")
    if sym not in all_known and bare not in all_known:
        obs(3, "BLOCKER", f"Expected instrument not found anywhere: {sym}")

if len(instr_symbols) == 12:
    obs(3, "GOOD", "instruments.parquet contains exactly 12 symbols")
else:
    obs(3, "NOTE", f"instruments.parquet has {len(instr_symbols)} symbols (expected 12)")

# Metadata sanity
print("\ninstruments metadata:")
print(instrs.to_string())

if "strike" in instrs.columns:
    strikes = sorted(instrs["strike"].dropna().unique())
    print(f"\n  strike values: {strikes}")
    if set(strikes) == {70000, 80000, 90000, 100000, 110000}:
        obs(3, "GOOD", "strikes = [70K,80K,90K,100K,110K] ✓")
    else:
        obs(3, "NOTE", f"unexpected strikes: {strikes}")

if "is_european" in instrs.columns:
    if instrs["is_european"].all():
        obs(3, "GOOD", "all options are European ✓")
    else:
        obs(3, "NOTE", "some options NOT European — check option style")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 4 — Missing timestamps / gaps in quote data
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*70)
print("STEP 4 — Gaps in quote data")
print("="*70)

t_min = quotes["time"].min()
t_max = quotes["time"].max()
expected_rows = int((t_max - t_min).total_seconds() / 5) + 1

print(f"\nExpected rows per instrument (5s grid, {t_min} → {t_max}): {expected_rows:,}")

gap_summary = []
for instr, grp in quotes.groupby("instrument"):
    actual = len(grp)
    pct_present = actual / expected_rows * 100
    pct_missing = 100 - pct_present

    # Largest single gap
    sorted_t = grp["time"].sort_values()
    diffs = sorted_t.diff().dt.total_seconds().dropna()
    max_gap_s = diffs.max() if len(diffs) else 0

    gap_summary.append({
        "instrument": instr,
        "actual_rows": actual,
        "expected_rows": expected_rows,
        "pct_missing": round(pct_missing, 2),
        "max_gap_s": round(max_gap_s, 0),
        "max_gap_h": round(max_gap_s / 3600, 2),
    })

gap_df = pd.DataFrame(gap_summary).sort_values("pct_missing", ascending=False)
print("\nPer-instrument gap summary (sorted worst → best):")
print(gap_df.to_string(index=False))

for _, row in gap_df.iterrows():
    if row["pct_missing"] > 20:
        obs(4, "NOTE", f"{row['instrument']}: {row['pct_missing']:.1f}% missing, max gap {row['max_gap_h']:.1f}h")
    elif row["pct_missing"] > 5:
        obs(4, "NOTE", f"{row['instrument']}: {row['pct_missing']:.1f}% missing")
    else:
        obs(4, "GOOD", f"{row['instrument']}: {row['pct_missing']:.1f}% missing ✓")

    if row["max_gap_s"] > 3600:
        obs(4, "NOTE", f"{row['instrument']}: longest gap = {row['max_gap_h']:.1f}h — backtest must handle stale price")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 5 — NaN / zero bid or ask prices
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*70)
print("STEP 5 — NaN / zero bid or ask prices")
print("="*70)

price_issues = []
for instr, grp in quotes.groupby("instrument"):
    n = len(grp)
    bid_nan  = grp["bid_price"].isna().sum()
    ask_nan  = grp["ask_price"].isna().sum()
    bid_zero = (grp["bid_price"] == 0).sum()
    ask_zero = (grp["ask_price"] == 0).sum()
    crossed  = (grp["bid_price"] > grp["ask_price"]).sum()
    no_liq   = ((grp["bid_size"] == 0) & grp["bid_price"].notna()).sum()

    price_issues.append({
        "instrument": instr,
        "n": n,
        "bid_nan%": round(bid_nan / n * 100, 2),
        "ask_nan%": round(ask_nan / n * 100, 2),
        "bid_zero": bid_zero,
        "ask_zero": ask_zero,
        "crossed": crossed,
        "no_liq_bid": no_liq,
    })

pi_df = pd.DataFrame(price_issues).sort_values("bid_nan%", ascending=False)
print("\nPrice quality per instrument:")
print(pi_df.to_string(index=False))

for _, row in pi_df.iterrows():
    instr = row["instrument"]
    if row["bid_nan%"] > 50 or row["ask_nan%"] > 50:
        tag = "BLOCKER" if FORWARD in instr else "NOTE"
        obs(5, tag, f"{instr}: {row['bid_nan%']:.0f}% bid NaN, {row['ask_nan%']:.0f}% ask NaN")
    if row["crossed"] > 10:
        obs(5, "NOTE", f"{instr}: {row['crossed']} crossed quotes (bid > ask)")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 6 — BTC forward price plot
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*70)
print("STEP 6 — BTC forward price")
print("="*70)

fwd = quotes[quotes["instrument"] == FORWARD].copy()
fwd["mid"] = (fwd["bid_price"] + fwd["ask_price"]) / 2
fwd["spread"] = fwd["ask_price"] - fwd["bid_price"]
fwd = fwd.sort_values("time")

print(f"\nForward price stats:")
print(fwd[["time", "bid_price", "ask_price", "mid", "spread"]].describe())

fig, axes = plt.subplots(2, 1, figsize=(14, 7), sharex=True)
axes[0].plot(fwd["time"], fwd["mid"], lw=0.6, color="steelblue", label="mid")
axes[0].set_ylabel("Price (USD)")
axes[0].set_title("BTC-30JAN26-future Mid Price")
axes[0].legend()
axes[0].yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x:,.0f}"))

axes[1].plot(fwd["time"], fwd["spread"], lw=0.5, color="orange", label="ask−bid")
axes[1].set_ylabel("Spread (USD)")
axes[1].set_title("Forward Bid-Ask Spread")
axes[1].legend()
axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
fig.autofmt_xdate()
plt.tight_layout()
fig.savefig(PLOTS / "btc_forward_price.png", dpi=150)
plt.close(fig)
print(f"  → saved {PLOTS / 'btc_forward_price.png'}")

mid_min, mid_max = fwd["mid"].min(), fwd["mid"].max()
if mid_min < 50_000 or mid_max > 200_000:
    obs(6, "NOTE", f"Forward mid outside expected range: min={mid_min:.0f}, max={mid_max:.0f}")
else:
    obs(6, "GOOD", f"Forward mid range: ${mid_min:,.0f} – ${mid_max:,.0f} ✓")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 7 — Bid-ask spread per option
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*70)
print("STEP 7 — Bid-ask spread per option")
print("="*70)

options_q = quotes[quotes["instrument"].str.endswith("-option")].copy()
options_q["spread"] = options_q["ask_price"] - options_q["bid_price"]
options_q["mid"]    = (options_q["bid_price"] + options_q["ask_price"]) / 2
options_q["rel_spread"] = options_q["spread"] / options_q["mid"].replace(0, np.nan)

spread_stats = options_q.groupby("instrument")[["spread", "rel_spread"]].agg(
    ["median", "mean", "max"]
)
print("\nSpread stats per option:")
print(spread_stats.to_string())

option_list = sorted(options_q["instrument"].unique())
n_opts = len(option_list)
cols = 2
rows = (n_opts + 1) // cols

fig, axes = plt.subplots(rows, cols, figsize=(16, rows * 3), sharex=False)
axes_flat = axes.flatten() if n_opts > 1 else [axes]

for ax, instr in zip(axes_flat, option_list):
    grp = options_q[options_q["instrument"] == instr].sort_values("time")
    ax.plot(grp["time"], grp["spread"], lw=0.4, alpha=0.8)
    ax.set_title(instr.replace("deribit-", ""), fontsize=8)
    ax.set_ylabel("Spread $", fontsize=7)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
    ax.tick_params(axis="both", labelsize=6)

for ax in axes_flat[n_opts:]:
    ax.set_visible(False)

fig.suptitle("Bid-Ask Spread per Option Over Time", fontsize=11)
plt.tight_layout()
fig.savefig(PLOTS / "option_spreads.png", dpi=150)
plt.close(fig)
print(f"  → saved {PLOTS / 'option_spreads.png'}")

for instr in option_list:
    grp = options_q[options_q["instrument"] == instr]
    med_rel = grp["rel_spread"].median()
    if pd.isna(med_rel):
        obs(7, "NOTE", f"{instr}: all-NaN spread (no quotes)")
    elif med_rel > 0.10:
        obs(7, "NOTE", f"{instr}: median relative spread = {med_rel:.1%} — wide, mid-fill assumption optimistic")
    else:
        obs(7, "GOOD", f"{instr}: median relative spread = {med_rel:.1%} ✓")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 8 — Put-call parity sanity check
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*70)
print("STEP 8 — Put-call parity sanity check")
print("="*70)

RISK_FREE = 0.0

# Sample ~10 timestamps evenly across the period
all_times = quotes["time"].sort_values().unique()
sample_idx = np.linspace(0, len(all_times) - 1, 10, dtype=int)
sample_times = all_times[sample_idx]

# Pivot quotes to wide format for a quick join
pivot = (
    quotes
    .assign(mid=lambda d: (d["bid_price"] + d["ask_price"]) / 2)
    [["time", "instrument", "mid", "bid_price", "ask_price"]]
)

parity_rows = []
for ts in sample_times:
    snap = pivot[pivot["time"] == ts].set_index("instrument")
    if FORWARD not in snap.index:
        continue
    F = snap.loc[FORWARD, "mid"]
    T = (EXPIRY - pd.Timestamp(ts)).total_seconds() / (365.25 * 86400)

    for strike in [70000, 80000, 90000, 100000, 110000]:
        # Try to find matching call/put — search for strike and C/P in instrument name
        calls = [i for i in snap.index if f"-{strike}-" in i and i.endswith("-C-option")]
        puts  = [i for i in snap.index if f"-{strike}-" in i and i.endswith("-P-option")]
        if not calls or not puts:
            continue

        C_mid = snap.loc[calls[0], "mid"]
        P_mid = snap.loc[puts[0], "mid"]
        K     = strike

        parity_lhs = C_mid - P_mid
        parity_rhs = F - K * np.exp(-RISK_FREE * T)
        deviation  = parity_lhs - parity_rhs
        deviation_pct = deviation / F * 100

        # Parity band using bid/ask
        C_bid, C_ask = snap.loc[calls[0], "bid_price"], snap.loc[calls[0], "ask_price"]
        P_bid, P_ask = snap.loc[puts[0], "bid_price"], snap.loc[puts[0], "ask_price"]
        # Lower bound: buy call at ask, sell put at bid → C_ask - P_bid ≥ F - K (no arb)
        # Upper bound: sell call at bid, buy put at ask → C_bid - P_ask ≤ F - K
        lower = C_ask - P_bid - parity_rhs
        upper = C_bid - P_ask - parity_rhs

        parity_rows.append({
            "time": ts,
            "strike": strike,
            "F": round(F, 2),
            "C_mid": round(C_mid, 4) if not pd.isna(C_mid) else np.nan,
            "P_mid": round(P_mid, 4) if not pd.isna(P_mid) else np.nan,
            "parity_lhs": round(parity_lhs, 2) if not pd.isna(parity_lhs) else np.nan,
            "parity_rhs": round(parity_rhs, 2),
            "deviation": round(deviation, 2) if not pd.isna(deviation) else np.nan,
            "deviation_%": round(deviation_pct, 3) if not pd.isna(deviation_pct) else np.nan,
            "arb_lower": round(lower, 2) if not pd.isna(lower) else np.nan,
            "arb_upper": round(upper, 2) if not pd.isna(upper) else np.nan,
        })

par_df = pd.DataFrame(parity_rows)
if par_df.empty:
    obs(8, "BLOCKER", "No parity rows computed — check instrument name patterns")
else:
    print("\nPut-call parity check (sample timestamps):")
    print(par_df.to_string(index=False))

    # Plot deviation % by strike
    fig, ax = plt.subplots(figsize=(12, 5))
    for strike, grp in par_df.groupby("strike"):
        ax.plot(grp["time"], grp["deviation_%"], marker="o", ms=4, lw=1, label=f"K={strike//1000}K")
    ax.axhline(0, color="black", lw=0.8, ls="--")
    ax.set_ylabel("C - P - (F - K·e^{-rT})  as % of F")
    ax.set_title("Put-Call Parity Deviation by Strike")
    ax.legend(fontsize=8)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
    fig.autofmt_xdate()
    plt.tight_layout()
    fig.savefig(PLOTS / "parity_deviation.png", dpi=150)
    plt.close(fig)
    print(f"  → saved {PLOTS / 'parity_deviation.png'}")

    max_dev = par_df["deviation_%"].abs().max()
    if max_dev > 5:
        obs(8, "NOTE", f"Max parity deviation = {max_dev:.2f}% of F — investigate if systematic")
    else:
        obs(8, "GOOD", f"Max parity deviation = {max_dev:.2f}% of F ✓")

    # Check if arbitrage band is ever crossed
    arb_crossed = par_df[par_df["arb_lower"] > 0.01]
    if not arb_crossed.empty:
        obs(8, "NOTE", f"{len(arb_crossed)} parity band breaches (lower > 0) — likely spread artefact, not real arb")
    else:
        obs(8, "GOOD", "No parity band breaches found ✓")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 9 — Observations summary
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*70)
print("STEP 9 — Observations Summary")
print("="*70)

blockers = [o for o in OBS if "BLOCKER" in o]
notes    = [o for o in OBS if "NOTE" in o]
goods    = [o for o in OBS if "GOOD" in o]

print(f"\nBLOCKERS ({len(blockers)}):")
for b in blockers: print(f"  {b}")

print(f"\nNOTES ({len(notes)}):")
for n in notes: print(f"  {n}")

print(f"\nGOOD ({len(goods)}):")
for g in goods: print(f"  {g}")

obs_path = PLOTS.parent / "phase0_observations.txt"
with open(obs_path, "w") as f:
    f.write("Phase 0 Observations\n")
    f.write("=" * 60 + "\n\n")
    f.write("BLOCKERS:\n")
    for b in blockers: f.write(f"  {b}\n")
    f.write("\nNOTES:\n")
    for n in notes: f.write(f"  {n}\n")
    f.write("\nGOOD:\n")
    for g in goods: f.write(f"  {g}\n")

print(f"\nObservations written to {obs_path}")
print("\nPlots saved to:", PLOTS)
print("  btc_forward_price.png")
print("  option_spreads.png")
print("  parity_deviation.png")
