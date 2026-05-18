"""
IV surface builder.

Pipeline:
  1. Load raw quotes (original events only, not forward-filled)
  2. Build a forward-filled forward price series for reference
  3. Solve IV only at original quote timestamps — no brentq on filled rows
  4. Pivot solved IVs to (timestamp × strike) DataFrames
  5. Forward-fill the IV surface onto the full 5s grid
  6. Mark is_stale = True where gap since last real IV > 15 min
  7. Cache to parquet; plot heatmaps

Usage:
    python pricing/iv_surface.py                  # build and cache
    python pricing/iv_surface.py --force-rebuild  # ignore cache and rebuild
"""

import argparse
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from pricing.iv_solver import solve_iv_for_quotes

DATA    = Path(__file__).parent.parent / "takehome_data"
CACHE   = Path(__file__).parent / "cache"
PLOTS   = Path(__file__).parent.parent / "explore" / "plots"
EXPIRY  = pd.Timestamp("2026-01-30 08:00:00", tz="UTC")
FORWARD = "deribit-BTC-30JAN26-future"
STALE_S = 15 * 60   # 15 minutes


def _to_utc(df: pd.DataFrame) -> pd.DataFrame:
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


def load_quotes(data_path: Path) -> pd.DataFrame:
    df = pd.read_parquet(data_path / "quotes_l1_5s.parquet")
    df = _to_utc(df)
    df["mid"] = (df["bid_price"] + df["ask_price"]) / 2
    return df


def build_forward_series(quotes: pd.DataFrame) -> pd.Series:
    """
    Forward-fill the forward price onto the full 5s grid.
    Returns a Series indexed by UTC timestamp.
    """
    fwd_raw = quotes[quotes["instrument"] == FORWARD][["time", "mid"]].copy()
    fwd_raw = fwd_raw.drop_duplicates("time").set_index("time").sort_index()

    t_min = quotes["time"].min().floor("5s")
    t_max = quotes["time"].max().ceil("5s")
    grid  = pd.date_range(t_min, t_max, freq="5s", tz="UTC")

    return fwd_raw["mid"].reindex(grid).ffill()


def _add_stale_flag(
    surface: pd.DataFrame,
    real_times_per_strike: dict[int, pd.Index],
) -> pd.DataFrame:
    """
    Add an is_stale_<K> column for each strike.
    A row is stale if it has been forward-filled beyond STALE_S seconds
    since the last real IV solve for that strike.
    """
    for K in surface.columns:
        real_ts = real_times_per_strike.get(K, pd.Index([]))
        # Build last-real-time series over the full index
        last_real = pd.Series(np.nan, index=surface.index, dtype="float64")
        if len(real_ts) > 0:
            valid = real_ts[real_ts.isin(surface.index)]
            last_real.loc[valid] = valid.asi8.astype(float) / 1e9
        last_real = last_real.ffill()
        gap_s = (surface.index.asi8.astype(float) / 1e9) - last_real.values
        surface[f"stale_{K}"] = (gap_s > STALE_S) | last_real.isna()

    return surface


def _ffill_surface_with_stale(
    iv_long: pd.DataFrame,
    flag: str,
) -> pd.DataFrame:
    """
    Pivot solved IVs for one option type (flag='c' or 'p') into a wide
    (timestamp × strike) DataFrame, forward-fill, and add stale flags.

    Returns a DataFrame with:
      - columns [70000, 80000, 90000, 100000, 110000] for IV values
      - columns [stale_70000, ...] for stale flags
    """
    strikes = [70_000, 80_000, 90_000, 100_000, 110_000]
    sub = iv_long[iv_long["flag"] == flag][["time", "strike", "iv"]].copy()

    # Pivot: one column per strike
    pivot = sub.pivot_table(index="time", columns="strike", values="iv", aggfunc="last")
    pivot = pivot.reindex(columns=strikes)

    # Track which (time, strike) pairs had real solves
    real_times_per_strike: dict[int, pd.Index] = {}
    for K in strikes:
        if K in pivot.columns:
            real_times_per_strike[K] = pivot.index[pivot[K].notna()]

    # Expand to full 5s grid
    t_min = sub["time"].min().floor("5s")
    t_max = sub["time"].max().ceil("5s")
    grid  = pd.date_range(t_min, t_max, freq="5s", tz="UTC")

    surface = pivot.reindex(grid)
    surface.index.name = "time"

    # Forward-fill IV values
    surface[strikes] = surface[strikes].ffill()

    # Add stale flags AFTER ffill
    surface = _add_stale_flag(surface, real_times_per_strike)

    return surface


def build_surface(force_rebuild: bool = False):
    CACHE.mkdir(exist_ok=True)
    PLOTS.mkdir(exist_ok=True)

    cache_c = CACHE / "iv_surface_calls.parquet"
    cache_p = CACHE / "iv_surface_puts.parquet"

    if not force_rebuild and cache_c.exists() and cache_p.exists():
        print("Loading IV surface from cache...")
        iv_c = pd.read_parquet(cache_c)
        iv_p = pd.read_parquet(cache_p)
        print(f"  calls: {iv_c.shape}  puts: {iv_p.shape}")
        return iv_c, iv_p

    # ── Step 1: Load raw quotes ──────────────────────────────────────────────
    print("Loading quotes...")
    quotes = load_quotes(DATA)
    print(f"  {len(quotes):,} total quote events across all instruments")

    # ── Step 2: Forward-filled forward price for currency conversion ─────────
    print("Building forward price series...")
    fwd_series = build_forward_series(quotes)
    print(f"  Forward: {fwd_series.min():.0f} – {fwd_series.max():.0f} USD "
          f"({fwd_series.notna().sum():,} valid slots)")

    # ── Step 3: Convert option mids to USD and solve ─────────────────────────
    option_quotes = quotes[quotes["instrument"].str.endswith("-option")].copy()
    # Align forward price to each quote's timestamp
    option_quotes["fwd_usd"] = option_quotes["time"].map(fwd_series)
    option_quotes["mid_usd"] = option_quotes["mid"] * option_quotes["fwd_usd"]

    # Drop rows where forward is unavailable
    option_quotes = option_quotes[option_quotes["fwd_usd"].notna() & (option_quotes["fwd_usd"] > 0)]

    print(f"\nSolving IV on {len(option_quotes):,} original option quote events...")
    iv_long = solve_iv_for_quotes(option_quotes, fwd_series, EXPIRY)

    # ── Step 4 & 5: Pivot, ffill, stale-flag ────────────────────────────────
    print("\nBuilding forward-filled IV surface with stale flags...")
    iv_c = _ffill_surface_with_stale(iv_long, "c")
    iv_p = _ffill_surface_with_stale(iv_long, "p")

    # Report stale coverage
    strikes = [70_000, 80_000, 90_000, 100_000, 110_000]
    for surf, label in [(iv_c, "calls"), (iv_p, "puts")]:
        n = len(surf)
        stale_cols = [f"stale_{K}" for K in strikes]
        any_stale = surf[stale_cols].any(axis=1).sum()
        print(f"  {label}: {n:,} rows, {any_stale:,} ({any_stale/n:.1%}) have at least one stale strike")

    # ── Step 6: Cache ────────────────────────────────────────────────────────
    print(f"\nSaving cache to {CACHE}...")
    iv_c.to_parquet(cache_c)
    iv_p.to_parquet(cache_p)
    print(f"  iv_surface_calls.parquet  {cache_c.stat().st_size / 1024:.0f} KB")
    print(f"  iv_surface_puts.parquet   {cache_p.stat().st_size / 1024:.0f} KB")

    return iv_c, iv_p


def print_surface_stats(iv_c: pd.DataFrame, iv_p: pd.DataFrame):
    strikes = [70_000, 80_000, 90_000, 100_000, 110_000]
    print("\nIV Surface Summary:")
    for surf, label in [(iv_c, "Calls"), (iv_p, "Puts")]:
        iv_cols = [c for c in strikes if c in surf.columns]
        medians = surf[iv_cols].median() * 100
        print(f"\n  {label} median IV / NaN count (full 5s grid):")
        for k in iv_cols:
            stale_col = f"stale_{k}"
            stale_n = surf[stale_col].sum() if stale_col in surf.columns else 0
            nan_n   = surf[k].isna().sum()
            print(f"    K={int(k)//1000}K : {medians[k]:.1f}%  NaN={nan_n:,}  stale={stale_n:,} / {len(surf):,}")


def plot_iv_heatmap(
    iv_df: pd.DataFrame,
    strikes: list[int],
    title: str,
    output_path: Path,
):
    """Plot IV values (ignoring stale rows) as a heatmap: x=date, y=strike."""
    iv_only = iv_df[strikes].copy()

    # Mask stale values so they don't colour the heatmap
    for K in strikes:
        stale_col = f"stale_{K}"
        if stale_col in iv_df.columns:
            iv_only.loc[iv_df[stale_col], K] = np.nan

    # Resample to hourly median
    iv_hourly = iv_only.resample("1h").median()
    data = iv_hourly.T * 100   # strikes × time, in percent

    fig, ax = plt.subplots(figsize=(16, 5))
    im = ax.imshow(
        data.values,
        aspect="auto",
        origin="lower",
        cmap="viridis",
        vmin=0,
        vmax=200,
    )

    n_cols = data.shape[1]
    tick_step = max(1, n_cols // 12)
    ax.set_xticks(range(0, n_cols, tick_step))
    ax.set_xticklabels(
        [t.strftime("%m-%d") for t in iv_hourly.index[::tick_step]],
        rotation=45, ha="right", fontsize=8,
    )
    ax.set_yticks(range(len(strikes)))
    ax.set_yticklabels([f"{int(k)//1000}K" for k in strikes], fontsize=9)

    cbar = fig.colorbar(im, ax=ax, pad=0.02)
    cbar.set_label("Implied Vol (%)", fontsize=9)
    ax.set_title(title, fontsize=11)
    ax.set_xlabel("Date", fontsize=9)
    ax.set_ylabel("Strike", fontsize=9)

    plt.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"  → saved {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--force-rebuild", action="store_true")
    args = parser.parse_args()

    iv_c, iv_p = build_surface(force_rebuild=args.force_rebuild)

    print_surface_stats(iv_c, iv_p)

    strikes = [70_000, 80_000, 90_000, 100_000, 110_000]
    PLOTS.mkdir(exist_ok=True)
    print("\nGenerating heatmaps...")
    plot_iv_heatmap(iv_c, strikes, "BTC Options — Call IV Surface (Dec 15–31 2025)", PLOTS / "iv_surface_calls.png")
    plot_iv_heatmap(iv_p, strikes, "BTC Options — Put IV Surface (Dec 15–31 2025)",  PLOTS / "iv_surface_puts.png")

    print("\nDone. Run 'python -m pricing.validation' to check smile and NaN counts.")
