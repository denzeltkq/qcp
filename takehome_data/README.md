# Crypto Markets Take-Home

Pick **one** of the two tracks below and build it end-to-end against the provided market data.

- **Track A — Mini Event-Driven Trading System.** Replay market data, run strategies on spot/perp and options, route orders through risk + OMS + a matching engine, track position and PnL.
- **Track B — Mini Event-Driven Options Backtester.** Build an event-driven backtester that prices an options book tick-by-tick, evaluates option strategies on the historical chain, and reports PnL and Greeks attribution.

Either track is a complete submission. Don't do both — depth beats breadth.

**Time budget:** ~7 days. Code quality and clear thinking matter more than feature breadth.

**Languages:** Python, Go, or Rust. Pick whichever lets you ship the cleanest design. Code samples below are shown in Python for brevity — translate the shape directly into idiomatic Go (`struct` + methods) or Rust (`struct` + `impl` / traits).

---

## What's in the box

```
candles_1m.parquet      OHLCV bars at 1-minute granularity
quotes_l1_5s.parquet    Top-of-book bid/ask snapshots, 5-second buckets
instruments.parquet     Instrument metadata (type, strike, expiry, ...)
```

All data is from **Deribit** (coin-margined / inverse), spans **2025-12-15 → 2025-12-31 UTC**.

### Universe (12 instruments)

| Type | Instrument |
|------|------------|
| Perpetual | `deribit-BTC-PERPETUAL-future` |
| Forward (1-month) | `deribit-BTC-30JAN26-future` |
| Options (10) | `deribit-BTC-30JAN26-{70000,80000,90000,100000,110000}-{C,P}-option` |

The forward and option chain share the same expiry (2026-01-30).

### Schemas

**candles_1m.parquet**
```
time                              UTC timestamp (datetime64)
instrument                        string
price_open/high/low/close/vwap    float
volume                            float (contracts)
candle_usd_volume                 float
candle_trades_count               int
```

**quotes_l1_5s.parquet**
```
time                                        UTC timestamp (5s floored)
instrument                                  string
bid_price, bid_size, ask_price, ask_size    float
```

**instruments.parquet** — per-instrument metadata:
`type, symbol, strike, option_type, expiration, contract_size, is_european, tick_size, size_asset, margin_asset`.

Parquet readers exist in all three target languages (e.g. `pyarrow` / `polars`, `parquet-go` or `arrow-go`, `arrow2` / `parquet` crate). CSV exports of the same data can be produced if you'd rather not pull in a Parquet dependency — just note it in your README.

---

# Track A — Mini Event-Driven Trading System

Build a small event-driven trading system that replays the data and feeds the pipeline:

```
event ──► strategy ──► risk ──► OMS ──► fill simulator ──► position / PnL
```

## What to build

### 1. Market data replay
Read the parquet files and feed events to the strategy and OMS. Chronological order is enough — you don't need to engineer a fancy interleaver. Either stream (candles, quotes, or both) is fine as long as your strategies have what they need to make decisions.

### 2. Strategies (at least 2)
- **Spot/perp strategy** on `deribit-BTC-PERPETUAL-future` — anything reasonable: MA cross, mean reversion, breakout. Doesn't need to be profitable.
- **Option strategy** — e.g. covered call (long perp/forward + short ATM call), short straddle, or delta-hedged single option.

### 3. Order Management System

```python
class Order:
    order_id: str
    symbol: str
    side: Literal["BUY", "SELL"]
    qty: int
    order_type: Literal["LIMIT", "MARKET"]
    limit_price: float | None
    status: OrderStatus       # ACKED | PARTIALLY_FILLED | FILLED | CANCELLED | REJECTED
    avg_fill_price: float
    filled_qty: int
    submitted_ts: int
    last_update_ts: int
```

Generate unique order IDs, manage lifecycle/transitions, track average fill price. In Go/Rust, model `OrderStatus` as an enum / sum type rather than a string.

### 4. Fill Simulator

No standalone matching engine — just a small simulator that turns an order + the current top-of-book into fills. Read the latest snapshot from `quotes_l1_5s.parquet`.

```python
class FillSimulator:
    def on_quote(self, quote) -> list[Fill]: ...   # check resting limits against new BBO
    def submit(self, order, bbo) -> list[Fill]: ... # immediate evaluation against current BBO
    def cancel(self, order_id) -> None: ...
```

Crossing rules (use these — no need to invent your own):
- **MARKET BUY** fills at `ask_price`, **MARKET SELL** fills at `bid_price`, sized by `min(order_qty, opposite_side_size)`. Unfilled remainder is rejected (no resting market orders).
- **LIMIT BUY** fills at `min(limit_price, ask_price)` if `limit_price >= ask_price`; otherwise rests until a future quote crosses. Symmetric for SELL.
- One quote → at most one fill per resting order.

No queue position modelling, no partial-fill probability — keep it simple and document any deviation.

### 5. Risk Monitor (must run pre-trade on every order)
- max order qty per symbol
- max position per symbol
- max gross notional (portfolio)
- max daily loss (portfolio)
- **max |portfolio delta|** (in BTC or USD — pick one and document)
- **max |portfolio gamma|** (per 1% move, or per 1 USD move — document the unit)
- kill switch

Delta and gamma checks are evaluated on the *post-trade* portfolio (i.e. simulate the fill, recompute portfolio Greeks, reject if the projected exposure breaches the limit). For options, use the Greeks from your pricer; for perp/forward, delta = signed contract notional, gamma = 0. If your strategy doesn't trade options, delta/gamma checks still need to be wired in — they just won't bind.

Reject orders that breach. Once the kill switch trips or daily-loss threshold is hit, reject all subsequent orders.

### 6. Position & PnL
Track per-symbol position and realized PnL. Print (or log) position + PnL whenever they change — that's enough. No need for fancy accounting layers, attribution, or live dashboards.

## Config (`config.yaml`)

```yaml
per_symbol:
  deribit-BTC-PERPETUAL-future:        {max_pos: 50, max_order_qty: 20}
  deribit-BTC-30JAN26-future:          {max_pos: 50, max_order_qty: 20}
  deribit-BTC-30JAN26-*-option:        {max_pos: 30, max_order_qty: 10}

portfolio:
  max_gross_notional: 500_000
  max_daily_loss: -10_000
  max_abs_delta_btc: 5.0          # |sum of deltas| in BTC equivalent
  max_abs_gamma_per_pct: 0.5      # |portfolio gamma| per 1% underlying move
  kill_switch: false
```

Hardcode or tweak — just make it loadable from a file.

## Out of scope (Track A)

- No funding-rate accounting on the perp.
- No expiry settlement — replay ends 2025-12-31, options expire 2026-01-30.

## Deliverables (Track A)

1. **Code** — runnable end-to-end (e.g. `python main.py`, `go run ./cmd/...`, or `cargo run --release`). Pin language version + dependencies.
2. **Output** — for each strategy, an end-of-replay summary (stdout or file):
   - Final positions, realized PnL, gross/net exposure
   - Order count by status (ACKED / FILLED / PARTIAL / CANCELLED / REJECTED)
   - Risk-rejection log
3. **Write-up** — a few paragraphs: design overview, key assumptions (fill rules, fees, slippage), what you'd improve with more time.

## Evaluation (Track A)

What we look at, in rough order:

1. Clear separation of concerns (strategy / risk / OMS / fills / portfolio).
2. Correct order lifecycle and PnL math.
3. Risk checks are pre-trade and unavoidable.
4. Faithful application of the spec'd fill rules.
5. Readable code, decent tests for the non-trivial bits (PnL, risk, fills).
6. Clean event-driven flow (no polling, no global state).

---

# Track B — Mini Event-Driven Options Backtester

Build a small **event-driven** backtester focused on the option chain. Less OMS / fill plumbing than Track A, more numerical + financial reasoning, but the engine still has to step through events one-at-a-time and never look ahead. No vectorised "compute everything in one pass over the whole DataFrame" shortcuts — the reviewer will be looking for a clear event loop with a single source of time.

## What to build

### 1. Pricing & Greeks
Implement Black-Scholes (or Black-76 on the forward) for European options. Compute price, delta, gamma, vega, theta. The instruments are flagged `is_european`, expiry is 2026-01-30, and `deribit-BTC-30JAN26-future` gives you the forward.

For implied vol: invert BSM from the mid-quote in `quotes_l1_5s.parquet` per timestamp per option. Cache an IV surface (strike × time) — you'll reuse it.

Risk-free rate: assume 0 (or document a constant). Carry comes from the forward.

### 2. Strategy library (at least 2)
Pick any two — keep them simple, well-defined, and reproducible:
- **Covered call** — long perp/forward + short ATM call, rolled or held.
- **Short straddle / strangle** — daily or weekly entry, fixed delta or ATM.
- **Delta-hedged short vol** — short an option, hedge delta with the perp on a fixed cadence (e.g. every quote bar, or when |delta drift| > threshold).

Each strategy is a function from market state → target position per instrument.

### 3. Event-driven backtest engine
Replay the data as a single chronological event stream (candles + quote snapshots interleaved by timestamp). Drive everything from one event loop — no batch passes that peek at future bars.

Suggested event flow:
```
event ──► market state update ──► strategy.on_event() ──► fill simulator ──► portfolio update ──► metrics
```

At each event:
1. Update market state for the affected instrument(s) (price, IV, Greeks).
2. Strategy receives the event and emits target positions or discrete orders.
3. Fill simulator turns intents into fills against the current top-of-book — fill at quote mid + configurable spread cost (e.g. cross half the spread). No knowledge of the next bar.
4. Update portfolio: positions, cash, mark-to-market PnL, realized PnL on closes.
5. Record Greeks exposure (portfolio delta, gamma, vega, theta) per event or on a fixed sampling cadence.

Hard rules:
- One source of time. The strategy can only see information whose timestamp ≤ the current event.
- Re-hedging, entries, and exits are triggered *by events* (a quote tick, a timer event you inject, a Greek crossing a threshold) — not by iterating a pre-built schedule that ignores the event stream.
- Document your fill model and any slippage / fee assumptions explicitly.

### 4. Reporting
At the end of the backtest, output per strategy:
- Equity curve (CSV or plot).
- Final and time-series PnL (realized, unrealized, total).
- Greeks exposure over time.
- Trade log (entry/exit, instrument, qty, price, PnL contribution).
- Summary stats: total PnL, max drawdown, hit rate.

### 5. Config (`config.yaml`)

```yaml
backtest:
  start: 2025-12-15
  end:   2025-12-31
  rehedge_interval_s: 60      # for delta-hedged strategies
  spread_cost_bps: 5          # half-spread paid on each fill
  fee_bps: 0
risk_free_rate: 0.0
strategies:
  short_straddle:
    enabled: true
    notional: 50_000
    entry_time_utc: "00:00"
  covered_call:
    enabled: true
    delta_target: 0.25
```

## Bonus (nice-to-have, not required)

- **PnL attribution** — decompose realized + unrealized PnL into delta-PnL, vega-PnL, theta-PnL (and optionally gamma / cross terms). Useful for telling whether a strategy actually got paid for the risk it ran.
- Stress / scenario PnL: shift spot ±X%, IV ±Y vol points, recompute portfolio value.
- Smile-aware IV interpolation across strikes (vs. per-option IV).

Don't attempt these until the core deliverables are solid.

## Out of scope (Track B)

- No expiry settlement — replay ends 2025-12-31, options expire 2026-01-30. Mark-to-market via IV surface is fine.
- No American-style early exercise (everything is European).
- No funding-rate accounting on the perp — document if you assume zero.
- No live order book modelling beyond top-of-book mid + half-spread.

## Deliverables (Track B)

1. **Code** — runnable end-to-end (`python main.py`, `go run ./cmd/...`, `cargo run --release`). Pin language version + deps.
2. **Output** — for each strategy: equity curve, summary stats, trade log, Greeks time series. CSV is fine.
3. **Write-up** — design overview, pricing/IV assumptions, fill/slippage model, sanity checks you ran (e.g. put-call parity, BSM vs. analytical sanity at known points), what you'd improve with more time.

## Evaluation (Track B)

What we look at, in rough order:

1. Clean event-driven loop — single source of time, no look-ahead, no batch shortcuts.
2. Correctness of pricing, Greeks, and IV inversion (sanity checks visible).
3. Honest fill model — fills decided from current state only, no free liquidity.
4. Clean separation: pricing / strategy / engine / reporting.
5. Sensible portfolio accounting (cash, mark-to-market, realized vs. unrealized).
6. Readable code, tests for pricing and PnL.
7. Thoughtful write-up — what the strategy is actually exposed to, and what the backtest can and can't tell you.

---

## Common notes

- Pin language toolchain version and dependencies (`requirements.txt` / `go.mod` / `Cargo.toml`).
- Treat the data as ground truth — no live network calls.
- Tests on the non-trivial bits (PnL, risk, fills, pricing) are expected; full coverage is not.
- We care more about how you reason about ambiguity than about hitting every bullet. State your assumptions and move on.
