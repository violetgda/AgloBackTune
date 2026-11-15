# AlgoBackTune

**AI Strategy External Backtester** for MetaTrader 5 Expert Advisors — backtest and
optimize *any* EA from its `.mq5` source, without opening MT5 and the terminal.

The EA source file can be translated to a Python strategy module by the
**MetaEditor AI Assistant** (or any LLM/AI tool you prefer — this app never calls
one itself). The translated module is pasted back into the app, and APPLY AS STRATEGY, 
then executed by the app's Numba-based tick-level backtest engine.

## How it works

1. **Copy the prompt** shown at the *STRATEGY* tab (the **Copy Prompt** button). 
   Paste it into the **MetaEditor AI Assistant** (or any AI tool you trust), 
   paste your .mq5 code underneath and execute
2. **Wait for the result**, paste the generated Python module into the STRATEGY tab, 
   and select **APPLY AS STRATEGY**.
   The code is then:
    - syntax-checked (`compile`),
    - required to define `simulate_range(...)` and `EXIT_LABEL`,
    - smoke-tested in-process on synthetic ticks (a couple of minutes max),
    - written to `%APPDATA%\AlgoBackTune\strategy_cache\strategy_<hash>.py`
      and auto-loaded for this EA.
3. **Backtest & optimize** as usual: the translated strategy runs on the same
   tick-level engine as the built-in default.

### ⚠ Important — read this first

AI-assistant-generated module is **not** guaranteed to be a correct 
re-implementation of the MQL5 original. 
**Always validate the generated strategy against an MT5 tester report** (run
the real EA once in MT5 with the same symbol/period/inputs/ticks and compare
net profit, deal counts, and per-trade results) before trusting any number
this app shows — and never use the result for live trading
without human review.

## Reference strategy module (no built-in default)

The app ships **no built-in default strategy**: a strategy module is
**required** before any backtest or optimization can run. The engine
raises a clear error if you try to run without one.

`ticklab_algo/strategy.py` is kept only as a **reference / test fixture**
for the strategy contract (a small EMA-fast / SMA-slow cross with
net-profit target / max-drawdown close-all). It is **not** matched to any
real EA and is not used as a fallback at runtime. To use it as a starting
point, paste its contents into the STRATEGY tab and click
**APPLY AS STRATEGY** — but always compare against your real EA's results
in the MT5 tester.

## Running

```bat
run.bat
```

(or `py -3 -m ticklab_algo`, or `run_tanpa_konsol.vbs` without a console window)

Requires: Python 3.10+, `numpy`, `numba`. MT5 terminal does **not** need to
be turn on — but the MT5 **tick cache** on this machine is required
(`%APPDATA%\MetaQuotes\Terminal` and `Tester`), i.e. the symbol/period must
have been backtested or downloaded at least once in MT5.

## Features

- **STRATEGY tab** (first tab): paste the AI-assistant-generated module for
  the loaded EA, APPLY AS STRATEGY (syntax check + synthetic smoke test +
  per-EA strategy cache). The paste field shows **line numbers** (gutter) so
  indentation / syntax errors can be located quickly.
- **Startup notice** (shown once on first launch, dismissible with
  "Don't show again"): reminds that results are not guaranteed accurate, that
  a `.mq5` EA source and an installed+logged-in MT5 are required, and that
  results must be validated against the MT5 Strategy Tester first.
- **EA SETTINGS tab**: parameters is auto load from strategy module, 
  one editable row per `.mq5` input (From/Step/To + OPT checkbox + int-aware counts).
- **SUMMARY / CHART / TRADES / OPTIMIZATION RESULTS / LOG** tabs.
- Real ticks only (100% ticks, via the local MT5 cache, no MT5 GUI).
- Streaming optimization with Stop button; Full grid and Quick (random +
  local) algorithms.
- Exports: CSV / XML / HTML report (trades table + metrics).
- **Cache**: each distinct `.mq5` source hashes to its own
  `strategy_<hash>.py` in `%APPDATA%\AlgoBackTune\strategy_cache\`
  (+ a `.json` sidecar with the generation metadata). Reload or re-select the
  EA and the cached strategy is used automatically — no re-generation needed.

## Layout

```
AlgoBackTune/
├── ticklab_algo/
│   ├── app.py          # GUI (tabs: STRATEGY + 6 others)
│   ├── strategy_cache.py # per-EA strategy cache, syntax check, smoke test
│   ├── config.py       # .mq5 input parser, Settings/EAParams, PARAM layout
│   ├── engine.py       # tick -> bars -> strategy simulation, metrics
│   ├── strategy.py     # reference / test fixture for the strategy contract (not a runtime fallback)
│   ├── optimizer.py    # grid / quick optimization
│   ├── report.py       # CSV/XML/HTML export
│   ├── bars.py         # tick aggregation, EMA/SMA
│   └── tickcache.py    # MT5 tick-cache reader (.tkc)
├── tests/test_smoke.py
└── run.bat / run_tanpa_konsol.vbs
```

## The strategy contract (what the AI assistant is asked to produce)

Generated code must define:

```python
EXIT_LABEL = {1: "...", 2: "..."}   # reason codes used in the report

@njit(cache=True)
def simulate_range(p, t_ms, bid, ask, bar_start,
                   m_time, m_open, m_high, m_low, m_close,
                   b_from, b_to, start_bar, tick_from,
                   equity, tr_entry_time, ..., npos_arr,
                   active_from_ms, progress_arr, live):
    ...
```

Parameter vector `p` layout: `p[0..6]` are fixed symbol/account fields
(`point, pip, pt_scale, contract, start_balance, commission_per_lot,
slippage_points`); `p[7..]` are the `.mq5` `input` parameters in declaration
order. The function processes bars `[b_from, b_to)` and must:

- write `equity[b]` for every bar it processes (final balance is read from
  `balance_arr`),
- append closed trades to the `tr_*` arrays (same order/columns the built-in
  strategy uses; `ntr_arr` carries the count across calls),
- also fill `live[ntr]` (17 columns) so the GUI can stream trades during the
  run,
- keep state (open positions `p_*`, `n_pos`) in the passed arrays.

See `strategy.py` (built-in example, fully annotated) and `engine.py` for the
exact, authoritative description. The STRATEGY tab's prompt asks the
assistant for a module with exactly this signature.

## Verification against the MT5 Tester (2026-11-12, built-in example strategy)

The tester report used as reference (produced by the original project this app was built from)
uses EURUSD M5, 2026.10.13→2026.11.12, 100% real ticks, all default inputs. The comparison
conclusions carried over to the built-in default strategy:

- **Identical bar count**: 6072 bars (this app) = "Bars: 6072" in the report.
- **The tick cache on disk diverges from the snapshot the tester used**:
  MT5 reports "Ticks: 4993534"; the current cache differs (see below).
  This is not an EA logic bug.
- **Early positions match consecutively and exactly** (first 16 in-deals,
  including a 5-position averaging chain), as do the largest profit trade
  and the maximal consecutive losses.
- **Current comparison on this cache**: 542 of 564 out-deal profits match
  exactly as a multiset; 496 of 569 app entries appear in the MT5 report.
  Net profit / total trades diverge because of the data-source difference.
- **To reproduce the tester report exactly, re-run the MT5 tester against
  the current tick cache and compare with this app again.**
- The profit/netProfit comparison uses `PROFIT_EPS = 2e-13` so closes happen
  on the same tick as MT5 at the exact target boundary.

## Notes

- Initial balance defaults to **100** (top-bar field); applied to both
  backtests and optimizations.
- Terminal MT5 must be closed before backtesting (single instance), though
  this app itself never opens MT5.
- The `.set` file is deliberately left empty: FROM/STEP/TO follow the `.mq5`
  defaults and OPT checkboxes are inactive. Fill in values manually to
  optimize.
- The initial symbol preset is **EURUSD** (5 digits, 1 lot = 100 000 units);
  the preset list in the ACCOUNT group covers the common symbols.
- Optimization methods: **Full** (grid search; > 20 000 combos warns; default on startup) and
  **Quick** (random sampling + local search, deduplicated).
- The app ships **no built-in default strategy** — loading a different
  `.mq5` and running without generating a strategy will fail with a clear
  "No strategy module provided" error. Generate / apply a strategy first.
# AgloBackTune
