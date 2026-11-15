"""Load ticks, prepare bars, run the strategy simulation, and compute metrics."""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from .bars import M1, Bars, resample, ticks_to_bars
from .config import Settings
from .tickcache import Ticks

PROGRESS = Callable[[str], None]

M5 = 5 * 60_000


def warmup_ms(settings: Settings, main_tf_ms: int = M5) -> int:
    """History to load before the start date so the indicators are mature.

    A generic 10x-max-period heuristic with a minimum of 2 days and a
    maximum of 30 days. EA strategies should rely on the same convention.
    """
    # Determine an approximate longest indicator from the EA params when
    # present, otherwise fall back to a safe default of 20 periods.
    longest = 20
    for s in settings.params.spec:
        if s.name.lower().endswith("period"):
            if int(settings.params[s.name]) > longest:
                longest = int(settings.params[s.name])
    bars = longest * 10
    # MN1 (monthly) uses a -1 sentinel, so normalize to a monthly estimate
    # before multiplying.
    tf_ms = int(main_tf_ms)
    if tf_ms <= 0:
        tf_ms = 30 * 86_400_000
    need = bars * max(tf_ms, 86_400_000)
    return int(min(max(need, 2 * 86_400_000), 30 * 86_400_000))


@dataclass
class TradeTable:
    entry_time: np.ndarray
    exit_time: np.ndarray
    dir: np.ndarray
    entry: np.ndarray
    exit: np.ndarray
    sl_init: np.ndarray
    tp: np.ndarray
    lot: np.ndarray
    pnl: np.ndarray
    reason: np.ndarray
    score: np.ndarray
    strength: np.ndarray
    risk: np.ndarray
    mae: np.ndarray
    mfe: np.ndarray
    entry_bar: np.ndarray
    exit_bar: np.ndarray

    def __len__(self) -> int:
        return int(self.pnl.size)


@dataclass
class Metrics:
    start_balance: float = 0.0
    end_balance: float = 0.0
    net_profit: float = 0.0
    return_pct: float = 0.0
    gross_profit: float = 0.0
    gross_loss: float = 0.0
    profit_factor: float = 0.0
    expectancy: float = 0.0
    win_rate: float = 0.0
    trades: int = 0
    wins: int = 0
    losses: int = 0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    best: float = 0.0
    worst: float = 0.0
    max_dd: float = 0.0
    max_dd_pct: float = 0.0
    recovery: float = 0.0
    sharpe: float = 0.0
    max_consec_loss: int = 0
    max_consec_win: int = 0
    avg_duration_min: float = 0.0
    trading_days: int = 0
    trades_per_day: float = 0.0
    final_equity: float = 0.0


@dataclass
class Prepared:
    ticks: Ticks
    m1: Bars
    main: Bars
    bar_start: np.ndarray
    main_tf_ms: int = M5
    key: tuple = ()

    @property
    def n_bars(self) -> int:
        return len(self.main)


@dataclass
class BacktestResult:
    settings: Settings
    ticks: Ticks
    m1: Bars
    bars: Bars
    trades: TradeTable
    equity_time: np.ndarray
    equity: np.ndarray
    end_balance: float
    metrics: Metrics
    source_label: str = ""
    meta: dict = field(default_factory=dict)

    @property
    def has_trades(self) -> bool:
        return len(self.trades) > 0


def indicator_key(settings: Settings, main_tf_ms: int) -> tuple:
    return (main_tf_ms,)


def prepare(ticks: Ticks, settings: Settings,
            main_tf_ms: int = M5) -> Prepared:
    if len(ticks) == 0:
        raise ValueError("No ticks in the selected range.")

    m1 = ticks_to_bars(ticks.time_ms, ticks.bid, ticks.ask, M1)
    main = m1 if main_tf_ms == M1 else resample(m1, main_tf_ms)
    bar_start = np.searchsorted(ticks.time_ms, main.time_ms, side="left").astype(np.int64)
    np.clip(bar_start, 0, max(0, len(ticks) - 1), out=bar_start)
    return Prepared(ticks, m1, main, bar_start, main_tf_ms,
                    indicator_key(settings, main_tf_ms))


LIVE_TRADE_COLS = 17
# Number of bars processed per engine slice when streaming results to a GUI.
# Small enough that a user interface can refresh between slices.
STREAM_CHUNK_BARS = 150


def trade_capacity(settings: Settings, n_bars: int) -> int:
    """Upper bound on closed trades the sim will produce (safe)."""
    max_pos = settings.max_positions
    cap = (n_bars + 2) * (max_pos + 1)
    if cap < 16:
        cap = 16
    if cap > 1_000_000:
        cap = 1_000_000
    return cap


def run_prepared(prep: Prepared, settings: Settings,
                 progress: Callable[[str], None] | None = None,
                 active_from_ms: int = 0,
                 trade_progress: np.ndarray | None = None,
                 trade_live_ctx: dict | None = None,
                 strategy_module=None) -> BacktestResult:
    """Run the strategy in bar slices.

    `strategy_module` must expose ``simulate_range`` with the exact
    signature of the strategy contract; it is REQUIRED — there is no
    built-in fallback. The module defines its own `PARAM_SPEC`, which
    becomes the parameter vector (p[7:]).
    """
    if not settings.params.spec:
        raise ValueError(
            "No strategy parameters configured. Paste a strategy module in the "
            "STRATEGY tab first — its PARAM_SPEC makes up the parameter "
            "vector (p[7:])."
        )

    if strategy_module is None or not hasattr(strategy_module, "simulate_range"):
        raise ValueError(
            "No strategy module provided. Paste a strategy module in the "
            "STRATEGY tab first (STRATEGY tab → APPLY AS STRATEGY)."
        )

    sim_fn = strategy_module.simulate_range

    ticks = prep.ticks
    main = prep.main
    if progress:
        progress(f"Simulating {len(ticks):,} ticks / {len(main):,} bars...".replace(",", "."))

    p = settings.to_array()

    n_bars = len(main)
    n_ticks = prep.ticks.time_ms.size

    # First bar that may be traded. Earlier bars are used only for
    # indicator warm-up, without trade/equity.
    start_bar = 1
    tick_from = 0
    if active_from_ms > 0:
        start_bar = n_bars
        for b in range(1, n_bars):
            if main.time_ms[b] >= active_from_ms:
                start_bar = b
                break
        lo = 0
        hi = n_ticks
        while lo < hi:
            mid = (lo + hi) // 2
            if ticks.time_ms[mid] < active_from_ms:
                lo = mid + 1
            else:
                hi = mid
        tick_from = lo

    max_pos = settings.max_positions
    if max_pos < 1:
        max_pos = 1
    if max_pos > 256:
        max_pos = 256
    cap = trade_capacity(settings, n_bars)

    # Preallocate outputs and carried state.
    equity = np.full(n_bars, settings.start_balance, np.float64)
    tr_entry_time = np.zeros(cap, np.int64)
    tr_exit_time = np.zeros(cap, np.int64)
    tr_dir = np.zeros(cap, np.int64)
    tr_entry = np.zeros(cap, np.float64)
    tr_exit = np.zeros(cap, np.float64)
    tr_sl = np.zeros(cap, np.float64)
    tr_tp = np.zeros(cap, np.float64)
    tr_lot = np.zeros(cap, np.float64)
    tr_pnl = np.zeros(cap, np.float64)
    tr_reason = np.zeros(cap, np.int64)
    tr_score = np.zeros(cap, np.float64)
    tr_strength = np.zeros(cap, np.float64)
    tr_risk = np.zeros(cap, np.float64)
    tr_mae = np.zeros(cap, np.float64)
    tr_mfe = np.zeros(cap, np.float64)
    tr_entry_bar = np.zeros(cap, np.int64)
    tr_exit_bar = np.zeros(cap, np.int64)

    p_dir = np.zeros(max_pos, np.int64)
    p_entry = np.zeros(max_pos, np.float64)
    p_lot = np.zeros(max_pos, np.float64)
    p_time = np.zeros(max_pos, np.int64)
    p_bar = np.zeros(max_pos, np.int64)
    p_mae = np.zeros(max_pos, np.float64)
    p_mfe = np.zeros(max_pos, np.float64)

    balance_arr = np.array([settings.start_balance], np.float64)
    ntr_arr = np.array([0], np.int64)
    npos_arr = np.array([0], np.int64)

    if trade_live_ctx is not None:
        live = np.zeros((cap, LIVE_TRADE_COLS), np.float64)
        trade_live_ctx["live"] = live
        trade_live_ctx["capacity"] = cap
    else:
        live = np.zeros((0, LIVE_TRADE_COLS), np.float64)

    progress_arr = (trade_progress if trade_progress is not None
                    else np.zeros(1, np.int64))

    sliced = trade_live_ctx is not None
    if sliced:
        total = max(1, (n_bars - start_bar + STREAM_CHUNK_BARS - 1)
                    // STREAM_CHUNK_BARS)
        done = 0
        for b0 in range(start_bar, n_bars, STREAM_CHUNK_BARS):
            b1 = b0 + STREAM_CHUNK_BARS
            if b1 > n_bars:
                b1 = n_bars
            sim_fn(
                p,
                ticks.time_ms.astype(np.int64), ticks.bid, ticks.ask, prep.bar_start,
                main.time_ms, main.open, main.high, main.low, main.close,
                b0, b1, start_bar, tick_from,
                equity,
                tr_entry_time, tr_exit_time, tr_dir,
                tr_entry, tr_exit, tr_sl, tr_tp,
                tr_lot, tr_pnl, tr_reason, tr_score, tr_strength, tr_risk,
                tr_mae, tr_mfe, tr_entry_bar, tr_exit_bar,
                p_dir, p_entry, p_lot, p_time, p_bar, p_mae, p_mfe,
                balance_arr, ntr_arr, npos_arr,
                active_from_ms, progress_arr, live,
            )
            done += 1
            if progress:
                progress(f"Simulating ({done}/{total})...")
    else:
        sim_fn(
            p,
            ticks.time_ms.astype(np.int64), ticks.bid, ticks.ask, prep.bar_start,
            main.time_ms, main.open, main.high, main.low, main.close,
            start_bar, n_bars, start_bar, tick_from,
            equity,
            tr_entry_time, tr_exit_time, tr_dir,
            tr_entry, tr_exit, tr_sl, tr_tp,
            tr_lot, tr_pnl, tr_reason, tr_score, tr_strength, tr_risk,
            tr_mae, tr_mfe, tr_entry_bar, tr_exit_bar,
            p_dir, p_entry, p_lot, p_time, p_bar, p_mae, p_mfe,
            balance_arr, ntr_arr, npos_arr,
            active_from_ms, progress_arr, live,
        )

    ntr = int(ntr_arr[0])
    end_balance = float(balance_arr[0])
    open_end = int(npos_arr[0])

    trades = TradeTable(
        tr_entry_time[:ntr], tr_exit_time[:ntr], tr_dir[:ntr],
        tr_entry[:ntr], tr_exit[:ntr], tr_sl[:ntr], tr_tp[:ntr],
        tr_lot[:ntr], tr_pnl[:ntr], tr_reason[:ntr], tr_score[:ntr],
        tr_strength[:ntr], tr_risk[:ntr], tr_mae[:ntr], tr_mfe[:ntr],
        tr_entry_bar[:ntr], tr_exit_bar[:ntr],
    )

    # Trim the output to the active period (drop the indicator warm-up).
    if active_from_ms > 0:
        b0 = int(np.searchsorted(main.time_ms, active_from_ms, side="left"))
        b0 = max(0, min(b0, len(main)))
    else:
        b0 = 0
    eq = equity[b0:] if b0 < equity.size else np.array([end_balance], dtype=np.float64)
    time_a = main.time_ms[b0:]
    bars_a = _slice_bars(main, b0)

    metrics = compute_metrics(eq, trades, settings.start_balance)
    metrics.end_balance = float(end_balance)
    metrics.net_profit = float(end_balance - settings.start_balance)
    metrics.return_pct = (metrics.net_profit / settings.start_balance * 100.0
                          if settings.start_balance > 0 else 0.0)
    metrics.recovery = (metrics.net_profit / metrics.max_dd) if metrics.max_dd > 0 else 0.0
    metrics.trading_days = _trading_days(time_a)
    metrics.trades_per_day = (metrics.trades / metrics.trading_days
                              if metrics.trading_days > 0 else 0.0)
    metrics.final_equity = float(eq[-1]) if eq.size else float(end_balance)

    if active_from_ms > 0:
        active_ticks = int(np.count_nonzero(ticks.time_ms >= active_from_ms))
    else:
        active_ticks = len(ticks)

    if progress:
        progress("Done.")

    exit_labels = getattr(strategy_module, "EXIT_LABEL", None) or {}

    return BacktestResult(
        settings=settings, ticks=ticks, m1=prep.m1, bars=bars_a,
        trades=trades, equity_time=time_a, equity=eq,
        end_balance=float(end_balance), metrics=metrics,
        source_label=ticks.source_label,
        meta={"active_from_ms": int(active_from_ms), "active_ticks": active_ticks,
              "warmup_ticks": len(ticks) - active_ticks,
              "open_positions_at_end": int(open_end),
              "exit_labels": exit_labels},
    )


def _slice_bars(bars: Bars, i0: int) -> Bars:
    if i0 <= 0:
        return bars
    return Bars(
        time_ms=bars.time_ms[i0:], open=bars.open[i0:], high=bars.high[i0:],
        low=bars.low[i0:], close=bars.close[i0:], spread=bars.spread[i0:],
        period_ms=bars.period_ms,
    )


def _trading_days(times: np.ndarray) -> int:
    if times.size == 0:
        return 0
    days = np.unique((times // 86_400_000).astype(np.int64))
    return int(days.size)


def compute_metrics(equity: np.ndarray, trades: TradeTable,
                    start_balance: float) -> Metrics:
    m = Metrics(start_balance=float(start_balance))
    m.trades = len(trades)
    equity = np.asarray(equity, dtype=np.float64)
    eq_series = (np.concatenate(([start_balance], equity)) if equity.size
                 else np.array([start_balance], dtype=np.float64))
    if m.trades == 0:
        m.end_balance = float(equity[-1]) if equity.size else start_balance
        m.net_profit = m.end_balance - start_balance
        m.return_pct = m.net_profit / start_balance * 100.0 if start_balance > 0 else 0.0
        m.max_dd, m.max_dd_pct = _drawdown(eq_series)
        m.final_equity = float(equity[-1]) if equity.size else float(start_balance)
        return m

    pnl = trades.pnl
    wins = pnl > 0
    losses = pnl < 0
    m.wins = int(wins.sum())
    m.losses = int(losses.sum())
    m.win_rate = m.wins / m.trades * 100.0
    m.gross_profit = float(pnl[wins].sum())
    m.gross_loss = float(-pnl[losses].sum())
    if m.gross_loss > 0:
        m.profit_factor = m.gross_profit / m.gross_loss
    elif m.gross_profit > 0:
        m.profit_factor = 99.99
    m.expectancy = float(pnl.mean())
    m.avg_win = float(pnl[wins].mean()) if m.wins else 0.0
    m.avg_loss = float(pnl[losses].mean()) if m.losses else 0.0
    m.best = float(pnl.max())
    m.worst = float(pnl.min())

    eq = np.concatenate(([start_balance], start_balance + np.cumsum(pnl)))
    m.end_balance = float(eq[-1])
    m.final_equity = float(equity[-1]) if equity.size else m.end_balance
    m.max_dd, m.max_dd_pct = _drawdown(eq_series)

    if pnl.size > 1 and pnl.std(ddof=1) > 0:
        m.sharpe = float(pnl.mean() / pnl.std(ddof=1) * np.sqrt(pnl.size))

    cl = cw = cur_l = cur_w = 0
    for x in pnl:
        if x < 0:
            cur_l += 1
            cur_w = 0
        elif x > 0:
            cur_w += 1
            cur_l = 0
        else:
            cur_l = cur_w = 0
        cl = max(cl, cur_l)
        cw = max(cw, cur_w)
    m.max_consec_loss = cl
    m.max_consec_win = cw

    dur = (trades.exit_time - trades.entry_time) / 60_000.0
    m.avg_duration_min = float(dur.mean()) if dur.size else 0.0
    return m


def _drawdown(eq: np.ndarray) -> tuple[float, float]:
    if eq.size == 0:
        return 0.0, 0.0
    peak = np.maximum.accumulate(eq)
    dd = peak - eq
    i = int(np.argmax(dd))
    mdd = float(dd[i])
    pct = float(dd[i] / peak[i] * 100.0) if peak[i] > 0 else 0.0
    return mdd, pct
