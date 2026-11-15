"""
Aggregate tick -> bar and MA indicators (EMA, SMA) for the loaded EA strategies.

MetaTrader semantics:
  * EMA : exponential smoothing with alpha = 2/(period+1)
  * SMA  : simple rolling window
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

try:  # numba optional; pure-numpy fallback still works
    from numba import njit
except Exception:  # pragma: no cover
    def njit(*args, **kwargs):  # type: ignore
        def deco(fn):
            return fn
        if args and callable(args[0]):
            return args[0]
        return deco


M1 = 60_000
M5 = 5 * M1
M15 = 15 * M1
M30 = 30 * M1
H1 = 60 * M1

W1 = 7 * 86_400_000        # 7 days, Monday-anchored
MN1 = -1                   # sentinel: calendar-month bars

_DAY_MS = np.int64(86_400_000)
_WEEK_MS = np.int64(W1)
_MONDAY_OFFSET_MS = np.int64(3 * 86_400_000)   # 1970-01-01 (epoch) was a Thursday


def _keys_and_starts(times: np.ndarray, period_ms: int):
    """Return (keys, bar_open_ms) for a time series.

    For fixed periods the classic integer-division bucketing is used.
    For W1 the keys are Monday-anchored week indices (epoch-Thursday offset).
    For MN1 (period_ms == MN1) the keys are calendar months since epoch and
    the bar opens are the timestamps of the first day of each month.
    """
    t = times.astype(np.int64)
    if period_ms == MN1:
        days = (t // _DAY_MS).astype("datetime64[D]")
        months = days.astype("datetime64[M]")
        keys = months.astype(np.int64)
        open_ms = months.astype("datetime64[D]").astype(np.int64) * _DAY_MS
        return keys, open_ms
    if period_ms >= W1:  # only W1 today
        keys = (t + _MONDAY_OFFSET_MS) // _WEEK_MS
        open_ms = keys * _WEEK_MS - _MONDAY_OFFSET_MS
        return keys, open_ms
    keys = t // np.int64(period_ms)
    return keys, keys * np.int64(period_ms)


@dataclass
class Bars:
    time_ms: np.ndarray      # bar open time (server time, in milliseconds)
    open: np.ndarray
    high: np.ndarray
    low: np.ndarray
    close: np.ndarray
    spread: np.ndarray       # average spread (price) within the bar
    period_ms: int

    def __len__(self) -> int:
        return int(self.time_ms.size)


def _aggregate(keys: np.ndarray, starts_ms: np.ndarray, bid: np.ndarray,
               spread: np.ndarray, period_ms: int) -> Bars:
    """Generic aggregation based on time groups."""
    if keys.size == 0:
        empty = np.empty(0, np.float64)
        return Bars(np.empty(0, np.int64), empty, empty, empty, empty, empty, period_ms)

    starts = np.empty_like(keys, dtype=np.bool_)
    starts[0] = True
    np.not_equal(keys[1:], keys[:-1], out=starts[1:])
    idx = np.flatnonzero(starts)
    ends = np.empty_like(idx)
    ends[:-1] = idx[1:]
    ends[-1] = keys.size

    open_ = bid[idx]
    close = bid[ends - 1]
    high = np.maximum.reduceat(bid, idx)
    low = np.minimum.reduceat(bid, idx)
    cnt = (ends - idx).astype(np.float64)
    sp = np.add.reduceat(spread, idx) / cnt

    t = starts_ms[idx]
    return Bars(t, open_, high, low, close, sp, period_ms)


def ticks_to_bars(times: np.ndarray, bid: np.ndarray, ask: np.ndarray,
                  period_ms: int = M1) -> Bars:
    """Convert bid/ask ticks into OHLC bars (bid price, like MT5)."""
    if times.size == 0:
        empty = np.empty(0, np.float64)
        return Bars(np.empty(0, np.int64), empty, empty, empty, empty, empty, period_ms)
    keys, starts = _keys_and_starts(times, period_ms)
    return _aggregate(keys, starts, bid, ask - bid, period_ms)


def resample(bars: Bars, period_ms: int) -> Bars:
    """Combine smaller bars into bigger bars (M1 -> M5/M15)."""
    if len(bars) == 0:
        return Bars(np.empty(0, np.int64), *(np.empty(0, np.float64),) * 5, period_ms)
    keys, bar_starts = _keys_and_starts(bars.time_ms, period_ms)
    starts = np.empty_like(keys, dtype=np.bool_)
    starts[0] = True
    np.not_equal(keys[1:], keys[:-1], out=starts[1:])
    idx = np.flatnonzero(starts)
    ends = np.empty_like(idx)
    ends[:-1] = idx[1:]
    ends[-1] = keys.size

    open_ = bars.open[idx]
    close = bars.close[ends - 1]
    high = np.maximum.reduceat(bars.high, idx)
    low = np.minimum.reduceat(bars.low, idx)
    cnt = (ends - idx).astype(np.float64)
    sp = np.add.reduceat(bars.spread, idx) / cnt
    t = bar_starts[idx]
    return Bars(t, open_, high, low, close, sp, period_ms)


@njit(cache=True)
def ema(values: np.ndarray, period: int) -> np.ndarray:
    n = values.size
    out = np.empty(n, np.float64)
    if n == 0:
        return out
    alpha = 2.0 / (period + 1.0)
    # seed = SMA of the first `period` bars (converges fast, like MT5 after warm-up)
    seed_n = period if period < n else n
    s = 0.0
    for i in range(seed_n):
        s += values[i]
    prev = s / seed_n
    for i in range(n):
        if i < seed_n:
            out[i] = prev
        else:
            prev = prev + alpha * (values[i] - prev)
            out[i] = prev
    return out


@njit(cache=True)
def sma(values: np.ndarray, period: int) -> np.ndarray:
    """Simple Moving Average (rolling window), MT5 semantics.

    The first valid value appears at index period-1; before that the value
    is 0.
    """
    n = values.size
    out = np.zeros(n, np.float64)
    if n == 0:
        return out
    if period < 1:
        period = 1
    csum = 0.0
    for i in range(n):
        csum += values[i]
        if i >= period:
            csum -= values[i - period]
        if i >= period - 1:
            out[i] = csum / period
    return out
