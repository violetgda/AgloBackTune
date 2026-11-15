"""
Reference strategy module for AlgoBackTune (MA Cross + micro-grid averaging).

This module is the structural reference / test fixture for the strategy
contract. The app itself ships NO built-in default: a strategy module is
REQUIRED (STRATEGY tab → APPLY AS STRATEGY) and its PARAM_SPEC defines the
parameter vector. The automated tests run this module explicitly.

The intended workflow:

    1. Copy the prompt from the STRATEGY tab into the MetaEditor AI Assistant
    2. Paste the resulting Python module back in and click APPLY AS STRATEGY.
       The pasted module replaces this reference.

The contract (call signature) is fixed and must be kept by any generated
strategy. The `p` vector layout is:

    [0..6]  point, pip, pt_scale, contract, start_balance, commission_per_lot, slippage_points
    [7..]   strategy parameters in PARAM_SPEC order (indices from
            PREFIX_LEN, see config.EAParams)
"""

from __future__ import annotations

import numpy as np
from numba import njit

# Exit reason codes
EXIT_TARGET = 1
EXIT_DRAWDOWN = 2

EXIT_LABEL = {
    EXIT_TARGET: "Target Profit",
    EXIT_DRAWDOWN: "Max Drawdown",
}

# Parameter specification for this strategy. Read by AlgoBackTune to build
# the parameter panel (and the p[PREFIX_LEN:] vector). Every pasted module
# applied in the STRATEGY tab must define the same module-level PARAM_SPEC.
# Order == p[PREFIX_LEN + i] order.
PARAM_SPEC = [
    ("InpTargetProfit", "float", 1.0, "Target Net Profit ($)"),
    ("InpMaxDrawdown", "float", 20.0, "Batas Rugi Maksimal ($)"),
    ("InpLotSize", "float", 0.01, "Ukuran Lot Tetap"),
    ("InpStepPips", "int", 4, "Jarak Buka Posisi Baru (Pips)"),
    ("InpMaxPositions", "int", 5, "Maksimal Posisi Terbuka"),
    ("InpmaFastHandle", "int", 5, "Fast EMA"),
    ("InpmaSlowHandle", "int", 20, "Slow EMA"),
    ("InpMagicNumber", "int", 33333, "Magic Number EA"),
]

# Small tolerance for the floating-point comparison in Target Profit /
# Max Drawdown. MT5 closes the position at the first tick whose net profit
# reaches the target exactly (e.g. 1.00), while double arithmetic often
# yields 0.9999999999998899.
PROFIT_EPS = 2e-13


try:
    from .config import PREFIX_LEN, P_POINT, P_PIP, P_PT_SCALE, P_CONTRACT, \
        P_START_BALANCE, P_COMMISSION_PER_LOT, P_SLIPPAGE_POINTS
except Exception:  # self-contained fallback (module pasted/imported standalone)
    PREFIX_LEN = 7
    P_POINT, P_PIP, P_PT_SCALE, P_CONTRACT = 0, 1, 2, 3
    P_START_BALANCE, P_COMMISSION_PER_LOT, P_SLIPPAGE_POINTS = 4, 5, 6


@njit(cache=True)
def ema(values, period: int):
    n = values.size
    out = np.empty(n, np.float64)
    if n == 0:
        return out
    alpha = 2.0 / (period + 1.0)
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
def sma(values, period: int):
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


@njit(cache=True)
def simulate_range(p, t_ms, bid, ask, bar_start,
                   m_time, m_open, m_high, m_low, m_close,
                   b_from: int, b_to: int,
                   start_bar: int, tick_from: int,
                   equity,
                   tr_entry_time, tr_exit_time, tr_dir,
                   tr_entry, tr_exit, tr_sl, tr_tp,
                   tr_lot, tr_pnl, tr_reason, tr_score, tr_strength, tr_risk,
                   tr_mae, tr_mfe, tr_entry_bar, tr_exit_bar,
                   p_dir, p_entry, p_lot, p_time, p_bar, p_mae, p_mfe,
                   balance_arr, ntr_arr, npos_arr,
                   active_from_ms, progress_arr, live):
    """Process bars [b_from, b_to) with fully carried state."""
    n_ticks = t_ms.size
    n_bars = m_time.size
    point = p[P_POINT]
    pip = p[P_PIP]
    contract = p[P_CONTRACT]
    comm = p[P_COMMISSION_PER_LOT]
    slip = p[P_SLIPPAGE_POINTS] * point

    # EA params offset — matches EA reported defaults:
    #   7  TargetProfit, 8 MaxDrawdown, 9 LotSize, 10 StepPips,
    #   11 MaxPositions, 12 maFastHandle, 13 maSlowHandle, 14 MagicNumber
    ea_target = float(p[PREFIX_LEN + 0])
    ea_max_dd = float(p[PREFIX_LEN + 1])
    lot_fixed = float(p[PREFIX_LEN + 2])
    step_pips = float(p[PREFIX_LEN + 3])
    max_pos = int(p[PREFIX_LEN + 4])
    if max_pos < 1:
        max_pos = 1
    if max_pos > 256:
        max_pos = 256
    ma_fast = int(p[PREFIX_LEN + 5])
    ma_slow = int(p[PREFIX_LEN + 6])
    if ma_fast < 1:
        ma_fast = 1
    if ma_slow < 1:
        ma_slow = 1

    ma_fast_arr = ema(m_close, ma_fast)
    ma_slow_arr = sma(m_close, ma_slow)

    max_pos = min(max_pos, p_dir.size)
    max_trades = tr_entry_time.size

    balance = balance_arr[0]
    ntr = int(ntr_arr[0])
    n_pos = int(npos_arr[0])

    for b in range(b_from, b_to):
        start = int(bar_start[b])
        if b == start_bar and start < tick_from:
            start = tick_from
        if start >= n_ticks:
            cum_equity = balance
            for xb in range(b, n_bars):
                equity[xb] = cum_equity
            break
        if b + 1 < n_bars:
            end = int(bar_start[b + 1])
        else:
            end = n_ticks
        if end <= start:
            end = start + 1
        if end > n_ticks:
            end = n_ticks

        k1 = b - 1
        k2 = b - 2
        if k2 < 0:
            k2 = 0
        f1 = ma_fast_arr[k1]
        s1 = ma_slow_arr[k1]
        f2 = ma_fast_arr[k2]
        s2 = ma_slow_arr[k2]
        sig_buy = (f1 <= s1) and (f2 > s2)
        sig_sell = (f1 >= s1) and (f2 < s2)

        for i in range(start, end):
            t = t_ms[i]
            bd = bid[i]
            ak = ask[i]

            # --- 1. compute profit & positions ---
            net_profit = 0.0
            for j in range(n_pos):
                if p_dir[j] > 0:
                    net_profit += (bd - p_entry[j]) * p_lot[j] * contract
                else:
                    net_profit += (p_entry[j] - ak) * p_lot[j] * contract
            current_type = -1
            last_price = 0.0
            if n_pos > 0:
                current_type = p_dir[0]
                last_price = p_entry[0]
            open_positions = n_pos

            # --- 2. exit logic ---
            if n_pos > 0 and (net_profit >= ea_target - PROFIT_EPS
                              or net_profit <= -ea_max_dd + PROFIT_EPS):
                if net_profit >= ea_target:
                    reason = EXIT_TARGET
                else:
                    reason = EXIT_DRAWDOWN
                for j in range(n_pos):
                    if p_dir[j] > 0:
                        exit_px = bd
                    else:
                        exit_px = ak
                    pnl = p_dir[j] * (exit_px - p_entry[j]) * p_lot[j] * contract - comm * p_lot[j]
                    balance += pnl
                    if ntr < max_trades:
                        tr_entry_time[ntr] = p_time[j]
                        tr_exit_time[ntr] = t
                        tr_dir[ntr] = p_dir[j]
                        tr_entry[ntr] = p_entry[j]
                        tr_exit[ntr] = exit_px
                        tr_sl[ntr] = 0.0
                        tr_tp[ntr] = 0.0
                        tr_lot[ntr] = p_lot[j]
                        tr_pnl[ntr] = pnl
                        tr_reason[ntr] = reason
                        tr_score[ntr] = 0.0
                        tr_strength[ntr] = 0.0
                        tr_risk[ntr] = 0.0
                        tr_mae[ntr] = p_mae[j]
                        tr_mfe[ntr] = p_mfe[j]
                        tr_entry_bar[ntr] = p_bar[j]
                        tr_exit_bar[ntr] = b
                        if ntr < live.shape[0]:
                            live[ntr, 0] = float(p_time[j])
                            live[ntr, 1] = float(t)
                            live[ntr, 2] = float(p_dir[j])
                            live[ntr, 3] = p_entry[j]
                            live[ntr, 4] = exit_px
                            live[ntr, 5] = 0.0
                            live[ntr, 6] = 0.0
                            live[ntr, 7] = p_lot[j]
                            live[ntr, 8] = pnl
                            live[ntr, 9] = float(reason)
                            live[ntr, 10] = 0.0
                            live[ntr, 11] = 0.0
                            live[ntr, 12] = 0.0
                            live[ntr, 13] = p_mae[j]
                            live[ntr, 14] = p_mfe[j]
                            live[ntr, 15] = float(p_bar[j])
                            live[ntr, 16] = float(b)
                        ntr += 1
                        progress_arr[0] = ntr
                n_pos = 0
                equity[b] = balance
                continue

            # --- 3. initial entry (MA crossover) ---
            if open_positions == 0:
                d = 0
                if sig_buy:
                    d = 1
                elif sig_sell:
                    d = -1
                if d != 0:
                    if d > 0:
                        entry = ak + slip
                    else:
                        entry = bd - slip
                    p_dir[n_pos] = d
                    p_entry[n_pos] = entry
                    p_lot[n_pos] = lot_fixed
                    p_time[n_pos] = t
                    p_bar[n_pos] = b
                    p_mae[n_pos] = 0.0
                    p_mfe[n_pos] = 0.0
                    n_pos += 1
                    balance -= comm * lot_fixed

            # --- 4. averaging / micro-grid ---
            if open_positions > 0 and open_positions < max_pos:
                if current_type > 0:
                    distance = (last_price - ak) / pip
                    if distance >= step_pips:
                        p_dir[n_pos] = 1
                        p_entry[n_pos] = ak + slip
                        p_lot[n_pos] = lot_fixed
                        p_time[n_pos] = t
                        p_bar[n_pos] = b
                        p_mae[n_pos] = 0.0
                        p_mfe[n_pos] = 0.0
                        n_pos += 1
                        balance -= comm * lot_fixed
                elif current_type < 0:
                    distance = (bd - last_price) / pip
                    if distance >= step_pips:
                        p_dir[n_pos] = -1
                        p_entry[n_pos] = bd - slip
                        p_lot[n_pos] = lot_fixed
                        p_time[n_pos] = t
                        p_bar[n_pos] = b
                        p_mae[n_pos] = 0.0
                        p_mfe[n_pos] = 0.0
                        n_pos += 1
                        balance -= comm * lot_fixed

            # --- 5. MAE / MFE ---
            for j in range(n_pos):
                if p_dir[j] > 0:
                    fav = bd - p_entry[j]
                else:
                    fav = p_entry[j] - ak
                if fav < 0.0:
                    if -fav > p_mae[j]:
                        p_mae[j] = -fav
                if fav > p_mfe[j]:
                    p_mfe[j] = fav

        equity[b] = balance

    balance_arr[0] = balance
    ntr_arr[0] = ntr
    npos_arr[0] = n_pos
