"""Parameter optimization: grid search / random search."""

from __future__ import annotations

import itertools
import math
import random
import threading
import time
from dataclasses import dataclass, field
from typing import Callable, Iterable

from .config import Settings
from .engine import (
    BacktestResult, Metrics, Prepared, indicator_key, prepare, run_prepared,
)
from .tickcache import Ticks

OBJECTIVES = (
    "Net Profit",
    "Profit Factor",
    "Recovery Factor",
    "Return / Drawdown",
    "Balanced Score",
)


@dataclass
class OptimizeRequest:
    base: Settings
    # Grid keys are strategy PARAM_SPEC names (e.g. "InpLotSize"). Values are lists of floats.
    grid: dict[str, list[float]]
    objective: str = "Balanced Score"
    min_trades: int = 10
    require_profit: bool = True
    method: str = "grid"
    iterations: int = 300
    seed: int = 42
    max_combos: int = 20000
    active_from_ms: int = 0


@dataclass
class OptimizeRow:
    params: dict[str, float]
    metrics: Metrics
    score: float
    settings: Settings = field(repr=False, default_factory=Settings)

    def short(self, keys: Iterable[str]) -> dict:
        d = {k: self.params.get(k, "") for k in keys}
        d.update({
            "Net profit": self.metrics.net_profit,
            "PF": self.metrics.profit_factor,
            "Trades": self.metrics.trades,
            "Win %": self.metrics.win_rate,
            "Max DD": self.metrics.max_dd,
            "Score": self.score,
        })
        return d


@dataclass
class OptimizeResult:
    rows: list[OptimizeRow]
    best: OptimizeRow | None
    best_result: BacktestResult | None
    total: int
    evaluated: int
    cancelled: bool
    elapsed: float


def combos_from_grid(grid: dict[str, list[float]]) -> list[dict[str, float]]:
    keys = list(grid.keys())
    values = [list(grid[k]) for k in keys]
    out: list[dict[str, float]] = []
    for combo in itertools.product(*values):
        out.append(dict(zip(keys, combo)))
    return out


def objective_score(metrics: Metrics, name: str, min_trades: int,
                    require_profit: bool) -> float:
    net = metrics.net_profit
    dd = metrics.max_dd
    pf = min(metrics.profit_factor, 20.0)
    if name == "Net Profit":
        score = net
    elif name == "Profit Factor":
        score = pf * 100.0
    elif name == "Recovery Factor":
        score = (net / dd) if dd > 0 else net
    elif name == "Return / Drawdown":
        score = (metrics.return_pct / metrics.max_dd_pct) if metrics.max_dd_pct > 0 else metrics.return_pct
    else:
        rec = (net / dd) if dd > 0 else max(net, 0.0)
        trade_factor = math.sqrt(min(metrics.trades, 200) / max(1, min_trades))
        score = rec * trade_factor + 0.1 * pf

    if metrics.trades < min_trades:
        score -= 1_000_000.0 * (min_trades - metrics.trades + 1)
    if require_profit and net <= 0.0:
        score -= 1_000_000.0
    return float(score)


def _param_kind(base: Settings, name: str) -> str:
    for s in base.params.spec:
        if s.name == name:
            return s.kind
    # Optimize grids declared by the UI only contain EA spec names.
    return "float"


def _apply(base: Settings, combo: dict[str, float]) -> Settings:
    data = base.params.as_dict()
    for k, v in combo.items():
        kind = _param_kind(base, k)
        if kind == "int":
            data[k] = int(round(v))
        elif kind == "bool":
            data[k] = bool(int(round(v)))
        else:
            data[k] = float(v)
    ea = base.params
    new_params = type(ea)(ea.spec, [data[s.name] for s in ea.spec])
    settings = Settings(
        point=base.point, pip=base.pip, pt_scale=base.pt_scale,
        contract=base.contract, start_balance=base.start_balance,
        commission_per_lot=base.commission_per_lot,
        slippage_points=base.slippage_points,
        params=new_params,
    )
    return settings


def optimize(
    ticks: Ticks,
    request: OptimizeRequest,
    progress: Callable[[int, int, OptimizeRow], None] | None = None,
    cancel: threading.Event | None = None,
    strategy_module=None,
) -> OptimizeResult:
    t0 = time.perf_counter()
    base = request.base
    keys = list(request.grid.keys())

    if request.method == "grid":
        est = 1
        for k in keys:
            est *= len(request.grid[k])
            if est > request.max_combos:
                raise ValueError(
                    f"Too many combinations (>{request.max_combos:,}). "
                    f"Reduce the grid or use the Quick algorithm."
                )
        all_combos = combos_from_grid(request.grid)
        combos = all_combos
    elif request.method == "fast":
        # Fast algorithm: random sampling + local search around the best.
        # Random draws can collide, so keep a `seen` set to avoid sampling
        # the same combination twice (and avoid duplicate result rows).
        rnd = random.Random(request.seed)
        n_sample = max(10, min(request.iterations, 500))
        try:
            max_unique = max(1, math.prod(len(request.grid[k]) for k in keys))
        except Exception:
            max_unique = 10**18
        target = min(n_sample, max_unique)
        seen: set[tuple] = set()
        sampled: list[dict[str, float]] = []
        tries = 0
        limit = max(100, target * 40)
        while len(sampled) < target and tries < limit:
            tries += 1
            c = dict(zip(keys, [rnd.choice(request.grid[k]) for k in keys]))
            keyt = tuple(sorted(c.items()))
            if keyt in seen:
                continue
            seen.add(keyt)
            sampled.append(c)
        neighbors = []
        for c in list(sampled):
            for k in keys:
                vals = request.grid[k]
                if c[k] not in vals:
                    continue
                idx = vals.index(c[k])
                for di in (-1, 1):
                    ni = idx + di
                    if 0 <= ni < len(vals):
                        nc = dict(c)
                        nc[k] = vals[ni]
                        keyt = tuple(sorted(nc.items()))
                        if keyt in seen:
                            continue
                        seen.add(keyt)
                        neighbors.append(nc)
        combos = list(sampled) + neighbors
    else:
        rnd = random.Random(request.seed)
        count = max(1, request.iterations)
        seen = set()
        combos = []
        tries = 0
        while len(combos) < count and tries < max(count * 40, 100):
            tries += 1
            c = dict(zip(keys, [rnd.choice(request.grid[k]) for k in keys]))
            keyt = tuple(sorted(c.items()))
            if keyt in seen:
                continue
            seen.add(keyt)
            combos.append(c)

    total = len(combos)
    prep: Prepared = prepare(ticks, _apply(base, combos[0]) if combos else base)
    rows: list[OptimizeRow] = []
    best: OptimizeRow | None = None
    best_result: BacktestResult | None = None
    cancelled = False

    for i, combo in enumerate(combos):
        if cancel is not None and cancel.is_set():
            cancelled = True
            break
        settings = _apply(base, combo)
        if indicator_key(settings, prep.main_tf_ms) != prep.key:
            prep = prepare(ticks, settings)
        result = run_prepared(prep, settings,
                              active_from_ms=request.active_from_ms,
                              strategy_module=strategy_module)
        score = objective_score(result.metrics, request.objective,
                                request.min_trades, request.require_profit)
        row = OptimizeRow(params=combo, metrics=result.metrics, score=score,
                          settings=settings)
        rows.append(row)
        if best is None or score > best.score:
            best = row
            best_result = result
        if progress is not None:
            progress(i + 1, total, row)

    return OptimizeResult(
        rows=rows, best=best, best_result=best_result, total=total,
        evaluated=len(rows), cancelled=cancelled,
        elapsed=time.perf_counter() - t0,
    )


def build_values(lo: float, hi: float, step: float, is_int: bool) -> list[float]:
    if step <= 0:
        step = 1.0 if is_int else 0.1
    n = int(math.floor((hi - lo) / step + 1e-9)) + 1
    vals = [lo + i * step for i in range(max(1, n))]
    if is_int:
        vals = sorted({float(round(v)) for v in vals})
    return vals


def _read_set_text(path: str) -> str | None:
    raw = None
    try:
        raw = open(path, "rb").read()
    except Exception:
        return None
    if raw.startswith(b"\xff\xfe") or raw.startswith(b"\xfe\xff"):
        encs = ["utf-16"]
    elif b"\x00" in raw[:200]:
        encs = ["utf-16", "utf-8"]
    else:
        encs = ["utf-8", "utf-16"]
    for enc in encs:
        try:
            return raw.decode(enc)
        except Exception:
            continue
    return None


def _known_names(base: Settings | None) -> set[str] | None:
    if base is None:
        return None
    return set(base.params.names())


def load_set_ranges(path: str, base: Settings | None = None) -> dict:
    text = _read_set_text(path)
    out = {}
    known = _known_names(base)
    if text is None:
        return out
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith(";") or "=" not in line:
            continue
        name, rest = line.split("=", 1)
        name = name.strip()
        if known is not None and name not in known:
            continue
        if "||" in rest:
            parts = [p.strip() for p in rest.split("||")]
        else:
            parts = [p.strip() for p in rest.split("|")]
        if len(parts) >= 4:
            try:
                lo = float(parts[1].replace(",", "."))
                step = float(parts[2].replace(",", "."))
                hi = float(parts[3].replace(",", "."))
            except ValueError:
                continue
            flag = parts[4].upper() if len(parts) >= 5 else "Y"
            if flag not in ("Y", "YES", "1", "TRUE", ""):
                continue
            if hi <= lo:
                continue
            if step <= 0:
                step = 1.0
            out[name] = (lo, hi, step)
    return out


def load_set_defaults(path: str, base: Settings | None = None) -> dict:
    text = _read_set_text(path)
    out = {}
    known = _known_names(base)
    if text is None:
        return out
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith(";") or "=" not in line:
            continue
        name, rest = line.split("=", 1)
        name = name.strip()
        if known is not None and name not in known:
            continue
        parts = [p.strip() for p in (rest.split("||") if "||" in rest else rest.split("|"))]
        if not parts:
            continue
        val = parts[0].strip().strip('"').strip("'")
        out[name] = val
    return out
