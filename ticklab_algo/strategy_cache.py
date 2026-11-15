"""
Strategy cache and validation utilities for AlgoBackTune.

Per-strategy cached modules pasted into the app are stored in
the AlgoBackTune strategy_cache folder under %APPDATA% and
re-imported automatically on the next launch.
"""

from __future__ import annotations

import json
import os
import re
import traceback
from datetime import datetime
from pathlib import Path

import numpy as np

APP_DIR = Path(os.environ.get("APPDATA", str(Path.home()))) / "AlgoBackTune"
CACHE_DIR = APP_DIR / "strategy_cache"


def extract_python_code(text: str) -> str:
    """Pull the Python module out of a model response (```python fences or raw)."""
    fence = re.search(r"```(?:python|py)?\s*\n(.*?)```", text, re.DOTALL | re.IGNORECASE)
    if fence:
        return fence.group(1).strip() + "\n"
    return text.strip() + "\n"


def normalize_code(code: str) -> str:
    """Return the code unchanged.

    Auto-repair of indentation was attempted but found to silently produce
    wrong results (e.g. zero trades) because heuristic re-indentation cannot
    reliably recover block boundaries from dedented LLM output. To preserve
    backtest accuracy, indentation is NOT auto-repaired; the user fixes it
    manually using the line-numbered paste field. The prompt and gutter are
    the supported aids.
    """
    return code

def validate_syntax(code: str) -> tuple[bool, str]:
    """Cheap syntax + contract check before anything is saved or executed."""
    try:
        compile(code, "<generated_strategy>", "exec")
    except SyntaxError:
        return False, traceback.format_exc()
    required = ["def simulate_range", "EXIT_LABEL"]
    missing = [item for item in required if item not in code]
    if missing:
        return False, f"Missing required definitions: {', '.join(missing)}"
    return True, ""


# ---------------------------------------------------------------------------
# Per-EA strategy cache
# ---------------------------------------------------------------------------

def ea_cache_key(mq5_source: str) -> str:
    import hashlib
    return hashlib.sha1(mq5_source.encode("utf-8", errors="replace")).hexdigest()[:12]


def ea_cache_path(mq5_source: str) -> Path:
    return CACHE_DIR / f"strategy_{ea_cache_key(mq5_source)}.py"


def load_ea_strategy(mq5_source: str):
    """Import the cached generated strategy for this EA, or None."""
    path = ea_cache_path(mq5_source)
    if not path.exists():
        return None
    return _import_from_file(path, f"algo_cached_{ea_cache_key(mq5_source)}")


def _import_from_file(path: Path, name: str):
    import importlib.util
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)  # type: ignore[union-attr]
    except Exception:
        return None
    return module


def save_generated_strategy(code: str, mq5_source: str, ea_name: str,
                            model: str) -> Path:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = ea_cache_path(mq5_source)
    path.write_text(code, encoding="utf-8")
    meta = {
        "ea_name": ea_name,
        "cache_key": ea_cache_key(mq5_source),
        "model": model,
        "saved_at": __import__("datetime").datetime.now().isoformat(timespec="seconds"),
    }
    path.with_suffix(".json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return path


def import_generated_code(code: str, tag: str = "generated"):
    """Write code to a temp file and import it (forces nothing yet)."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = CACHE_DIR / f"algo_tmp_{tag}.py"
    path.write_text(code, encoding="utf-8")
    module = _import_from_file(path, f"algo_tmp_{tag}_{os.getpid()}")
    try:
        path.unlink()
    except OSError:
        pass
    return module





def smoke_test_module(module, settings, main_tf_ms: int = 5 * 60_000) -> tuple[bool, str]:
    """Run the generated module on synthetic ticks.

    Builds ~2 days of random-walk M5 bars with steady bid/ask, runs the full
    engine against it, and returns (ok, error_text). This forces the Numba
    JIT to compile the new code, so njit-incompatible constructs surface
    here instead of silently corrupting a real backtest.
    """
    import math

    rng = np.random.default_rng(20240501)
    n_bars = 220
    period = main_tf_ms
    t0 = 1_700_000_000_000  # fixed epoch (2023-11-14, Tue) so runs are repeatable
    bar_ms = period
    price = 1.1000
    m_open = np.empty(n_bars, np.float64)
    m_high = np.empty(n_bars, np.float64)
    m_low = np.empty(n_bars, np.float64)
    m_close = np.empty(n_bars, np.float64)
    m_time = np.empty(n_bars, np.int64)
    tick_time: list[int] = []
    tick_bid: list[float] = []
    tick_ask: list[float] = []
    point = 0.00001
    for b in range(n_bars):
        shock = float(rng.normal(0.0, 0.0006))
        open_p = price
        close_p = price + shock
        high_p = max(open_p, close_p) + abs(float(rng.normal(0.0, 0.0002)))
        low_p = min(open_p, close_p) - abs(float(rng.normal(0.0, 0.0002)))
        ts = t0 + b * bar_ms
        m_time[b] = ts
        m_open[b], m_high[b], m_low[b], m_close[b] = open_p, high_p, low_p, close_p
        # ~one tick every 5 seconds inside the bar
        for k in range(0, bar_ms, 5_000):
            frac = k / bar_ms
            px = open_p + (close_p - open_p) * frac
            spread = 2 * point
            tick_time.append(ts + k)
            tick_bid.append(px - spread / 2.0)
            tick_ask.append(px + spread / 2.0)
        price = close_p

    from .tickcache import Ticks
    from .engine import prepare, run_prepared

    ticks = Ticks(np.asarray(tick_time, np.int64), np.asarray(tick_bid),
                  np.asarray(tick_ask), "synthetic-smoke")
    try:
        prep = prepare(ticks, settings, main_tf_ms)
        run_prepared(prep, settings, strategy_module=module)
    except Exception:
        return False, traceback.format_exc()
    return True, ""


# ---------------------------------------------------------------------------
# Content-keyed strategies (paste-apply flow, no .mq5)
# ---------------------------------------------------------------------------

def code_cache_key(code: str) -> str:
    """Stable short key for strategy source code."""
    return ea_cache_key(code)


def parse_param_spec(module) -> list:
    """Read the module-level PARAM_SPEC list.

    Returns a list of (name, kind, default, comment) tuples in declaration
    order, matching the p[PREFIX_LEN + i] layout of the parameter vector.
    kind is one of "int" | "float" | "bool" | "str".
    """
    raw = getattr(module, "PARAM_SPEC", None)
    if raw is None:
        return []
    out = []
    for item in raw:
        try:
            if len(item) == 3:
                name, kind, default = item
                comment = ""
            elif len(item) >= 4:
                name, kind, default, comment = item[0], item[1], item[2], str(item[3] or "")
            else:
                continue
            name = str(name)
            kind = str(kind).strip().lower() or "float"
            if kind == "bool":
                d = bool(default) if isinstance(default, bool) else str(default).strip().lower() in ("1", "true", "yes", "on")
            elif kind == "int":
                d = int(round(float(default)))
            elif kind == "str":
                d = str(default)
            else:
                kind = "float"
                d = float(default)
            out.append((name, kind, d, comment))
        except Exception:
            continue
    return out


def save_strategy_for_code(code: str, ea_name: str = "pasted (AlgoBackTune)") -> Path:
    """Validate-free save; caller is expected to have run smoke_test_module first."""
    key = code_cache_key(code)
    path = CACHE_DIR / f"strategy_{key}.py"
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path.write_text(code, encoding="utf-8")
    meta = {
        "ea_name": ea_name,
        "cache_key": key,
        "mode": "pasted",
        "saved_at": datetime.now().isoformat(timespec="seconds"),
    }
    path.with_suffix(".json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    save_last_applied(path)
    return path


def save_last_applied(path) -> None:
    pointer = CACHE_DIR / "last_applied.json"
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    pointer.write_text(json.dumps({"path": str(path)}), encoding="utf-8")


def load_last_strategy():
    """Import the most recently applied strategy, or return None."""
    pointer = CACHE_DIR / "last_applied.json"
    try:
        data = json.loads(pointer.read_text(encoding="utf-8"))
        path = Path(data.get("path", ""))
        if not path.exists():
            return None
        meta_file = path.with_suffix(".json")
        meta = {}
        if meta_file.exists():
            try:
                meta = json.loads(meta_file.read_text(encoding="utf-8"))
            except Exception:
                meta = {}
        module = _import_from_file(path, "algo_applied_strategy")
        if module is None:
            return None
        return module, meta
    except Exception:
        return None

