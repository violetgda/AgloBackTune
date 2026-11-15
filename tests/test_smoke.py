"""Smoke test for AlgoBackTune: parsing, config, engine, optimizer, reports, LLM helpers.

Run from the AlgoBackTune folder:
    python -m unittest discover -s tests -v

The tests use synthetic data plus an inline example .mq5 snippet, so the
MetaTrader cache is not required. The first Numba compilation can take
15-70 seconds; after that the cache makes the tests run quickly.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ticklab_algo import strategy_cache  # noqa: E402
from ticklab_algo.bars import M1, ema, sma, ticks_to_bars  # noqa: E402
from ticklab_algo.config import (  # noqa: E402
    PREFIX_LEN, EAParams, ParamSpec, Settings, parse_mq5_inputs, strategy_spec_from_mq5,
)
from ticklab_algo.engine import compute_metrics, prepare, run_prepared  # noqa: E402
from ticklab_algo.optimizer import OptimizeRequest, optimize  # noqa: E402
from ticklab_algo.report import (  # noqa: E402
    save_html_report, save_optimizer_csv, save_summary_csv, save_trades_csv,
    summary_text,
)
from ticklab_algo.strategy import EXIT_LABEL  # noqa: E402
from ticklab_algo import strategy as bundled_strategy  # noqa: E402
from ticklab_algo.tickcache import (  # noqa: E402
    discover_sources, parse_datetime_ms, read_source,
)
from ticklab_algo.tickcache import Ticks  # noqa: E402

EXAMPLE_MQ5 = """//--- Input Parameters
input double   InpTargetProfit = 1.0;      // Target Net Profit ($)
input double   InpMaxDrawdown  = 20.0;     // Batas Rugi Maksimal ($)
input double   InpLotSize      = 0.01;     // Ukuran Lot Tetap
input int      InpStepPips     = 4;        // Jarak Buka Posisi Baru (Pips)
input int      InpMaxPositions = 5;        // Maksimal Posisi Terbuka
input int      InpmaFastHandle = 5;        // Fast EMA
input int      InpmaSlowHandle = 20;       // Slow EMA
input int      InpMagicNumber  = 33333;    // Magic Number EA"""


def example_source() -> str:
    return EXAMPLE_MQ5


def load_example_settings() -> Settings:
    """Build the Settings the app would use for the bundled example EA."""
    params = strategy_spec_from_mq5(example_source())
    return Settings(params=params)


def synthetic_ticks(days: int = 6, step_seconds: int = 5, seed: int = 7) -> Ticks:
    """Synthetic EURUSD ticks with oscillation so MA crosses occur."""
    rng = np.random.default_rng(seed)
    n = int(days * 86_400 / step_seconds)
    start = parse_datetime_ms("2025-01-06 00:00")
    times = start + np.arange(n, dtype=np.int64) * step_seconds * 1000
    t = np.arange(n, dtype=np.float64)
    wave = 0.0012 * np.sin(t * 2.0 * np.pi / 2400.0)
    wave += 0.0004 * np.sin(t * 2.0 * np.pi / 480.0)
    steps = np.gradient(wave) + rng.normal(0.0, 0.00004, n)
    bid = 1.10000 + np.cumsum(steps)
    spread = rng.uniform(0.00001, 0.00003, n)
    return Ticks(times, bid, bid + spread, "synthetic")


class TestParseMQ5Inputs(unittest.TestCase):
    def test_finds_example_inputs_in_order(self):
        specs = parse_mq5_inputs(example_source())
        names = [s.name for s in specs]
        self.assertEqual(names, [
            "InpTargetProfit", "InpMaxDrawdown", "InpLotSize", "InpStepPips",
            "InpMaxPositions", "InpmaFastHandle", "InpmaSlowHandle", "InpMagicNumber",
        ])
        kinds = {s.name: s.kind for s in specs}
        self.assertEqual(kinds["InpTargetProfit"], "float")
        self.assertEqual(kinds["InpLotSize"], "float")
        self.assertEqual(kinds["InpStepPips"], "int")
        self.assertEqual(kinds["InpMaxPositions"], "int")
        self.assertEqual(kinds["InpmaFastHandle"], "int")
        self.assertEqual(kinds["InpMagicNumber"], "int")
        defaults = {s.name: s.default for s in specs}
        self.assertAlmostEqual(defaults["InpTargetProfit"], 1.0)
        self.assertEqual(defaults["InpMaxPositions"], 5)
        self.assertEqual(defaults["InpmaSlowHandle"], 20)

    def test_strategy_spec_values(self):
        params = strategy_spec_from_mq5(example_source())
        self.assertEqual(len(params), 8)
        self.assertAlmostEqual(params["InpLotSize"], 0.01)
        self.assertEqual(params["InpStepPips"], 4)

    def test_comment_is_captured(self):
        specs = parse_mq5_inputs(example_source())
        comments = {s.name: s.comment for s in specs}
        self.assertIn("Net Profit", comments["InpTargetProfit"])


class TestBars(unittest.TestCase):
    def test_aggregation_ohlc(self):
        base = parse_datetime_ms("2025-01-02 00:00")
        sec = 1000
        times = np.array([
            base, base + 10 * sec, base + 20 * sec,    # minute 0
            base + 60 * sec, base + 70 * sec,          # minute 1
        ], dtype=np.int64)
        bid = np.array([10.0, 12.0, 9.0, 11.0, 8.0])
        ask = bid + 0.5
        bars = ticks_to_bars(times, bid, ask, M1)
        self.assertEqual(len(bars), 2)
        np.testing.assert_allclose(bars.open, [10.0, 11.0])
        np.testing.assert_allclose(bars.high, [12.0, 11.0])
        np.testing.assert_allclose(bars.low, [9.0, 8.0])
        np.testing.assert_allclose(bars.close, [9.0, 8.0])
        np.testing.assert_allclose(bars.spread, [0.5, 0.5])

    def test_ema_constant(self):
        values = np.full(200, 5.0)
        out = ema(values, 21)
        np.testing.assert_allclose(out, 5.0, rtol=0, atol=1e-12)

    def test_sma_constant(self):
        values = np.full(200, 5.0)
        out = sma(values, 20)
        np.testing.assert_allclose(out[19:], 5.0, rtol=0, atol=1e-12)
        np.testing.assert_allclose(out[:19], 0.0, rtol=0, atol=1e-12)

    def test_mn1_month_bars(self):
        from ticklab_algo.bars import MN1
        t0 = parse_datetime_ms("2024-01-15 10:00")
        t1 = parse_datetime_ms("2024-02-10 12:00")
        t2 = parse_datetime_ms("2024-03-05 08:00")
        times = np.array([t0, t0 + 1000, t1, t1 + 1000, t2], dtype=np.int64)
        bid = np.array([10.0, 11.0, 20.0, 21.0, 30.0])
        ask = bid + 0.5
        bars = ticks_to_bars(times, bid, ask, MN1)
        self.assertEqual(len(bars), 3)
        self.assertEqual(bars.time_ms[0], parse_datetime_ms("2024-01-01 00:00"))
        self.assertEqual(bars.time_ms[1], parse_datetime_ms("2024-02-01 00:00"))
        self.assertEqual(bars.time_ms[2], parse_datetime_ms("2024-03-01 00:00"))
        np.testing.assert_allclose(bars.close, [11.0, 21.0, 30.0])

    def test_w1_week_bars(self):
        from ticklab_algo.bars import W1
        monday = parse_datetime_ms("2024-06-03 00:00")  # 2024-06-03 is a Monday
        next_week = monday + 7 * 86_400_000
        times = np.array([monday + 1000, monday + 2000, next_week + 1000], dtype=np.int64)
        bid = np.array([10.0, 12.0, 20.0])
        ask = bid + 0.5
        bars = ticks_to_bars(times, bid, ask, W1)
        self.assertEqual(len(bars), 2)
        self.assertEqual(bars.time_ms[0], monday)
        self.assertEqual(bars.time_ms[1], next_week)
        np.testing.assert_allclose(bars.high, [12.0, 20.0])


class TestConfig(unittest.TestCase):
    def test_prefix_and_ea_layout(self):
        st = load_example_settings()
        arr = st.to_array()
        self.assertEqual(arr.shape[0], PREFIX_LEN + len(st.params))
        self.assertAlmostEqual(arr[0], st.point)
        self.assertAlmostEqual(arr[4], st.start_balance)
        # EA params start at PREFIX_LEN in .mq5 declaration order
        self.assertAlmostEqual(arr[PREFIX_LEN + 0], 1.0)    # InpTargetProfit
        self.assertAlmostEqual(arr[PREFIX_LEN + 2], 0.01)   # InpLotSize
        self.assertAlmostEqual(arr[PREFIX_LEN + 4], 5.0)    # InpMaxPositions

    def test_max_positions_detected(self):
        st = load_example_settings()
        self.assertEqual(st.max_positions, 5)

    def test_payload_roundtrip(self):
        st = load_example_settings()
        back = Settings.from_payload(st.to_payload())
        self.assertAlmostEqual(back.start_balance, st.start_balance)
        self.assertAlmostEqual(back.point, st.point)
        self.assertEqual(back.params.names(), st.params.names())
        for name in st.params.names():
            self.assertAlmostEqual(float(back.params[name]), float(st.params[name]))

    def test_int_indices(self):
        st = load_example_settings()
        idx = st.params.int_indices()
        self.assertEqual(idx, {3, 4, 5, 6, 7})

    def test_bool_and_string_encoding(self):
        spec = (
            ParamSpec("InpFlag", "bool", True),
            ParamSpec("InpText", "str", "abc"),
            ParamSpec("InpNum", "float", 1.5),
        )
        params = EAParams(spec)
        arr = params.to_array()
        np.testing.assert_allclose(arr, [1.0, 0.0, 1.5])


class TestEngine(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ticks = synthetic_ticks(days=6, step_seconds=5)
        cls.settings = load_example_settings()
        cls.prep = prepare(cls.ticks, cls.settings)
        cls.result = run_prepared(cls.prep, cls.settings, strategy_module=bundled_strategy)

    def test_empty_ea_params_raise_clear_error(self):
        from ticklab_algo.config import Settings
        bare = Settings()
        prep = prepare(self.ticks, bare)
        with self.assertRaises(ValueError):
            run_prepared(prep, bare)

    def test_bars_prepared(self):
        self.assertGreater(len(self.result.bars), 500)
        self.assertEqual(len(self.result.equity), len(self.result.bars))
        self.assertTrue(np.isfinite(self.result.equity).all())

    def test_balance_consistency(self):
        r = self.result
        total = float(r.trades.pnl.sum()) if len(r.trades) else 0.0
        self.assertAlmostEqual(r.end_balance,
                               self.settings.start_balance + total, places=6)
        self.assertEqual(r.metrics.trades, len(r.trades))

    def test_metrics_drawdown_nonnegative(self):
        m = compute_metrics(self.result.equity, self.result.trades,
                            self.settings.start_balance)
        self.assertGreaterEqual(m.max_dd, 0.0)
        self.assertGreaterEqual(m.max_dd_pct, 0.0)

    def test_trades_have_valid_fields(self):
        t = self.result.trades
        if len(t) == 0:
            self.skipTest("Synthetic data produced no trades")
        self.assertTrue((t.exit_time >= t.entry_time).all())
        self.assertTrue((t.lot > 0).all())
        self.assertTrue((t.pnl != 0).all())
        self.assertTrue(np.isin(t.reason, list(EXIT_LABEL)).all())

    def test_unclosed_positions_excluded(self):
        # equity must contain realized trades only; any positions still open
        # at end of data must be reported and must not touch the balance.
        r = self.result
        self.assertIn("open_positions_at_end", r.meta)
        self.assertTrue(np.isfinite(r.equity).all())
        total = float(r.trades.pnl.sum()) if len(r.trades) else 0.0
        self.assertAlmostEqual(r.end_balance,
                               self.settings.start_balance + total, places=6)


class TestOptimizer(unittest.TestCase):
    def test_grid(self):
        ticks = synthetic_ticks(days=4, step_seconds=10, seed=3)
        request = OptimizeRequest(
            base=load_example_settings(),
            grid={"InpStepPips": [2.0, 4.0], "InpMaxPositions": [3.0, 5.0]},
            objective="Balanced Score",
            min_trades=0,
            require_profit=False,
            method="grid",
        )
        result = optimize(ticks, request, strategy_module=bundled_strategy)
        self.assertEqual(result.total, 4)
        self.assertEqual(result.evaluated, 4)
        self.assertIsNotNone(result.best)
        self.assertGreaterEqual(result.best.score, -1e9)

    def test_fast_no_duplicate_combos(self):
        ticks = synthetic_ticks(days=3, step_seconds=10, seed=5)
        request = OptimizeRequest(
            base=load_example_settings(),
            grid={"InpStepPips": [2.0, 4.0], "InpMaxPositions": [3.0, 5.0]},
            objective="Balanced Score",
            min_trades=0,
            require_profit=False,
            method="fast",
            iterations=10,
            seed=42,
        )
        result = optimize(ticks, request, strategy_module=bundled_strategy)
        keys = [tuple(sorted(r.params.items())) for r in result.rows]
        self.assertEqual(len(keys), len(set(keys)), "Quick method produced duplicate combinations")


class TestReports(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.settings = load_example_settings()
        cls.result = run_prepared(
            prepare(synthetic_ticks(days=4, step_seconds=10), cls.settings),
            cls.settings, strategy_module=bundled_strategy)

    def test_exports(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            trades = save_trades_csv(self.result, tmp / "trades.csv")
            summary = save_summary_csv(self.result, tmp / "summary.csv")
            html = save_html_report(self.result, tmp / "report.html")
            opt = save_optimizer_csv([], tmp / "opt.csv")
            for path in (trades, summary, html, opt):
                self.assertTrue(path.exists())
                self.assertGreater(path.stat().st_size, 0)
            text = trades.read_text(encoding="utf-8-sig")
            self.assertIn("Entry Time", text.splitlines()[0])
            doc = html.read_text(encoding="utf-8")
            self.assertIn("<html", doc)
            self.assertIn("Equity Curve", doc)

    def test_summary_text(self):
        text = summary_text(self.result)
        self.assertIn("Net profit", text)
        self.assertIn("Trades", text)


class TestStrategyCache(unittest.TestCase):
    """Offline parts of the strategy cache module (no network, no API keys)."""

    def test_extract_python_code_from_fence(self):
        text = 'Here you go:\n```python\nimport numpy as np\n\ndef simulate_range():\n    pass\n```\nDone.'
        code = strategy_cache.extract_python_code(text)
        self.assertIn("def simulate_range", code)
        self.assertNotIn("```", code)

    def test_extract_python_code_plain(self):
        text = "import numpy as np\n\ndef simulate_range(p):\n    return p\n"
        code = strategy_cache.extract_python_code(text)
        self.assertIn("def simulate_range", code)

    def test_validate_syntax_ok(self):
        strategy_py = (Path(__file__).resolve().parents[1]
                       / "ticklab_algo" / "strategy.py").read_text(encoding="utf-8")
        ok, err = strategy_cache.validate_syntax(strategy_py)
        self.assertTrue(ok, err)

    def test_validate_syntax_rejects_garbage(self):
        ok, err = strategy_cache.validate_syntax("this is not python (")
        self.assertFalse(ok)
        self.assertTrue(err)

    def test_validate_syntax_requires_contract(self):
        ok, err = strategy_cache.validate_syntax("import numpy as np\nx = 1\n")
        self.assertFalse(ok)

    def test_cache_key_stable_and_distinct(self):
        a = strategy_cache.ea_cache_key("source-a")
        b = strategy_cache.ea_cache_key("source-b")
        self.assertEqual(a, strategy_cache.ea_cache_key("source-a"))
        self.assertNotEqual(a, b)

    def test_load_ea_strategy_unknown_cache(self):
        self.assertIsNone(strategy_cache.load_ea_strategy("definitely-unknown-mq5-source-xyz"))


class TestRealTickCache(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.sources = [s for s in discover_sources()
                       if s.symbol.upper().startswith("EURUSD")]

    def test_read_one_day(self):
        if not self.sources:
            self.skipTest("No .tkc MetaTrader cache on this machine")
        source = self.sources[0]
        start = source.last_time_ms - 2 * 86_400_000
        ticks = read_source(source, start, source.last_time_ms)
        if len(ticks) == 0:
            self.skipTest("Last range is empty")
        self.assertTrue((ticks.time_ms[1:] >= ticks.time_ms[:-1]).all())
        self.assertTrue((ticks.ask >= ticks.bid).mean() > 0.99)
        self.assertGreater(float(np.median(ticks.ask - ticks.bid)), 0.0)


class TestSymbolSpecs(unittest.TestCase):
    def test_forex_major(self):
        from ticklab_algo.symbol_specs import symbol_specs
        s = symbol_specs("EURUSD")
        self.assertEqual(s["point"], 0.00001)
        self.assertEqual(s["pip"], 0.0001)
        self.assertEqual(s["contract"], 100000.0)

    def test_jpy_pair(self):
        from ticklab_algo.symbol_specs import symbol_specs
        s = symbol_specs("USDJPY")
        self.assertEqual(s["point"], 0.001)
        self.assertEqual(s["pip"], 0.01)
        self.assertEqual(s["contract"], 100000.0)

    def test_gold(self):
        from ticklab_algo.symbol_specs import symbol_specs
        s = symbol_specs("XAUUSD")
        self.assertEqual(s["point"], 0.01)
        self.assertEqual(s["pip"], 0.1)
        self.assertEqual(s["contract"], 100.0)

    def test_index(self):
        from ticklab_algo.symbol_specs import symbol_specs
        self.assertEqual(symbol_specs("US30")["point"], 0.1)
        self.assertEqual(symbol_specs("US30")["contract"], 1.0)
        self.assertEqual(symbol_specs("US500")["point"], 0.01)

    def test_unknown_falls_back_to_forex(self):
        from ticklab_algo.symbol_specs import symbol_specs
        s = symbol_specs("SOMETHINGODD")
        self.assertEqual(s["point"], 0.00001)
        self.assertEqual(s["contract"], 100000.0)

if __name__ == "__main__":
    unittest.main(verbosity=2)

