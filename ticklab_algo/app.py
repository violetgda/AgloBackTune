"""TickLab-AlgoBackTune - backtest & optimization of pasted strategy modules (real ticks).

Workflow (no .mq5 file needed):
    1. STRATEGY tab      : paste the strategy module the MetaEditor AI Assistant
                            produces, APPLY AS STRATEGY.
    2. EA SETTINGS tab    : tweak / optimize the strategy parameters parsed
                            from the module's PARAM_SPEC.
    3. Backtest / Optimize
"""

from __future__ import annotations

import os
import queue
import sys
import threading
import time
import tkinter as tk
from datetime import datetime
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
import tkinter.font as tkfont

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from ticklab_algo import strategy_cache
from ticklab_algo.symbol_specs import symbol_specs
from ticklab_algo.config import EAParams, Settings, ParamSpec
from ticklab_algo.engine import prepare, run_prepared, warmup_ms
from ticklab_algo.optimizer import (
    OBJECTIVES, OptimizeRequest,
    build_values, load_set_defaults, load_set_ranges, optimize,
)
from ticklab_algo.report import metrics_pairs, save_html_report, save_optimizer_csv, summary_text
from ticklab_algo.tickcache import discover_sources, fmt_ms, human_int, parse_datetime_ms, read_source

APP_TITLE = "TickLab-AlgoBackTune - AI Strategy Backtester (Real Ticks)"
BG = "#0b1220"
PANEL = "#121b2c"
PANEL_ALT = "#16213a"
BORDER = "#1e2b42"
TEXT = "#dfe7f2"
MUTED = "#8b9bb4"
ACCENT = "#3b82f6"
GREEN = "#22c55e"
RED = "#ef4444"
PERIODS = ["M1", "M5", "M15", "M30", "H1", "H4", "D1", "W1", "MN1"]
FOREX_SYMBOLS = (
    "EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCAD", "USDCHF", "NZDUSD",
    "EURGBP", "EURJPY", "EURCHF", "EURAUD", "EURCAD", "EURNZD",
    "GBPJPY", "GBPCHF", "GBPAUD", "GBPCAD", "GBPNZD",
    "AUDJPY", "AUDCHF", "AUDCAD", "AUDNZD",
    "NZDJPY", "NZDCHF", "NZDCAD",
    "CADJPY", "CADCHF", "CHFJPY",
    "USDCNH", "USDSGD", "USDHKD", "USDSEK", "USDNOK", "USDDKK",
    "USDTRY", "USDZAR", "USDMXN",
)
INDICES_SYMBOLS = (
    "US30", "US500", "NAS100", "US2000", "UK100", "GER40", "FRA40", "EU50",
    "ESP35", "ITA40", "SWI20", "JP225", "HK50", "CHINA50", "AUS200",
)
METALS_SYMBOLS = (
    "XAUUSD", "XAGUSD", "XPTUSD", "XPDUSD",
)
ALL_SYMBOLS = FOREX_SYMBOLS + INDICES_SYMBOLS + METALS_SYMBOLS
PERIOD_MS = {
    "M1": 60_000, "M5": 5 * 60_000, "M15": 15 * 60_000, "M30": 30 * 60_000,
    "H1": 60 * 60_000, "H4": 4 * 60 * 60_000, "D1": 24 * 60 * 60_000,
    "W1": 7 * 24 * 60 * 60_000, "MN1": -1,
}

class EquityChart(tk.Canvas):
    def __init__(self, master, **kwargs):
        super().__init__(master, bg="#0f1724", highlightthickness=0, bd=0, **kwargs)
        self.times = None
        self.equity = None
        self.bind("<Configure>", lambda _e: self.redraw())

    def set_data(self, times, equity):
        self.times = list(times) if times is not None else None
        self.equity = np.asarray(equity, dtype=float) if equity is not None else None
        self.redraw()

    def clear(self):
        self.times = None
        self.equity = None
        self.redraw()

    def redraw(self):
        self.delete("all")
        w = max(120, self.winfo_width())
        h = max(80, self.winfo_height())
        left, right, top, bottom = 70, 18, 18, 30
        x0, y0 = left, top
        x1, y1 = w - right, h - bottom
        self.create_rectangle(x0, y0, x1, y1, outline=BORDER, fill="#0f1724")
        if self.equity is None or self.equity.size < 2:
            self.create_text((x0 + x1) / 2, (y0 + y1) / 2, text="No data yet", fill=MUTED, font=("Segoe UI", 10))
            return
        eq = self.equity
        n = eq.size
        lo = float(eq.min())
        hi = float(eq.max())
        pad = (hi - lo) * 0.08 or max(abs(hi) * 0.001, 1.0)
        lo -= pad
        hi += pad
        span = max(1e-9, hi - lo)
        for k in range(5):
            v = lo + span * k / 4.0
            y = y1 - (y1 - y0) * (v - lo) / span
            self.create_line(x0, y, x1, y, fill="#233046")
            self.create_text(x0 - 8, y, text=f"{v:,.0f}".replace(",", "."), fill=MUTED, anchor="e", font=("Consolas", 8))
        xs = x0 + (x1 - x0) * np.arange(n) / max(1, n - 1)
        ys = y1 - (y1 - y0) * (eq - lo) / span
        pts = [c for pair in zip(xs, ys) for c in pair]
        color = GREEN if eq[-1] >= eq[0] else RED
        self.create_line(*pts, fill=color, width=2)
        if self.times:
            self.create_text(x0, y1 + 6, text=str(self.times[0]), anchor="nw", fill=MUTED, font=("Consolas", 8))
            self.create_text(x1, y1 + 6, text=str(self.times[-1]), anchor="ne", fill=MUTED, font=("Consolas", 8))

class AlgoBackTuneApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title(APP_TITLE)
        self.geometry("1380x640")
        self.minsize(1180, 580)
        try:
            self.state("zoomed")
        except Exception:
            pass
        self.configure(bg=BG)

        self.q = queue.Queue()
        self.worker = None
        self.cancel_event = threading.Event()
        self.backtest_cancel = threading.Event()
        self._run_gen = 0
        self.sources = []
        self.source = None
        self.custom_roots = []
        self.ticks_cache_key = None
        self.ticks_cache = None
        self.last_result = None
        self.last_opt = None
        self._busy = False
        self._opt_busy = False
        self._backtest_running = False
        self._trade_progress = None
        self._last_trade_count = 0
        self.vars = {}
        self.opt_flags = {}
        self.grid_lo = {}
        self.iter_var = tk.StringVar(value="")
        self.grid_step = {}
        self.grid_hi = {}
        self.grid_count = {}
        self._metric_vars = []
        # EA / strategy state
        self.ea_source_path: str | None = None
        self.ea_name = "none"
        self.ea_source: str | None = None
        self.ea_params = None
        self.strategy_module = None      # no strategy module applied yet (STRATEGY tab)
        self.strategy_mode = "none"  # "none" | "generated" | "cached"
        self._ai_busy = False

        self._init_style()
        self._build_layout()
        self.after(80, self._poll_queue)
        self._start_discovery()

    def _init_style(self):
        style = ttk.Style(self)
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass
        self.option_add("*Font", ("Segoe UI", 9))
        style.configure(".", background=BG, foreground=TEXT, fieldbackground=PANEL,
                        bordercolor=BORDER, lightcolor=BORDER, darkcolor=BORDER,
                        troughcolor=PANEL, focuscolor=ACCENT)
        style.configure("TFrame", background=BG)
        style.configure("Panel.TFrame", background=PANEL)
        style.configure("TLabel", background=BG, foreground=TEXT)
        style.configure("Muted.TLabel", background=BG, foreground=MUTED)
        style.configure("Panel.TLabel", background=PANEL, foreground=TEXT)
        style.configure("PanelMuted.TLabel", background=PANEL, foreground=MUTED)
        style.configure("Title.TLabel", background=BG, foreground=TEXT, font=("Segoe UI Semibold", 15))
        style.configure("TButton", background=PANEL_ALT, foreground=TEXT, bordercolor=BORDER, padding=(10, 5))
        style.map("TButton", background=[("pressed", "#0f2747"), ("active", "#1c2a44")])
        style.configure("Accent.TButton", background=ACCENT, foreground="#ffffff", bordercolor=ACCENT, padding=(14, 6))
        style.map("Accent.TButton", background=[("pressed", "#1d4ed8"), ("active", "#2563eb")])
        style.configure("Danger.TButton", background="#3b1d22", foreground="#fca5a5", bordercolor="#5b2b33", padding=(10, 5))
        style.configure("TEntry", fieldbackground=PANEL, foreground=TEXT, insertcolor=TEXT, bordercolor=BORDER, padding=3)
        style.configure("TCombobox", fieldbackground=PANEL, foreground=TEXT, background=PANEL_ALT, arrowcolor=TEXT, padding=2)
        style.map("TCombobox", fieldbackground=[("readonly", PANEL)], foreground=[("readonly", TEXT)])
        style.configure("TNotebook", background=BG, bordercolor=BORDER, tabmargins=(2, 4, 2, 0))
        style.configure("TNotebook.Tab", background=PANEL, foreground=MUTED, padding=(14, 7), bordercolor=BORDER)
        style.map("TNotebook.Tab", background=[("selected", PANEL_ALT), ("active", "#1a2740")], foreground=[("selected", TEXT)])
        style.configure("Treeview", background=PANEL, fieldbackground=PANEL, foreground=TEXT, bordercolor=BORDER, rowheight=20, padding=(12, 0))
        style.configure("Treeview.Heading", background=PANEL_ALT, foreground="#9ec5ff", bordercolor=BORDER, relief="flat", padding=(4, 3))
        style.configure("TProgressbar", background=ACCENT, troughcolor=PANEL, bordercolor=BORDER)
        style.configure("TCheckbutton", background=PANEL, foreground=TEXT)

    def _build_layout(self):
        root = ttk.Frame(self, padding=(8, 6, 8, 4))
        root.pack(fill="both", expand=True)
        root.rowconfigure(1, weight=1)
        root.columnconfigure(0, weight=1)

        bar = ttk.Frame(root, style="Panel.TFrame", padding=(10, 8))
        bar.grid(row=0, column=0, sticky="ew", pady=(0, 6))

        ttk.Label(bar, text="Symbol", style="PanelMuted.TLabel").grid(row=0, column=0, sticky="w", padx=(0, 4))
        self.symbol_var = tk.StringVar(value="EURUSD")
        self.symbol_combo = ttk.Combobox(bar, textvariable=self.symbol_var, state="normal", values=ALL_SYMBOLS, width=12)
        self.symbol_combo.grid(row=0, column=1, sticky="w", padx=(0, 8))
        self.symbol_combo.bind("<<ComboboxSelected>>", lambda _e: self._apply_symbol_specs())

        ttk.Label(bar, text="Timeframe", style="PanelMuted.TLabel").grid(row=0, column=2, sticky="w", padx=(0, 4))
        self.tf_var = tk.StringVar(value="M5")
        ttk.Combobox(bar, textvariable=self.tf_var, state="readonly", values=PERIODS, width=6).grid(row=0, column=3, sticky="w", padx=(0, 8))

        ttk.Label(bar, text="Initial Balance", style="PanelMuted.TLabel").grid(row=0, column=4, sticky="w", padx=(0, 4))
        self.balance_var = tk.StringVar(value="100")
        ttk.Entry(bar, textvariable=self.balance_var, width=8).grid(row=0, column=5, sticky="w", padx=(0, 8))

        # ---- Account settings (right side) ----
        self.prefix_vars = {}
        acct = ttk.Frame(bar, style="Panel.TFrame", padding=(6, 2))
        acct.grid(row=0, column=7, rowspan=5, sticky="ne", padx=(6, 0))
        ttk.Label(acct, text="ACCOUNT", style="PanelMuted.TLabel",
                  font=("Segoe UI Semibold", 9)).grid(row=0, column=0, columnspan=4, sticky="w")

        self.leverage_var = tk.StringVar(value="500")
        ttk.Label(acct, text="Leverage", style="PanelMuted.TLabel").grid(
            row=1, column=0, sticky="w", padx=(0, 4), pady=(4, 0))
        ttk.Entry(acct, textvariable=self.leverage_var, width=10).grid(
            row=1, column=1, sticky="w", padx=(0, 10), pady=(4, 0))
        ttk.Label(acct, text="Contract", style="PanelMuted.TLabel").grid(
            row=2, column=0, sticky="w", padx=(0, 4), pady=(4, 0))
        self.prefix_vars["contract"] = tk.StringVar(value="100000")
        ttk.Entry(acct, textvariable=self.prefix_vars["contract"], width=10).grid(
            row=2, column=1, sticky="w", padx=(0, 10), pady=(4, 0))
        ttk.Label(acct, text="Point", style="PanelMuted.TLabel").grid(
            row=2, column=2, sticky="w", padx=(0, 4), pady=(4, 0))
        self.prefix_vars["point"] = tk.StringVar(value="0.00001")
        ttk.Entry(acct, textvariable=self.prefix_vars["point"], width=10).grid(
            row=2, column=3, sticky="w", padx=(0, 10), pady=(4, 0))
        ttk.Label(acct, text="Commission/Lot", style="PanelMuted.TLabel").grid(
            row=3, column=0, sticky="w", padx=(0, 4), pady=(4, 0))
        self.prefix_vars["commission_per_lot"] = tk.StringVar(value="0.0")
        ttk.Entry(acct, textvariable=self.prefix_vars["commission_per_lot"], width=10).grid(
            row=3, column=1, sticky="w", padx=(0, 10), pady=(4, 0))
        ttk.Label(acct, text="Pip", style="PanelMuted.TLabel").grid(
            row=3, column=2, sticky="w", padx=(0, 4), pady=(4, 0))
        self.prefix_vars["pip"] = tk.StringVar(value="0.0001")
        ttk.Entry(acct, textvariable=self.prefix_vars["pip"], width=10).grid(
            row=3, column=3, sticky="w", padx=(0, 10), pady=(4, 0))
        ttk.Label(acct, text="Slippage", style="PanelMuted.TLabel").grid(
            row=4, column=0, sticky="w", padx=(0, 4), pady=(4, 0))
        self.prefix_vars["slippage_points"] = tk.StringVar(value="0.0")
        ttk.Entry(acct, textvariable=self.prefix_vars["slippage_points"], width=10).grid(
            row=4, column=1, sticky="w", padx=(0, 10), pady=(4, 0))
        ttk.Label(acct, text="Pt Scale", style="PanelMuted.TLabel").grid(
            row=4, column=2, sticky="w", padx=(0, 4), pady=(4, 0))
        self.prefix_vars["pt_scale"] = tk.StringVar(value="1.0")
        ttk.Entry(acct, textvariable=self.prefix_vars["pt_scale"], width=10).grid(
            row=4, column=3, sticky="w", padx=(0, 10), pady=(4, 0))

        ttk.Label(bar, text="Tick Source", style="PanelMuted.TLabel").grid(row=1, column=0, sticky="w", pady=(6, 0))
        self.source_var = tk.StringVar()
        self.source_combo = ttk.Combobox(bar, textvariable=self.source_var, state="readonly", width=80)
        self.source_combo.grid(row=1, column=1, columnspan=5, sticky="ew", pady=(6, 0), padx=(0, 8))
        self.source_combo.bind("<<ComboboxSelected>>", lambda _e: self._on_source_change())
        ttk.Button(bar, text="Refresh", command=self._reload_sources).grid(row=1, column=6, pady=(6, 0), padx=(0, 4))

        ttk.Label(bar, text="From", style="PanelMuted.TLabel").grid(row=2, column=0, sticky="w", pady=(6, 0), padx=(0, 4))
        self.from_var = tk.StringVar(value="")
        ttk.Entry(bar, textvariable=self.from_var, width=12).grid(row=2, column=1, sticky="w", pady=(6, 0), padx=(0, 8))
        ttk.Label(bar, text="To", style="PanelMuted.TLabel").grid(row=2, column=2, sticky="w", pady=(6, 0), padx=(0, 4))
        self.to_var = tk.StringVar(value="")
        ttk.Entry(bar, textvariable=self.to_var, width=12).grid(row=2, column=3, sticky="w", pady=(6, 0), padx=(0, 8))
        qf = ttk.Frame(bar, style="Panel.TFrame")
        qf.grid(row=2, column=4, columnspan=3, sticky="ew", pady=(6, 0))
        for txt, days in (("3 Days", 3), ("1 Week", 7), ("1 Month", 30), ("1 Year", 365)):
            ttk.Button(qf, text=txt, width=9, command=lambda d=days: self._quick_range(d)).pack(side="left", padx=(0, 4))

        ttk.Label(bar, text="Objective", style="PanelMuted.TLabel").grid(row=3, column=0, sticky="w", pady=(6, 0))
        self.objective_var = tk.StringVar(value="Balanced Score")
        ttk.Combobox(bar, textvariable=self.objective_var, state="readonly", values=list(OBJECTIVES), width=18).grid(row=3, column=1, sticky="w", pady=(6, 0), padx=(0, 8))
        ttk.Label(bar, text="Method", style="PanelMuted.TLabel").grid(row=3, column=2, sticky="w", pady=(6, 0))
        self.method_var = tk.StringVar(value="Full")
        ttk.Combobox(bar, textvariable=self.method_var, state="readonly", values=["Full", "Quick"], width=22).grid(row=3, column=3, sticky="w", pady=(6, 0), padx=(0, 8))
        self.method_var.trace_add("write", lambda *_: self._method_changed())

        ttk.Label(bar, text="Iterations (quick)", style="PanelMuted.TLabel").grid(row=3, column=4, sticky="w", pady=(6, 0), padx=(8, 4))
        ttk.Entry(bar, textvariable=self.iter_var, width=8).grid(row=3, column=5, sticky="w", pady=(6, 0))

        self.status_var = tk.StringVar(value="Ready.")

        btns = ttk.Frame(bar, style="Panel.TFrame")
        btns.grid(row=4, column=0, columnspan=7, sticky="ew", pady=(8, 0))
        self.btn_bt = ttk.Button(btns, text="Backtest", style="Accent.TButton", command=self.start_backtest)
        self.btn_bt.pack(side="left")
        self.btn_opt = ttk.Button(btns, text="Optimize", command=self.start_optimize)
        self.btn_opt.pack(side="left", padx=8)
        self.btn_stop = ttk.Button(btns, text="Stop", style="Danger.TButton", command=self.cancel_all)
        self.btn_stop.pack(side="left", padx=8)
        self.progress = ttk.Progressbar(btns, orient="horizontal", mode="determinate")
        self.progress.pack(side="left", fill="x", expand=True, padx=(8, 0))


        # notebook
        self.nb = ttk.Notebook(root)
        self.nb.grid(row=1, column=0, sticky="nsew")

        self.tab_ai = ttk.Frame(self.nb)
        self.tab_param = ttk.Frame(self.nb)
        self.tab_summary = ttk.Frame(self.nb)
        self.tab_chart = ttk.Frame(self.nb)
        self.tab_trade = ttk.Frame(self.nb)
        self.tab_opt = ttk.Frame(self.nb)
        self.tab_log = ttk.Frame(self.nb)

        self.nb.add(self.tab_ai, text="  STRATEGY  ")
        self.nb.add(self.tab_param, text="  EA SETTINGS  ")
        self.nb.add(self.tab_summary, text="  SUMMARY  ")
        self.nb.add(self.tab_chart, text="  CHART  ")
        self.nb.add(self.tab_trade, text="  TRADES  ")
        self.nb.add(self.tab_opt, text="  OPTIMIZATION RESULTS  ")
        self.nb.add(self.tab_log, text="  LOG  ")

        self._build_strategy_tab()
        self._build_param_tab()
        self._build_summary_tab()
        self._build_chart_tab()
        self._build_trade_tab()
        self._build_opt_tab()
        self._build_log_tab()

        self._boot_strategy()

        self.symbol_var.trace_add("write", lambda *_: self._reload_sources())
        if not self._should_suppress_startup_warning():
            self.after(150, self._show_startup_warning)

    _STARTUP_NOTICE_FLAG = Path(os.environ.get("APPDATA", str(Path.home()))) / "AlgoBackTune" / "startup_notice_hidden"

    def _should_suppress_startup_warning(self) -> bool:
        try:
            return self._STARTUP_NOTICE_FLAG.exists()
        except OSError:
            return False

    def _hide_startup_warning_forever(self):
        try:
            self._STARTUP_NOTICE_FLAG.parent.mkdir(parents=True, exist_ok=True)
            self._STARTUP_NOTICE_FLAG.write_text("hidden\n", encoding="utf-8")
        except OSError:
            pass

    def _show_startup_warning(self):
        """First-run information dialog shown in English."""
        notice = tk.Toplevel(self)
        notice.title("AlgoBackTune — Important Notice")
        notice.configure(bg=PANEL)
        notice.resizable(False, False)
        notice.transient(self)
        notice.grab_set()
        notice.attributes("-topmost", True)

        # center the dialog on the screen
        w, h = 560, 392
        sw = self.winfo_screenwidth()
        sh = self.winfo_screenheight()
        x = max((sw - w) // 2, 0)
        y = max((sh - h) // 2, 0)
        notice.geometry(f"{w}x{h}+{x}+{y}")

        outer = tk.Frame(notice, bg=PANEL, padx=22, pady=18)
        outer.pack(fill="both", expand=True)

        tk.Label(outer, text="IMPORTANT NOTICE",
                 bg=PANEL, fg="#f6c343",
                 font=("Segoe UI Semibold", 15, "bold")).pack(anchor="w")
        tk.Label(outer, text="Please read before using the backtester",
                 bg=PANEL, fg=MUTED,
                 font=("Segoe UI", 9, "italic")).pack(anchor="w", pady=(0, 12))

        body = (
            "AlgoBackTune is designed to help you backtest and optimize\n"
            "trading strategies outside MetaTrader 5, faster.\n\n"
            "However, please be aware of the following:\n\n"
            "  •  The application may fail, and its results are NOT always accurate.\n"
            "  •  You must have the EA source code from the .mq5 file.\n"
            "  •  MT5 must already be installed and logged into your account, so\n"
            "     you can match the settings (Contract, Point, Pip, Commission/Lot,\n"
            "     Slippage, etc.) with your broker.\n\n"
            "!!!  ALWAYS validate the results against the MT5 Strategy Tester first.\n"
            "!!!  NEVER trade real money before you have done so.\n"
        )
        tk.Label(outer, text=body, bg=PANEL, fg=TEXT,
                 font=("Segoe UI", 9), justify="left",
                 anchor="w").pack(anchor="w", fill="x")

        btn_row = tk.Frame(outer, bg=PANEL)
        btn_row.pack(anchor="e", side="bottom", fill="x", pady=(10, 0))
        ttk.Button(btn_row, text="Don't show again",
                  command=lambda: (self._hide_startup_warning_forever(), notice.destroy())).pack(side="left")
        ttk.Button(btn_row, text="I understand",
                  command=notice.destroy, style="Accent.TButton").pack(side="right")
        notice.protocol("WM_DELETE_WINDOW", notice.destroy)
        self.update_idletasks()

    def _boot_strategy(self):
        """Start with an empty parameter table.

        Strategy parameters are taken from the applied strategy module's
        PARAM_SPEC, so they only appear after a strategy is pasted and
        APPLIED in the STRATEGY tab.
        """
        self.ea_name = "(no strategy applied)"
        self.strategy_module = None
        self.strategy_mode = "none"
        self.ea_params = EAParams(spec=())
        self.ea_source = None
        self.ea_source_path = None
        self._load_param_panel()
        self._update_ai_labels()

    def _build_param_tab(self):
        f = ttk.Frame(self.tab_param, padding=8)
        f.pack(fill="both", expand=True)

        btns = ttk.Frame(f)
        btns.pack(fill="x", pady=(0, 6))
        ttk.Button(btns, text="LOAD FULL SET", command=self._load_set_all).pack(side="left")
        ttk.Button(btns, text="LOAD OPT SET", command=self._load_set_opt).pack(side="left", padx=6)
        ttk.Button(btns, text="SAVE SET", command=self._save_set).pack(side="left", padx=6)
        ttk.Button(btns, text="RESET DEFAULTS", command=self._reset_defaults).pack(side="left", padx=6)
        self.ea_hint_var = tk.StringVar(
            value="No EA strategy has been loaded, please apply a strategy in the STRATEGY tab")
        ttk.Label(btns, textvariable=self.ea_hint_var, style="Muted.TLabel").pack(side="right")
        self.opt_steps_var = tk.StringVar(value="0 steps")
        ttk.Label(btns, textvariable=self.opt_steps_var, style="Muted.TLabel").pack(side="right", padx=(0, 12))

        hdr = ttk.Frame(f, style="Panel.TFrame", padding=(6, 4))
        hdr.pack(fill="x")
        for i, h in enumerate(("OPT", "VARIABLE", "COMMENT", "DEFAULT", "FROM", "STEP", "TO", "#")):
            ttk.Label(hdr, text=h, style="PanelMuted.TLabel", font=("Segoe UI Semibold", 9)).grid(row=0, column=i, padx=4, sticky="w")

        self.param_canvas = tk.Canvas(f, bg=BG, highlightthickness=0)
        self.param_scroll = ttk.Scrollbar(f, orient="vertical", command=self.param_canvas.yview)
        self.param_inner = ttk.Frame(self.param_canvas)
        self.param_inner.bind("<Configure>", lambda e: self.param_canvas.configure(scrollregion=self.param_canvas.bbox("all")))
        self.param_canvas.create_window((0, 0), window=self.param_inner, anchor="nw")
        self.param_canvas.configure(yscrollcommand=self.param_scroll.set)
        self.param_canvas.bind("<MouseWheel>", lambda e: self.param_canvas.yview_scroll(int(-e.delta/120), "units"))
        self.param_canvas.pack(side="left", fill="both", expand=True)
        self.param_scroll.pack(side="right", fill="y")

        self.param_rows = {}
        self.vars = {}
        self.opt_flags = {}
        self.grid_lo = {}
        self.grid_step = {}
        self.grid_hi = {}
        self.grid_count = {}


    def _method_changed(self):
        pass

    # ---- current EA state helpers ------------------------------------------
    def _ea_kind(self, name):
        if self.ea_params is not None:
            for s in self.ea_params.spec:
                if s.name == name:
                    return s.kind
        return "float"

    def _opt_value_is_int(self, name):
        return self._ea_kind(name) in ("int", "bool")

    def _exit_labels(self):
        if self.strategy_module is None:
            return {}
        return getattr(self.strategy_module, "EXIT_LABEL", {}) or {}

    def _ea_settings_values(self):
        """Current EA param values as {name: value} (floats/ints/bools/str)."""
        values = {}
        if self.ea_params is None:
            return values
        for spec in self.ea_params.spec:
            var = self.vars.get(spec.name)
            if var is None:
                continue
            raw = str(var.get()).strip().replace(",", ".")
            try:
                if spec.kind == "bool":
                    values[spec.name] = 1 if raw.lower() in ("true", "1", "yes", "on") else 0
                elif spec.kind == "int":
                    values[spec.name] = int(float(raw))
                elif spec.kind == "str":
                    values[spec.name] = raw.strip('"').strip("'")
                else:
                    values[spec.name] = float(raw)
            except ValueError:
                values[spec.name] = spec.default
        return values

    def _reset_defaults(self):
        if self.ea_params is None:
            return
        for spec in self.ea_params.spec:
            var = self.vars.get(spec.name)
            if var is not None:
                d = spec.default
                if isinstance(d, bool):
                    var.set("1" if d else "0")
                else:
                    var.set(f"{d:g}" if isinstance(d, float) else str(d))
        self.log("Strategy parameters reset to defaults.")
        for spec in self.ea_params.spec:
            var = self.vars.get(spec.name)
            if var is not None:
                d = spec.default
                if isinstance(d, bool):
                    var.set("1" if d else "0")
                else:
                    var.set(f"{d:g}" if isinstance(d, float) else str(d))
        self.log("EA parameters reset to .mq5 defaults.")

    def _load_param_panel(self):
        set_path = None

        # parse the .set: ranges + flags + defaults (keyed by EA input name)
        set_data = {}
        if set_path:
            raw = Path(set_path).read_bytes()
            encs = ["utf-16"] if (raw.startswith(b"\xff\xfe") or b"\x00" in raw[:200]) else ["utf-8"]
            text = raw.decode(encs[0], errors="replace")
            for line in text.splitlines():
                line = line.strip()
                if not line or line.startswith(";") or "=" not in line:
                    continue
                name, rest = line.split("=", 1)
                name = name.strip()
                parts = [p.strip() for p in (rest.split("||") if "||" in rest else rest.split("|"))]
                if not parts:
                    continue
                default = parts[0].strip().strip('"').strip("'")
                lo = parts[1] if len(parts) >= 2 else default
                step = parts[2] if len(parts) >= 3 else "1"
                hi = parts[3] if len(parts) >= 4 else default
                flag = parts[4].upper() if len(parts) >= 5 else "Y"
                set_data[name] = (default, lo, step, hi, flag)

        for child in self.param_inner.winfo_children():
            child.destroy()
        self.vars = {}
        self.opt_flags = {}
        self.grid_lo = {}
        self.grid_step = {}
        self.grid_hi = {}
        self.grid_count = {}


        # ---- EA INPUTS group ----------------------------------------------
        row_idx = 0
        if self.ea_params is not None:
            lbl = ttk.Label(self.param_inner, text="STRATEGY PARAMETERS  (from the pasted strategy module)",
                            style="Panel.TLabel", font=("Segoe UI Semibold", 10))
            lbl.grid(row=row_idx, column=0, columnspan=8, sticky="w", pady=(10, 2))
            row_idx += 1
            if len(self.ea_params.spec) == 0:
                ttk.Label(self.param_inner, text="(empty - parameters appear after a strategy is applied in the STRATEGY tab)",
                          style="Muted.TLabel").grid(row=row_idx, column=0, columnspan=8, sticky="w", padx=4)
                row_idx += 1
            for spec in self.ea_params.spec:
                ea_name = spec.name
                sd = set_data.get(ea_name)
                d = spec.default
                if isinstance(d, bool):
                    default_val = (sd[0] if sd else ("1" if d else "0"))
                else:
                    default_val = (sd[0] if sd else (f"{d:g}" if isinstance(d, float) else str(d)))
                lo = sd[1] if sd else default_val
                step = sd[2] if sd else ("1" if spec.kind in ("int", "bool") else "0.1")
                hi = sd[3] if sd else default_val
                flag = sd[4].upper() if sd else "Y"
                comment = (spec.comment or "").strip()
                if len(comment) > 60:
                    comment = comment[:57] + "..."
                v_default = tk.StringVar(value=default_val)
                v_lo = tk.StringVar(value=lo if lo else default_val)
                v_step = tk.StringVar(value=step if step else "1")
                v_hi = tk.StringVar(value=hi if hi else default_val)
                v_opt = tk.IntVar(value=0)
                self.vars[ea_name] = v_default
                self.opt_flags[ea_name] = v_opt
                self.grid_lo[ea_name] = v_lo
                self.grid_step[ea_name] = v_step
                self.grid_hi[ea_name] = v_hi
                opt_ok = spec.kind in ("int", "float", "bool")
                for _v in (v_opt, v_lo, v_step, v_hi):
                    _v.trace_add("write", lambda *_: self._update_opt_steps())
                if opt_ok:
                    ttk.Checkbutton(self.param_inner, variable=v_opt).grid(row=row_idx, column=0, sticky="w", padx=4)
                else:
                    ttk.Checkbutton(self.param_inner, state="disabled").grid(row=row_idx, column=0, sticky="w", padx=4)
                ttk.Label(self.param_inner, text=ea_name, style="PanelMuted.TLabel",
                          font=("Consolas", 9)).grid(row=row_idx, column=1, sticky="w", padx=4)
                ttk.Label(self.param_inner, text=comment, style="Muted.TLabel").grid(row=row_idx, column=2, sticky="w", padx=4)
                ttk.Entry(self.param_inner, textvariable=v_default, width=11).grid(row=row_idx, column=3, sticky="w", padx=4)
                ttk.Entry(self.param_inner, textvariable=v_lo, width=11).grid(row=row_idx, column=4, sticky="w", padx=4)
                ttk.Entry(self.param_inner, textvariable=v_step, width=9).grid(row=row_idx, column=5, sticky="w", padx=4)
                ttk.Entry(self.param_inner, textvariable=v_hi, width=11).grid(row=row_idx, column=6, sticky="w", padx=4)
                cnt_var = tk.StringVar(value="")
                self.grid_count[ea_name] = cnt_var
                ttk.Label(self.param_inner, textvariable=cnt_var, style="Muted.TLabel", width=6).grid(row=row_idx, column=7, sticky="w", padx=4)
                row_idx += 1
        self._update_opt_steps()

    def _update_opt_steps(self):
        total = 1
        overflow = False
        for name, flag in self.opt_flags.items():
            lo = self.grid_lo[name].get().strip()
            step = self.grid_step[name].get().strip()
            hi = self.grid_hi[name].get().strip()
            n = 0
            if flag.get() and lo and step and hi:
                try:
                    lof, stepf, hif = float(lo), float(step), float(hi)
                    if hif > lof and stepf > 0:
                        n = int((hif - lof) / stepf + 1e-9) + 1
                except ValueError:
                    n = 0
            if name in self.grid_count:
                self.grid_count[name].set(str(n) if n > 0 else "")
            if not flag.get():
                continue
            if n <= 0:
                continue
            if total > 10**12:
                overflow = True
            else:
                total *= max(1, n)
        self.opt_steps_var.set("very large (Full will confirm)" if (overflow or total > 10**12) else f"{total} steps")
        self._sync_opt_columns()

    def _load_set_all(self):
        p = filedialog.askopenfilename(filetypes=[("SET", "*.set")])
        if not p:
            return
        from ticklab_algo.optimizer import load_set_defaults, load_set_ranges
        base = self._read_settings()
        defaults = load_set_defaults(p, base=base)
        ranges = load_set_ranges(p, base=base)
        # apply defaults to vars
        for k, v in defaults.items():
            if k in self.vars:
                self.vars[k].set(v)
        # apply ranges to grid fields (only Y already filtered in ranges)
        for name, (lo, hi, step) in ranges.items():
            if name in self.grid_lo:
                self.grid_lo[name].set(f"{lo:g}")
                self.grid_step[name].set(f"{step:g}")
                self.grid_hi[name].set(f"{hi:g}")
                if name in self.opt_flags:
                    self.opt_flags[name].set(1)
        self._update_opt_steps()
        self.log(f"Loaded full set: {len(defaults)} defaults, {len(ranges)} ranges from {p}")

    def _load_set_opt(self):
        p = filedialog.askopenfilename(filetypes=[("SET", "*.set")])
        if not p:
            return
        from ticklab_algo.optimizer import load_set_ranges
        ranges = load_set_ranges(p, base=self._read_settings())
        for name, (lo, hi, step) in ranges.items():
            if name in self.grid_lo:
                self.grid_lo[name].set(f"{lo:g}")
                self.grid_step[name].set(f"{step:g}")
                self.grid_hi[name].set(f"{hi:g}")
                if name in self.opt_flags:
                    self.opt_flags[name].set(1)
        self._update_opt_steps()
        self.log(f"Loaded optimization set: {len(ranges)} ranges from {p}")

    def _save_set(self):
        p = filedialog.asksaveasfilename(defaultextension=".set", filetypes=[("SET", "*.set")])
        if not p:
            return
        lines = ["; TickLab-AlgoBackTune optimization set", "; name=default||from||step||to||Y/N", ""]
        for name in self.vars:
            default = self.vars[name].get().strip()
            flag = "Y" if (name in self.opt_flags and self.opt_flags[name].get()) else "N"
            lo = self.grid_lo.get(name, tk.StringVar(value="")).get().strip() if name in self.grid_lo else ""
            step = self.grid_step.get(name, tk.StringVar(value="")).get().strip() if name in self.grid_step else ""
            hi = self.grid_hi.get(name, tk.StringVar(value="")).get().strip() if name in self.grid_hi else ""
            if not (lo and step and hi):
                lo = hi = step = default
            lines.append(f"{name}={default}||{lo}||{step}||{hi}||{flag}")
        Path(p).write_text("\n".join(lines) + "\n", encoding="utf-8")
        self.log(f"Saved: {p}")

    def _apply_symbol_specs(self, symbol: str | None = None):
        """Auto-fill the ACCOUNT panel (point/pip/pt_scale/contract) from the

        selected symbol's standard MT5 specification. Commission, slippage
        and leverage stay user-editable (broker/account-side).
        """
        sym = (symbol or self.symbol_var.get()).strip().upper()
        spec = symbol_specs(sym)
        mapping = {"point": spec["point"], "pip": spec["pip"],
                   "pt_scale": spec["pt_scale"], "contract": spec["contract"]}
        applied = []
        def _fmt(v):
            s = f"{v:.10f}".rstrip("0").rstrip(".")
            return s if s else "0"
        for key, value in mapping.items():
            var = self.prefix_vars.get(key)
            if var is not None:
                var.set(_fmt(value))
                applied.append(key)
        self.log(f"Account applied for {sym}: "
                 + ", ".join(f"{k}={_fmt(mapping[k])}" for k in applied))

    def _reload_sources(self):
        sym = self.symbol_var.get().strip().upper()
        all_srcs = discover_sources(self.custom_roots)
        # Mirror the symbols discovered in the MT5 cache into the Symbol dropdown
        standard = list(dict.fromkeys(ALL_SYMBOLS))
        extras = sorted({s.symbol.upper() for s in all_srcs} - set(standard))
        if getattr(self, "symbol_combo", None) is not None and extras:
            self.symbol_combo["values"] = standard + extras
        self.sources = [s for s in all_srcs if sym in s.symbol.upper()] if sym else all_srcs
        if not self.sources:
            self.source = None
            self.source_combo["values"] = [s.label() for s in all_srcs]
            messagebox.showwarning("Ticks", f"No tick cache for symbol {sym}. Download ticks in MT5 first, then Refresh.")
            return
        if self.sources:
            self.source_combo["values"] = [s.label() for s in self.sources]
            self.source_combo.current(0)
            self._on_source_change()
            # check time coverage
            try:
                start = parse_datetime_ms(self.from_var.get().strip() + " 00:00:00") if len(self.from_var.get().strip()) <= 10 else parse_datetime_ms(self.from_var.get().strip())
                end = parse_datetime_ms(self.to_var.get().strip() + " 23:59:59") if len(self.to_var.get().strip()) <= 10 else parse_datetime_ms(self.to_var.get().strip())
            except ValueError:
                start, end = 0, 0
            src = self.source
            if src and start and end:
                try:
                    _t = read_source(src, start, end)
                    if len(_t) == 0:
                        self.status_var.set("No ticks for this symbol/time range. Download ticks in MT5.")
                        messagebox.showwarning("Ticks", "No ticks for this symbol/time range. Download ticks in MT5, then Refresh.")
                    else:
                        # Check full coverage
                        first = int(_t.time_ms[0])
                        last = int(_t.time_ms[-1])
                        tol = 2 * 86_400_000  # 2 days tolerance (weekend)
                        if first > start + tol or last < end - tol:
                            self.status_var.set("Ticks do not cover the full required range. Download ticks in MT5.")
                            messagebox.showwarning("Ticks", "Ticks do not cover the full required range. Download ticks in MT5, then Refresh.")
                except Exception:
                    pass

        else:
            messagebox.showwarning("Ticks", f"No tick cache for {sym}. Download ticks in MT5 first, then Refresh.")

    def _on_source_change(self):
        idx = self.source_combo.current()
        if 0 <= idx < len(self.sources):
            self.source = self.sources[idx]
            s = self.source
            self.status_var.set(f"{s.origin} | {s.server} | {s.symbol} | {human_int(s.total_ticks)} tick")
            if not self.from_var.get().strip():
                self._quick_range(7)

    def _quick_range(self, days):
        if not self.source:
            return
        s = self.source
        end = s.last_time_ms
        start = max(s.first_time_ms, end - days * 86_400_000)
        self.from_var.set(fmt_ms(start, with_time=False))
        self.to_var.set(fmt_ms(end, with_time=False))

    # ---- tabs ----
    def _build_summary_tab(self):
        f = ttk.Frame(self.tab_summary, padding=6)
        f.pack(fill="both", expand=True)
        f.rowconfigure(1, weight=1)
        f.columnconfigure(0, weight=1)

        # Button row always pinned to the top
        btns = ttk.Frame(f)
        btns.grid(row=0, column=0, sticky="ew", pady=(0, 6))
        ttk.Button(btns, text="Save Report", command=self._export_all).pack(side="left")
        ttk.Button(btns, text="Save Summary", command=self._export_summary).pack(side="left", padx=6)

        # Summary area (scrollable, shrinks so buttons stay visible)
        body = ttk.Frame(f, style="Panel.TFrame")
        body.grid(row=1, column=0, sticky="nsew")
        body.rowconfigure(0, weight=1)
        body.columnconfigure(0, weight=1)
        self.summary_text = tk.Text(body, bg="#0f1724", fg=TEXT, bd=0, relief="flat",
                                    wrap="word", font=("Consolas", 9), padx=10, pady=6,
                                    height=8)
        self.summary_text.grid(row=0, column=0, sticky="nsew")
        svs = ttk.Scrollbar(body, orient="vertical", command=self.summary_text.yview)
        svs.grid(row=0, column=1, sticky="ns")
        self.summary_text.configure(yscrollcommand=svs.set)

    def _build_chart_tab(self):
        f = ttk.Frame(self.tab_chart, padding=8)
        f.pack(fill="both", expand=True)
        btns = ttk.Frame(f)
        btns.pack(fill="x", pady=(0, 6))
        ttk.Button(btns, text="Save Image", command=self._save_chart).pack(side="left")
        self.chart = EquityChart(f)
        self.chart.pack(fill="both", expand=True)

    def _build_trade_tab(self):
        f = ttk.Frame(self.tab_trade, padding=8)
        f.pack(fill="both", expand=True)
        btns = ttk.Frame(f)
        btns.pack(fill="x", pady=(0, 6))
        ttk.Button(btns, text="Save Trades", command=self._export_trades).pack(side="left")
        self.trade_count_var = tk.StringVar(value="0 trade")
        ttk.Label(btns, textvariable=self.trade_count_var, style="Muted.TLabel").pack(side="right")
        cols = ("no", "entry_time", "exit_time", "dir", "entry", "exit", "sl", "tp", "lot", "pnl", "reason", "score")
        heads = ("#", "ENTRY TIME", "EXIT TIME", "DIR", "ENTRY", "EXIT", "SL", "TP", "LOT", "P/L", "REASON", "SCORE")
        widths = (40, 130, 130, 50, 70, 70, 70, 70, 45, 70, 110, 50)
        wrap = ttk.Frame(f)
        wrap.pack(fill="both", expand=True)
        wrap.rowconfigure(0, weight=1)
        wrap.columnconfigure(0, weight=1)
        self.trade_tree = ttk.Treeview(wrap, columns=cols, show="headings")
        self._trade_data = []
        self._trade_sort = (0, True)
        for idx, (c, h, w) in enumerate(zip(cols, heads, widths)):
            self.trade_tree.heading(c, text=h, anchor="center", command=lambda col=c: self._sort_trade(col))
            self.trade_tree.column(c, width=w, anchor="w" if c in ("entry_time", "exit_time", "dir", "reason") else "e")
        self.trade_tree.grid(row=0, column=0, sticky="nsew")
        vsb = ttk.Scrollbar(wrap, orient="vertical", command=self.trade_tree.yview)
        vsb.grid(row=0, column=1, sticky="ns")
        self.trade_tree.configure(yscrollcommand=vsb.set)

    def _trade_display(self, r):
        """Format raw tuple -> display tuple for the TRADE table."""
        labels = self._exit_labels()
        return (
            str(int(r[0])), _fmt_dt(int(r[1])), _fmt_dt(int(r[2])),
            "BUY" if r[3] > 0 else "SELL", _fmt_num(r[4]), _fmt_num(r[5]),
            _fmt_num(r[6]), _fmt_num(r[7]), _fmt_num(r[8], 2),
            f"{r[9]:+.2f}", labels.get(int(r[10]), str(int(r[10]))),
            _fmt_num(r[11], 1),
        )

    _TRADE_KEYMAP = {"no": 0, "entry_time": 1, "exit_time": 2, "dir": 3,
                     "entry": 4, "exit": 5, "sl": 6, "tp": 7, "lot": 8,
                     "pnl": 9, "reason": 10, "score": 11}
    _TRADE_COLS = ("no", "entry_time", "exit_time", "dir", "entry", "exit",
                   "sl", "tp", "lot", "pnl", "reason", "score")

    def _render_trades(self):
        self.trade_tree.delete(*self.trade_tree.get_children())
        for r in self._trade_data:
            self.trade_tree.insert("", "end", values=self._trade_display(r))

    def _sort_trade_data(self):
        idx, asc = self._trade_sort
        if not (0 <= idx < len(self._TRADE_COLS)):
            idx = 0
        ki = self._TRADE_KEYMAP[self._TRADE_COLS[idx]]
        self._trade_data.sort(key=lambda r: self._safe_sort_key(r[ki]),
                              reverse=not asc)

    def _sort_trade(self, col):
        idx = self._TRADE_COLS.index(col)
        asc = not self._trade_sort[1] if self._trade_sort[0] == idx else True
        self._trade_sort = (idx, asc)
        if not self._trade_data:
            return
        self._sort_trade_data()
        self._render_trades()

    # =================================================================
    # Optimization results table: separate column per optimized parameter
    # (active as soon as optimization starts), English UI.
    # =================================================================
    _BASE_OPT_COLS = ("no", "score", "trades", "win", "net", "pf", "dd")
    _BASE_OPT_HEADS = {"no": "#", "score": "SCORE", "trades": "TRADES",
                       "win": "WIN %", "net": "PROFIT", "pf": "PF",
                       "dd": "Max DD"}
    _BASE_OPT_WIDTHS = {"no": 40, "score": 70, "trades": 60, "win": 60,
                        "net": 90, "pf": 60, "dd": 80}

    def _build_opt_tab(self):
        f = ttk.Frame(self.tab_opt, padding=8)
        f.pack(fill="both", expand=True)
        btns = ttk.Frame(f)
        btns.pack(fill="x", pady=(0, 6))
        ttk.Button(btns, text="Apply Row", command=self._apply_opt_row).pack(side="left")
        ttk.Button(btns, text="Save Optimization", command=self._export_opt).pack(side="left", padx=6)
        ttk.Button(btns, text="Load Optimization", command=self._load_opt).pack(side="left", padx=6)
        self.opt_note_var = tk.StringVar(value="No optimization yet.")
        ttk.Label(btns, textvariable=self.opt_note_var, style="Muted.TLabel").pack(side="right")
        self.opt_wrap = ttk.Frame(f)
        self.opt_wrap.pack(fill="both", expand=True)
        self.opt_wrap.rowconfigure(0, weight=1)
        self.opt_wrap.columnconfigure(0, weight=1)
        # state
        self.opt_tree = None
        self.vsb_opt = None
        self._opt_rows = []        # list of record dicts
        self._opt_records = {}     # iid -> record dict
        self._opt_param_cols = []  # parameter column names shown (ordered)
        self._opt_sort = ("score", False)   # (column, ascending)
        self._make_opt_tree()

    # ---- column management -------------------------------------------------
    def _checked_opt_params(self):
        return [n for n in self.opt_flags if self.opt_flags[n].get()]

    def _desired_opt_param_cols(self):
        cols = list(self._checked_opt_params())
        seen = set(cols)
        for rec in self._opt_rows:
            for k in rec["params"]:
                if k not in seen:
                    seen.add(k)
                    cols.append(k)
        return cols

    def _opt_all_cols(self):
        return self._BASE_OPT_COLS + tuple(self._opt_param_cols)

    def _make_opt_tree(self):
        top_rec = self._viewport_top_record()
        if self.opt_tree is not None:
            self.opt_tree.destroy()
            self.vsb_opt.destroy()
        cols = self._opt_all_cols()
        self.opt_tree = ttk.Treeview(self.opt_wrap, columns=cols, show="headings")
        for c in cols:
            h = self._BASE_OPT_HEADS.get(c, c)
            w = self._BASE_OPT_WIDTHS.get(c, 90)
            self.opt_tree.heading(c, text=h, anchor="center", command=lambda col=c: self._sort_opt(col))
            self.opt_tree.column(c, width=w, anchor="e")
        self.opt_tree.grid(row=0, column=0, sticky="nsew")
        self.vsb_opt = ttk.Scrollbar(self.opt_wrap, orient="vertical",
                                     command=self.opt_tree.yview)
        self.vsb_opt.grid(row=0, column=1, sticky="ns")
        self.opt_tree.configure(yscrollcommand=self.vsb_opt.set)
        self._render_opt_rows(top_rec)

    def _viewport_top_record(self):
        """The record currently shown at the top of the scrollable view."""
        if not self._opt_rows:
            return None
        frac = 0.0
        if self.opt_tree is not None:
            try:
                frac = self.opt_tree.yview()[0]
            except Exception:
                frac = 0.0
        idx = min(len(self._opt_rows) - 1, int(frac * len(self._opt_rows)))
        return self._opt_rows[idx]

    def _render_opt_rows(self, top_rec=None):
        if top_rec is None:
            top_rec = self._viewport_top_record()
        self.opt_tree.delete(*self.opt_tree.get_children())
        self._opt_records = {}
        cols = self._opt_all_cols()
        for rec in self._opt_rows:
            values = tuple(self._opt_cell(c, rec) for c in cols)
            iid = self.opt_tree.insert("", "end", values=values)
            self._opt_records[iid] = rec
        if top_rec is not None and self._opt_rows:
            try:
                idx = self._opt_rows.index(top_rec)
                self.opt_tree.yview_moveto(idx / max(1, len(self._opt_rows)))
            except Exception:
                pass

    def _opt_cell(self, c, rec):
        base = rec["params"]
        if c in base:
            v = base[c]
            try:
                return f"{float(v):.6g}"
            except (TypeError, ValueError):
                return str(v)
        v = rec.get(c)
        if v is None:
            return ""
        if c == "no":
            return str(int(v))
        if c in ("score", "pf", "dd"):
            return f"{float(v):.2f}"
        if c == "trades":
            return str(int(v))
        if c == "win":
            return f"{float(v):.1f}"
        if c == "net":
            return f"{float(v):+.2f}"
        return str(v)

    def _sync_opt_columns(self):
        desired = self._desired_opt_param_cols()
        if desired == self._opt_param_cols:
            return
        self._opt_param_cols = desired
        self._make_opt_tree()

    # ---- helpers -----------------------------------------------------------
    @staticmethod
    def _safe_sort_key(v):
        """Numeric-safe key: numbers sort numerically, text alphabetically."""
        try:
            return (0, float(v))
        except (TypeError, ValueError):
            return (1, str(v))

    def _opt_sort_key(self, col, rec):
        if col in self._BASE_OPT_HEADS:
            return self._safe_sort_key(rec.get(col, ""))
        return self._safe_sort_key(rec["params"].get(col, ""))

    def _opt_insert_index(self, rec):
        """Position where `rec` belongs under the last column sort chosen."""
        col, ascending = self._opt_sort
        rk = self._opt_sort_key(col, rec)
        lo, hi = 0, len(self._opt_rows)
        while lo < hi:
            mid = (lo + hi) // 2
            km = self._opt_sort_key(col, self._opt_rows[mid])
            if (km <= rk) if ascending else (km >= rk):
                lo = mid + 1
            else:
                hi = mid
        return lo

    # ---- sort ---------------------------------------------------------------
    def _sort_opt(self, col):
        cols = self._opt_all_cols()
        if col not in cols:
            return
        cur_col, cur_asc = self._opt_sort
        ascending = not cur_asc if cur_col == col else True
        self._opt_sort = (col, ascending)
        if not self._opt_rows:
            # remember the sort even when the table is empty; new rows
            # will be inserted at the sorted position during streaming
            return
        self._opt_rows.sort(key=lambda rec: self._opt_sort_key(col, rec),
                            reverse=not ascending)
        self._render_opt_rows()

    def _load_opt(self):
        p = filedialog.askopenfilename(filetypes=[("CSV", "*.csv"), ("HTML", "*.html")])
        if not p:
            return
        if not p.lower().endswith(".csv"):
            messagebox.showinfo("Load", "HTML format is not supported. Use CSV.")
            return
        import csv
        with open(p, newline="", encoding="utf-8-sig") as f:
            rows = list(csv.reader(f))
        if not rows:
            return
        header = [h.strip() for h in rows[0]]
        base_alias = {"no": "no", "net profit": "net", "pf": "pf", "max dd": "dd",
                      "trades": "trades", "win %": "win", "score": "score",
                      "return %": "ret_pct", "return/drawdown": "ret_pct",
                      "skor": "score", "skor seimbang": "score"}
        cols = [base_alias.get(h.lower(), h) for h in header]
        param_cols = [c for c in cols if c not in self._BASE_OPT_COLS]
        self._opt_param_cols = param_cols
        self._opt_rows = []
        for idx, line in enumerate(rows[1:], 1):
            if not line:
                continue
            rec = {"no": idx, "score": 0.0, "trades": 0, "win": 0.0,
                   "net": 0.0, "pf": 0.0, "dd": 0.0, "params": {}}
            for j, key in enumerate(cols):
                if j >= len(line):
                    break
                v = line[j].strip()
                if key in self._BASE_OPT_COLS:
                    try:
                        if key == "trades":
                            rec[key] = int(float(v.replace(",", ".")))
                        else:
                            rec[key] = float(v.replace(",", "."))
                    except ValueError:
                        pass
                else:
                    try:
                        rec["params"][key] = float(v.replace(",", "."))
                    except ValueError:
                        rec["params"][key] = v
            self._opt_rows.append(rec)
        # honor the last column sort chosen by the user
        col, ascending = self._opt_sort
        if col:
            self._opt_rows.sort(key=lambda r: self._opt_sort_key(col, r),
                                reverse=not ascending)
        self._make_opt_tree()
        self.opt_note_var.set(f"Loaded {len(self._opt_rows)} rows from {Path(p).name}")
        self.log(f"Loaded optimization: {len(self._opt_rows)} rows from {p}")

    def _build_log_tab(self):
        f = ttk.Frame(self.tab_log, padding=(8, 6, 8, 6))
        f.pack(fill="both", expand=True)
        btns = ttk.Frame(f)
        btns.pack(fill="x", pady=(0, 6))
        ttk.Button(btns, text="Clear Log", command=lambda: self.log_text.delete("1.0", "end")).pack(side="left")
        ttk.Button(btns, text="Save Log", command=self._save_log).pack(side="left", padx=6)
        self.log_text = tk.Text(f, bg="#0f1724", fg=TEXT, bd=0, relief="flat", wrap="word", font=("Consolas", 9), padx=10, pady=8)
        self.log_text.pack(fill="both", expand=True)

    # =================================================================
    # STRATEGY tab
    # =================================================================
    def _build_strategy_tab(self):
        outer = ttk.Frame(self.tab_ai, padding=8)
        outer.pack(fill="both", expand=True)
        outer.columnconfigure(0, weight=1)
        outer.columnconfigure(1, weight=1)
        outer.rowconfigure(0, weight=1)

        # ---------- LEFT: big paste area ----------------------------------
        left = ttk.Frame(outer, style="Panel.TFrame", padding=0)
        left.grid(row=0, column=0, sticky="nsew", padx=(0, 6))
        paste_wrap = ttk.Frame(left, style="Panel.TFrame", padding=0)
        paste_wrap.grid(row=0, column=0, sticky="nsew")
        paste_wrap.rowconfigure(0, weight=1)
        paste_wrap.columnconfigure(0, weight=0)
        paste_wrap.columnconfigure(1, weight=1)
        self.paste_linenums = tk.Canvas(paste_wrap, width=44, bg="#0b1220", highlightthickness=0, takefocus=False)
        self.paste_linenums.grid(row=0, column=0, sticky="ns")
        self.paste_text = tk.Text(paste_wrap, bg="#0f1724", fg=TEXT, bd=0, relief="flat",
                                  wrap="none", font=("Consolas", 9), padx=10, pady=8)
        pvs = ttk.Scrollbar(paste_wrap, orient="vertical", command=self.paste_text.yview)
        phs = ttk.Scrollbar(paste_wrap, orient="horizontal", command=self.paste_text.xview)
        self.paste_text.configure(yscrollcommand=self._paste_yview, xscrollcommand=self._paste_xview)
        self.paste_text.grid(row=0, column=1, sticky="nsew")
        pvs.grid(row=0, column=2, sticky="ns")
        phs.grid(row=1, column=1, sticky="ew")
        for seq in ("<Key>", "<KeyRelease>", "<Button-1>"):
            self.paste_text.bind(seq, self._on_paste_edit, add="+")
        for seq in ("<MouseWheel>", "<Button-4>", "<Button-5>"):
            self.paste_text.bind(seq, self._on_paste_scroll, add="+")
        self.paste_text.bind("<Configure>", self._on_paste_edit, add="+")
        left.rowconfigure(0, weight=1)
        left.columnconfigure(0, weight=1)

        # ---------- RIGHT: apply panel -------------------------------------
        right = ttk.Frame(outer, style="Panel.TFrame", padding=10)
        right.grid(row=0, column=1, sticky="nsew")

        # Kept as a StringVar so apply/generation events can report status via the LOG.
        self.gen_status_var = tk.StringVar(
            value="Paste the module the MetaEditor AI Assistant gives you, then APPLY AS STRATEGY.")

        # Prompt loaded from the bundled translation prompt file.
        prompt_path = Path(__file__).resolve().parents[1] / "MQL5_to_AlgoBackTune_prompt.txt"
        try:
            self.ai_prompt = prompt_path.read_text(encoding="utf-8")
        except OSError:
            self.ai_prompt = (
                "You are an expert MQL5\u2192Python translator for the AlgoBackTune backtester.\n"
                "You output ONE complete, self-contained Python module and nothing else.\n"
                "No explanations, no markdown fences unless asked, no prose outside the code.\n"
                "The module must implement the requested strategy exactly, in the exact ABI\n"
                "described, runnable by Numba @njit(cache=True).\n"
            )

        # Large navy prompt box with a FIXED height (like the example) so the
        # controls below it stay visible. Shows the full translation prompt
        # file automatically.
        prompt_wrap = ttk.Frame(right, style="Panel.TFrame", padding=0)
        prompt_wrap.pack(anchor="w", fill="x", pady=(0, 2))
        prompt_box = tk.Text(prompt_wrap, bg="#16213a", fg=TEXT, bd=0, relief="flat",
                             wrap="word", font=("Consolas", 8), padx=10, pady=8, height=12)
        pvs2 = ttk.Scrollbar(prompt_wrap, orient="vertical", command=prompt_box.yview)
        prompt_box.configure(yscrollcommand=pvs2.set)
        prompt_box.insert("1.0", self.ai_prompt)
        prompt_box.configure(state="disabled")
        prompt_box.grid(row=0, column=0, sticky="nsew")
        pvs2.grid(row=0, column=1, sticky="ns")
        prompt_wrap.columnconfigure(0, weight=1)
        ttk.Button(right, text="Copy Prompt", command=self._copy_prompt).pack(anchor="w",
                                                                               pady=(4, 10))

        ttk.Label(right, text="Copy the prompt above into the MetaEditor AI Assistant (or any AI tool),\n"
                              "paste your .mq5 code underneath and execute\n"
                              "Paste the generated module back here and select APPLY AS STRATEGY.",
                  style="Panel.TLabel", justify="left").pack(anchor="w", pady=(0, 12))

        btn_row = ttk.Frame(right, style="Panel.TFrame")
        btn_row.pack(anchor="w", fill="x")
        self.btn_apply = ttk.Button(btn_row, text="APPLY AS STRATEGY", command=self._on_apply_pasted)
        self.btn_apply.pack(side="left")
        ttk.Button(btn_row, text="Clear",
                   command=lambda: (self.paste_text.delete("1.0", "end"), self._refresh_paste_linenums())).pack(side="left", padx=6)

        ttk.Label(right, text="Apply runs a syntax check and a synthetic smoke backtest first; "
                              "the code is cached and activated only if it passes.\n"
                              "\u26a0  Always validate against the MT5 Tester before trusting results, "
                              "and never trade live with it unchecked.",
                  style="Muted.TLabel", wraplength=420, justify="left",
                  font=("Segoe UI", 8, "italic")).pack(anchor="w", pady=(10, 0))

        self._update_ai_labels()

    def _update_ai_labels(self):
        hint = getattr(self, "ea_hint_var", None)
        if hint is None:
            return
        params = getattr(self, "ea_params", None)
        n = len(params.spec) if params is not None else 0
        if n:
            hint.set(f"Strategy: {self.ea_name}  \u2022  params: {n}")
        else:
            hint.set("No strategy applied yet - apply one in the STRATEGY tab")

    def _copy_prompt(self):
        self.clipboard_clear()
        self.clipboard_append(self.ai_prompt)
        self.log("Prompt copied to the clipboard.")

    def _paste_yview(self, *args):
        self._refresh_paste_linenums()

    def _paste_xview(self, *args):
        pass

    def _on_paste_edit(self, event=None):
        self._refresh_paste_linenums()

    def _on_paste_scroll(self, event=None):
        try:
            delta = getattr(event, "delta", 0) or 0
            if delta:
                self.paste_text.yview_scroll(int(-delta / 120), "units")
        except tk.TclError:
            pass
        self._refresh_paste_linenums()

    def _refresh_paste_linenums(self):
        """Redraw the line-number gutter for the current paste content."""
        try:
            nlines = len(self.paste_text.get("1.0", "end-1c").splitlines())
            # Use a FIXED line height from the text font so alignment is stable
            # before/during scroll (dlineinfo() returns None for off-screen lines).
            text_font = tkfont.Font(font=self.paste_text["font"])
            num_font = tkfont.Font(family="Consolas", size=7)
            line_h = float(text_font.metrics("linespace"))
            text_ascent = float(text_font.metrics("ascent"))
            num_ascent = float(num_font.metrics("ascent"))
            pady = int(self.paste_text["pady"] or 0)
            self.paste_linenums.delete("all")
            for i in range(1, nlines + 1):
                y = pady + (i - 1) * line_h
                # align the number's BASELINE with the text's baseline
                number_y = y + (text_ascent - num_ascent)
                self.paste_linenums.create_text(40, number_y, anchor="ne", text=str(i),
                                                fill="#6b7a99", font=("Consolas", 7), justify="right")
            total_h = pady + nlines * line_h + pady
            self.paste_linenums.configure(scrollregion=(0, 0, 44, total_h))
            # keep the gutter scrolled in sync with the text viewport
            frac = self.paste_text.yview()[0]
            self.paste_linenums.yview_moveto(frac)
        except tk.TclError:
            pass

    def _on_apply_pasted(self):
        if self._ai_busy:
            return
        text = self.paste_text.get("1.0", "end").strip()
        if not text:
            messagebox.showwarning("Apply Strategy", "Paste the generated code first.")
            return
        settings = self._read_settings()   # snapshot on the main thread
        self._ai_busy = True
        self.btn_apply.state(["disabled"])
        self.gen_status_var.set("Applying pasted code... - syntax check, smoke backtest.")
        self.log("Strategy: applying pasted code...")

        def emit(ev):
            ev = dict(ev)
            ev["gen"] = self._run_gen
            self.q.put(ev)

        def work():
            code = strategy_cache.extract_python_code(text)
            repaired = strategy_cache.normalize_code(code)
            if repaired != code:
                code = repaired
                emit({"t": "ai_log", "text": "Apply: auto-repaired indentation of the pasted module."})
            ok, err = strategy_cache.validate_syntax(code)
            if not ok:
                emit({"t": "ai_done", "ok": False,
                      "message": f"The pasted code failed the syntax/contract check:\n{err}"})
                return
            module = strategy_cache.import_generated_code(code, "smoke")
            if module is None:
                emit({"t": "ai_done", "ok": False,
                      "message": "The module could not be imported (module-level error). "
                                 "Make sure all code is inside @njit functions, numpy/numba only."})
                return
            emit({"t": "ai_status", "text": "Compiling & running synthetic backtest..."})
            # Build settings from the NEW module's PARAM_SPEC (not the
            # currently-applied one) so the smoke test has a valid param
            # vector even when no strategy is applied yet.
            new_spec = strategy_cache.parse_param_spec(module)
            smoke_params = EAParams(tuple(ParamSpec(n, k, d, c) for (n, k, d, c) in new_spec))
            smoke_settings = Settings(
                point=settings.point, pip=settings.pip, pt_scale=settings.pt_scale,
                contract=settings.contract, start_balance=settings.start_balance,
                commission_per_lot=settings.commission_per_lot,
                slippage_points=settings.slippage_points,
                params=smoke_params,
            )
            ok, err = strategy_cache.smoke_test_module(module, smoke_settings)
            if not ok:
                tail = "\n".join(err.strip().splitlines()[-12:]) if err.strip() else "unknown error"
                last = err.strip().splitlines()[-1] if err.strip() else "unknown error"
                emit({"t": "ai_log", "text": f"Apply: smoke test failed - {last}"})
                emit({"t": "ai_done", "ok": False,
                      "message": f"The pasted strategy failed the synthetic smoke backtest:\n{tail}"})
                return
            path = strategy_cache.save_strategy_for_code(code, "pasted (AlgoBackTune)")
            spec = new_spec
            emit({"t": "ai_done", "ok": True, "path": str(path),
                  "code": code, "module": module, "spec": spec})

        threading.Thread(target=work, daemon=True).start()

    # ---- helpers ----
    def log(self, text):
        stamp = datetime.now().strftime("%H:%M:%S")
        self.log_text.insert("end", f"[{stamp}] {text}\n")
        self.log_text.see("end")

    def _set_busy(self, busy):
        self._busy = busy
        if busy:
            self.btn_bt.state(["disabled"])
            self.btn_opt.state(["disabled"])
            self.progress.configure(value=0)
        else:
            self.btn_bt.state(["!disabled"])
            self.btn_opt.state(["!disabled"])
            self.progress.configure(value=0)

    def _start_discovery(self):
        sym = self.symbol_var.get().strip().upper()
        all_srcs = discover_sources(self.custom_roots)
        self.sources = [s for s in all_srcs if sym in s.symbol.upper()] if sym else all_srcs
        self.source_combo["values"] = [s.label() for s in self.sources]
        if self.sources:
            self.source_combo.current(0)
            self._on_source_change()

    # ---- backtest / optimize (engine: strategy_module required) ----
    def start_backtest(self):
        if self._busy:
            return
        if not self.source:
            messagebox.showwarning("Data", "Select a tick source.")
            return
        # check tick availability for range
        try:
            start = parse_datetime_ms(self.from_var.get().strip() + " 00:00:00") if len(self.from_var.get().strip()) <= 10 else parse_datetime_ms(self.from_var.get().strip())
            end = parse_datetime_ms(self.to_var.get().strip() + " 23:59:59") if len(self.to_var.get().strip()) <= 10 else parse_datetime_ms(self.to_var.get().strip())
        except ValueError as exc:
            messagebox.showerror("Input", str(exc))
            return
        if self.source.last_time_ms < start or self.source.first_time_ms > end:
            self.status_var.set("No ticks for this symbol/time range. Download ticks in MT5.")
            messagebox.showwarning("Ticks", "No ticks for this symbol/time range. Download ticks in MT5, then Refresh.")
            return
        settings = self._read_settings()
        tf_ms = PERIOD_MS.get(self.tf_var.get(), 5 * 60_000)
        self.cancel_event.clear()
        self.backtest_cancel.clear()
        self._run_gen += 1
        self._set_busy(True)
        self._backtest_running = True
        self._trade_progress = np.zeros(1, np.int64)
        self._last_trade_count = 0
        self._trade_live_ctx = {}
        self._trade_data = []
        self._render_trades()
        self.trade_count_var.set("0 trade(s)")
        self.log(f"Starting backtest (strategy: {self.strategy_mode})...")

        gen = self._run_gen

        def emit(ev):
            ev = dict(ev)
            ev["gen"] = gen
            self.q.put(ev)

        def work():
            try:
                load_start = max(self.source.first_time_ms, start - warmup_ms(settings, tf_ms))
                def prog(name, i, total):
                    if i == total or i % 5 == 0:
                        emit({"t": "status", "text": f"Reading {name} ({i}/{total})..."})
                        pct = (i / total) * 50.0 if total else 0
                        emit({"t": "progress", "pct": pct})
                ticks = read_source(self.source, load_start, end, progress=prog)
                if len(ticks) == 0:
                    emit({"t": "warn", "message": "No ticks for this symbol/time range. Download ticks in MT5, then Refresh."})
                    return
                first = int(ticks.time_ms[0]); last = int(ticks.time_ms[-1])
                tol = 2 * 86_400_000
                if first > start + tol or last < end - tol:
                    emit({"t": "warn", "message": "Ticks do not cover the full required range. Download ticks in MT5."})
                emit({"t": "status", "text": "Preparing bars & indicators..."})
                emit({"t": "progress", "pct": 60})
                pref = prepare(ticks, settings, tf_ms)
                emit({"t": "progress", "pct": 80})
                emit({"t": "status", "text": "Running tick-level simulation..."})
                result = run_prepared(pref, settings, active_from_ms=start,
                                      trade_progress=self._trade_progress,
                                      trade_live_ctx=self._trade_live_ctx,
                                      strategy_module=self.strategy_module)
                emit({"t": "progress", "pct": 100})
                emit({"t": "result", "result": result})
            except Exception as exc:
                import traceback
                emit({"t": "error", "message": str(exc), "detail": traceback.format_exc()})
            finally:
                emit({"t": "done"})

        threading.Thread(target=work, daemon=True).start()

    def start_optimize(self):
        if self._busy:
            return
        # collect grid
        grid = {}
        for name, flag in self.opt_flags.items():
            if not flag.get():
                continue
            lo = self.grid_lo[name].get().strip()
            step = self.grid_step[name].get().strip()
            hi = self.grid_hi[name].get().strip()
            if not (lo and step and hi):
                continue
            try:
                grid[name] = (float(lo.replace(",", ".")), float(hi.replace(",", ".")), float(step.replace(",", ".")))
            except ValueError:
                continue
        if not grid:
            messagebox.showwarning("Optimization", "No parameters checked.")
            return
        try:
            start = parse_datetime_ms(self.from_var.get().strip() + " 00:00:00") if len(self.from_var.get().strip()) <= 10 else parse_datetime_ms(self.from_var.get().strip())
            end = parse_datetime_ms(self.to_var.get().strip() + " 23:59:59") if len(self.to_var.get().strip()) <= 10 else parse_datetime_ms(self.to_var.get().strip())
        except ValueError as exc:
            messagebox.showerror("Input", str(exc))
            return
        settings = self._read_settings()
        obj = self.objective_var.get()
        method = "fast" if "Quick" in self.method_var.get() else "grid"
        max_combos = 20000
        if method == "grid":
            est = 1
            try:
                for name, (lo, hi, step) in grid.items():
                    est *= max(1, len(build_values(lo, hi, step, self._opt_value_is_int(name))))
                    if est > 10**15:
                        break
            except Exception:
                est = 10**15
            if est > 20000:
                if not messagebox.askyesno(
                    "Full Optimization",
                    f"Full optimization will evaluate about {est:,.0f} combinations. "
                    "This can take a very long time (and may not finish).\n\n"
                    "Quick mode finds good results much faster.\n\n"
                    "Continue with Full anyway?",
                ):
                    return
                max_combos = int(est)
        iters = int(float(self.iter_var.get() or 300))
        tf_ms = PERIOD_MS.get(self.tf_var.get(), 5 * 60_000)
        self.cancel_event.clear()
        self._run_gen += 1
        self._set_busy(True)
        self._backtest_running = False
        # reset the optimization results table for the new run
        self._opt_rows = []
        self._opt_records = {}
        self._opt_param_cols = list(self._desired_opt_param_cols())
        self._make_opt_tree()
        self.opt_note_var.set("Starting optimization...")
        self.opt_steps_var.set("evaluating...")
        self.log(f"Starting optimization (strategy: {self.strategy_mode})...")

        gen = self._run_gen

        def emit(ev):
            ev = dict(ev)
            ev["gen"] = gen
            self.q.put(ev)

        def work():
            try:
                load_start = max(self.source.first_time_ms, start - warmup_ms(settings, tf_ms))
                def prog_read(name, i, total):
                    if i == total or i % 5 == 0:
                        pct = (i / total) * 30.0 if total else 0
                        emit({"t": "progress", "pct": pct})
                ticks = read_source(self.source, load_start, end, progress=prog_read)
                g = {}
                for name, (lo, hi, step) in grid.items():
                    is_int = self._opt_value_is_int(name)
                    g[name] = build_values(lo, hi, step, is_int)
                emit({"t": "progress", "pct": 40})
                req = OptimizeRequest(base=settings, grid=g, objective=obj, method=method,
                                      iterations=iters, active_from_ms=start,
                                      max_combos=max_combos)
                def prog(i, total, row):
                    pct = 40 + (i / total) * 60.0 if total else 40
                    emit({"t": "progress", "pct": pct})
                    emit({"t": "opt_progress", "i": i, "total": total, "row": row})
                res = optimize(ticks, req, progress=prog, cancel=self.cancel_event,
                               strategy_module=self.strategy_module)
                emit({"t": "progress", "pct": 100})
                emit({"t": "opt_done", "result": res})
            except Exception as exc:
                import traceback
                emit({"t": "error", "message": str(exc), "detail": traceback.format_exc()})
            finally:
                emit({"t": "done"})

        threading.Thread(target=work, daemon=True).start()

    def cancel_all(self):
        self.cancel_event.set()
        self.backtest_cancel.set()
        self._run_gen += 1        # invalidate events from the stopped run
        self._backtest_running = False
        self._set_busy(False)
        self.log("Cancelled.")

    def _handle_event(self, ev):
        gen = ev.get("gen")
        if gen is not None and gen != self._run_gen:
            return
        kind = ev.get("t")
        if kind == "log":
            self.log(ev["text"])
        elif kind == "status":
            self.status_var.set(ev["text"])
        elif kind == "progress":
            self.progress.configure(value=ev.get("pct", 0))
        elif kind == "result":
            self.progress.configure(value=100)
            self._backtest_running = False
            self._show_result(ev["result"])
        elif kind == "opt_progress":
            self._opt_progress(ev["i"], ev["total"], ev["row"])
        elif kind == "opt_done":
            self._show_opt_result(ev["result"])
        elif kind == "ai_status":
            self.gen_status_var.set(ev.get("text", ""))
        elif kind == "ai_log":
            self.log(ev.get("text", ""))
        elif kind == "ai_done":
            self._ai_busy = False
            try:
                self.btn_apply.state(["!disabled"])
            except Exception:
                pass
            if ev.get("ok"):
                module = ev.get("module")
                spec = ev.get("spec") or []
                self.strategy_module = module
                self.strategy_mode = "generated"
                if spec:
                    self.ea_params = EAParams(tuple(ParamSpec(*s) for s in spec))
                self.ea_name = f"pasted ({Path(ev.get('path', '')).stem})"
                self.ea_source_path = None
                self.ea_source = None
                self._load_param_panel()
                self.gen_status_var.set("Strategy applied & cached (pasted code).")
                self.log(f"Strategy: saved to {ev.get('path')} and loaded.")
                self._update_ai_labels()
                messagebox.showinfo(
                    "Apply Strategy",
                    "Strategy applied successfully.\n\n"
                    f"Cached to:\n{ev.get('path')}\n\n"
                    "The strategy is now active.")
            else:
                self.gen_status_var.set("Apply failed - see LOG.")
                self.log(f"Strategy: apply failed:\n{ev.get('message', 'unknown error')}")
                err = str(ev.get("message") or "unknown error")
                tail = "\n".join(err.strip().splitlines()[-8:])
                messagebox.showerror(
                    "Apply Strategy",
                    f"Strategy apply FAILED.\n\n{tail}\n\nSee the LOG tab for the full details.")
        elif kind == "warn":
            self._backtest_running = False
            self._set_busy(False)
            self.status_var.set(ev.get("message", "Warning"))
            self.log(ev.get("message", "Warning"))
            messagebox.showwarning("Ticks", ev.get("message", "Warning"))
        elif kind == "error":
            self._backtest_running = False
            self._set_busy(False)
            self.status_var.set(ev.get("message", "Error"))
            self.log(ev.get("message", "Error"))
            if ev.get("detail"):
                self.log(ev["detail"])
            messagebox.showerror("Error", ev.get("message", "Error"))
        elif kind == "done":
            self._backtest_running = False
            self._set_busy(False)

    def _show_result(self, result):
        self.last_result = result
        pairs = dict(metrics_pairs(result.metrics))
        for label, var in self._metric_vars:
            var.set(pairs.get(label, "-"))
        self.summary_text.delete("1.0", "end")
        self.summary_text.insert("end", summary_text(result, leverage=self.leverage_var.get()))
        self.chart.set_data(result.equity_time, result.equity)
        # trades
        t = result.trades
        self._trade_data = []
        for i in range(len(t)):
            pnl = float(t.pnl[i])
            row = (
                i + 1, int(t.entry_time[i]), int(t.exit_time[i]),
                1 if t.dir[i] > 0 else -1, float(t.entry[i]), float(t.exit[i]),
                float(t.sl_init[i]), float(t.tp[i]), float(t.lot[i]),
                float(pnl), int(t.reason[i]), float(t.score[i]),
            )
            self._trade_data.append(row)
        # keep the display order of the last column sort chosen by the user
        self._sort_trade_data()
        self._render_trades()
        self.trade_count_var.set(f"{len(t)} trade(s)")
        open_end = int(result.meta.get("open_positions_at_end", 0))
        self.status_var.set(
            f"Backtest finished: {result.metrics.trades} trade(s), "
            f"net {result.metrics.net_profit:+.2f}"
        )
        if open_end:
            self.log(f"Open positions at end of data: {open_end} "
                     f"(excluded from results)")

    def _opt_progress(self, i, total, row):
        mm = row.metrics
        self.opt_steps_var.set(f"{i + 1}/{total} steps")
        self.opt_note_var.set(f"Evaluating {i + 1}/{total}...")
        rec = {
            "no": i + 1,
            "score": float(row.score),
            "trades": int(mm.trades),
            "win": float(mm.win_rate),
            "net": float(mm.net_profit),
            "pf": float(mm.profit_factor),
            "dd": float(mm.max_dd),
            "params": {k: row.params.get(k, "") for k in self._opt_param_cols},
        }
        insert_at = self._opt_insert_index(rec)
        view_frac = self.opt_tree.yview()[0] if self._opt_rows else 0.0
        top_idx = int(view_frac * len(self._opt_rows))
        self._opt_rows.insert(insert_at, rec)
        cols = self._opt_all_cols()
        iid = self.opt_tree.insert(
            "", insert_at, values=tuple(self._opt_cell(c, rec) for c in cols))
        self._opt_records[iid] = rec
        if view_frac > 0.0 and len(self._opt_rows) > 1:
            # scrolled away from the top: keep the same top row in view so
            # the table does not jump up and down as rows stream in
            if insert_at <= top_idx:
                top_idx += 1
            self.opt_tree.yview_moveto(top_idx / len(self._opt_rows))
        # at the very top (frac == 0): do not scroll; new best rows simply
        # stream in at the top while the view stays put

    def _show_opt_result(self, result):
        self.last_opt = result
        # Parameter columns = checked parameters + any parameter present in
        # rows (union, preserving order).
        param_cols = list(self._checked_opt_params())
        seen = set(param_cols)
        for row in result.rows:
            for k in row.params:
                if k not in seen:
                    seen.add(k)
                    param_cols.append(k)
        self._opt_param_cols = param_cols
        self._make_opt_tree()   # rebuild columns, keeps existing rows
        self._opt_rows = []
        rows = sorted(result.rows, key=lambda r: r.score, reverse=True)
        for rank, row in enumerate(rows[:800], 1):
            mm = row.metrics
            rec = {
                "no": rank,
                "score": float(row.score),
                "trades": int(mm.trades),
                "win": float(mm.win_rate),
                "net": float(mm.net_profit),
                "pf": float(mm.profit_factor),
                "dd": float(mm.max_dd),
                "params": {k: row.params.get(k, "") for k in param_cols},
            }
            self._opt_rows.append(rec)
        # honor the last column sort chosen by the user for the display
        col, ascending = self._opt_sort
        if col:
            self._opt_rows.sort(key=lambda r: self._opt_sort_key(col, r),
                                reverse=not ascending)
        self._render_opt_rows()
        self._update_opt_steps()   # restore the configured grid-size label
        self.opt_note_var.set(
            f"{result.evaluated}/{result.total} evaluated in {result.elapsed:.1f} s"
            + (" (stopped)" if result.cancelled else ""))
        if result.best and result.best_result:
            self._show_result(result.best_result)

    def _apply_opt_row(self):
        sel = self.opt_tree.selection()
        if not sel:
            messagebox.showinfo("Apply Row", "Select an optimization result row first.")
            return
        rec = self._opt_records.get(sel[0])
        if rec is None:
            messagebox.showinfo("Apply Row", "The selected row is no longer available.")
            return
        params = rec.get("params", {})
        if not params:
            messagebox.showinfo("Apply Row", "The selected row has no parameter values.")
            return
        applied, skipped = [], []
        for k, v in params.items():
            if k not in self.vars:
                skipped.append(k)
                continue
            kind = self._ea_kind(k)
            try:
                fv = float(v)
            except (TypeError, ValueError):
                continue
            target = self.vars[k]
            if kind == "bool":
                target.set("1" if fv else "0")
            elif kind == "int":
                target.set(str(int(round(fv))))
            else:
                target.set(f"{fv:g}")
            applied.append((k, fv))
            if k in self.grid_lo:
                self.grid_lo[k].set(f"{fv:g}")
                self.grid_hi[k].set(f"{fv:g}")
                self.grid_step[k].set("1")
        if applied:
            self.log("Parameters applied from optimization row: "
                     + ", ".join(f"{k}={v:g}" for k, v in applied))
            self.status_var.set("Parameters applied from selected row.")
        if skipped:
            self.log(f"Skipped unknown parameters: {', '.join(skipped)}")

    def _export_all(self):
        if not self.last_result:
            messagebox.showinfo("Export", "No results yet.")
            return
        p = filedialog.asksaveasfilename(defaultextension=".html", filetypes=[("HTML", "*.html")])
        if not p:
            return
        save_html_report(self.last_result, Path(p))
        self.log(f"Report (Summary+Chart+Trades) saved: {p}")

    def _export_summary(self):
        if not self.last_result:
            return
        p = filedialog.asksaveasfilename(defaultextension=".xml", filetypes=[("XML", "*.xml")])
        if p:
            from ticklab_algo.report import save_summary_xml
            save_summary_xml(self.last_result, Path(p))
            self.log(f"Summary exported: {p}")

    def _export_trades(self):
        if not self.last_result:
            return
        p = filedialog.asksaveasfilename(defaultextension=".xml", filetypes=[("XML", "*.xml")])
        if p:
            from ticklab_algo.report import save_trades_xml
            save_trades_xml(self.last_result, Path(p))
            self.log(f"Trades exported: {p}")

    def _export_opt(self):
        if not self.last_opt or not self.last_opt.rows:
            return
        p = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV", "*.csv")])
        if p:
            save_optimizer_csv(sorted(self.last_opt.rows, key=lambda r: r.score, reverse=True), Path(p))
            self.log(f"Optimization exported: {p}")

    def _save_log(self):
        text = self.log_text.get("1.0", "end").strip()
        if not text:
            return
        p = filedialog.asksaveasfilename(defaultextension=".txt", filetypes=[("Text", "*.txt")])
        if p:
            Path(p).write_text(text, encoding="utf-8")
            self.log(f"Log saved: {p}")

    def _save_chart(self):
        p = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG", "*.png"), ("JPEG", "*.jpg"), ("HTML", "*.html")])
        if not p:
            return
        try:
            from PIL import ImageGrab
            x = self.chart.winfo_rootx()
            y = self.chart.winfo_rooty()
            w = self.chart.winfo_width()
            h = self.chart.winfo_height()
            img = ImageGrab.grab(bbox=(x, y, x + w, y + h))
            img.save(p)
            self.log(f"Chart saved: {p}")
        except Exception as exc:
            # fallback: postscript
            try:
                self.chart.postscript(file=p, colormode="color")
                self.log(f"Chart saved (PostScript): {p}")
            except Exception as e:
                messagebox.showerror("Error", str(e))

    def _poll_queue(self):
        try:
            while True:
                ev = self.q.get_nowait()
                self._handle_event(ev)
        except queue.Empty:
            pass
        if self._backtest_running and self._trade_progress is not None:
            n = int(self._trade_progress[0])
            if n != self._last_trade_count:
                self._append_live_trades(n)
                self._last_trade_count = n
        self.after(80, self._poll_queue)

    def _append_live_trades(self, count):
        """Stream new closed trades from the shared live buffer into the table."""
        ctx = getattr(self, "_trade_live_ctx", None)
        if not ctx:
            self.trade_count_var.set(f"{count} trade(s)")
            return
        live = ctx.get("live")
        if live is None:
            self.trade_count_var.set(f"{count} trade(s)")
            return
        cur = len(self._trade_data)
        cap = live.shape[0]
        hi = min(count, cap)
        if cur < hi:
            for i in range(cur, hi):
                row = live[i]
                self._trade_data.append((
                    i + 1,                     # No.
                    int(row[0]),               # entry time
                    int(row[1]),               # exit time
                    1 if row[2] > 0 else -1,   # dir
                    float(row[3]),             # entry
                    float(row[4]),             # exit
                    float(row[5]),             # sl_init
                    float(row[6]),             # tp
                    float(row[7]),             # lot
                    float(row[8]),             # pnl
                    int(row[9]),               # reason
                    float(row[10]),            # score
                ))
            self._sort_trade_data()
            self._render_trades()
        self.trade_count_var.set(f"{count} trade(s)")

    def _read_settings(self) -> Settings:
        data = {}
        try:
            balance = float(str(self.balance_var.get()).strip().replace(",", "."))
            data["start_balance"] = balance if balance > 0 else 100.0
        except ValueError:
            data["start_balance"] = 100.0

        _prefix_defaults = (("point", 0.00001), ("pip", 0.0001), ("pt_scale", 1.0),
                            ("contract", 100000.0), ("commission_per_lot", 0.0),
                            ("slippage_points", 0.0))
        for name, default in _prefix_defaults:
            var = getattr(self, "prefix_vars", {}).get(name)
            try:
                v = float(str(var.get()).strip().replace(",", ".")) if var is not None else default
                if name in ("point", "pip", "contract") and v <= 0:
                    v = default
                data[name] = v
            except ValueError:
                data[name] = default

        values = self._ea_settings_values()
        spec = self.ea_params.spec if self.ea_params is not None else ()
        ea_values = [values.get(s.name, s.default) for s in spec]
        data["params"] = EAParams(tuple(spec), ea_values)
        return Settings(**data)


def _fmt_dt(ms):
    if not ms:
        return "-"
    return datetime.fromtimestamp(ms / 1000).strftime("%Y-%m-%d %H:%M")

def _fmt_num(v, d=2):
    return f"{float(v):,.{d}f}".replace(",", "X").replace(".", ",").replace("X", ".")


def main():
    app = AlgoBackTuneApp()
    app.mainloop()

if __name__ == "__main__":
    main()
