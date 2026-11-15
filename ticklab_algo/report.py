"""
Export backtest / optimization results to CSV and HTML.

The HTML report is self-contained (no external files): metrics, parameters,
an SVG equity curve, and a trades table. It can be opened in any browser or
printed to PDF.
"""

from __future__ import annotations

import csv
import html
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from .config import Settings
from .engine import BacktestResult, Metrics, TradeTable

# Exit-reason label defaults. The engine now requires a strategy module
# (whose EXIT_LABEL is stored in the result meta); this local constant is
# only a rendering fallback when no labels are present.
EXIT_LABEL: dict = {1: "Target Profit", 2: "Max Drawdown"}

TRADE_HEADERS = (
    "No", "Entry Time", "Exit Time", "Dir", "Entry", "Exit", "Init SL",
    "TP", "Lot", "PnL ($)", "Reason", "Score", "Strength", "Risk", "MAE", "MFE",
)


def _exit_labels(result: BacktestResult) -> dict:
    labels = (result.meta or {}).get("exit_labels")
    if isinstance(labels, dict) and labels:
        return labels
    return EXIT_LABEL


def _fmt_dt(ms: int) -> str:
    if not ms:
        return "-"
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S")


def _fmt_num(value: float, digits: int = 2) -> str:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not np.isfinite(v):
        return "-"
    return f"{v:,.{digits}f}".replace(",", "X").replace(".", ",").replace("X", ".")


def metrics_pairs(metrics: Metrics) -> list[tuple[str, str]]:
    """Return a list of (label, value) pairs ready to display."""
    pnl_sign = "+" if metrics.net_profit >= 0 else ""
    return [
        ("Initial balance", f"{_fmt_num(metrics.start_balance)} $"),
        ("Net profit", f"{pnl_sign}{_fmt_num(metrics.net_profit)} $"),
        ("Return", f"{_fmt_num(metrics.return_pct)} %"),
        ("End balance", f"{_fmt_num(metrics.end_balance)} $"),
        ("Gross profit", f"+{_fmt_num(metrics.gross_profit)} $"),
        ("Gross loss", f"{_fmt_num(metrics.gross_loss)} $"),
        ("Profit factor", _fmt_num(metrics.profit_factor)),
        ("Win rate", f"{_fmt_num(metrics.win_rate)} %"),
        ("Trades", f"{metrics.trades}"),
        ("Wins / losses", f"{metrics.wins} / {metrics.losses}"),
        ("Expectancy", f"{_fmt_num(metrics.expectancy)} $"),
        ("Max drawdown", f"{_fmt_num(metrics.max_dd)} $ ({_fmt_num(metrics.max_dd_pct)} %)"),
        ("Recovery factor", _fmt_num(metrics.recovery)),
        ("Sharpe / trade", _fmt_num(metrics.sharpe)),
        ("Max consec. losses", f"{metrics.max_consec_loss}"),
        ("Avg duration", f"{_fmt_num(metrics.avg_duration_min, 1)} min"),
        ("Trades / day", _fmt_num(metrics.trades_per_day, 1)),
    ]


def _tf_label(period_ms: int) -> str:
    minutes = int(period_ms // 60_000)
    if minutes < 60:
        return f"Bar M{minutes}"
    if minutes < 1440:
        return f"Bar H{minutes // 60}"
    return f"Bar D{minutes // 1440}"


_WEEKDAYS = ("Monday", "Tuesday", "Wednesday", "Thursday", "Friday",
             "Saturday", "Sunday")
_MONTHS = ("January", "February", "March", "April", "May", "June", "July",
           "August", "September", "October", "November", "December")


def _breakdown_tables(result: BacktestResult) -> list[str]:
    """Trade counts and P/L grouped by entry hour / day / month."""
    t = result.trades
    if len(t) == 0:
        return []
    hour_n = np.zeros(24, np.int64)
    hour_pnl = np.zeros(24, np.float64)
    day_n = np.zeros(7, np.int64)
    day_pnl = np.zeros(7, np.float64)
    month_n = np.zeros(12, np.int64)
    month_pnl = np.zeros(12, np.float64)
    times = np.asarray(t.entry_time, dtype=np.int64)
    pnl = np.asarray(t.pnl, dtype=np.float64)
    for ms, p in zip(times, pnl):
        dt = datetime.fromtimestamp(int(ms) / 1000, tz=timezone.utc)
        hour_n[dt.hour] += 1
        hour_pnl[dt.hour] += p
        day_n[dt.weekday()] += 1
        day_pnl[dt.weekday()] += p
        month_n[dt.month - 1] += 1
        month_pnl[dt.month - 1] += p

    def table(title: str, names, counts, pnls) -> list[str]:
        width = max(len(n) for n in names)
        lines = ["", title]
        lines.append(f"{'Name':<{width}}  {'Trades':>6}  {'P/L':>12}")
        for name, cnt, p in zip(names, counts, pnls):
            lines.append(f"{name:<{width}}  {int(cnt):>6}  {p:>+12.2f}")
        return lines

    out: list[str] = []
    out += table("Trades by Hours (entry time)", [f"{h:02d}" for h in range(24)],
                 hour_n, hour_pnl)
    out += table("Trades by Days (entry time)", _WEEKDAYS,
                 day_n, day_pnl)
    out += table("Trades by Months (entry time)", _MONTHS,
                 month_n, month_pnl)
    return out


def summary_text(result: BacktestResult, leverage: str | None = None) -> str:
    m = result.metrics
    lv = (leverage or "500").strip().upper()
    if lv and not lv.startswith("1:"):
        lv = f"1:{lv}"
    lines = [
        f"Data source : {result.source_label or '-'}",
        f"Period     : {_fmt_dt(int(result.bars.time_ms[0])) if len(result.bars) else '-'}"
        f"  ..  {_fmt_dt(int(result.bars.time_ms[-1])) if len(result.bars) else '-'}",
        f"{_tf_label(result.bars.period_ms):<12}: {len(result.bars):,}".replace(",", "."),
        f"Active ticks : {int(result.meta.get('active_ticks', len(result.ticks))):,}".replace(",", "."),
        (f"Warm-up     : {int(result.meta.get('warmup_ticks', 0)):,} indicator ticks".replace(",", ".")
         if result.meta.get("warmup_ticks") else "Warm-up     : -"),
        f"Open positions at end of data : {int(result.meta.get('open_positions_at_end', 0))} (excluded)",
        "",
        f"Initial balance : {m.start_balance:.2f} $",
        f"Leverage        : {lv or '-'}",
        f"Gross profit    : +{m.gross_profit:.2f} $",
        f"Gross loss      : {m.gross_loss:.2f} $",
        f"Net profit  : {m.net_profit:+.2f} $  ({m.return_pct:+.2f} %)",
        f"End balance : {m.end_balance:.2f} $",
        f"Profit factor: {m.profit_factor:.2f}",
        f"Win rate    : {m.win_rate:.1f} %  ({m.wins} wins / {m.losses} losses)",
        f"Max DD      : {m.max_dd:.2f} $  ({m.max_dd_pct:.2f} %)",
        f"Trades      : {m.trades}  ({m.trades_per_day:.1f} per day)",
        f"Expectancy  : {m.expectancy:+.2f} $ per trade",
    ]
    if len(result.trades):
        labels = _exit_labels(result)
        reasons: dict[str, int] = {}
        for r in result.trades.reason:
            label = labels.get(int(r), str(int(r)))
            reasons[label] = reasons.get(label, 0) + 1
        lines.append("Exit reasons : " + ", ".join(f"{k} ({v})" for k, v in reasons.items()))
    lines += _breakdown_tables(result)
    return "\n".join(lines)


def _trade_row_values(t: TradeTable, i: int, human: bool = True,
                      labels: dict | None = None) -> list[str]:
    exit_labels = labels or EXIT_LABEL

    def num(value: float, digits: int) -> str:
        return _fmt_num(value, digits) if human else f"{float(value):.{digits}f}"

    return [
        str(i + 1),
        _fmt_dt(int(t.entry_time[i])),
        _fmt_dt(int(t.exit_time[i])),
        "BUY" if t.dir[i] > 0 else "SELL",
        num(t.entry[i], 2),
        num(t.exit[i], 2),
        num(t.sl_init[i], 2),
        num(t.tp[i], 2),
        num(t.lot[i], 2),
        num(t.pnl[i], 2),
        exit_labels.get(int(t.reason[i]), str(int(t.reason[i]))),
        num(t.score[i], 1),
        num(t.strength[i], 1),
        num(t.risk[i], 2),
        num(t.mae[i], 2),
        num(t.mfe[i], 2),
    ]


def save_trades_csv(result: BacktestResult, path: str | Path) -> Path:
    path = Path(path)
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f, delimiter=",", quotechar='"', quoting=csv.QUOTE_MINIMAL)
        w.writerow(TRADE_HEADERS)
        t = result.trades
        labels = _exit_labels(result)
        for i in range(len(t)):
            w.writerow(_trade_row_values(t, i, human=False, labels=labels))
    return path


def save_summary_csv(result: BacktestResult, path: str | Path) -> Path:
    path = Path(path)
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f, delimiter=",", quoting=csv.QUOTE_MINIMAL)
        w.writerow(["Metric", "Value"])
        for label, value in metrics_pairs(result.metrics):
            w.writerow([label, value])
        w.writerow([])
        w.writerow(["Parameter", "Value"])
        for name, value in _settings_pairs(result.settings):
            w.writerow([name, value])
    return path


def _settings_pairs(settings: Settings) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    payload = settings.to_payload()
    for name, value in payload.items():
        if name == "params":
            continue
        if isinstance(value, float):
            out.append((name, f"{value:g}"))
        else:
            out.append((name, str(value)))
    for name, value in payload.get("params", {}).items():
        if isinstance(value, float):
            out.append((name, f"{value:g}"))
        else:
            out.append((name, str(value)))
    return out


def _equity_svg(result: BacktestResult, width: int = 960, height: int = 300) -> str:
    times = np.asarray(result.equity_time, dtype=np.float64)
    eq = np.asarray(result.equity, dtype=np.float64)
    if eq.size < 2:
        return ""
    n = eq.size
    start_balance = float(result.settings.start_balance)
    lo = float(min(eq.min(), start_balance))
    hi = float(max(eq.max(), start_balance))
    pad = (hi - lo) * 0.08 or max(hi * 0.001, 1.0)
    lo -= pad
    hi += pad

    left, right, top, bottom = 68, 18, 16, 36
    plot_w = width - left - right
    plot_h = height - top - bottom

    def x_of(i: float) -> float:
        return left + plot_w * (i / max(1.0, n - 1))

    def y_of(v: float) -> float:
        return top + plot_h * (1.0 - (v - lo) / max(1e-9, hi - lo))

    step = max(1, n // 1200)
    pts = " ".join(f"{x_of(i):.1f},{y_of(eq[i]):.1f}" for i in range(0, n, step))
    if (n - 1) % step:
        pts += f" {x_of(n - 1):.1f},{y_of(eq[-1]):.1f}"

    up = eq[-1] >= start_balance
    color = "#22c55e" if up else "#ef4444"
    base_y = y_of(start_balance)
    grid_lines = []
    for k in range(5):
        v = lo + (hi - lo) * k / 4.0
        y = y_of(v)
        grid_lines.append(
            f'<line x1="{left}" y1="{y:.1f}" x2="{width - right}" y2="{y:.1f}" '
            f'stroke="#233046" stroke-width="1"/>'
            f'<text x="{left - 8}" y="{y + 4:.1f}" fill="#8b9bb4" font-size="11" '
            f'text-anchor="end">{_fmt_num(v, 0)}</text>'
        )

    area_pts = f"{left},{top + plot_h} {pts} {x_of(n - 1):.1f},{top + plot_h}"
    return f"""<svg viewBox="0 0 {width} {height}" width="100%" preserveAspectRatio="none" role="img">
  <rect x="0" y="0" width="{width}" height="{height}" fill="#0f1724"/>
  {''.join(grid_lines)}
  <line x1="{left}" y1="{base_y:.1f}" x2="{width - right}" y2="{base_y:.1f}"
        stroke="#64748b" stroke-width="1" stroke-dasharray="5 4"/>
  <polygon points="{area_pts}" fill="{color}" opacity="0.10"/>
  <polyline points="{pts}" fill="none" stroke="{color}" stroke-width="2" stroke-linejoin="round"/>
</svg>"""


def save_html_report(
    result: BacktestResult,
    path: str | Path,
    title: str = "TickLab-AlgoBackTune Backtest Report",
    optimizer_rows: list | None = None,
) -> Path:
    path = Path(path)
    m = result.metrics
    metric_cards = "".join(
        f'<div class="card"><div class="k">{html.escape(k)}</div>'
        f'<div class="v">{html.escape(v)}</div></div>'
        for k, v in metrics_pairs(m)
    )

    params_rows = "".join(
        f"<tr><td>{html.escape(str(k))}</td><td>{html.escape(str(v))}</td></tr>"
        for k, v in _settings_pairs(result.settings)
    )

    t = result.trades
    labels = _exit_labels(result)
    trade_rows_html = "".join(
        "<tr>" + "".join(f"<td>{html.escape(c)}</td>" for c in _trade_row_values(t, i, labels=labels)) + "</tr>"
        for i in range(len(t))
    )
    if not trade_rows_html:
        trade_rows_html = f'<tr><td colspan="{len(TRADE_HEADERS)}">No trades.</td></tr>'

    trade_head = "".join(f"<th>{html.escape(h)}</th>" for h in TRADE_HEADERS)

    opt_html = ""
    if optimizer_rows:
        head = ("No", "Parameter", "Net profit", "PF", "Max DD", "Trades", "Win %", "Score")
        body = []
        for i, row in enumerate(optimizer_rows, 1):
            params = ", ".join(f"{k}={v:g}" if isinstance(v, (int, float)) else f"{k}={v}"
                               for k, v in row.params.items())
            mm = row.metrics
            body.append(
                "<tr>"
                f"<td>{i}</td><td class='params'>{html.escape(params)}</td>"
                f"<td>{_fmt_num(mm.net_profit)}</td>"
                f"<td>{_fmt_num(mm.profit_factor)}</td>"
                f"<td>{_fmt_num(mm.max_dd)}</td>"
                f"<td>{mm.trades}</td><td>{_fmt_num(mm.win_rate, 1)}</td>"
                f"<td>{_fmt_num(row.score, 2)}</td>"
                "</tr>"
            )
        opt_html = (
            "<h2>Optimization Results</h2><table><thead><tr>"
            + "".join(f"<th>{html.escape(h)}</th>" for h in head)
            + "</tr></thead><tbody>" + "".join(body) + "</tbody></table>"
        )

    generated = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    svg = _equity_svg(result)
    source = html.escape(result.source_label or "-")
    period = "..".join(
        _fmt_dt(int(x)) for x in (result.bars.time_ms[0], result.bars.time_ms[-1])
    ) if len(result.bars) else "-"

    doc = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html.escape(title)}</title>
<style>
  :root {{ color-scheme: dark; }}
  * {{ box-sizing: border-box; }}
  body {{ margin: 0; padding: 28px; background: #0b1220; color: #dfe7f2;
         font: 14px/1.5 "Segoe UI", system-ui, sans-serif; }}
  h1 {{ margin: 0 0 4px; font-size: 26px; }}
  h2 {{ margin: 32px 0 12px; font-size: 18px; color: #9ec5ff; }}
  .sub {{ color: #8b9bb4; margin-bottom: 22px; }}
  .cards {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(190px, 1fr));
            gap: 10px; }}
  .card {{ background: #121b2c; border: 1px solid #1e2b42; border-radius: 10px;
           padding: 12px 14px; }}
  .card .k {{ color: #8b9bb4; font-size: 12px; text-transform: uppercase;
              letter-spacing: .04em; }}
  .card .v {{ font-size: 17px; font-weight: 600; margin-top: 3px; }}
  table {{ width: 100%; border-collapse: collapse; background: #121b2c;
           border: 1px solid #1e2b42; border-radius: 10px; overflow: hidden; }}
  th, td {{ padding: 7px 10px; border-bottom: 1px solid #1a263c; text-align: right;
            white-space: nowrap; }}
  th {{ background: #16213a; color: #9ec5ff; font-weight: 600; position: sticky; top: 0; }}
  td.params {{ text-align: left; white-space: normal; font-size: 12px; }}
  tr:last-child td {{ border-bottom: none; }}
  tbody tr:hover {{ background: #16213a; }}
  .scroll {{ max-height: 560px; overflow: auto; border-radius: 10px; }}
  .chart {{ background: #0f1724; border: 1px solid #1e2b42; border-radius: 10px;
            padding: 8px; }}
  details {{ background: #121b2c; border: 1px solid #1e2b42; border-radius: 10px;
             padding: 12px 16px; }}
  summary {{ cursor: pointer; color: #9ec5ff; font-weight: 600; }}
  .note {{ color: #8b9bb4; font-size: 12px; margin-top: 18px; }}
</style>
</head>
<body>
<h1>{html.escape(title)}</h1>
<div class="sub">{source} &middot; {period}</div>
<div class="cards">{metric_cards}</div>

<h2>Equity Curve</h2>
<div class="chart">{svg}</div>

<h2>Trades ({len(result.trades)})</h2>
<div class="scroll"><table>
<thead><tr>{trade_head}</tr></thead>
<tbody>{trade_rows_html}</tbody>
</table></div>

{opt_html}

<h2>Parameters</h2>
<details><summary>Show all parameters</summary>
<table><thead><tr><th style="text-align:left">Parameter</th>
<th style="text-align:left">Value</th></tr></thead>
<tbody>{params_rows}</tbody></table>
</details>

<div class="note">
  Generated by TickLab-AlgoBackTune on {generated}. Simulated on real ticks
  from the MetaTrader 5 cache with the Python strategy currently loaded in the
  app (not an .ex5 execution).
</div>
</body>
</html>
"""
    path.write_text(doc, encoding="utf-8")
    return path


def save_optimizer_csv(rows: list, path: str | Path) -> Path:
    """Save optimization results (OptimizeRow list) to CSV."""
    path = Path(path)
    if not rows:
        path.write_text("", encoding="utf-8-sig")
        return path
    param_keys = sorted({k for row in rows for k in row.params})
    headers = ["No"] + param_keys + ["Net profit", "PF", "Max DD", "Trades",
                                    "Win %", "Return %", "Score"]
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(headers)
        for i, row in enumerate(rows, 1):
            values: list[object] = [i]
            values += [row.params.get(k, "") for k in param_keys]
            mm = row.metrics
            values += [round(mm.net_profit, 4), round(mm.profit_factor, 4),
                       round(mm.max_dd, 4), mm.trades, round(mm.win_rate, 2),
                       round(mm.return_pct, 4), round(row.score, 4)]
            w.writerow(values)
    return path


__all__ = [
    "metrics_pairs", "summary_text", "save_trades_csv", "save_summary_csv",
    "save_html_report", "save_optimizer_csv", "save_trades_xml", "save_summary_xml",
]


def save_trades_xml(result, path):
    from pathlib import Path
    import xml.etree.ElementTree as ET

    path = Path(path)
    root = ET.Element("trades")
    t = result.trades
    for i in range(len(t)):
        tr = ET.SubElement(root, "trade")
        tr.set("no", str(i + 1))
        tr.set("entry_time", _fmt_dt(int(t.entry_time[i])))
        tr.set("exit_time", _fmt_dt(int(t.exit_time[i])))
        tr.set("dir", "BUY" if t.dir[i] > 0 else "SELL")
        tr.set("entry", "%.2f" % t.entry[i])
        tr.set("exit", "%.2f" % t.exit[i])
        tr.set("sl", "%.2f" % t.sl_init[i])
        tr.set("tp", "%.2f" % t.tp[i])
        tr.set("lot", "%.2f" % t.lot[i])
        tr.set("pnl", "%.2f" % t.pnl[i])
        tr.set("reason", _exit_labels(result).get(int(t.reason[i]), str(int(t.reason[i]))))
        tr.set("score", "%.1f" % t.score[i])

    tree = ET.ElementTree(root)
    ET.indent(tree, space="  ")
    tree.write(path, encoding="utf-8", xml_declaration=True)
    return path


def save_summary_xml(result, path):
    from pathlib import Path
    import xml.etree.ElementTree as ET

    path = Path(path)
    root = ET.Element("summary")
    m = result.metrics
    for label, value in metrics_pairs(m):
        el = ET.SubElement(root, "metric")
        el.set("name", label)
        el.set("value", value)

    params = ET.SubElement(root, "parameters")
    for name, value in _settings_pairs(result.settings):
        el = ET.SubElement(params, "param")
        el.set("name", name)
        el.set("value", str(value))

    tree = ET.ElementTree(root)
    ET.indent(tree, space="  ")
    tree.write(path, encoding="utf-8", xml_declaration=True)
    return path
