"""
Symbol specifications for the ACCOUNT toolbar panel.

Values follow standard MetaTrader 5 conventions per symbol class:

    point      = minimum price increment (1 / 10**digits)
    pip        = point x (10 if the symbol uses fractional pips, else 1)
    pt_scale   = pip / point ratio (kept for the ACCOUNT panel display)
    contract   = contract size for one lot

Used by the GUI to auto-fill the ACCOUNT fields when the user picks a
symbol in the toolbar. The values are plain defaults - every field can be
edited by the user afterwards.
"""

from __future__ import annotations

# Standard contract sizes per MT5 class.
_FOREX = {"point": 0.00001, "pip": 0.0001, "pt_scale": 10.0, "contract": 100000.0}
_FOREX_JPY = {"point": 0.001, "pip": 0.01, "pt_scale": 10.0, "contract": 100000.0}
_INDEX = {"point": 0.1, "pip": 1.0, "pt_scale": 10.0, "contract": 1.0}
_INDEX_2DP = {"point": 0.01, "pip": 0.1, "pt_scale": 10.0, "contract": 1.0}
_METALS = {
    "XAUUSD": {"point": 0.01, "pip": 0.1, "pt_scale": 10.0, "contract": 100.0},
    "XAGUSD": {"point": 0.001, "pip": 0.01, "pt_scale": 10.0, "contract": 5000.0},
    "XPTUSD": {"point": 0.001, "pip": 0.01, "pt_scale": 10.0, "contract": 100.0},
    "XPDUSD": {"point": 0.001, "pip": 0.01, "pt_scale": 10.0, "contract": 100.0},
}

_INDEX_SYMBOLS = {
    "US30", "US500", "NAS100", "US2000", "UK100", "GER40", "FRA40", "EU50",
    "ESP35", "ITA40", "SWI20", "JP225", "HK50", "CHINA50", "AUS200",
}

_INDEX_2DP_SYMBOLS = {"US500"}


def symbol_specs(symbol: str) -> dict:
    """Return {point, pip, pt_scale, contract} for a symbol name.

    Unknown symbols fall back to 5-digit FX defaults. Matching is
    case-insensitive.
    """
    s = (symbol or "").strip().upper()
    if not s:
        return dict(_FOREX)
    if s in _METALS:
        return dict(_METALS[s])
    if s in _INDEX_2DP_SYMBOLS:
        return dict(_INDEX_2DP)
    if s in _INDEX_SYMBOLS:
        return dict(_INDEX)
    if "JPY" in s:
        return dict(_FOREX_JPY)
    return dict(_FOREX)
