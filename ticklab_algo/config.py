"""
Parameter configuration for AlgoBackTune.

The param vector is:
    [prefix (7)]  point, pip, pt_scale, contract, start_balance, commission_per_lot, slippage_points
    [EA params]   in .mq5 declaration order

Prefix indices are fixed constants (0..6). EA parameter indices start at
PREFIX_LEN == 7. `StrategySpec` carries the EA params with their declared
names, types and defaults (parsed from the .mq5 `input ...` lines).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
import numpy as np

# Prefix indices (fixed).
P_POINT = 0
P_PIP = 1
P_PT_SCALE = 2
P_CONTRACT = 3
P_START_BALANCE = 4
P_COMMISSION_PER_LOT = 5
P_SLIPPAGE_POINTS = 6
PREFIX_LEN = 7
PREFIX_NAMES = ("point", "pip", "pt_scale", "contract", "start_balance",
                "commission_per_lot", "slippage_points")


@dataclass(frozen=True)
class ParamSpec:
    name: str                  # raw name from the .mq5 (e.g. InpLotSize)
    kind: str                  # "int" | "float" | "bool" | "str"
    default: Any               # parsed default (float/int/bool/str)
    comment: str = ""


@dataclass
class EAParams:
    """Ordered EA parameter bag built from a .mq5 spec (or defaults)."""
    spec: tuple[ParamSpec, ...]
    values: list[Any] = field(default_factory=list)

    def __post_init__(self):
        if self.values:
            if len(self.values) != len(self.spec):
                raise ValueError("values length does not match spec")
        else:
            self.values = [s.default for s in self.spec]

    def __len__(self) -> int:
        return len(self.spec)

    def names(self) -> tuple[str, ...]:
        return tuple(s.name for s in self.spec)

    def __getitem__(self, key):
        for i, s in enumerate(self.spec):
            if s.name == key:
                return self.values[i]
        raise KeyError(key)

    def __setitem__(self, key, value):
        for i, s in enumerate(self.spec):
            if s.name == key:
                self.values[i] = value
                return
        raise KeyError(key)

    def as_dict(self) -> dict[str, Any]:
        return {s.name: v for s, v in zip(self.spec, self.values)}

    def to_array(self) -> np.ndarray:
        arr = []
        for v in self.values:
            if isinstance(v, bool):
                arr.append(1.0 if v else 0.0)
            elif isinstance(v, (int, float)):
                arr.append(float(v))
            else:
                arr.append(0.0)
        return np.array(arr, dtype=np.float64)

    def replace(self, **kwargs) -> "EAParams":
        data = {s.name: v for s, v in zip(self.spec, self.values)}
        data.update(kwargs)
        return EAParams(self.spec, [data[s.name] for s in self.spec])

    def int_indices(self) -> set[int]:
        return {i for i, s in enumerate(self.spec) if s.kind == "int"}


@dataclass
class Settings:
    point: float = 0.00001
    pip: float = 0.0001
    pt_scale: float = 1.0
    contract: float = 100000.0
    start_balance: float = 100.0
    commission_per_lot: float = 0.0
    slippage_points: float = 0.0
    params: EAParams = field(default_factory=lambda: EAParams(spec=()))

    def prefix_array(self) -> np.ndarray:
        return np.array([self.point, self.pip, self.pt_scale,
                         self.contract, self.start_balance,
                         self.commission_per_lot, self.slippage_points],
                        dtype=np.float64)

    def to_array(self) -> np.ndarray:
        return np.concatenate([self.prefix_array(), self.params.to_array()])

    @property
    def max_positions(self) -> int:
        """Upper bound on open positions the strategy may hold.

        Used to size internal arrays. Falls back to a safe cap when the
        loaded EA declares no max-positions-like parameter.
        """
        for s in self.params.spec:
            if "maxposition" in s.name.lower():
                try:
                    return int(float(self.params[s.name]))
                except (TypeError, ValueError):
                    pass
        return 256

    def to_payload(self) -> dict[str, Any]:
        return {
            "point": self.point, "pip": self.pip, "pt_scale": self.pt_scale,
            "contract": self.contract, "start_balance": self.start_balance,
            "commission_per_lot": self.commission_per_lot,
            "slippage_points": self.slippage_points,
            "params": self.params.as_dict(),
        }

    @classmethod
    def from_payload(cls, data: dict[str, Any]) -> "Settings":
        ea = data.get("params", {})
        spec: list[ParamSpec] = [
            ParamSpec(name=name, kind="float", default=float(val))
            for name, val in ea.items()
        ]
        return cls(
            point=float(data.get("point", 0.00001)),
            pip=float(data.get("pip", 0.0001)),
            pt_scale=float(data.get("pt_scale", 1.0)),
            contract=float(data.get("contract", 100000.0)),
            start_balance=float(data.get("start_balance", 100.0)),
            commission_per_lot=float(data.get("commission_per_lot", 0.0)),
            slippage_points=float(data.get("slippage_points", 0.0)),
            params=EAParams(tuple(spec), [float(ea[n]) for n in ea]),
        )


def parse_mq5_inputs(text: str) -> list[ParamSpec]:
    """Extract ordered `input ... Name = Default; // Comment` lines from .mq5 code."""
    import re
    specs: list[ParamSpec] = []
    # enum types treated as int, input string preserved as str.
    # MQL5 puts the optional display comment after the ';' on the same line:
    #     input double InpX = 1.0;      // Comment text
    pattern = re.compile(
        r'^\s*input\s+(?P<type>[A-Za-z_][A-Za-z_0-9]*(?:\s*\*)?)\s+'
        r'(?P<name>[A-Za-z_][A-Za-z_0-9]*)\s*=\s*'
        r'(?P<default>[^;]+);'
        r'(?P<tail>[^\n]*)',
        re.MULTILINE)
    for m in pattern.finditer(text):
        raw_type = m.group('type').strip()
        name = m.group('name').strip()
        default = m.group('default').strip()
        tail = m.group('tail').strip()
        comment = ""
        if '//' in default:
            default_comment = default.split('//', 1)
            default = default_comment[0].strip()
            comment = default_comment[1].strip()
        elif tail.startswith('//'):
            comment = tail[2:].strip()
        # strip trailing commas / whitespace
        default = default.rstrip(',').strip()
        kind, value = _classify(raw_type, default)
        specs.append(ParamSpec(name=name, kind=kind, default=value, comment=comment))
    return specs


def _classify(raw_type: str, default: str) -> tuple[str, Any]:
    t = raw_type.lower()
    low = default.lower()
    if low in ("true", "false"):
        return "bool", low == "true"
    if t == "string":
        return "str", default.strip().strip('"')
    if t in ("int", "long", "enum", "datetime", "color") or t.endswith("_int") \
            or t.startswith("input_int"):
        # numeric int
        try:
            return "int", int(float(default.replace(".", ".")))
        except ValueError:
            return "int", 0
    # enum & everything else numeric double
    if t.startswith("double") or t.startswith("float"):
        try:
            return "float", float(default)
        except ValueError:
            return "float", 0.0
    try:
        return "float", float(default)
    except ValueError:
        try:
            return "int", int(default)
        except ValueError:
            return "str", default.strip('"')


def strategy_spec_from_mq5(text: str) -> EAParams:
    specs = parse_mq5_inputs(text)
    return EAParams(tuple(specs))
