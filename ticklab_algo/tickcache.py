"""
Reader for MetaTrader 5 tick cache (.tkc) files without running the MT5 terminal.

The supported format is the MT5 v509 tick cache (commonly used by recent
builds). Each file contains a per-day block index plus several zlib streams:
  0 : timestamp (uint64, delta ms; the first element is the absolute time)
  1 : last price (used by MT5, usually zero for forex/gold)
  2 : volume / internal field
  3 : bid (float64)
  4 : ask (float64)
  5 : internal field
  6 : flags (1 byte per tick)
  7 : real volume

This module only reads data already present in the MT5 data folders, so the
terminal does not need to be opened.
"""

from __future__ import annotations

import mmap
import os
import struct
import zlib
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Sequence

import numpy as np

TKC_INDEX_OFFSET = 0xE4
TKC_INDEX_SIZE = 48
TKC_DATA_OFFSET = 419
TKC_SYMBOL_OFFSET = 0xA4
TKC_SYMBOL_SIZE = 64
TKC_MAX_INDEX = 128


class TickCacheError(RuntimeError):
    pass


@dataclass(frozen=True)
class TickBlock:
    offset: int
    size: int
    count: int
    last_time_ms: int


@dataclass
class TickFileInfo:
    path: Path
    symbol: str
    blocks: tuple[TickBlock, ...]
    total_ticks: int
    first_time_ms: int
    last_time_ms: int


@dataclass
class TickSource:
    key: str
    origin: str  # "Tester" / "Terminal" / "Custom"
    server: str
    symbol: str
    files: list[TickFileInfo] = field(default_factory=list)

    @property
    def total_ticks(self) -> int:
        return sum(f.total_ticks for f in self.files)

    @property
    def first_time_ms(self) -> int:
        return min((f.first_time_ms for f in self.files if f.first_time_ms), default=0)

    @property
    def last_time_ms(self) -> int:
        return max((f.last_time_ms for f in self.files if f.last_time_ms), default=0)

    def label(self) -> str:
        first = fmt_ms(self.first_time_ms)
        last = fmt_ms(self.last_time_ms)
        ticks = human_int(self.total_ticks)
        return f"{self.origin} | {self.server} | {self.symbol} | {first} .. {last} | {ticks} tick"


@dataclass
class Ticks:
    time_ms: np.ndarray
    bid: np.ndarray
    ask: np.ndarray
    source_label: str = ""

    def __len__(self) -> int:
        return int(self.time_ms.size)


def human_int(n: int) -> str:
    if n >= 1_000_000_000:
        return f"{n / 1_000_000_000:.2f} M"
    if n >= 1_000_000:
        return f"{n / 1_000_000:.2f} M"
    if n >= 1_000:
        return f"{n / 1_000:.1f} k"
    return str(n)


def fmt_ms(ms: int, with_time: bool = True) -> str:
    if not ms:
        return "-"
    dt = datetime.fromtimestamp(ms / 1000, tz=timezone.utc)
    return dt.strftime("%Y-%m-%d %H:%M") if with_time else dt.strftime("%Y-%m-%d")


def parse_datetime_ms(text: str) -> int:
    """Parse 'YYYY-MM-DD[ HH:MM[:SS]]' as server time (naive, UTC-like)."""
    text = text.strip()
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M", "%Y-%m-%d"):
        try:
            dt = datetime.strptime(text, fmt)
            return int(dt.replace(tzinfo=timezone.utc).timestamp() * 1000)
        except ValueError:
            pass
    raise ValueError(f"Invalid time format: {text!r}. Use YYYY-MM-DD HH:MM")


def _decode_utf16(buf: bytes) -> str:
    return buf.decode("utf-16le", errors="ignore").split("\x00", 1)[0].strip()


def _is_zlib_header(data: bytes | memoryview) -> bool:
    if len(data) < 2:
        return False
    cmf, flg = data[0], data[1]
    return (cmf & 0x0F) == 8 and (cmf * 256 + flg) % 31 == 0


def inspect_tkc(path: str | os.PathLike[str]) -> TickFileInfo:
    path = Path(path)
    try:
        with path.open("rb") as f:
            head = f.read(TKC_INDEX_OFFSET + TKC_INDEX_SIZE * TKC_MAX_INDEX)
            file_size = path.stat().st_size
    except OSError as exc:
        raise TickCacheError(f"Cannot read {path}: {exc}") from exc

    if len(head) < TKC_INDEX_OFFSET + TKC_INDEX_SIZE:
        raise TickCacheError(f"Tick cache file too short: {path}")

    symbol = _decode_utf16(head[TKC_SYMBOL_OFFSET : TKC_SYMBOL_OFFSET + TKC_SYMBOL_SIZE]) or path.parent.name

    blocks: list[TickBlock] = []
    for i in range(TKC_MAX_INDEX):
        base = TKC_INDEX_OFFSET + i * TKC_INDEX_SIZE
        if base + TKC_INDEX_SIZE > len(head):
            break
        vals = struct.unpack_from("<12I", head, base)
        # an empty record marks the end of the index
        if vals[0] == 0 and vals[1] == 0 and vals[2] == 0:
            break
        size = int(vals[2])
        offset = int(vals[3])
        # times in the index are stored in Unix seconds.
        last_time = int(vals[1]) * 1000
        count = int(vals[6])
        if offset <= 0 or size <= 0 or count <= 0:
            continue
        if offset + size > file_size + 1:
            continue
        blocks.append(TickBlock(offset, size, count, last_time))

    if not blocks:
        raise TickCacheError(f"No tick blocks found in {path}")

    blocks.sort(key=lambda b: b.last_time_ms)
    # First month estimate comes from the YYYYMM file name, then use the last block's timestamp.
    first = blocks[0].last_time_ms
    try:
        month = datetime.strptime(path.stem, "%Y%m").replace(tzinfo=timezone.utc)
        first = max(int(month.timestamp() * 1000), blocks[0].last_time_ms - 3 * 86400_000)
    except ValueError:
        pass

    return TickFileInfo(
        path=path,
        symbol=symbol,
        blocks=tuple(blocks),
        total_ticks=sum(b.count for b in blocks),
        first_time_ms=first,
        last_time_ms=blocks[-1].last_time_ms,
    )


def _decompress_stream(data: memoryview | bytes) -> tuple[bytes, bytes]:
    d = zlib.decompressobj()
    try:
        out = d.decompress(data)
        out += d.flush()
    except zlib.error as exc:
        raise TickCacheError(f"Failed to decompress tick stream: {exc}") from exc
    return out, d.unused_data


def _find_data_offset(mm: mmap.mmap, block: TickBlock) -> int:
    candidate = block.offset + TKC_DATA_OFFSET
    if candidate + 2 <= block.offset + block.size and _is_zlib_header(mm[candidate : candidate + 2]):
        return candidate

    # Fallback for other format versions: find the first zlib header
    # whose decompressed time stream is exactly count * 8 bytes.
    limit = min(block.offset + 4096, block.offset + block.size - 2)
    for off in range(block.offset + 64, limit):
        if not _is_zlib_header(mm[off : off + 2]):
            continue
        try:
            d = zlib.decompressobj()
            out = d.decompress(mm[off : block.offset + block.size])
            if len(out) == block.count * 8:
                return off
        except zlib.error:
            continue
    raise TickCacheError(f"Tick stream not found in block at offset {block.offset}")


def _decode_new_format_block(mm, block) -> tuple:
    """.tkc new-format block (latest MT5 build).

    Layout: a list of groups; each group contains 7 zlib streams:
      s0 time u64 (element 0 = absolute ms, the rest are ms deltas)
      s1 volume u64, s2 last/point u64, s3 bid f64, s4 ask f64, s5 u64, s6 flags u32
    The older (8-stream) format cannot be used for these blocks.
    """
    data = mm[block.offset : block.offset + block.size]
    streams: list[tuple[int, bytes]] = []
    i = 0
    n = len(data)
    while i < n - 1:
        if data[i] == 0x78 and data[i + 1] in (0x01, 0x20, 0x5E, 0x9C, 0xDA):
            try:
                d = zlib.decompressobj()
                out = d.decompress(data[i:])
                used = n - i - len(d.unused_data)
                if len(out) > 16:
                    streams.append((len(out), out))
                    i += used
                    continue
            except zlib.error:
                pass
        i += 1

    groups: list[list[tuple[int, bytes]]] = []
    i = 0
    while i + 6 < len(streams):
        same = all(streams[i + j][0] == streams[i][0] for j in range(6))
        if same and streams[i + 6][0] * 2 == streams[i][0]:
            groups.append(streams[i : i + 7])
            i += 7
        else:
            i += 1
    if not groups:
        raise TickCacheError("New-format block cannot be decoded")

    times_all: list[np.ndarray] = []
    bid_all: list[np.ndarray] = []
    ask_all: list[np.ndarray] = []
    for g in groups:
        cnt = g[0][0] // 8
        tv = np.frombuffer(g[0][1], dtype="<u8", count=cnt)
        bid = np.frombuffer(g[3][1], dtype="<f8", count=cnt)
        ask = np.frombuffer(g[4][1], dtype="<f8", count=cnt)
        ts = np.empty(cnt, dtype=np.int64)
        ts[0] = np.int64(tv[0])
        if cnt > 1:
            ts[1:] = ts[0] + np.cumsum(tv[1:].astype(np.int64))
        times_all.append(ts)
        bid_all.append(bid)
        ask_all.append(ask)

    times = np.concatenate(times_all)
    bid = np.concatenate(bid_all)
    ask = np.concatenate(ask_all)
    if times.size > 1 and np.any(times[1:] < times[:-1]):
        order = np.argsort(times, kind="mergesort")
        times, bid, ask = times[order], bid[order], ask[order]
    return times, bid, ask


def read_tkc(
    info: TickFileInfo,
    start_ms: int = 0,
    end_ms: int = 2**62,
    progress: Callable[[int, int], None] | None = None,
) -> Ticks:
    """Read a single .tkc file and return bid/ask ticks within the time range."""
    path = info.path
    out_times: list[np.ndarray] = []
    out_bid: list[np.ndarray] = []
    out_ask: list[np.ndarray] = []

    blocks = [
        b
        for b in info.blocks
        if b.last_time_ms >= start_ms - 36 * 3600_000 and b.last_time_ms <= end_ms + 36 * 3600_000
    ]
    total = len(blocks)
    try:
        with path.open("rb") as f:
            with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
                for idx, block in enumerate(blocks):
                    try:
                        data_off = _find_data_offset(mm, block)
                        stream = mm[data_off : block.offset + block.size]

                        # stream 0: time, 1-2: internal field, 3: bid, 4: ask
                        t_raw, stream = _decompress_stream(stream)
                        if len(t_raw) != block.count * 8:
                            raise TickCacheError(
                                f"Time stream length mismatch in {path.name} block {block.offset}: "
                                f"{len(t_raw)} != {block.count * 8}"
                            )
                        deltas = np.frombuffer(t_raw, dtype="<u8", count=block.count)
                        times = np.cumsum(deltas, dtype=np.uint64)

                        # If this block does not overlap the range, do not
                        # decompress the large price streams.
                        if int(times[-1]) < start_ms or int(times[0]) > end_ms:
                            if progress:
                                progress(idx + 1, total)
                            del t_raw
                            continue

                        _vol_raw, stream = _decompress_stream(stream)
                        _last_raw, stream = _decompress_stream(stream)
                        bid_raw, stream = _decompress_stream(stream)
                        ask_raw, stream = _decompress_stream(stream)
                        if len(bid_raw) != block.count * 8 or len(ask_raw) != block.count * 8:
                            raise TickCacheError(
                                f"Price stream length mismatch in {path.name} block {block.offset}"
                            )

                        bid = np.frombuffer(bid_raw, dtype="<f8", count=block.count)
                        ask = np.frombuffer(ask_raw, dtype="<f8", count=block.count)

                        mask = (times >= start_ms) & (times <= end_ms) & (bid > 0) & (ask > 0)
                        if mask.any():
                            out_times.append(times[mask].astype(np.int64, copy=True))
                            out_bid.append(bid[mask].astype(np.float64, copy=True))
                            out_ask.append(ask[mask].astype(np.float64, copy=True))
                        del t_raw, _vol_raw, _last_raw, bid_raw, ask_raw
                    except (TickCacheError, zlib.error, OSError, ValueError):
                        # Fallback: new-format block (latest MT5 build).
                        try:
                            t_new, b_new, a_new = _decode_new_format_block(mm, block)
                            m_new = ((t_new >= start_ms) & (t_new <= end_ms)
                                     & (b_new > 0) & (a_new > 0))
                            if m_new.any():
                                out_times.append(t_new[m_new])
                                out_bid.append(b_new[m_new])
                                out_ask.append(a_new[m_new])
                        except Exception:
                            pass
                    if progress:
                        progress(idx + 1, total)
    except OSError as exc:
        raise TickCacheError(f"Failed to read {path}: {exc}") from exc

    if not out_times:
        return Ticks(np.empty(0, np.int64), np.empty(0, np.float64),
                     np.empty(0, np.float64), str(path))

    # blocks in the index are already chronological after sorting; make sure order anyway.
    times = np.concatenate(out_times)
    bid = np.concatenate(out_bid)
    ask = np.concatenate(out_ask)
    if times.size > 1 and np.any(times[1:] < times[:-1]):
        order = np.argsort(times, kind="mergesort")
        times, bid, ask = times[order], bid[order], ask[order]

    return Ticks(times, bid, ask, str(path))


def read_source(
    source: TickSource,
    start_ms: int,
    end_ms: int,
    progress: Callable[[str, int, int], None] | None = None,
) -> Ticks:
    chunks: list[Ticks] = []
    files = sorted(source.files, key=lambda f: f.path.name)
    for i, info in enumerate(files):
        # skip months clearly outside the range
        if info.last_time_ms < start_ms:
            continue
        if info.first_time_ms > end_ms:
            continue
        if progress:
            progress(info.path.stem, i, len(files))
        part = read_tkc(info, start_ms, end_ms)
        if len(part):
            chunks.append(part)
        if progress:
            progress(info.path.stem, i + 1, len(files))

    if not chunks:
        return Ticks(np.empty(0, np.int64), np.empty(0, np.float64),
                     np.empty(0, np.float64), source.label())

    times = np.concatenate([c.time_ms for c in chunks])
    bid = np.concatenate([c.bid for c in chunks])
    ask = np.concatenate([c.ask for c in chunks])
    if times.size > 1 and np.any(times[1:] < times[:-1]):
        order = np.argsort(times, kind="mergesort")
        times, bid, ask = times[order], bid[order], ask[order]
    return Ticks(times, bid, ask, source.label())


def _mt5_data_roots() -> list[tuple[str, Path]]:
    roots: list[tuple[str, Path]] = []
    appdata = os.environ.get("APPDATA")
    if not appdata:
        return roots
    for origin, sub in (("Tester", "Tester"), ("Terminal", "Terminal")):
        base = Path(appdata) / "MetaQuotes" / sub
        if base.exists():
            roots.append((origin, base))
    return roots


def discover_sources(custom_roots: Sequence[str | os.PathLike[str]] = ()) -> list[TickSource]:
    """Discover all .tkc sources in the Terminal and Tester data folders."""
    roots = _mt5_data_roots()
    for custom in custom_roots:
        p = Path(custom).expanduser()
        if p.exists():
            roots.append(("Custom", p))

    grouped: dict[tuple[str, str, str], list[TickFileInfo]] = {}
    for origin, root in roots:
        # Terminal/<hash>/bases/<server>/ticks/<symbol>/*.tkc
        # Tester  /<hash>/bases/<server>/ticks/<symbol>/*.tkc
        paths = list(root.glob("*/bases/*/ticks/*/*.tkc"))
        if not paths:
            paths = list(root.glob("**/ticks/*/*.tkc"))
        for path in paths:
            try:
                info = inspect_tkc(path)
            except (TickCacheError, OSError):
                continue
            parts = path.parts
            try:
                bases_i = parts.index("bases")
                server = parts[bases_i + 1]
            except (ValueError, IndexError):
                server = path.parent.parent.name if path.parent.parent else "MT5"
            key = (origin, server, info.symbol)
            grouped.setdefault(key, []).append(info)

    sources: list[TickSource] = []
    for (origin, server, symbol), infos in grouped.items():
        # Drop duplicate months (can appear from two terminal IDs).
        by_month: dict[str, TickFileInfo] = {}
        for info in infos:
            old = by_month.get(info.path.stem)
            if old is None or info.path.stat().st_mtime > old.path.stat().st_mtime:
                by_month[info.path.stem] = info
        files = sorted(by_month.values(), key=lambda x: x.path.stem)
        if not files:
            continue
        source = TickSource(
            key=f"{origin}|{server}|{symbol}",
            origin=origin,
            server=server,
            symbol=symbol,
            files=files,
        )
        sources.append(source)

    sources.sort(key=lambda s: (s.origin != "Tester", -s.total_ticks, s.symbol))
    return sources
