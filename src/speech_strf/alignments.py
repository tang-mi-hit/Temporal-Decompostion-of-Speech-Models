from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class Interval:
    tier: str
    start: float
    end: float
    label: str


_ITEM = re.compile(r"item\s*\[\d+\]:\s*(.*?)(?=\n\s*item\s*\[\d+\]:|\Z)", re.S)
_INTERVAL = re.compile(
    r"intervals\s*\[\d+\]:\s*xmin\s*=\s*([-+.\deE]+)\s*"
    r"xmax\s*=\s*([-+.\deE]+)\s*text\s*=\s*\"((?:[^\"]|\"\")*)\"",
    re.S,
)


def parse_textgrid(path: str | Path) -> list[Interval]:
    """Parse Praat long-text interval tiers while retaining empty labels."""
    text = Path(path).read_text(encoding="utf-8-sig")
    rows: list[Interval] = []
    for item in _ITEM.findall(text):
        name_match = re.search(r'name\s*=\s*"((?:[^"]|"")*)"', item)
        class_match = re.search(r'class\s*=\s*"([^"]+)"', item)
        if not name_match or not class_match or class_match.group(1) != "IntervalTier":
            continue
        tier = name_match.group(1).replace('""', '"')
        for start, end, label in _INTERVAL.findall(item):
            rows.append(Interval(tier, float(start), float(end), label.replace('""', '"')))
    if not rows:
        raise ValueError(f"No IntervalTier intervals parsed from {path}")
    return rows


def validate_intervals(
    intervals: list[Interval],
    audio_duration: float,
    empty_end_tolerance_seconds: float = 0.0,
) -> list[dict]:
    if empty_end_tolerance_seconds < 0:
        raise ValueError("empty_end_tolerance_seconds must be nonnegative")
    issues: list[dict] = []
    by_tier: dict[str, list[Interval]] = {}
    for interval in intervals:
        by_tier.setdefault(interval.tier, []).append(interval)
        if interval.start < 0 or interval.end < interval.start:
            issues.append(
                {"severity": "error", "code": "negative_or_reversed_duration",
                 **asdict(interval)}
            )
        if interval.end > audio_duration + 1e-6:
            overhang = interval.end - audio_duration
            tolerated = not interval.label.strip() and overhang <= empty_end_tolerance_seconds
            issues.append(
                {
                    "severity": "warning" if tolerated else "error",
                    "code": (
                        "empty_trailing_interval_overhang"
                        if tolerated
                        else "outside_audio_duration"
                    ),
                    "overhang_seconds": overhang,
                    **asdict(interval),
                }
            )
    for tier, rows in by_tier.items():
        for previous, current in zip(rows, rows[1:]):
            if current.start < previous.start:
                issues.append(
                    {"severity": "error", "code": "non_monotonic_times", "tier": tier,
                     "previous": asdict(previous), "current": asdict(current)}
                )
    return issues

