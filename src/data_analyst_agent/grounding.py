"""Number extraction and grounding checks.

Used at runtime by the agent (to verify an answer's figures before returning
it) and by the eval scorer. A figure is "grounded" if some tool result rounds
to it at the precision it was written with.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any, Iterable

# A number not glued to a word or another number: 7,900  -3.5  .25  1.2e-5  45%
_NUMBER = re.compile(
    r"(?<![\w.])"
    r"(?P<sign>-)?"
    r"(?P<body>\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?|\.\d+)"
    r"(?P<exp>[eE][-+]?\d+)?"
    r"(?P<pct>%)?"
    r"(?!\w)"
)
_COMPARATOR = re.compile(r"(?:[<>≤≥]=?|=<|=>)\s*$")


@dataclass(frozen=True)
class Number:
    value: float
    decimals: int  # digits after the decimal point as written; drives rounding tolerance
    percent: bool
    threshold: bool  # written after <, >, <=, >= (e.g. "p < 0.05"): a cutoff, not a reported figure
    text: str
    start: int = 0


def extract_number_spans(text: str) -> list[Number]:
    out = []
    for m in _NUMBER.finditer(text or ""):
        body = m.group("body").replace(",", "")
        literal = (m.group("sign") or "") + body + (m.group("exp") or "")
        try:
            value = float(literal)
        except ValueError:
            continue
        if not math.isfinite(value):
            continue
        # Decimal places implied by how it was written: 562.7 -> 1, 7,900 -> 0,
        # 1.2e-5 -> 6 (one mantissa decimal shifted by the exponent).
        decimals = len(body.split(".")[1]) if "." in body else 0
        if m.group("exp"):
            decimals -= int(m.group("exp")[1:])
        before = text[max(0, m.start() - 4) : m.start()]
        out.append(
            Number(
                value=value,
                decimals=decimals,
                percent=bool(m.group("pct")),
                threshold=bool(_COMPARATOR.search(before)),
                text=m.group(0),
                start=m.start(),
            )
        )
    return out


def extract_numbers(text: str) -> list[float]:
    """All numbers in ``text`` as floats, in order. Thousands separators are removed."""
    return [n.value for n in extract_number_spans(text)]


def collect_numbers(obj: Any) -> list[float]:
    """Every numeric leaf in a (nested) tool result."""
    out: list[float] = []
    if isinstance(obj, bool) or obj is None:
        return out
    if isinstance(obj, (int, float)):
        if math.isfinite(obj):
            out.append(float(obj))
    elif isinstance(obj, dict):
        for v in obj.values():
            out.extend(collect_numbers(v))
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            out.extend(collect_numbers(v))
    return out


def _grounded(n: Number, sources: list[float]) -> bool:
    """A figure written with d decimals is grounded if some source rounds to it.

    Percentages also match a source fraction (12.5% <- 0.125).
    """
    tol = 0.5 * 10.0 ** (-n.decimals) * (1 + 1e-9)
    if any(abs(s - n.value) <= tol for s in sources):
        return True
    return n.percent and any(abs(s * 100 - n.value) <= tol for s in sources)


def ungrounded_numbers(answer: str | None, tool_results: Iterable[Any], question: str = "") -> list[str]:
    """Numbers in ``answer`` that do not come from any tool result.

    Ignored: thresholds ("p < 0.05") and numbers that already appear in the
    question. Everything else must match a tool figure after rounding.
    """
    sources: list[float] = []
    for r in tool_results:
        sources.extend(collect_numbers(r))
    in_question = set(extract_numbers(question))
    bad = []
    for n in extract_number_spans(answer or ""):
        if n.threshold or n.value in in_question:
            continue
        if not _grounded(n, sources):
            bad.append(n.text)
    return bad
