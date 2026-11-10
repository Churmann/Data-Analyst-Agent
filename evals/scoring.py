"""Pure scoring functions for the evaluation set.

Nothing here calls a model or reads files: every function takes plain values
and returns plain values, so the scorer itself is easy to test.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
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


def numeric_match(expected: float, candidates: Iterable[float], tolerance: float = 0.0) -> bool:
    """True if any candidate is within ``tolerance`` (absolute) of ``expected``."""
    return any(abs(c - expected) <= tolerance + 1e-9 for c in candidates)


REFUSAL_PATTERNS = [
    r"\bcan(?:not|'t|’t)\b.{0,40}\b(?:answer|determine|calculate|compute|tell|provide|forecast|predict)",
    r"\b(?:unable|not able) to\b",
    r"\bnot (?:possible|available|included|present|recorded|contained)\b",
    r"\b(?:does not|doesn't|doesn’t|do not|don't|don’t) (?:have|contain|include|record|track)\b",
    r"\bno (?:column|data|information|field|cost|date|time|customer)\w*\b",
    r"\b(?:outside|beyond) (?:the )?(?:scope|data|dataset)\b",
    r"\bout of scope\b",
    r"\bisn['’]t (?:in|part of|possible|available)\b",
]
_REFUSAL = re.compile("|".join(REFUSAL_PATTERNS), re.IGNORECASE | re.DOTALL)

CLARIFY_PATTERNS = [
    r"\bdo you mean\b",
    r"\bcould you (?:clarify|specify|tell me)\b",
    r"\bcan you (?:clarify|specify)\b",
    r"\bwhat do you mean\b",
    r"\bby (?:what|which) (?:measure|metric|criteria|criterion)\b",
    r"\bwhich (?:measure|metric)\b",
    r"\b(?:clarify|clarification)\b",
    r"\bwould you like\b",
]
_CLARIFY = re.compile("|".join(CLARIFY_PATTERNS), re.IGNORECASE)


def is_refusal(text: str | None) -> bool:
    """Does the answer decline because the data cannot answer the question?"""
    if not text:
        return False
    return _REFUSAL.search(text) is not None


def is_clarification(text: str | None) -> bool:
    """Does the answer ask the user what they mean? Requires an actual question mark."""
    if not text or "?" not in text:
        return False
    return _CLARIFY.search(text) is not None


# -- grounding ----------------------------------------------------------------- #


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


# -- item scoring -------------------------------------------------------------- #


@dataclass
class ItemScore:
    id: str
    kind: str
    passed: bool
    reasons: list[str] = field(default_factory=list)  # why it failed; empty when passed
    ungrounded: list[str] = field(default_factory=list)
    steps: int = 0
    tool_errors: int = 0


def score_item(item: dict[str, Any], result: dict[str, Any]) -> ItemScore:
    """Score one agent run against one eval item.

    ``result`` is ``AgentResult.to_dict()`` (or the same shape): it needs
    ``answer``, ``stopped_reason``, ``question`` and ``steps`` (each with ``ok``
    and ``result``).
    """
    kind = item.get("kind", "answer")
    answer = result.get("answer") or ""
    steps = result.get("steps", [])
    tool_results = [s.get("result") for s in steps if s.get("ok")]
    score = ItemScore(
        id=item["id"],
        kind=kind,
        passed=False,
        steps=len(steps),
        tool_errors=sum(1 for s in steps if not s.get("ok")),
    )
    reasons = score.reasons

    if result.get("stopped_reason") != "answered":
        reasons.append(f"stopped: {result.get('stopped_reason')}")

    if kind == "answer":
        numbers = extract_numbers(answer)
        for exp in item.get("expect_numbers", []):
            if not numeric_match(exp["value"], numbers, exp.get("tolerance", 0)):
                reasons.append(f"missing number {exp['value']} (+/-{exp.get('tolerance', 0)})")
        lower = answer.lower()
        for sub in item.get("expect_substrings", []):
            if sub.lower() not in lower:
                reasons.append(f"missing text '{sub}'")
        score.ungrounded = ungrounded_numbers(answer, tool_results, result.get("question", item.get("question", "")))
        if score.ungrounded:
            reasons.append(f"ungrounded numbers: {score.ungrounded}")
    elif kind == "refuse":
        if not is_refusal(answer):
            reasons.append("did not decline")
    elif kind == "clarify":
        if not is_clarification(answer):
            reasons.append("did not ask for clarification")
    else:
        reasons.append(f"unknown item kind '{kind}'")

    score.passed = not reasons
    return score


def summarise(scores: list[ItemScore]) -> dict[str, Any]:
    """Aggregate pass rates overall and per kind."""
    by_kind: dict[str, list[ItemScore]] = {}
    for s in scores:
        by_kind.setdefault(s.kind, []).append(s)
    total = len(scores)
    passed = sum(s.passed for s in scores)
    return {
        "total": total,
        "passed": passed,
        "pass_rate": passed / total if total else 0.0,
        "by_kind": {k: {"total": len(v), "passed": sum(s.passed for s in v)} for k, v in sorted(by_kind.items())},
        "ungrounded_items": sum(1 for s in scores if s.ungrounded),
        "mean_steps": sum(s.steps for s in scores) / total if total else 0.0,
        "tool_errors": sum(s.tool_errors for s in scores),
    }
