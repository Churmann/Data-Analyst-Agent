"""Pure scoring functions for the evaluation set.

Nothing here calls a model or reads files: every function takes plain values
and returns plain values, so the scorer itself is easy to test. Number
extraction and grounding live in the package (data_analyst_agent.grounding)
because the agent uses them at runtime too; they are re-exported here.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Iterable

from data_analyst_agent.grounding import (
    Number,
    collect_numbers,
    extract_number_spans,
    extract_numbers,
    ungrounded_numbers,
)

__all__ = [
    "ItemScore",
    "Number",
    "collect_numbers",
    "extract_number_spans",
    "extract_numbers",
    "is_clarification",
    "is_refusal",
    "numeric_match",
    "score_item",
    "summarise",
    "ungrounded_numbers",
]

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
