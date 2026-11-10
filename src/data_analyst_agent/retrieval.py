"""Keyword retrieval over the data dictionary.

Given a question, return the column entries most likely to matter, scored by
word overlap with each column's name, description, allowed values and notes.
Deliberately simple: no embeddings, fully deterministic and explainable.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

STOPWORDS = frozenset(
    """a an and are as at be by did do does for from has have how i in is it its
    of on or per than that the their there these this to was were what when where
    which who why with across between each all any much many most more less
    give show tell me us our we you your""".split()
)

# Weights: a column named in the question matters most, then a category value
# it contains (e.g. "North" -> region), then any descriptive word.
NAME_WEIGHT = 3.0
VALUE_WEIGHT = 2.0
TEXT_WEIGHT = 1.0


@dataclass(frozen=True)
class ColumnNote:
    name: str
    score: float
    spec: dict[str, Any] = field(compare=False)
    matched: tuple[str, ...] = ()

    def render(self) -> str:
        """One readable block describing the column, for the system prompt."""
        s = self.spec
        lines = [f"- {self.name} ({s.get('type', 'unknown')}): {s.get('description', '').strip()}"]
        for key in ("unit", "allowed_values", "derivation", "notes"):
            if key in s and s[key] not in (None, ""):
                value = ", ".join(map(str, s[key])) if isinstance(s[key], list) else " ".join(str(s[key]).split())
                lines.append(f"    {key}: {value}")
        if s.get("nullable"):
            lines.append("    may contain missing values")
        return "\n".join(lines)


def load_data_dictionary(path: str | Path) -> dict[str, Any]:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict) or not isinstance(data.get("columns"), dict):
        raise ValueError(f"{path} is not a data dictionary: expected a 'columns' mapping.")
    return data


def _stem(word: str) -> str:
    """Crude plural folding: categories -> category, boxes -> box, prices -> price."""
    if len(word) <= 3 or word.endswith("ss"):
        return word
    if word.endswith("ies"):
        return word[:-3] + "y"
    if word.endswith(("ses", "xes", "zes", "ches", "shes")):
        return word[:-2]
    if word.endswith("s"):
        return word[:-1]
    return word


def tokens(text: str) -> set[str]:
    words = re.findall(r"[a-z0-9]+", text.lower().replace("_", " "))
    return {_stem(w) for w in words if w not in STOPWORDS and len(w) > 1}


def _flatten(value: Any) -> str:
    if isinstance(value, (list, tuple)):
        return " ".join(map(str, value))
    return "" if value is None else str(value)


def retrieve_columns(question: str, dictionary: dict[str, Any], k: int = 3) -> list[ColumnNote]:
    """Top-``k`` columns by keyword overlap with ``question``; only columns that match at all."""
    q = tokens(question)
    if not q:
        return []

    scored: list[tuple[float, int, ColumnNote]] = []
    for order, (name, spec) in enumerate(dictionary["columns"].items()):
        spec = spec or {}
        # The name only counts if every part of it is in the question
        # ("unit price" -> unit_price); a partial hit ("order" vs order_id)
        # is scored like any other descriptive word.
        name_tokens = tokens(name)
        name_hits = name_tokens if name_tokens <= q else set()
        value_hits = q & tokens(_flatten(spec.get("allowed_values")))
        text = " ".join([name] + [_flatten(spec.get(key)) for key in ("description", "unit", "derivation", "notes")])
        text_hits = (q & tokens(text)) - name_hits - value_hits

        score = NAME_WEIGHT * len(name_hits) + VALUE_WEIGHT * len(value_hits) + TEXT_WEIGHT * len(text_hits)
        if score > 0:
            matched = tuple(sorted(name_hits | value_hits | text_hits))
            scored.append((score, order, ColumnNote(name, score, spec, matched)))

    scored.sort(key=lambda t: (-t[0], t[1]))  # best score first, ties in dictionary order
    return [note for _, _, note in scored[:k]]
