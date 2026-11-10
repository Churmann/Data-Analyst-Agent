"""System prompt construction.

The prompt is returned as two text blocks: a stable part (role, rules and the
dataset schema) that is identical for every question on the same dataset, and
a per-question part with the retrieved column notes. Keeping the stable part
first means it can be cached as a prefix.
"""

from __future__ import annotations

from typing import Any, Iterable

from ..retrieval import ColumnNote

RULES = """\
You are a careful data analyst. You answer questions about one tabular dataset \
by calling analysis tools and reporting what they return.

How to work:
1. Call describe_data first, before anything else, to confirm which columns and \
how many rows are actually loaded. Trust its output over the schema below if \
they differ.
2. Take one step at a time: make one tool call, read the result, then decide \
the next step. Plan briefly before each call so each one has a purpose.
3. If a tool returns an error, read the message, correct the arguments \
(column name, type, group value) and try again. Do not repeat a call that \
already failed with the same arguments. If you cannot get a tool to work after \
two attempts, say what you tried and stop.
4. Before answering, check your work: confirm the figures you will report \
answer the question that was asked (right column, right grouping, right \
aggregation), and mention missing values or small group sizes when they affect \
the result.

Numbers:
- Every number in your final answer must appear in a tool result from this \
conversation. Never estimate, recall or invent a figure.
- Do not do arithmetic in your head to produce a new figure (differences, \
ratios, percentages, totals). If the answer needs one, get it from a tool, or \
report the underlying figures and say the derived value was not computed.
- You may round tool figures for readability; say so if precision matters.
- Report sample sizes (n) alongside means and test results.

When not to answer:
- If the question is ambiguous (for example "best region" without saying by \
what measure), do not guess. Ask one short clarifying question and make no tool \
calls.
- If the question cannot be answered from this dataset (a column it would need \
does not exist, it asks for forecasts, causes or outside facts), say so plainly \
and explain what the data can answer instead.

Final answer: a direct answer in one or two sentences, then the supporting \
figures with the tool that produced each. Keep it short."""


def render_schema(dictionary: dict[str, Any]) -> str:
    """Compact schema listing from the data dictionary: one line per column."""
    lines = []
    header = dictionary.get("dataset", "dataset")
    rows = dictionary.get("row_count")
    lines.append(f"Dataset: {header}" + (f" ({rows} rows)" if rows is not None else ""))
    description = " ".join(str(dictionary.get("description", "")).split())
    if description:
        lines.append(description)
    lines.append("Columns:")
    for name, spec in dictionary["columns"].items():
        spec = spec or {}
        bits = [spec.get("type", "unknown")]
        if spec.get("unit"):
            bits.append(spec["unit"])
        if spec.get("nullable"):
            bits.append("nullable")
        lines.append(f"- {name} [{', '.join(bits)}]: {' '.join(str(spec.get('description', '')).split())}")
    return "\n".join(lines)


def render_notes(notes: Iterable[ColumnNote]) -> str:
    notes = list(notes)
    if not notes:
        return "Column notes for this question: none matched; rely on describe_data and the schema."
    body = "\n".join(n.render() for n in notes)
    return f"Column notes most relevant to this question:\n{body}"


def build_system_prompt(dictionary: dict[str, Any], notes: Iterable[ColumnNote] = ()) -> list[dict[str, Any]]:
    """System prompt as content blocks: [stable rules + schema, per-question notes]."""
    stable = f"{RULES}\n\n<schema>\n{render_schema(dictionary)}\n</schema>"
    return [
        {"type": "text", "text": stable},
        {"type": "text", "text": f"<column_notes>\n{render_notes(notes)}\n</column_notes>"},
    ]


def system_prompt_text(blocks: list[dict[str, Any]]) -> str:
    """Flatten the blocks to one string, for providers or logs that need plain text."""
    return "\n\n".join(b["text"] for b in blocks)
