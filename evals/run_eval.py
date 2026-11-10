"""Run the agent over the eval set and print a scorecard.

    python evals/run_eval.py                      # all items, live API
    python evals/run_eval.py --ids row_count forecast --verbose
    python evals/run_eval.py --json-out outputs/eval.json

Each item gets a fresh agent, so runs do not share history.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable, Sequence

# Allow `python evals/run_eval.py` without installing the package.
ROOT = Path(__file__).resolve().parent.parent
for p in (ROOT, ROOT / "src"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import pandas as pd  # noqa: E402
import yaml  # noqa: E402

from data_analyst_agent.agent import Agent  # noqa: E402
from data_analyst_agent.llm import DEFAULT_MODEL, LLM, AnthropicLLM, LLMError  # noqa: E402
from data_analyst_agent.retrieval import load_data_dictionary  # noqa: E402
from data_analyst_agent.tools import Toolbox  # noqa: E402
from evals.scoring import ItemScore, score_item, summarise  # noqa: E402

DEFAULT_ITEMS = ROOT / "evals" / "eval_questions.yaml"


def load_items(path: str | Path = DEFAULT_ITEMS, ids: Sequence[str] | None = None) -> list[dict[str, Any]]:
    """Load eval items, optionally keeping only ``ids`` (in file order)."""
    items: list[dict[str, Any]] = yaml.safe_load(Path(path).read_text(encoding="utf-8"))["items"]
    seen: set[str] = set()
    for item in items:
        if item["id"] in seen:
            raise ValueError(f"Duplicate eval id '{item['id']}'")
        seen.add(item["id"])
    if ids:
        unknown = sorted(set(ids) - seen)
        if unknown:
            raise ValueError(f"Unknown eval id(s): {unknown}")
        items = [i for i in items if i["id"] in set(ids)]
    return items


def run_items(
    items: list[dict[str, Any]],
    make_agent: Callable[[dict[str, Any]], Agent],
    on_item: Callable[[dict[str, Any], dict[str, Any], ItemScore], None] | None = None,
) -> list[tuple[dict[str, Any], dict[str, Any], ItemScore]]:
    """Run and score each item. An API failure on one item is scored as a failure, not raised."""
    rows = []
    for item in items:
        agent = make_agent(item)
        start = time.perf_counter()
        try:
            result = agent.run(item["question"]).to_dict()
        except LLMError as e:
            result = {"question": item["question"], "answer": None, "stopped_reason": f"error: {e}", "steps": []}
        result["seconds"] = round(time.perf_counter() - start, 2)
        score = score_item(item, result)
        rows.append((item, result, score))
        if on_item:
            on_item(item, result, score)
    return rows


def format_scorecard(scores: list[ItemScore]) -> str:
    summary = summarise(scores)
    width = max([len(s.id) for s in scores] + [4])
    lines = [f"{'item':<{width}}  {'kind':<8} {'result':<6} {'steps':>5}  notes", "-" * (width + 40)]
    for s in scores:
        status = "PASS" if s.passed else "FAIL"
        lines.append(f"{s.id:<{width}}  {s.kind:<8} {status:<6} {s.steps:>5}  {'; '.join(s.reasons)}")
    lines.append("-" * (width + 40))
    by_kind = ", ".join(f"{k} {v['passed']}/{v['total']}" for k, v in summary["by_kind"].items())
    lines.append(
        f"passed {summary['passed']}/{summary['total']} ({summary['pass_rate']:.0%})  [{by_kind}]"
    )
    lines.append(
        f"answers with ungrounded numbers: {summary['ungrounded_items']}  "
        f"mean steps: {summary['mean_steps']:.1f}  tool errors: {summary['tool_errors']}"
    )
    return "\n".join(lines)


def parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Score the agent on the eval set.")
    p.add_argument("--items", type=Path, default=DEFAULT_ITEMS)
    p.add_argument("--ids", nargs="+", help="Only run these item ids.")
    p.add_argument("--csv", type=Path, default=ROOT / "data" / "sales.csv")
    p.add_argument("--dictionary", type=Path, default=ROOT / "data" / "data_dictionary.yaml")
    p.add_argument("--output-dir", type=Path, default=ROOT / "outputs" / "eval")
    p.add_argument("--max-steps", type=int, default=12)
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--effort", default="high", choices=["low", "medium", "high", "xhigh", "max"])
    p.add_argument("--json-out", type=Path, help="Write every answer, step and score to this file.")
    p.add_argument("--verbose", "-v", action="store_true", help="Print each answer as it completes.")
    return p.parse_args(argv)


def main(argv: Sequence[str] | None = None, llm_factory: Callable[[dict[str, Any]], LLM] | None = None) -> int:
    args = parse_args(argv)
    items = load_items(args.items, args.ids)
    df = pd.read_csv(args.csv)
    dictionary = load_data_dictionary(args.dictionary)
    factory = llm_factory or (lambda _item: AnthropicLLM(model=args.model, effort=args.effort))

    def make_agent(item: dict[str, Any]) -> Agent:
        # Fresh agent, history and plot folder per item; a copy of the data so
        # no item can affect another.
        toolbox = Toolbox(df.copy(), output_dir=args.output_dir / item["id"])
        return Agent(factory(item), toolbox, dictionary, max_steps=args.max_steps)

    def progress(item: dict[str, Any], result: dict[str, Any], score: ItemScore) -> None:
        mark = "PASS" if score.passed else "FAIL"
        print(f"[{mark}] {item['id']} ({result['seconds']}s)", file=sys.stderr)
        if args.verbose:
            print(f"    Q: {item['question'].strip()}\n    A: {result.get('answer')}\n", file=sys.stderr)

    rows = run_items(items, make_agent, on_item=progress)
    scores = [s for _, _, s in rows]
    print(format_scorecard(scores))

    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "summary": summarise(scores),
            "items": [{"item": i, "result": r, "score": asdict(s)} for i, r, s in rows],
        }
        args.json_out.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    return 0 if all(s.passed for s in scores) else 1


if __name__ == "__main__":
    sys.exit(main())
