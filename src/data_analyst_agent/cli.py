"""Command-line entry point: answer one question about a CSV."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

from .agent import Agent, AgentResult, Step
from .llm import DEFAULT_MODEL, LLM, AnthropicLLM, LLMError
from .retrieval import load_data_dictionary
from .tools import Toolbox

ROOT = Path(__file__).resolve().parents[2]

EXIT_ANSWERED, EXIT_ERROR, EXIT_NOT_ANSWERED = 0, 1, 2


def _positive_int(text: str) -> int:
    value = int(text)
    if value < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return value


def parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Answer a question about a CSV dataset with an LLM agent.")
    p.add_argument("--question", "-q", required=True, help="Natural-language question about the data.")
    p.add_argument("--csv", type=Path, default=ROOT / "data" / "sales.csv")
    p.add_argument("--dictionary", type=Path, default=ROOT / "data" / "data_dictionary.yaml")
    p.add_argument("--output-dir", type=Path, default=ROOT / "outputs", help="Where plots are saved.")
    p.add_argument("--max-steps", type=_positive_int, default=12, help="Maximum number of tool calls.")
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--effort", default="high", choices=["low", "medium", "high", "xhigh", "max"])
    p.add_argument("--verbose", "-v", action="store_true", help="Print each step as it runs.")
    p.add_argument("--json", action="store_true", help="Print the full result as JSON instead of text.")
    return p.parse_args(argv)


def _short(value: object, limit: int = 300) -> str:
    text = json.dumps(value, default=str)
    return text if len(text) <= limit else text[: limit - 3] + "..."


def print_step(step: Step) -> None:
    if step.thought:
        print(f"  thinking: {step.thought}", file=sys.stderr)
    status = "ok" if step.ok else "ERROR"
    print(f"[{step.index}] {step.tool}({_short(step.arguments, 200)}) -> {status}", file=sys.stderr)
    body = step.result if step.ok else step.error
    print(f"    {_short(body)}", file=sys.stderr)


def print_result(result: AgentResult) -> None:
    if result.answer:
        print(result.answer)
    if result.stopped_reason == "step_budget":
        print(f"\n(Stopped: step budget of {len(result.steps)} tool calls used up before an answer.)")
    elif result.stopped_reason == "refused":
        print("\n(Stopped: the model declined this request.)")


def main(argv: Sequence[str] | None = None, llm: LLM | None = None) -> int:
    args = parse_args(argv)
    try:
        toolbox = Toolbox.from_csv(args.csv, output_dir=args.output_dir)
        dictionary = load_data_dictionary(args.dictionary)
    except (OSError, ValueError) as e:
        print(f"error: {e}", file=sys.stderr)
        return EXIT_ERROR

    agent = Agent(
        llm or AnthropicLLM(model=args.model, effort=args.effort),
        toolbox,
        dictionary,
        max_steps=args.max_steps,
        on_step=print_step if args.verbose else None,
    )
    if args.verbose:
        print(f"question: {args.question}", file=sys.stderr)
        print(f"rows: {len(toolbox.df)}  max steps: {args.max_steps}", file=sys.stderr)

    try:
        result = agent.run(args.question)
    except LLMError as e:
        print(f"error: {e}", file=sys.stderr)
        return EXIT_ERROR

    if args.verbose:
        print(
            f"stopped: {result.stopped_reason} after {len(result.steps)} step(s), {result.turns} turn(s); "
            f"retrieved columns: {result.retrieved_columns}; usage: {result.usage}\n",
            file=sys.stderr,
        )
    if args.json:
        print(result.to_json(indent=2, default=str))
    else:
        print_result(result)
    return EXIT_ANSWERED if result.stopped_reason == "answered" else EXIT_NOT_ANSWERED


if __name__ == "__main__":
    sys.exit(main())
