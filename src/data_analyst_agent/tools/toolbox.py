"""Analysis tools exposed to the agent.

Each tool is a plain pandas/scipy computation over the loaded DataFrame that
returns a JSON-serialisable dict. Every tool has a matching JSON schema in
``TOOL_SPECS`` for the Messages API ``tools`` parameter.

Bad input (unknown tool, unknown column, wrong dtype, bad arguments) raises
``ToolError`` with a message written for the model to read and correct itself,
so the agent loop can return it as an error result instead of crashing.
"""

from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Any, Callable

import matplotlib

matplotlib.use("Agg")  # headless: write files, never open windows
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.axes import Axes  # noqa: E402
from scipy import stats  # noqa: E402

# Allowed values for enum arguments; the schemas below are built from these.
AGGREGATIONS = ("mean", "sum", "median", "min", "max", "count", "std")
TESTS = ("welch_t", "mann_whitney")
ALTERNATIVES = ("two-sided", "less", "greater")
PLOT_KINDS = ("bar", "hist", "scatter")
PLOT_COLOR = "#4C72B0"


class ToolError(Exception):
    """A recoverable tool failure; the message is shown to the model."""


# --------------------------------------------------------------------------- #
# Schemas
# --------------------------------------------------------------------------- #

TOOL_SPECS: list[dict[str, Any]] = [
    {
        "name": "describe_data",
        "description": (
            "Overview of the dataset: row count, and for each column its dtype, "
            "number of missing values and number of distinct values. Call this "
            "first to learn which columns exist."
        ),
        "input_schema": {"type": "object", "properties": {}, "additionalProperties": False},
    },
    {
        "name": "column_stats",
        "description": (
            "Summary statistics for one column. Numeric columns: count, missing, "
            "mean, std, min, quartiles, max. Categorical columns: count, missing "
            "and value counts."
        ),
        "input_schema": {
            "type": "object",
            "properties": {"column": {"type": "string", "description": "Column name."}},
            "required": ["column"],
            "additionalProperties": False,
        },
    },
    {
        "name": "aggregate",
        "description": (
            "Group rows by one or more columns and aggregate a numeric column. "
            "Missing values in the value column are skipped; each group reports "
            "how many non-missing values it used (n)."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "group_by": {
                    "type": "array",
                    "items": {"type": "string"},
                    "minItems": 1,
                    "description": "Columns to group by.",
                },
                "value": {"type": "string", "description": "Numeric column to aggregate."},
                "agg": {"type": "string", "enum": list(AGGREGATIONS)},
            },
            "required": ["group_by", "value", "agg"],
            "additionalProperties": False,
        },
    },
    {
        "name": "correlation",
        "description": (
            "Pearson correlation between two numeric columns, using rows where "
            "both are present. Returns r, the two-sided p-value and n."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "x": {"type": "string", "description": "First numeric column."},
                "y": {"type": "string", "description": "Second numeric column."},
            },
            "required": ["x", "y"],
            "additionalProperties": False,
        },
    },
    {
        "name": "hypothesis_test",
        "description": (
            "Compare a numeric column between two groups of a categorical column. "
            "welch_t: Welch's t-test (unequal variances) on means. mann_whitney: "
            "Mann-Whitney U test on distributions. Missing values are dropped."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "value": {"type": "string", "description": "Numeric column to compare."},
                "group_col": {"type": "string", "description": "Categorical column defining groups."},
                "group_a": {"type": "string", "description": "First group's value in group_col."},
                "group_b": {"type": "string", "description": "Second group's value in group_col."},
                "test": {"type": "string", "enum": list(TESTS)},
                "alternative": {
                    "type": "string",
                    "enum": list(ALTERNATIVES),
                    "description": "Alternative hypothesis for group_a vs group_b. Default two-sided.",
                },
            },
            "required": ["value", "group_col", "group_a", "group_b", "test"],
            "additionalProperties": False,
        },
    },
    {
        "name": "plot",
        "description": (
            "Save a chart as a PNG and return its path. bar: mean of y per "
            "category of x. hist: distribution of x. scatter: x against y."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "kind": {"type": "string", "enum": list(PLOT_KINDS)},
                "x": {"type": "string", "description": "Column for the x axis (categorical for bar)."},
                "y": {"type": "string", "description": "Numeric column for the y axis (bar and scatter)."},
                "bins": {"type": "integer", "minimum": 1, "maximum": 200, "description": "hist only. Default 20."},
                "title": {"type": "string"},
                "filename": {"type": "string", "description": "Optional file name; letters, digits, - and _ only."},
            },
            "required": ["kind", "x"],
            "additionalProperties": False,
        },
    },
]

_SPECS_BY_NAME = {spec["name"]: spec for spec in TOOL_SPECS}


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _num(x: Any) -> float | int | None:
    """Convert numpy scalars to plain Python numbers; NaN/inf become None."""
    if x is None:
        return None
    if isinstance(x, (np.integer, int)) and not isinstance(x, bool):
        return int(x)
    x = float(x)
    return None if math.isnan(x) or math.isinf(x) else x


def _key(value: Any) -> Any:
    return _num(value) if isinstance(value, (np.number, int, float)) else str(value)


_TYPE_CHECKS: dict[str, Callable[[Any], bool]] = {
    "string": lambda v: isinstance(v, str),
    "integer": lambda v: isinstance(v, int) and not isinstance(v, bool),
    "array": lambda v: isinstance(v, list),
    "object": lambda v: isinstance(v, dict),
}


def _validate_args(name: str, args: Any) -> dict[str, Any]:
    """Check args against the tool's schema (required, unknown, type, enum, bounds)."""
    if args is None:
        args = {}
    if not isinstance(args, dict):
        raise ToolError(f"Arguments for '{name}' must be an object, got {type(args).__name__}.")
    schema = _SPECS_BY_NAME[name]["input_schema"]
    props = schema["properties"]

    unknown = sorted(set(args) - set(props))
    if unknown:
        raise ToolError(f"Unknown argument(s) for '{name}': {unknown}. Allowed: {sorted(props)}.")
    missing = [p for p in schema.get("required", []) if p not in args]
    if missing:
        raise ToolError(f"Missing required argument(s) for '{name}': {missing}.")

    for key, value in args.items():
        spec = props[key]
        check = _TYPE_CHECKS[spec["type"]]
        if not check(value):
            raise ToolError(f"Argument '{key}' for '{name}' must be of type {spec['type']}.")
        if "enum" in spec and value not in spec["enum"]:
            raise ToolError(f"Argument '{key}' must be one of {spec['enum']}, got {value!r}.")
        if spec["type"] == "array":
            if len(value) < spec.get("minItems", 0):
                raise ToolError(f"Argument '{key}' needs at least {spec['minItems']} item(s).")
            if not all(isinstance(v, str) for v in value):
                raise ToolError(f"Argument '{key}' must be a list of strings.")
        if spec["type"] == "integer":
            lo, hi = spec.get("minimum"), spec.get("maximum")
            if (lo is not None and value < lo) or (hi is not None and value > hi):
                raise ToolError(f"Argument '{key}' must be between {lo} and {hi}, got {value}.")
    return args


# --------------------------------------------------------------------------- #
# Toolbox
# --------------------------------------------------------------------------- #


class Toolbox:
    """Analysis tools bound to one DataFrame."""

    def __init__(self, df: pd.DataFrame, output_dir: str | Path = "outputs") -> None:
        self.df = df
        self.output_dir = Path(output_dir)
        self._plot_count = 0
        self._tools: dict[str, Callable[..., dict[str, Any]]] = {
            "describe_data": self.describe_data,
            "column_stats": self.column_stats,
            "aggregate": self.aggregate,
            "correlation": self.correlation,
            "hypothesis_test": self.hypothesis_test,
            "plot": self.plot,
        }

    @classmethod
    def from_csv(cls, path: str | Path, **kwargs: Any) -> "Toolbox":
        return cls(pd.read_csv(path), **kwargs)

    # -- dispatch ----------------------------------------------------------- #

    @staticmethod
    def specs() -> list[dict[str, Any]]:
        """Tool schemas in the format expected by the Messages API ``tools`` parameter."""
        return [dict(spec) for spec in TOOL_SPECS]

    def run(self, name: str, args: dict[str, Any] | None = None) -> dict[str, Any]:
        """Validate arguments and run tool ``name``. Raises ToolError on bad input."""
        if name not in self._tools:
            raise ToolError(f"Unknown tool '{name}'. Available tools: {sorted(self._tools)}.")
        args = _validate_args(name, args)
        return self._tools[name](**args)

    # -- validation --------------------------------------------------------- #

    def _column(self, name: str) -> pd.Series:
        if name not in self.df.columns:
            raise ToolError(f"Unknown column '{name}'. Available columns: {list(self.df.columns)}.")
        return self.df[name]

    def _numeric(self, name: str) -> pd.Series:
        col = self._column(name)
        if not pd.api.types.is_numeric_dtype(col) or pd.api.types.is_bool_dtype(col):
            raise ToolError(f"Column '{name}' is not numeric (dtype {col.dtype}).")
        return col

    def _categorical(self, name: str) -> pd.Series:
        col = self._column(name)
        if pd.api.types.is_float_dtype(col):
            raise ToolError(
                f"Column '{name}' is continuous (dtype {col.dtype}) and cannot be used to define groups."
            )
        return col

    # -- tools -------------------------------------------------------------- #

    def describe_data(self) -> dict[str, Any]:
        columns = []
        for name in self.df.columns:
            col = self.df[name]
            columns.append(
                {
                    "name": name,
                    "dtype": str(col.dtype),
                    "numeric": bool(pd.api.types.is_numeric_dtype(col)),
                    "missing": int(col.isna().sum()),
                    "distinct": int(col.nunique(dropna=True)),
                }
            )
        return {"rows": len(self.df), "columns": columns}

    def column_stats(self, column: str) -> dict[str, Any]:
        col = self._column(column)
        result: dict[str, Any] = {
            "column": column,
            "count": int(col.notna().sum()),
            "missing": int(col.isna().sum()),
        }
        if pd.api.types.is_numeric_dtype(col) and not pd.api.types.is_bool_dtype(col):
            q = col.quantile([0.25, 0.5, 0.75])
            result.update(
                {
                    "mean": _num(col.mean()),
                    "std": _num(col.std()),
                    "min": _num(col.min()),
                    "p25": _num(q[0.25]),
                    "median": _num(q[0.5]),
                    "p75": _num(q[0.75]),
                    "max": _num(col.max()),
                    "sum": _num(col.sum()),
                }
            )
        else:
            counts = col.value_counts(dropna=True)
            result["value_counts"] = {str(k): int(v) for k, v in counts.items()}
        return result

    def aggregate(self, group_by: list[str], value: str, agg: str) -> dict[str, Any]:
        for g in group_by:
            self._categorical(g)
        self._numeric(value)
        if value in group_by:
            raise ToolError(f"Column '{value}' cannot be both a group_by column and the value.")

        grouped = self.df.groupby(group_by, observed=True, dropna=False)[value]
        table = grouped.agg(_result=agg, _n="count").reset_index()
        groups = []
        for _, row in table.iterrows():
            groups.append(
                {
                    "group": {g: _key(row[g]) for g in group_by},
                    agg: _num(row["_result"]),
                    "n": int(row["_n"]),
                }
            )
        return {
            "group_by": group_by,
            "value": value,
            "agg": agg,
            "missing_values_skipped": int(self.df[value].isna().sum()),
            "groups": groups,
        }

    def correlation(self, x: str, y: str) -> dict[str, Any]:
        self._numeric(x)
        self._numeric(y)
        pair = self.df[[x, y]].dropna()
        if len(pair) < 3:
            raise ToolError(f"Need at least 3 rows with both '{x}' and '{y}' present; found {len(pair)}.")
        if pair[x].nunique() < 2 or pair[y].nunique() < 2:
            raise ToolError("Correlation is undefined when a column is constant.")
        r, p = stats.pearsonr(pair[x], pair[y])
        return {"x": x, "y": y, "method": "pearson", "r": _num(r), "p_value": _num(p), "n": len(pair)}

    def hypothesis_test(
        self,
        value: str,
        group_col: str,
        group_a: str,
        group_b: str,
        test: str,
        alternative: str = "two-sided",
    ) -> dict[str, Any]:
        self._numeric(value)
        groups = self._categorical(group_col)
        if group_a == group_b:
            raise ToolError("group_a and group_b must be different.")
        labels = groups.astype(str).where(groups.notna())
        available = sorted(labels.dropna().unique())
        samples = {}
        for g in (group_a, group_b):
            if g not in available:
                raise ToolError(f"Group '{g}' not found in '{group_col}'. Available: {available}.")
            s = self.df.loc[labels == g, value].dropna()
            if len(s) < 2:
                raise ToolError(f"Group '{g}' has fewer than 2 non-missing '{value}' values.")
            samples[g] = s

        a, b = samples[group_a], samples[group_b]
        if test == "welch_t":
            res = stats.ttest_ind(a, b, equal_var=False, alternative=alternative)
            statistic_name = "t"
        else:
            res = stats.mannwhitneyu(a, b, alternative=alternative)
            statistic_name = "U"

        def summary(s: pd.Series) -> dict[str, Any]:
            return {"n": len(s), "mean": _num(s.mean()), "median": _num(s.median()), "std": _num(s.std())}

        return {
            "test": test,
            "value": value,
            "group_col": group_col,
            "alternative": alternative,
            "groups": {group_a: summary(a), group_b: summary(b)},
            "statistic_name": statistic_name,
            "statistic": _num(res.statistic),
            "p_value": _num(res.pvalue),
        }

    def plot(
        self,
        kind: str,
        x: str,
        y: str | None = None,
        bins: int = 20,
        title: str | None = None,
        filename: str | None = None,
    ) -> dict[str, Any]:
        if filename is not None and not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", filename):
            # Also stops path traversal: the name can never contain / \ or ..
            raise ToolError("filename may contain only letters, digits, '-' and '_' (max 64 chars).")

        # Validate every column before creating a figure, so bad input leaves
        # no file behind; then pick the drawer for this kind.
        draw: Callable[[Axes], dict[str, Any]]
        if kind == "hist":
            if y is not None:
                raise ToolError("Plot kind 'hist' takes only 'x'.")
            self._numeric(x)
            draw = lambda ax: self._draw_hist(ax, x, bins)  # noqa: E731
        else:
            if y is None:
                raise ToolError(f"Plot kind '{kind}' requires 'y'.")
            y_col: str = y
            if kind == "bar":
                self._categorical(x)
                self._numeric(y_col)
                draw = lambda ax: self._draw_bar(ax, x, y_col)  # noqa: E731
            else:
                self._numeric(x)
                self._numeric(y_col)
                draw = lambda ax: self._draw_scatter(ax, x, y_col)  # noqa: E731

        fig, ax = plt.subplots(figsize=(7, 4.5))
        try:
            summary = draw(ax)
            ax.set_title(title or (f"{kind}: {x}" + (f" vs {y}" if y else "")))
            ax.spines[["top", "right"]].set_visible(False)
            fig.tight_layout()

            self._plot_count += 1
            stem = filename or f"plot_{self._plot_count:02d}_{kind}"
            self.output_dir.mkdir(parents=True, exist_ok=True)
            path = self.output_dir / f"{stem}.png"
            fig.savefig(path, dpi=120)
        finally:
            plt.close(fig)  # never leak figures, even if drawing fails

        return {"kind": kind, "x": x, "y": y, "path": str(path), **summary}

    # Each drawer returns the figures it plotted, so the model can cite them
    # without reading the image.

    def _draw_bar(self, ax: Axes, x: str, y: str) -> dict[str, Any]:
        means = self.df.groupby(x, observed=True)[y].mean().sort_values(ascending=False)
        ax.bar(means.index.astype(str), means.to_numpy(), color=PLOT_COLOR)
        ax.set_xlabel(x)
        ax.set_ylabel(f"mean {y}")
        return {"bars": {str(k): _num(v) for k, v in means.items()}}

    def _draw_hist(self, ax: Axes, x: str, bins: int) -> dict[str, Any]:
        data = self.df[x].dropna()
        ax.hist(data, bins=bins, color=PLOT_COLOR, edgecolor="white")
        ax.set_xlabel(x)
        ax.set_ylabel("count")
        return {"n": len(data), "bins": bins}

    def _draw_scatter(self, ax: Axes, x: str, y: str) -> dict[str, Any]:
        pair = self.df[[x, y]].dropna()
        ax.scatter(pair[x], pair[y], s=12, alpha=0.6, color=PLOT_COLOR)
        ax.set_xlabel(x)
        ax.set_ylabel(y)
        return {"n": len(pair)}
