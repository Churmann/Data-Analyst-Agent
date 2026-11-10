import json
from pathlib import Path

import pandas as pd
import pytest
from scipy import stats

from data_analyst_agent.tools import TOOL_SPECS, Toolbox, ToolError

from .test_sample_data import (
    CSV,
    EXPECTED_REGION_MEAN_REVENUE,
    EXPECTED_SEGMENT_MEAN_REVENUE,
)


@pytest.fixture
def tb(tmp_path: Path) -> Toolbox:
    return Toolbox.from_csv(CSV, output_dir=tmp_path)


# -- specs / dispatch -------------------------------------------------------- #


def test_specs_cover_every_tool(tb):
    names = [s["name"] for s in tb.specs()]
    assert names == ["describe_data", "column_stats", "aggregate", "correlation", "hypothesis_test", "plot"]
    for spec in tb.specs():
        assert set(spec) == {"name", "description", "input_schema"}
        assert spec["input_schema"]["type"] == "object"


def test_specs_returns_copies(tb):
    tb.specs()[0]["name"] = "mutated"
    assert TOOL_SPECS[0]["name"] == "describe_data"


@pytest.mark.parametrize(
    "name,args",
    [
        ("describe_data", {}),
        ("column_stats", {"column": "region"}),
        ("aggregate", {"group_by": ["region"], "value": "revenue", "agg": "mean"}),
        ("correlation", {"x": "units", "y": "revenue"}),
        ("hypothesis_test", {"value": "revenue", "group_col": "region", "group_a": "East",
                             "group_b": "South", "test": "mann_whitney"}),
        ("plot", {"kind": "hist", "x": "units"}),
    ],
)
def test_results_are_json_serialisable(tb, name, args):
    json.dumps(tb.run(name, args), allow_nan=False)


# -- correctness against the known sample-data means ------------------------ #


def test_describe_data(tb):
    out = tb.run("describe_data", {})
    assert out["rows"] == 600
    revenue = next(c for c in out["columns"] if c["name"] == "revenue")
    assert revenue["missing"] == 24 and revenue["numeric"]


def test_column_stats_numeric_and_categorical(tb):
    rev = tb.run("column_stats", {"column": "revenue"})
    assert rev["count"] == 576 and rev["missing"] == 24
    assert rev["mean"] == pytest.approx(5464 / 12)
    seg = tb.run("column_stats", {"column": "segment"})
    assert seg["value_counts"] == {"Consumer": 200, "Corporate": 200, "SMB": 200}


def test_aggregate_mean_by_region(tb):
    out = tb.run("aggregate", {"group_by": ["region"], "value": "revenue", "agg": "mean"})
    got = {g["group"]["region"]: g["mean"] for g in out["groups"]}
    assert got == pytest.approx(EXPECTED_REGION_MEAN_REVENUE)
    assert all(g["n"] == 144 for g in out["groups"])
    assert out["missing_values_skipped"] == 24


def test_aggregate_two_keys(tb):
    out = tb.run("aggregate", {"group_by": ["region", "segment"], "value": "units", "agg": "count"})
    assert len(out["groups"]) == 12
    assert all(g["count"] == 50 for g in out["groups"])


def test_aggregate_by_segment(tb):
    out = tb.run("aggregate", {"group_by": ["segment"], "value": "revenue", "agg": "mean"})
    got = {g["group"]["segment"]: g["mean"] for g in out["groups"]}
    assert got == pytest.approx(EXPECTED_SEGMENT_MEAN_REVENUE)


def test_correlation_matches_scipy(tb):
    out = tb.run("correlation", {"x": "units", "y": "revenue"})
    df = pd.read_csv(CSV).dropna()
    r, p = stats.pearsonr(df["units"], df["revenue"])
    assert out["n"] == 576
    assert out["r"] == pytest.approx(r) and out["p_value"] == pytest.approx(p)


def test_welch_t_matches_scipy(tb):
    out = tb.run("hypothesis_test", {"value": "revenue", "group_col": "segment", "group_a": "Corporate",
                                     "group_b": "Consumer", "test": "welch_t"})
    df = pd.read_csv(CSV)
    a = df.loc[df.segment == "Corporate", "revenue"].dropna()
    b = df.loc[df.segment == "Consumer", "revenue"].dropna()
    expected = stats.ttest_ind(a, b, equal_var=False)
    assert out["statistic"] == pytest.approx(expected.statistic)
    assert out["p_value"] == pytest.approx(expected.pvalue)
    assert out["groups"]["Corporate"]["mean"] == pytest.approx(730.0)
    assert out["groups"]["Corporate"]["n"] == 192


def test_mann_whitney_matches_scipy(tb):
    out = tb.run("hypothesis_test", {"value": "units", "group_col": "region", "group_a": "East",
                                     "group_b": "South", "test": "mann_whitney", "alternative": "greater"})
    df = pd.read_csv(CSV)
    expected = stats.mannwhitneyu(df.loc[df.region == "East", "units"], df.loc[df.region == "South", "units"],
                                  alternative="greater")
    assert out["statistic_name"] == "U" and out["alternative"] == "greater"
    assert out["statistic"] == pytest.approx(expected.statistic)
    assert out["p_value"] == pytest.approx(expected.pvalue)
    assert out["groups"]["East"]["n"] == 150


def test_column_stats_numeric_quantiles(tb):
    out = tb.run("column_stats", {"column": "units"})
    units = pd.read_csv(CSV)["units"]
    assert out["sum"] == 7900 and out["median"] == units.median()
    assert out["min"] == units.min() and out["max"] == units.max()


def test_plot_writes_png(tb, tmp_path):
    for args in (
        {"kind": "bar", "x": "region", "y": "revenue"},
        {"kind": "hist", "x": "revenue", "bins": 30},
        {"kind": "scatter", "x": "units", "y": "revenue", "filename": "units_vs_rev"},
    ):
        out = tb.run("plot", args)
        path = Path(out["path"])
        assert path.parent == tmp_path and path.stat().st_size > 0
    assert (tmp_path / "units_vs_rev.png").exists()


# -- recoverable errors ------------------------------------------------------ #


@pytest.mark.parametrize(
    "name,args,fragment",
    [
        ("nonexistent", {}, "Unknown tool 'nonexistent'"),
        ("describe_data", {"column": "units"}, "Unknown argument"),
        ("hypothesis_test", {"value": "revenue", "group_col": "region", "group_a": "North",
                             "group_b": "South", "test": "anova"}, "must be one of"),
        ("hypothesis_test", {"value": "revenue", "group_col": "unit_price", "group_a": "1",
                             "group_b": "2", "test": "welch_t"}, "continuous"),
        ("correlation", {"x": "units", "y": "nope"}, "Unknown column 'nope'"),
        ("plot", {"kind": "scatter", "x": "units", "y": "region"}, "not numeric"),
        ("column_stats", {"column": "profit"}, "Unknown column 'profit'"),
        ("column_stats", {}, "Missing required argument"),
        ("column_stats", {"column": "units", "extra": 1}, "Unknown argument"),
        ("column_stats", {"column": 3}, "must be of type string"),
        ("aggregate", {"group_by": ["region"], "value": "segment", "agg": "mean"}, "not numeric"),
        ("aggregate", {"group_by": ["revenue"], "value": "units", "agg": "mean"}, "continuous"),
        ("aggregate", {"group_by": ["region"], "value": "units", "agg": "mode"}, "must be one of"),
        ("aggregate", {"group_by": [], "value": "units", "agg": "mean"}, "at least 1"),
        ("correlation", {"x": "region", "y": "revenue"}, "not numeric"),
        ("hypothesis_test", {"value": "revenue", "group_col": "region", "group_a": "North",
                             "group_b": "Central", "test": "welch_t"}, "Group 'Central' not found"),
        ("hypothesis_test", {"value": "revenue", "group_col": "region", "group_a": "North",
                             "group_b": "North", "test": "welch_t"}, "must be different"),
        ("plot", {"kind": "bar", "x": "region"}, "requires 'y'"),
        ("plot", {"kind": "pie", "x": "region"}, "must be one of"),
        ("plot", {"kind": "hist", "x": "units", "filename": "../evil"}, "filename may contain"),
        ("plot", {"kind": "hist", "x": "units", "bins": 0}, "between"),
    ],
)
def test_bad_input_raises_tool_error(tb, name, args, fragment):
    with pytest.raises(ToolError, match=None) as exc:
        tb.run(name, args)
    assert fragment in str(exc.value)


def test_args_must_be_object(tb):
    with pytest.raises(ToolError, match="must be an object"):
        tb.run("describe_data", ["not", "a", "dict"])


def test_failed_plot_leaves_no_file(tb, tmp_path):
    with pytest.raises(ToolError):
        tb.run("plot", {"kind": "bar", "x": "region", "y": "segment"})
    assert list(tmp_path.iterdir()) == []
