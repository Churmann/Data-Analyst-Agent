"""Sanity checks that the sample dataset has the shape and means it was designed with.

Expected values are worked out by hand from the cell definitions
(mean revenue per cell = mean_units * unit_price), not read from the CSV.
"""

from pathlib import Path

import pandas as pd
import pytest
import yaml

ROOT = Path(__file__).resolve().parent.parent
CSV = ROOT / "data" / "sales.csv"
DICTIONARY = ROOT / "data" / "data_dictionary.yaml"

EXPECTED_REGION_MEAN_REVENUE = {
    "North": 1417 / 3,  # (245 + 756 + 416) / 3
    "South": 370.5,  # (176 + 600 + 335.5) / 3
    "East": 1688 / 3,  # (312 + 900 + 476) / 3
    "West": 1247.5 / 3,  # (211.5 + 664 + 372) / 3
}
EXPECTED_SEGMENT_MEAN_REVENUE = {
    "Consumer": 236.125,
    "Corporate": 730.0,
    "SMB": 399.875,
}
EXPECTED_OVERALL_MEAN_REVENUE = 5464 / 12


@pytest.fixture(scope="module")
def df() -> pd.DataFrame:
    return pd.read_csv(CSV)


@pytest.fixture(scope="module")
def dictionary() -> dict:
    return yaml.safe_load(DICTIONARY.read_text())


def test_shape_and_columns(df):
    assert df.shape == (600, 6)
    assert list(df.columns) == ["order_id", "region", "segment", "units", "unit_price", "revenue"]
    assert df["order_id"].is_unique


def test_missing_revenue_only(df):
    assert df["revenue"].isna().sum() == 24
    assert df.drop(columns="revenue").notna().all().all()


def test_revenue_is_units_times_price(df):
    present = df.dropna(subset=["revenue"])
    assert (present["revenue"] - present["units"] * present["unit_price"]).abs().max() < 1e-9


def test_balanced_groups(df):
    assert df.groupby(["region", "segment"]).size().eq(50).all()


@pytest.mark.parametrize("complete", [False, True], ids=["observed", "reconstructed"])
def test_group_means(df, complete):
    revenue = df["units"] * df["unit_price"] if complete else df["revenue"]
    by_region = revenue.groupby(df["region"]).mean()
    by_segment = revenue.groupby(df["segment"]).mean()
    for k, v in EXPECTED_REGION_MEAN_REVENUE.items():
        assert by_region[k] == pytest.approx(v)
    for k, v in EXPECTED_SEGMENT_MEAN_REVENUE.items():
        assert by_segment[k] == pytest.approx(v)
    assert revenue.mean() == pytest.approx(EXPECTED_OVERALL_MEAN_REVENUE)


def test_dictionary_matches_csv(df, dictionary):
    assert list(dictionary["columns"]) == list(df.columns)
    assert dictionary["row_count"] == len(df)
    for name, spec in dictionary["columns"].items():
        assert spec["nullable"] == bool(df[name].isna().any()), name
        if "allowed_values" in spec:
            assert set(df[name]) == set(spec["allowed_values"]), name
