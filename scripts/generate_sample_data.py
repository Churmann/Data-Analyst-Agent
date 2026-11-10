"""Generate the deterministic sample sales dataset (data/sales.csv).

The data is built constructively rather than sampled, so every group mean is
known exactly in advance:

* 4 regions x 3 segments = 12 cells, 50 orders per cell (600 rows).
* Within a cell, unit_price is fixed and units = mean_units + offset, where the
  offsets are +/-1..+/-5, each repeated 5 times. Offsets are symmetric, so the
  cell's mean units is exactly ``mean_units`` and its mean revenue is exactly
  ``mean_units * unit_price``.
* revenue = units * unit_price for every row.
* In each cell, revenue is blanked for one order at offset +d and one at -d.
  Removing a symmetric pair leaves the mean unchanged, so group means are the
  same whether or not missing values are skipped. 12 cells x 2 = 24 missing.
* Rows are shuffled with a fixed seed and order_ids assigned afterwards.

Re-running this script always produces a byte-identical file.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

SEED = 42
ROWS_PER_CELL = 50
OFFSETS = [o for o in (-5, -4, -3, -2, -1, 1, 2, 3, 4, 5) for _ in range(5)]

REGIONS = ["North", "South", "East", "West"]
SEGMENTS = ["Consumer", "Corporate", "SMB"]

# (region, segment) -> (mean units per order, unit price)
CELLS: dict[tuple[str, str], tuple[int, float]] = {
    ("North", "Consumer"): (10, 24.50),
    ("South", "Consumer"): (8, 22.00),
    ("East", "Consumer"): (12, 26.00),
    ("West", "Consumer"): (9, 23.50),
    ("North", "Corporate"): (18, 42.00),
    ("South", "Corporate"): (15, 40.00),
    ("East", "Corporate"): (20, 45.00),
    ("West", "Corporate"): (16, 41.50),
    ("North", "SMB"): (13, 32.00),
    ("South", "SMB"): (11, 30.50),
    ("East", "SMB"): (14, 34.00),
    ("West", "SMB"): (12, 31.00),
}

OUTPUT = Path(__file__).resolve().parent.parent / "data" / "sales.csv"


def build() -> pd.DataFrame:
    rows = []
    for cell_idx, ((region, segment), (mean_units, price)) in enumerate(CELLS.items()):
        d = 1 + cell_idx % 5  # which symmetric pair loses its revenue
        blanked = {+d: False, -d: False}
        for offset in OFFSETS:
            units = mean_units + offset
            revenue: float | None = round(units * price, 2)
            if offset in blanked and not blanked[offset]:
                blanked[offset] = True
                revenue = None
            rows.append(
                {
                    "region": region,
                    "segment": segment,
                    "units": units,
                    "unit_price": price,
                    "revenue": revenue,
                }
            )

    df = pd.DataFrame(rows)
    rng = np.random.default_rng(SEED)
    df = df.iloc[rng.permutation(len(df))].reset_index(drop=True)
    df.insert(0, "order_id", [f"ORD-{i:04d}" for i in range(1, len(df) + 1)])
    return df


def main() -> None:
    df = build()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT, index=False, float_format="%.2f", lineterminator="\n")
    print(f"wrote {len(df)} rows ({df['revenue'].isna().sum()} missing revenue) to {OUTPUT}")


if __name__ == "__main__":
    main()
