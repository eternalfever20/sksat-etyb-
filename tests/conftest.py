from pathlib import Path

import pandas as pd
import pytest

from dataset.loader import load_dataset


@pytest.fixture
def demo():
    path = Path(__file__).parents[1] / "data/demo_sales.csv"
    return load_dataset(path.name, path.read_bytes())


@pytest.fixture
def simple_frame():
    frame = pd.DataFrame(
        {
            "value": [1, 2, 3, 100, None],
            "group": ["a", "a", "b", "b", "b"],
            "day": pd.to_datetime(
                ["2025-01-01", "2025-01-02", "2025-02-01", "2025-02-02", None]
            ),
            "flag": [True, False, True, False, None],
        }
    )
    return frame, {
        "value": "numeric",
        "group": "categorical",
        "day": "datetime",
        "flag": "boolean",
    }
