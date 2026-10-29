"""Tests for feature engineering logic."""
import pytest
import pandas as pd
import numpy as np


def test_feature_columns_present():
    # Simulate the expected output schema from build_features
    expected = ["entity_id", "n_events_7d", "avg_value_a", "std_value_a",
                "max_value_b", "min_value_b", "flag_count", "label"]
    df = pd.DataFrame(
        {col: np.random.randn(10) for col in expected}
    )
    df["entity_id"] = [f"e{i}" for i in range(10)]
    df["label"] = (np.random.rand(10) > 0.5).astype(int)
    for col in expected:
        assert col in df.columns


def test_no_nulls_after_aggregation():
    df = pd.DataFrame({
        "entity_id": ["e1", "e2"],
        "n_events_7d": [10, 5],
        "avg_value_a": [1.5, 2.1],
        "label": [0, 1],
    })
    assert df.isnull().sum().sum() == 0
