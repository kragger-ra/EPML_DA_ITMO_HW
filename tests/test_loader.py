"""Tests for data loader."""

import pandas as pd


def test_example():
    """Example test."""
    df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    assert len(df) == 3
    assert list(df.columns) == ["a", "b"]
