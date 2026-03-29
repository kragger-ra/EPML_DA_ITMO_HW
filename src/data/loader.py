"""Data loading utilities."""

from pathlib import Path

import pandas as pd


def load_csv(file_path: str | Path) -> pd.DataFrame:
    """Load data from CSV file.

    Args:
        file_path: Path to CSV file

    Returns:
        DataFrame with loaded data
    """
    return pd.read_csv(file_path)


def save_csv(df: pd.DataFrame, file_path: str | Path) -> None:
    """Save DataFrame to CSV file.

    Args:
        df: DataFrame to save
        file_path: Path where to save the file
    """
    df.to_csv(file_path, index=False)
