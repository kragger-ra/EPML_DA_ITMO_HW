"""Tests for reproducibility of ML pipeline."""

import json
from pathlib import Path


def test_metrics_file_exists():
    """Test that metrics file exists."""
    metrics_path = Path("metrics.json")
    assert metrics_path.exists(), "metrics.json file not found"


def test_metrics_values():
    """Test that metrics are within expected ranges."""
    metrics_path = Path("metrics.json")
    with open(metrics_path) as f:
        metrics = json.load(f)

    assert "accuracy" in metrics, "accuracy metric not found"
    assert "roc_auc" in metrics, "roc_auc metric not found"
    assert "f1_score" in metrics, "f1_score metric not found"

    assert 0 <= metrics["accuracy"] <= 1, "accuracy out of range"
    assert 0 <= metrics["roc_auc"] <= 1, "roc_auc out of range"
    assert 0 <= metrics["f1_score"] <= 1, "f1_score out of range"


def test_model_exists():
    """Test that model file exists."""
    model_path = Path("models/churn_model.pkl")
    assert model_path.exists(), "Model file not found"
    assert model_path.stat().st_size > 0, "Model file is empty"


def test_processed_data_exists():
    """Test that processed data files exist."""
    processed_dir = Path("data/processed")
    assert processed_dir.exists(), "Processed data directory not found"

    required_files = [
        "X_train.csv",
        "X_test.csv",
        "y_train.csv",
        "y_test.csv",
    ]

    for file_name in required_files:
        file_path = processed_dir / file_name
        assert file_path.exists(), f"{file_name} not found"
        assert file_path.stat().st_size > 0, f"{file_name} is empty"
