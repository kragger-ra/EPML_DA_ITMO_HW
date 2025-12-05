"""Train LightGBM model for customer churn prediction."""

import json
from pathlib import Path
from typing import Any, cast

import joblib
import lightgbm as lgb
import pandas as pd
import yaml
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score


def load_params() -> dict[str, Any]:
    """Load parameters from params.yaml."""
    params_path = Path("params.yaml")
    with open(params_path) as f:
        return cast(dict[str, Any], yaml.safe_load(f))


def train_model() -> tuple[Any, dict[str, float]]:
    """Train LightGBM model."""
    params = load_params()

    processed_dir = Path("data/processed")
    X_train = pd.read_csv(processed_dir / "X_train.csv")
    X_test = pd.read_csv(processed_dir / "X_test.csv")
    y_train = pd.read_csv(processed_dir / "y_train.csv").values.ravel()
    y_test = pd.read_csv(processed_dir / "y_test.csv").values.ravel()

    model = lgb.LGBMClassifier(**params["model"]["params"])
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "roc_auc": float(roc_auc_score(y_test, y_pred_proba)),
        "f1_score": float(f1_score(y_test, y_pred)),
    }

    models_dir = Path("models")
    models_dir.mkdir(parents=True, exist_ok=True)
    model_path = models_dir / "churn_model.pkl"
    joblib.dump(model, model_path)

    metrics_path = Path("metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=4)

    print("Model trained successfully!")
    print(f"Metrics: {metrics}")
    print(f"Model saved to: {model_path}")

    return model, metrics


if __name__ == "__main__":
    train_model()
