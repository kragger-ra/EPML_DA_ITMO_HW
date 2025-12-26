"""Data preprocessing for customer churn prediction."""

from pathlib import Path
from typing import Any, cast

import pandas as pd
import yaml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


def load_params() -> dict[str, Any]:
    """Load parameters from params.yaml."""
    params_path = Path("params.yaml")
    with open(params_path) as f:
        return cast(dict[str, Any], yaml.safe_load(f))


def preprocess_data() -> None:
    """Preprocess customer churn dataset."""
    params = load_params()

    data_path = Path("data/raw/customer_churn.csv")
    df = pd.read_csv(data_path)

    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].mean())

    df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

    le = LabelEncoder()
    for col in params["features"]["categorical"]:
        if col in df.columns:
            df[col] = le.fit_transform(df[col].astype(str))

    features = params["features"]["categorical"] + params["features"]["numerical"]
    X = df[features]
    y = df["Churn"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=params["data"]["test_size"],
        random_state=params["data"]["random_state"],
    )

    processed_dir = Path("data/processed")
    processed_dir.mkdir(parents=True, exist_ok=True)

    X_train.to_csv(processed_dir / "X_train.csv", index=False)
    X_test.to_csv(processed_dir / "X_test.csv", index=False)
    y_train.to_csv(processed_dir / "y_train.csv", index=False)
    y_test.to_csv(processed_dir / "y_test.csv", index=False)

    print("Data preprocessed successfully!")
    print(f"Train size: {X_train.shape}")
    print(f"Test size: {X_test.shape}")


if __name__ == "__main__":
    preprocess_data()
