"""Quick lightweight experiments."""

import sys
from pathlib import Path

import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.tree import DecisionTreeClassifier

sys.path.insert(0, str(Path(__file__).parent.parent))


def load_data():
    """Load processed data."""
    processed_dir = Path("data/processed")
    X_train = pd.read_csv(processed_dir / "X_train.csv")
    X_test = pd.read_csv(processed_dir / "X_test.csv")
    y_train = pd.read_csv(processed_dir / "y_train.csv").values.ravel()
    y_test = pd.read_csv(processed_dir / "y_test.csv").values.ravel()
    return X_train, X_test, y_train, y_test


def evaluate_model(model, X_test, y_test):
    """Evaluate model and return metrics."""
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    return {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred)),
        "recall": float(recall_score(y_test, y_pred)),
        "f1_score": float(f1_score(y_test, y_pred)),
        "roc_auc": float(roc_auc_score(y_test, y_pred_proba)),
    }


if __name__ == "__main__":
    mlflow.set_experiment("churn-prediction")
    mlflow.set_tracking_uri("file:./mlruns")

    X_train, X_test, y_train, y_test = load_data()

    print("Running 15 lightweight experiments...")
    print("=" * 80)

    experiments = [
        # Logistic Regression - 5 experiments
        (
            "LogisticRegression_C1.0",
            LogisticRegression,
            {"max_iter": 1000, "C": 1.0, "random_state": 42},
        ),
        (
            "LogisticRegression_C0.1",
            LogisticRegression,
            {"max_iter": 1000, "C": 0.1, "random_state": 42},
        ),
        (
            "LogisticRegression_C10",
            LogisticRegression,
            {"max_iter": 1000, "C": 10.0, "random_state": 42},
        ),
        (
            "LogisticRegression_C0.01",
            LogisticRegression,
            {"max_iter": 1000, "C": 0.01, "random_state": 42},
        ),
        (
            "LogisticRegression_C100",
            LogisticRegression,
            {"max_iter": 1000, "C": 100.0, "random_state": 42},
        ),
        # Decision Tree - 5 experiments
        (
            "DecisionTree_depth5",
            DecisionTreeClassifier,
            {"max_depth": 5, "random_state": 42},
        ),
        (
            "DecisionTree_depth10",
            DecisionTreeClassifier,
            {"max_depth": 10, "random_state": 42},
        ),
        (
            "DecisionTree_depth15",
            DecisionTreeClassifier,
            {"max_depth": 15, "random_state": 42},
        ),
        (
            "DecisionTree_depth20",
            DecisionTreeClassifier,
            {"max_depth": 20, "random_state": 42},
        ),
        (
            "DecisionTree_depth3",
            DecisionTreeClassifier,
            {"max_depth": 3, "random_state": 42},
        ),
        # Random Forest - 5 experiments
        (
            "RandomForest_n50",
            RandomForestClassifier,
            {"n_estimators": 50, "max_depth": 10, "random_state": 42},
        ),
        (
            "RandomForest_n100",
            RandomForestClassifier,
            {"n_estimators": 100, "max_depth": 10, "random_state": 42},
        ),
        (
            "RandomForest_n150",
            RandomForestClassifier,
            {"n_estimators": 150, "max_depth": 15, "random_state": 42},
        ),
        (
            "RandomForest_n50_d5",
            RandomForestClassifier,
            {"n_estimators": 50, "max_depth": 5, "random_state": 42},
        ),
        (
            "RandomForest_n100_d15",
            RandomForestClassifier,
            {"n_estimators": 100, "max_depth": 15, "random_state": 42},
        ),
    ]

    for i, (name, model_class, params) in enumerate(experiments, 1):
        print(f"[{i}/15] {name}")
        with mlflow.start_run(run_name=name):
            model = model_class(**params)
            model.fit(X_train, y_train)

            metrics = evaluate_model(model, X_test, y_test)

            mlflow.log_params(params)
            mlflow.log_metrics(metrics)
            mlflow.sklearn.log_model(model, "model")

            print(
                f"  Metrics: acc={metrics['accuracy']:.3f}, roc_auc={metrics['roc_auc']:.3f}"
            )

    print("=" * 80)
    print("All 15 experiments completed!")
    print("View results: mlflow ui --backend-store-uri file:./mlruns")
