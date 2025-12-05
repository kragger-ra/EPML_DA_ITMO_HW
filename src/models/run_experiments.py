"""Quick experiments runner."""

from models.train_mlflow import (
    load_data,
    mlflow,
    train_catboost,
    train_decision_tree,
    train_lightgbm,
    train_logistic_regression,
    train_random_forest,
    train_xgboost,
)

if __name__ == "__main__":
    mlflow.set_experiment("churn-prediction")
    mlflow.set_tracking_uri("file:./mlruns")

    X_train, X_test, y_train, y_test = load_data()

    print("Running 18 experiments...")
    print("=" * 80)

    # Logistic Regression - 3 experiments
    print("[1/18] LogisticRegression (C=1.0)")
    train_logistic_regression(
        X_train, y_train, X_test, y_test, {"max_iter": 1000, "random_state": 42}
    )

    print("[2/18] LogisticRegression (C=0.1)")
    train_logistic_regression(
        X_train,
        y_train,
        X_test,
        y_test,
        {"max_iter": 1000, "C": 0.1, "random_state": 42},
    )

    print("[3/18] LogisticRegression (C=10.0)")
    train_logistic_regression(
        X_train,
        y_train,
        X_test,
        y_test,
        {"max_iter": 1000, "C": 10.0, "random_state": 42},
    )

    # Decision Tree - 3 experiments
    print("[4/18] DecisionTree (max_depth=5)")
    train_decision_tree(
        X_train, y_train, X_test, y_test, {"max_depth": 5, "random_state": 42}
    )

    print("[5/18] DecisionTree (max_depth=10)")
    train_decision_tree(
        X_train, y_train, X_test, y_test, {"max_depth": 10, "random_state": 42}
    )

    print("[6/18] DecisionTree (max_depth=20)")
    train_decision_tree(
        X_train, y_train, X_test, y_test, {"max_depth": 20, "random_state": 42}
    )

    # Random Forest - 3 experiments
    print("[7/18] RandomForest (n_estimators=50)")
    train_random_forest(
        X_train,
        y_train,
        X_test,
        y_test,
        {"n_estimators": 50, "max_depth": 10, "random_state": 42},
    )

    print("[8/18] RandomForest (n_estimators=100)")
    train_random_forest(
        X_train,
        y_train,
        X_test,
        y_test,
        {"n_estimators": 100, "max_depth": 15, "random_state": 42},
    )

    print("[9/18] RandomForest (n_estimators=200)")
    train_random_forest(
        X_train,
        y_train,
        X_test,
        y_test,
        {"n_estimators": 200, "max_depth": 20, "random_state": 42},
    )

    # LightGBM - 3 experiments
    print("[10/18] LightGBM (lr=0.05)")
    train_lightgbm(
        X_train,
        y_train,
        X_test,
        y_test,
        {
            "num_leaves": 31,
            "learning_rate": 0.05,
            "n_estimators": 100,
            "random_state": 42,
        },
    )

    print("[11/18] LightGBM (lr=0.1)")
    train_lightgbm(
        X_train,
        y_train,
        X_test,
        y_test,
        {
            "num_leaves": 50,
            "learning_rate": 0.1,
            "n_estimators": 150,
            "random_state": 42,
        },
    )

    print("[12/18] LightGBM (lr=0.01)")
    train_lightgbm(
        X_train,
        y_train,
        X_test,
        y_test,
        {
            "num_leaves": 20,
            "learning_rate": 0.01,
            "n_estimators": 200,
            "random_state": 42,
        },
    )

    # XGBoost - 3 experiments
    print("[13/18] XGBoost (depth=6)")
    train_xgboost(
        X_train,
        y_train,
        X_test,
        y_test,
        {"max_depth": 6, "learning_rate": 0.1, "n_estimators": 100, "random_state": 42},
    )

    print("[14/18] XGBoost (depth=8)")
    train_xgboost(
        X_train,
        y_train,
        X_test,
        y_test,
        {
            "max_depth": 8,
            "learning_rate": 0.05,
            "n_estimators": 150,
            "random_state": 42,
        },
    )

    print("[15/18] XGBoost (depth=4)")
    train_xgboost(
        X_train,
        y_train,
        X_test,
        y_test,
        {"max_depth": 4, "learning_rate": 0.2, "n_estimators": 50, "random_state": 42},
    )

    # CatBoost - 3 experiments
    print("[16/18] CatBoost (depth=6)")
    train_catboost(
        X_train,
        y_train,
        X_test,
        y_test,
        {"depth": 6, "learning_rate": 0.1, "iterations": 100, "random_state": 42},
    )

    print("[17/18] CatBoost (depth=8)")
    train_catboost(
        X_train,
        y_train,
        X_test,
        y_test,
        {"depth": 8, "learning_rate": 0.05, "iterations": 150, "random_state": 42},
    )

    print("[18/18] CatBoost (depth=4)")
    train_catboost(
        X_train,
        y_train,
        X_test,
        y_test,
        {"depth": 4, "learning_rate": 0.2, "iterations": 50, "random_state": 42},
    )

    print("=" * 80)
    print("All 18 experiments completed!")
    print("View results: mlflow ui --backend-store-uri file:./mlruns")
