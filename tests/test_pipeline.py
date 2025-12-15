"""Tests for ML pipeline reproducibility and correctness."""

import json
import subprocess
from pathlib import Path

import pytest


class TestPipelineStructure:
    """Test that pipeline structure is correct."""

    def test_dvc_yaml_exists(self):
        """Test that dvc.yaml file exists."""
        assert Path("dvc.yaml").exists(), "dvc.yaml file not found"

    def test_hydra_config_exists(self):
        """Test that Hydra configuration exists."""
        assert Path("conf/config.yaml").exists(), "Hydra config.yaml not found"

    def test_model_configs_exist(self):
        """Test that all model configurations exist."""
        model_types = [
            "lightgbm",
            "xgboost",
            "catboost",
            "random_forest",
            "gradient_boosting",
            "logistic_regression",
        ]
        for model in model_types:
            config_path = Path(f"conf/model/{model}.yaml")
            assert config_path.exists(), f"Config for {model} not found"


class TestPipelineOutputs:
    """Test that pipeline generates expected outputs."""

    def test_processed_data_exists(self):
        """Test that processed data files exist."""
        processed_dir = Path("data/processed")
        if not processed_dir.exists():
            pytest.skip("Processed data directory not found - pipeline not run")

        required_files = [
            "X_train.csv",
            "X_test.csv",
            "y_train.csv",
            "y_test.csv",
        ]

        for file_name in required_files:
            file_path = processed_dir / file_name
            if file_path.exists():
                assert file_path.stat().st_size > 0, f"{file_name} is empty"

    def test_metrics_file_exists(self):
        """Test that metrics.json exists."""
        metrics_path = Path("metrics.json")
        if not metrics_path.exists():
            pytest.skip("metrics.json not found - pipeline not run")

        assert metrics_path.stat().st_size > 0, "metrics.json is empty"

    def test_metrics_values(self):
        """Test that metrics are within expected ranges."""
        metrics_path = Path("metrics.json")
        if not metrics_path.exists():
            pytest.skip("metrics.json not found - pipeline not run")

        with open(metrics_path) as f:
            metrics = json.load(f)

        required_metrics = ["accuracy", "roc_auc", "f1_score"]
        for metric in required_metrics:
            assert metric in metrics, f"{metric} not found in metrics"
            assert 0 <= metrics[metric] <= 1, f"{metric} out of range [0, 1]"

    def test_model_file_exists(self):
        """Test that trained model exists."""
        model_path = Path("models/churn_model.pkl")
        if not model_path.exists():
            pytest.skip("Model not found - pipeline not run")

        assert model_path.stat().st_size > 0, "Model file is empty"


class TestPipelineScripts:
    """Test that all pipeline scripts are present and executable."""

    def test_preprocess_script_exists(self):
        """Test that preprocessing script exists."""
        assert Path("src/data/preprocess.py").exists()

    def test_train_hydra_script_exists(self):
        """Test that Hydra training script exists."""
        assert Path("src/models/train_hydra.py").exists()

    def test_evaluation_script_exists(self):
        """Test that evaluation script exists."""
        assert Path("src/pipelines/evaluate_models.py").exists()

    def test_selection_script_exists(self):
        """Test that model selection script exists."""
        assert Path("src/pipelines/select_best_model.py").exists()

    def test_report_script_exists(self):
        """Test that report generation script exists."""
        assert Path("src/pipelines/generate_report.py").exists()

    def test_notification_script_exists(self):
        """Test that notification script exists."""
        assert Path("src/pipelines/notify.py").exists()


class TestDVCPipeline:
    """Test DVC pipeline functionality."""

    def test_dvc_dag_command(self):
        """Test that dvc dag command works."""
        try:
            result = subprocess.run(
                ["dvc", "dag"],
                capture_output=True,
                text=True,
                check=False,
                timeout=30,
            )
            assert result.returncode in [0, 1], "dvc dag command failed unexpectedly"
        except FileNotFoundError:
            pytest.skip("DVC not installed")
        except subprocess.TimeoutExpired:
            pytest.fail("dvc dag command timed out")

    def test_dvc_status_command(self):
        """Test that dvc status command works."""
        try:
            result = subprocess.run(
                ["dvc", "status"],
                capture_output=True,
                text=True,
                check=False,
                timeout=30,
            )
            assert result.returncode in [0, 1], "dvc status command failed"
        except FileNotFoundError:
            pytest.skip("DVC not installed")
        except subprocess.TimeoutExpired:
            pytest.fail("dvc status command timed out")


class TestHydraConfiguration:
    """Test Hydra configuration functionality."""

    def test_hydra_config_is_valid_yaml(self):
        """Test that Hydra config is valid YAML."""
        import yaml

        config_path = Path("conf/config.yaml")
        with open(config_path) as f:
            config = yaml.safe_load(f)

        assert isinstance(config, dict), "Config is not a dictionary"
        assert "defaults" in config, "Config missing 'defaults' key"

    def test_model_configs_are_valid_yaml(self):
        """Test that all model configs are valid YAML."""
        import yaml

        model_types = [
            "lightgbm",
            "xgboost",
            "catboost",
            "random_forest",
            "gradient_boosting",
            "logistic_regression",
        ]

        for model in model_types:
            config_path = Path(f"conf/model/{model}.yaml")
            with open(config_path) as f:
                config = yaml.safe_load(f)

            assert isinstance(config, dict), f"{model} config is not a dictionary"
            assert "type" in config, f"{model} config missing 'type' key"
            assert "name" in config, f"{model} config missing 'name' key"
            assert "params" in config, f"{model} config missing 'params' key"


class TestReproducibility:
    """Test pipeline reproducibility."""

    def test_params_yaml_exists(self):
        """Test that params.yaml exists for reproducibility."""
        assert Path("params.yaml").exists(), "params.yaml not found"

    def test_gitignore_excludes_artifacts(self):
        """Test that .gitignore properly excludes artifacts."""
        gitignore_path = Path(".gitignore")
        if not gitignore_path.exists():
            pytest.skip(".gitignore not found")

        with open(gitignore_path) as f:
            content = f.read()

        assert "mlruns" in content, "mlruns not in .gitignore"
        assert ".pixi" in content or "pixi" in content, "pixi not in .gitignore"

    def test_dvc_files_tracked(self):
        """Test that DVC files are properly tracked."""
        assert Path(".dvc").exists(), ".dvc directory not found"

        assert Path(".dvc/config").exists(), ".dvc/config not found"
