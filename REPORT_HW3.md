# Отчет о выполнении HW_3

## 1. Настройка MLflow (4 балла)

### Установка зависимостей

Добавлены зависимости в `pixi.toml`:

```toml
mlflow = ">=2.9.0,<3.0"
xgboost = ">=2.0.0,<3.0"
catboost = ">=1.2.0,<2.0"
optuna = ">=3.5.0,<4.0"
pyyaml = ">=6.0.0,<7.0"
```

Установка:

```bash
pixi install
```

### Настройка tracking URI

MLflow настроен на локальное хранилище в `src/utils/mlflow_utils.py`:

```python
def setup_mlflow(
    tracking_uri: str = "./mlruns",
    experiment_name: str = "customer-churn-prediction",
) -> None:
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)
```

Конфигурация:
- **Tracking URI**: `./mlruns`
- **Experiment Name**: `customer-churn-prediction`
- **Backend Store**: SQLite (автоматически)
- **Artifact Store**: Локальная файловая система

### Команды для работы

Добавлены задачи в `pixi.toml`:

```bash
pixi run mlflow-ui              # UI на http://127.0.0.1:5000
pixi run run-experiments
```

## 2. Проведение экспериментов (4 балла)

### Реализованные эксперименты

Создано 16 экспериментов с различными алгоритмами в `experiments/experiment_configs.py`:

#### LightGBM (4 эксперимента)

- **lightgbm_baseline** - базовая конфигурация (num_leaves: 31, lr: 0.05, n_estimators: 100)
- **lightgbm_high_lr** - увеличенный learning rate (0.1)
- **lightgbm_deep** - глубокие деревья (num_leaves: 63, max_depth: 10, n_estimators: 150)
- **lightgbm_regularized** - с L1/L2 регуляризацией (reg_alpha: 0.1, reg_lambda: 0.1)

#### XGBoost (3 эксперимента)

- **xgboost_baseline** - базовая конфигурация (max_depth: 6, lr: 0.05)
- **xgboost_shallow** - мелкие деревья (max_depth: 3, n_estimators: 150)
- **xgboost_deep** - глубокие деревья (max_depth: 10, lr: 0.03, n_estimators: 200)

#### CatBoost (2 эксперимента)

- **catboost_baseline** - базовая конфигурация (depth: 6, lr: 0.05, iterations: 100)
- **catboost_aggressive** - агрессивное обучение (depth: 8, lr: 0.1, iterations: 150)

#### Random Forest (3 эксперимента)

- **random_forest_baseline** - базовая конфигурация (n_estimators: 100, max_depth: 10)
- **random_forest_deep** - глубокий лес (n_estimators: 200, max_depth: 20)
- **random_forest_shallow** - мелкий лес (max_depth: 5, min_samples_split: 10)

#### Gradient Boosting (2 эксперимента)

- **gradient_boosting_baseline** - базовая конфигурация (n_estimators: 100, lr: 0.1)
- **gradient_boosting_conservative** - консервативное обучение (lr: 0.05, n_estimators: 150)

#### Logistic Regression (2 эксперимента)

- **logistic_regression_l2** - L2 регуляризация (C: 1.0)
- **logistic_regression_strong_reg** - сильная регуляризация (C: 0.1)

### Логирование

**Метрики** - для каждого эксперимента:

```python
metrics = {
    "accuracy": accuracy_score(y_test, y_pred),
    "precision": precision_score(y_test, y_pred),
    "recall": recall_score(y_test, y_pred),
    "f1_score": f1_score(y_test, y_pred),
    "roc_auc": roc_auc_score(y_test, y_pred_proba),
}
```

**Параметры** - автоматическое логирование:

```python
mlflow.log_param("model_type", model_type)
mlflow.log_param("train_size", len(X_train))
mlflow.log_param("test_size", len(X_test))
mlflow.log_param("n_features", X_train.shape[1])
log_params_from_dict(model_params, prefix="model.")
```

**Артефакты** - сохранение:

```python
mlflow.sklearn.log_model(model, "model")
mlflow.log_artifact(str(model_path))
mlflow.log_artifact(str(metrics_path))
```

### Запуск экспериментов

Запуск всех экспериментов:

```bash
pixi run python experiments/run_experiments.py
```

Запуск отдельной модели:

```bash
pixi run python src/models/train_mlflow.py lightgbm
pixi run python src/models/train_mlflow.py xgboost
pixi run python src/models/train_mlflow.py catboost
```

### Система сравнения

Реализованы функции в `src/utils/mlflow_utils.py`:

**compare_runs()** - сравнение нескольких run'ов по метрикам

```python
def compare_runs(run_ids: list[str], metrics: list[str]) -> Dict[str, Dict[str, float]]:
    """Compare multiple runs based on specified metrics."""
```

**get_best_run()** - получение лучшего run'а по метрике

```python
def get_best_run(experiment_name: str, metric: str = "roc_auc") -> Optional[Run]:
    """Get the best run from an experiment based on a metric."""
```

### Фильтрация в MLflow UI

Примеры фильтрации:
- По модели: `params.model_type = "lightgbm"`
- По метрикам: `metrics.roc_auc > 0.85`
- По параметрам: `params.model.learning_rate > 0.05`

### Скриншоты

- `screenshots/hw3_mlflow_main.png` - главная страница MLflow UI
- `screenshots/hw3_experiments_list.png` - список всех run'ов
- `screenshots/hw3_comparison.png` - сравнение моделей
- `screenshots/hw3_parallel_coords.png` - Parallel Coordinates Plot
- `screenshots/hw3_scatter_plot.png` - Scatter plot метрик

## 3. Интеграция с кодом (2 балла)

### Утилиты MLflow

Создан модуль `src/utils/mlflow_utils.py` с 7 функциями:

1. **setup_mlflow()** - настройка tracking URI и experiment
2. **log_params_from_dict()** - логирование параметров из словаря
3. **log_metrics_from_dict()** - логирование метрик из словаря
4. **mlflow_run()** - декоратор для автоматического логирования
5. **load_experiment_config()** - загрузка конфигурации из YAML
6. **get_best_run()** - получение лучшего run'а по метрике
7. **compare_runs()** - сравнение нескольких run'ов

### Декоратор для автоматического логирования

Пример использования:

```python
from src.utils.mlflow_utils import mlflow_run

@mlflow_run(run_name="my_experiment", log_model=True)
def train_model(params):
    model = LGBMClassifier(**params)
    model.fit(X_train, y_train)
    return model, metrics
```

Декоратор автоматически:
- Создает MLflow run
- Логирует параметры
- Сохраняет модель
- Обрабатывает контекст

### Контекстные менеджеры

Использование контекстного менеджера:

```python
with mlflow.start_run(run_name=run_name):
    mlflow.log_param("model_type", model_type)
    log_params_from_dict(model_params, prefix="model.")

    model = create_model(model_type, model_params)
    model.fit(X_train, y_train)

    log_metrics_from_dict(metrics)
    mlflow.sklearn.log_model(model, "model")
```

### Model Factory

Создан модуль `src/models/model_factory.py` для унификации:

```python
def create_model(model_type: str, params: Dict[str, Any] = None) -> Any:
    """Create a machine learning model based on type."""
    models = {
        "lightgbm": lambda: lgb.LGBMClassifier(**params),
        "xgboost": lambda: xgb.XGBClassifier(**params),
        "catboost": lambda: CatBoostClassifier(**params, verbose=False),
        "random_forest": lambda: RandomForestClassifier(**params),
        "gradient_boosting": lambda: GradientBoostingClassifier(**params),
        "logistic_regression": lambda: LogisticRegression(**params, max_iter=1000),
        "decision_tree": lambda: DecisionTreeClassifier(**params),
    }
    return models[model_type]()
```

### Тренировочный код

`src/models/train_mlflow.py` - функция `train_model_with_mlflow()`:
- Загрузка данных
- Создание модели через factory
- Обучение с логированием
- Сохранение артефактов

### Автоматизация

`experiments/run_experiments.py` - автоматический запуск всех 16 экспериментов с сохранением результатов в CSV.

## 4. Отчет о проделанной работе (2 балла)

### Архитектура решения

```
.
├── experiments/
│   ├── experiment_configs.py    # 16 конфигураций
│   ├── run_experiments.py       # Автозапуск
│   └── experiment_results.csv   # Результаты
├── src/
│   ├── models/
│   │   ├── train_mlflow.py     # Обучение с MLflow
│   │   └── model_factory.py    # Фабрика моделей
│   └── utils/
│       └── mlflow_utils.py     # Утилиты MLflow
└── mlruns/                      # MLflow tracking store
```

### Инструкции по воспроизведению

**1. Установка**

```bash
pixi install
```

**2. Подготовка данных**

```bash
pixi run dvc pull
# или
pixi run python src/data/download_data.py
pixi run dvc repro preprocess
```

**3. Запуск экспериментов**

```bash
# Все эксперименты
pixi run python experiments/run_experiments.py

# Отдельные модели
pixi run python src/models/train_mlflow.py lightgbm
pixi run python src/models/train_mlflow.py xgboost
pixi run python src/models/train_mlflow.py catboost
```

**4. Просмотр результатов**

```bash
pixi run mlflow-ui
```

### Технические детали

**MLflow Configuration:**
- Backend Store: SQLite (`mlruns/`)
- Artifact Store: локальная файловая система
- Tracking Server: file-based
- Experiment Name: `customer-churn-prediction`

**Поддерживаемые модели:**
LightGBM, XGBoost, CatBoost, Random Forest, Gradient Boosting, Logistic Regression, Decision Tree

**Метрики:**
accuracy, precision, recall, f1_score, roc_auc (основная метрика для сравнения)

### Интеграция с DVC

Версионирование лучшей модели:

```bash
pixi run dvc repro
git add models/*.pkl.dvc
git commit -m "Add best model from MLflow experiments"
```
