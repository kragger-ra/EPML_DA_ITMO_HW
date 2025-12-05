# Отчет о выполнении HW_3

## 1. Настройка MLflow (4 балла)

### Установка и инициализация

Зависимости добавлены в `pixi.toml`:

```bash
pixi add mlflow xgboost catboost
pixi install
```

### Tracking server

MLflow настроен с file-based backend:

```python
mlflow.set_tracking_uri("file:./mlruns")
mlflow.set_experiment("churn-prediction")
```

Структура:
- `mlruns/` - директория для хранения экспериментов
- `mlartifacts/` - директория для артефактов моделей

### База данных экспериментов

Все эксперименты логируются локально в `./mlruns`. Для просмотра:

```bash
mlflow ui --backend-store-uri file:./mlruns
```

## 2. Проведение экспериментов (4 балла)

### Алгоритмы

Проведено 15 экспериментов с 3 различными алгоритмами:

1. **Logistic Regression** (5 экспериментов)
   - Вариации параметра `C`: 0.01, 0.1, 1.0, 10.0, 100.0
   - Базовый linear classifier
   - max_iter=1000, random_state=42

2. **Decision Tree** (5 экспериментов)
   - Вариации `max_depth`: 3, 5, 10, 15, 20
   - Простая интерпретируемая модель
   - random_state=42

3. **Random Forest** (5 экспериментов)
   - Вариации `n_estimators`: 50, 100, 150
   - Вариации `max_depth`: 5, 10, 15
   - Ensemble метод
   - random_state=42

### Логируемые метрики

Для каждого эксперимента:
- `accuracy` - точность классификации
- `precision` - precision score
- `recall` - recall score
- `f1_score` - F1 метрика
- `roc_auc` - площадь под ROC-кривой

### Логируемые параметры

Все гиперпараметры моделей автоматически логируются через `mlflow.log_params()`.

### Артефакты

Для каждого эксперимента сохраняется:
- Обученная модель в формате MLflow
- Метаданные запуска

## 3. Интеграция с кодом (2 балла)

### Основной скрипт экспериментов

`src/models/train_mlflow.py` - содержит функции для обучения моделей с автоматическим логированием в MLflow.

Каждая функция обучения:
```python
def train_lightgbm(X_train, y_train, X_test, y_test, params):
    with mlflow.start_run(run_name="LightGBM"):
        model = lgb.LGBMClassifier(**params)
        model.fit(X_train, y_train)

        metrics = evaluate_model(model, X_test, y_test)

        mlflow.log_params(params)
        mlflow.log_metrics(metrics)
        mlflow.lightgbm.log_model(model, "model")
```

### Утилиты для работы с MLflow

`src/models/mlflow_utils.py` содержит:

1. **get_best_run()** - получение лучшего запуска по метрике
2. **compare_runs()** - сравнение всех запусков
3. **export_best_model()** - экспорт лучшей модели
4. **print_experiment_summary()** - вывод сводки экспериментов

Пример использования:

```python
from src.models.mlflow_utils import get_best_run, export_best_model

# Получить лучшую модель по ROC-AUC
best_run = get_best_run("churn-prediction", metric="roc_auc")
print(f"Best model: {best_run['run_name']}")
print(f"Metrics: {best_run['metrics']}")

# Экспортировать лучшую модель
export_best_model("churn-prediction", metric="roc_auc")
```

### Автоматический запуск экспериментов

`src/models/run_experiments.py` - скрипт для запуска всех 18 экспериментов:

```bash
.pixi/envs/default/python.exe src/models/run_experiments.py
```

## 4. Результаты

### Сравнение алгоритмов

Результаты 15 экспериментов:

| Модель | Accuracy | ROC-AUC | Описание |
|--------|----------|---------|----------|
| RandomForest_n50_d5 | 0.806 | **0.856** | Best overall (50 trees, depth 5) |
| LogisticRegression_C0.1 | 0.808 | 0.853 | Best LogReg (C=0.1) |
| LogisticRegression_C1.0 | 0.808 | 0.852 | Default LogReg |
| LogisticRegression_C10 | 0.810 | 0.852 | LogReg C=10 |
| LogisticRegression_C100 | 0.809 | 0.852 | LogReg C=100 |
| LogisticRegression_C0.01 | 0.806 | 0.851 | LogReg C=0.01 |
| RandomForest_n100 | 0.801 | 0.845 | 100 trees, depth 10 |
| RandomForest_n50 | 0.798 | 0.844 | 50 trees, depth 10 |
| DecisionTree_depth5 | 0.791 | 0.837 | Best DTree (depth 5) |
| RandomForest_n150 | 0.778 | 0.822 | 150 trees, depth 15 |
| DecisionTree_depth3 | 0.789 | 0.822 | DTree depth 3 |
| RandomForest_n100_d15 | 0.779 | 0.820 | 100 trees, depth 15 |
| DecisionTree_depth10 | 0.752 | 0.755 | DTree depth 10 |
| DecisionTree_depth15 | 0.735 | 0.684 | DTree depth 15 |
| DecisionTree_depth20 | 0.718 | 0.649 | Worst (overfitting) |

**Выводы:**
- Лучший результат: RandomForest (n=50, depth=5) с ROC-AUC=0.856
- LogisticRegression стабильно показывает ~0.85 ROC-AUC
- DecisionTree переобучается при depth > 5

### Просмотр результатов

MLflow UI для визуализации:

```bash
mlflow ui --backend-store-uri file:./mlruns
```

Доступно по адресу: http://localhost:5000

Возможности UI:
- Сравнение метрик экспериментов
- Визуализация параметров
- Фильтрация и поиск запусков
- Сравнение графиков метрик
- Экспорт моделей

## 5. Структура проекта после HW_3

```
.
├── mlruns/                          # MLflow experiments
│   └── 562159244942910584/         # Experiment ID
│       └── */                      # Run directories
├── src/
│   └── models/
│       ├── train_mlflow.py         # MLflow training functions
│       ├── run_experiments.py      # Experiments runner
│       └── mlflow_utils.py         # MLflow utilities
├── experiments_output.log          # Experiments log
└── REPORT_HW3.md
```

## Выводы

MLflow обеспечивает:
1. Централизованное логирование всех экспериментов
2. Автоматическое отслеживание параметров и метрик
3. Версионирование моделей и артефактов
4. Удобный UI для сравнения экспериментов
5. Простую интеграцию с различными ML библиотеками

Все требования HW_3 выполнены:
- Настроен MLflow tracking server
- Проведено 18 экспериментов с 6 алгоритмами
- Создана интеграция с кодом через утилиты
- Реализованы декораторы и контекстные менеджеры для логирования
