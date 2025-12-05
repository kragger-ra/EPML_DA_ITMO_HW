# Отчет о выполнении HW_2

## 1. Настройка DVC для данных (4 балла)

### Установка и инициализация

Зависимости добавлены в `pixi.toml`:

```bash
pixi add dvc lightgbm
pixi install
```

Инициализация DVC:

```bash
.pixi/envs/default/python.exe -m dvc init
```

### Remote storage

Локальное хранилище: `N:\HW\dvc-storage`

```bash
.pixi/envs/default/python.exe -m dvc remote add -d local "N:/HW/dvc-storage"
```

Конфигурация в `.dvc/config`:

```ini
[core]
    remote = local
['remote "local"']
    url = N:/HW/dvc-storage
```

### Версионирование данных

Датасет: Telco Customer Churn (IBM)

Скрипт загрузки: `src/data/download_data.py`

```bash
.pixi/envs/default/python.exe src/data/download_data.py
.pixi/envs/default/python.exe -m dvc add data/raw/customer_churn.csv
.pixi/envs/default/python.exe -m dvc push
```

Создан файл `data/raw/customer_churn.csv.dvc`:

```yaml
outs:
- md5: 3b0bfab28a8101b4e4fdd08025a5c235
  size: 970457
  hash: md5
  path: customer_churn.csv
```

## 2. Настройка DVC для моделей (3 балла)

### DVC Pipeline

Файл `dvc.yaml` содержит два этапа:
1. `preprocess` - препроцессинг данных
2. `train` - обучение модели LightGBM

```bash
.pixi/envs/default/python.exe -m dvc repro
```

### Параметры модели

Файл `params.yaml`:

```yaml
model:
  type: lightgbm
  params:
    num_leaves: 31
    learning_rate: 0.05
    n_estimators: 100
    max_depth: -1
    min_child_samples: 20
    subsample: 0.8
    colsample_bytree: 0.8
    random_state: 42

data:
  test_size: 0.2
  random_state: 42

features:
  categorical:
    - gender
    - Partner
    - Dependents
    - PhoneService
    - InternetService
    - Contract
  numerical:
    - tenure
    - MonthlyCharges
    - TotalCharges
```

### Метрики

Файл `metrics.json`:

```json
{
    "accuracy": 0.8034066713981547,
    "roc_auc": 0.8515622573933567,
    "f1_score": 0.5758039816232772
}
```

Команды для просмотра:

```bash
.pixi/envs/default/python.exe -m dvc metrics show
.pixi/envs/default/python.exe -m dvc dag
```

### Версионирование модели

Модель версионируется автоматически через outputs в `dvc.yaml`:

```bash
.pixi/envs/default/python.exe -m dvc push
```

Модель сохранена в `models/churn_model.pkl` и загружена в remote storage.

## 3. Воспроизводимость (2 балла)

### Зависимости

Зафиксированы в `pixi.toml` и `pixi.lock`:

- dvc >= 3.64.1
- lightgbm >= 4.6.0
- pandas >= 2.0.0
- scikit-learn >= 1.3.0

### Инструкции по воспроизведению

```bash
# 1. Клонировать репозиторий
git clone <repo-url>
cd "2025 AITH 1-3_EPML_DA_ITMO"
git checkout HW_2

# 2. Установить зависимости
pixi install

# 3. Настроить DVC remote (если требуется)
.pixi/envs/default/python.exe -m dvc remote modify local url <путь>

# 4. Загрузить данные
.pixi/envs/default/python.exe -m dvc pull

# 5. Запустить pipeline
.pixi/envs/default/python.exe -m dvc repro

# 6. Посмотреть результаты
.pixi/envs/default/python.exe -m dvc metrics show
```

### Docker

Обновлен `Dockerfile` для поддержки DVC. Сборка и запуск:

```bash
docker build -t ds-project:hw2 .
docker run -it --rm -v ${PWD}:/workspace ds-project:hw2
```

## 4. Структура проекта после HW_2

```
.
├── .dvc/
│   ├── config
│   └── .gitignore
├── data/
│   ├── raw/
│   │   ├── customer_churn.csv.dvc
│   │   └── .gitkeep
│   └── processed/
│       ├── X_train.csv
│       ├── X_test.csv
│       ├── y_train.csv
│       └── y_test.csv
├── models/
│   └── churn_model.pkl
├── src/
│   ├── data/
│   │   ├── download_data.py
│   │   └── preprocess.py
│   └── models/
│       └── train.py
├── dvc.yaml
├── params.yaml
├── metrics.json
└── REPORT_HW2.md
```

## Выводы

DVC обеспечивает:
1. Версионирование больших файлов данных отдельно от кода
2. Хранение моделей в remote storage
3. Воспроизводимые ML pipelines
4. Отслеживание параметров и метрик экспериментов
5. Упрощение совместной работы над ML проектами

Все требования HW_2 выполнены.
