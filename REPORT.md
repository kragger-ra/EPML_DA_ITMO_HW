# Очет о выполнении HW_1

## 1. Структура проекта (2 балла)

### Создание структуры папок

```
.
├── data/
│   ├── raw/
│   ├── processed/
│   └── interim/
├── notebooks/
├── src/
│   ├── data/
│   ├── features/
│   ├── models/
│   └── visualization/
├── tests/
├── reports/
│   └── figures/
└── models/
```

## 2. Качество кода (2 балла)

### Pre-commit hooks

Настроены pre-commit hooks в файле `.pre-commit-config.yaml`.

**Базовые проверки:**
- Удаление trailing whitespace
- Проверка конца файла
- Валидация YAML, JSON, TOML
- Проверка больших файлов (макс. 10MB)
- Обнаружение конфликтов слияния
- Детекция приватных ключей

**Форматирование и линтинг:**
- **Black** (v23.12.1)
- **isort** (v5.13.2)
- **Ruff** (v0.1.9)
- **MyPy** (v1.7.1)
- **Bandit** (v1.7.6)

Установка:
```bash
pixi run pre-commit install
```

### Конфигурационные файлы

**pyproject.toml** :

1. **Black:**
   - line-length: 88
   - target-version: Python 3.10, 3.11
   - исключение специфичных директорий

2. **isort:**
   - profile: "black"
   - line_length: 88

3. **Ruff:**
   - Проверки: pycodestyle, pyflakes, isort, comprehensions, bugbear, pyupgrade
   - Игнорирование E501

4. **MyPy:**
   - python_version: "3.10"
   - Строгие проверки: warn_return_any, warn_unused_configs
   - Разрешены нетипизированные определения в тестах

5. **Bandit:**
   - exclude_dirs: tests, .venv, venv
   - skips: B101

## 3. Управление завмсимостями (2 балла)

### Pixi

**pixi.toml** содержит:

**Зависимости:**
- Python: >=3.10,<3.13
- Data Science библиотеки: numpy, pandas, scikit-learn, matplotlib, seaborn
- Jupyter: jupyter, notebook
- Инструменты качества кода: black, isort, ruff, mypy, bandit, pre-commit
- Тестирование: pytest, pytest-cov

**Задачи (tasks):**
```bash
pixi run jupyter
pixi run test
pixi run lint
pixi run format
pixi run typecheck
pixi run security
```

### Виртуальное окружение

Автоматически создается в `.pixi/envs/default/` при выполнении `pixi install`.

### Dockerfile

Dockerfile для контейнеризации:

**Особенности:**
- Base image: Ubuntu 22.04
- Pixi v0.59.0
- Зависимости из pixi.toml
- Экспорт порта 8888 под Jupyter Notebook

**Использование:**
```bash
# Сборка
docker build -t ds-project .

# Запуск
docker run -it --rm -v $(pwd):/workspace ds-project
```

## 4. Git workflow (1 балл)

### .gitignore

 Создан `.gitignore` с исключениями:

**Python:**
- `__pycache__/`, `*.pyc`, `*.pyo`
- `build/`, `dist/`, `*.egg-info/`

**Venv:**
- `.pixi/`, `.venv/`, `venv/`

**IDE:**
- `.vscode/`, `.idea/`, `.DS_Store`

**Jupyter:**
- `.ipynb_checkpoints/`

**Test:**
- `.pytest_cache/`, `.coverage`, `htmlcov/`

**ML:**
- `data/raw/*`, `data/processed/*`, `data/interim/*`
- `models/*.pkl`, `models/*.h5`, `models/*.pt`
- `reports/figures/*`


### Ветки

Основная ветка: `main`

Каждое дз будет иметь свою ветку.
HW_1: `HW_1`
и т.д.
