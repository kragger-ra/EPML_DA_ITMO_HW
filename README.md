# Data Science Project

ДЗ 1: Настройка рабочего места Data Scientist.

## Структура

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

## Установка

```bash
# Windows
iwr -useb https://pixi.sh/install.ps1 | iex

# Установка зависимостей
pixi install

# Настройка pre-commit
pixi run pre-commit install
```

## Использование

```bash
pixi run jupyter      # Запуск Jupyter
pixi run test         # Тесты
pixi run format       # Форматирование (Black, isort)
pixi run lint         # Линтинг (Ruff)
pixi run typecheck    # Проверка типов (MyPy)
pixi run security     # Проверка безопасности (Bandit)
```

## Docker

```bash
docker build -t ds-project .
docker run -it --rm -v ${PWD}:/workspace ds-project
```

## Инструменты

- **Управление зависимостями:** pixi
- **Форматирование:** Black, isort
- **Линтинг:** Ruff, MyPy, Bandit
- **Pre-commit hooks:** автоматические проверки перед коммитом

## Документация

- [REPORT.md](REPORT.md) - подробный отчет
- [QUICKSTART.md](QUICKSTART.md) - быстрый старт

---

**Репозиторий:** https://github.com/kragger-ra/EPML_DA_ITMO_HW_Repo
**Ветка:** HW_1
