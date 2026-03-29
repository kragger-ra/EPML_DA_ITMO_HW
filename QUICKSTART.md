# Быстрый старт

## Установка

1. Установите pixi:
   ```bash
   # Windows
   iwr -useb https://pixi.sh/install.ps1 | iex

   # Linux/macOS
   curl -fsSL https://pixi.sh/install.sh | bash
   ```

2. Установите зависимости:
   ```bash
   pixi install
   ```

3. Установите pre-commit hooks:
   ```bash
   pixi run pre-commit install
   ```

## Основные команды

### Разработка

```bash
# Jupyter Notebook
pixi run jupyter

# Тесты
pixi run test

# Форматирование
pixi run format

# Линтинг
pixi run lint

# Проверка типов
pixi run typecheck

# Проверка безопасности
pixi run security
```

### Pre-commit

```bash
# Запуск всех проверок вручную
pixi run pre-commit run --all-files

# Обновление хуков
pixi run pre-commit autoupdate
```

### Docker

```bash
# Сборка образа
docker build -t ds-project .

# Запуск контейнера
docker run -it --rm -v ${PWD}:/workspace ds-project

# Запуск Jupyter в контейнере
docker run -p 8888:8888 ds-project pixi run jupyter --ip=0.0.0.0 --allow-root
```

## Структура проекта

- `src/` - исходный код
- `tests/` - тесты
- `notebooks/` - Jupyter notebooks
- `data/` - данные (не коммитится)
- `models/` - сохраненные модели (не коммитится)
- `reports/` - отчеты и графики

## Проверка настройки

```bash
# Проверьте, что все инструменты работают
pixi run black --version
pixi run ruff --version
pixi run mypy --version
pixi run pytest --version

# Запустите тесты
pixi run test

# Запустите проверки качества кода
pixi run pre-commit run --all-files
```

Если все команды выполняются без ошибок - настройка завершена успешно.
