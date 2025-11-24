# EPML DA ITMO - Домашние работы

Репозиторий с домашними работами по курсу Data Science.

## Структура репозитория

Каждая домашняя работа находится в отдельной ветке:

- `hw1` - ДЗ 1: Настройка рабочего места Data Scientist
- `hw2` - ДЗ 2: (будет добавлено позже)
- `hw3` - ДЗ 3: (будет добавлено позже)

## Как переключаться между ДЗ

```bash
# Посмотреть все ветки
git branch -a

# Переключиться на ДЗ 1
git checkout hw1

# Переключиться на ДЗ 2
git checkout hw2

# Вернуться в основную ветку
git checkout master
```

## Как проверить домашнюю работу

1. Клонируйте репозиторий:
   ```bash
   git clone https://github.com/kragger-ra/EPML_DA_ITMO_HW_Repo
   cd EPML_DA_ITMO_HW_Repo
   ```

2. Переключитесь на нужную ветку:
   ```bash
   git checkout hw1
   ```

3. Следуйте инструкциям в README.md конкретной ветки

## Список домашних работ

### ДЗ 1: Настройка рабочего места Data Scientist
**Ветка:** `hw1`
**Статус:** Выполнено
**Дата сдачи:** 24 ноября 2025

**Содержание:**
- Структура проекта Data Science
- Настройка pixi для управления зависимостями
- Настройка pre-commit hooks (Black, isort, Ruff, MyPy, Bandit)
- Dockerfile для контейнеризации
- Git workflow

**Команды для проверки:**
```bash
git checkout hw1
cat REPORT.md  # Подробный отчет
cat QUICKSTART.md  # Быстрый старт
```

---

## Автор

Студент ITMO University
Курс: Enterprise ML & Data Analytics

## Репозиторий

https://github.com/kragger-ra/EPML_DA_ITMO_HW_Repo
