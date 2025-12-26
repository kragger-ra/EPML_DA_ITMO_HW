@echo off
setlocal enabledelayedexpansion

set PYTHON=.pixi\envs\default\python.exe
set PYTHONPATH=%CD%

echo ================================================================================
echo ML PIPELINE HW_4 - AUTOMATED EXECUTION
echo ================================================================================
echo Start time: %date% %time%
echo ================================================================================
echo.

REM Установка проекта
echo [SETUP] Installing project in editable mode...
%PYTHON% -m pip install -e . > nul 2>&1
echo [OK] Setup completed
echo.

REM Stage 1: Preprocessing
echo ================================================================================
echo [1/6] DATA PREPROCESSING
echo ================================================================================
%PYTHON% src\data\preprocess.py
if errorlevel 1 goto error
echo [OK] Preprocessing completed
echo.

REM Stage 2: Training
echo ================================================================================
echo [2/6] TRAINING ALL MODELS (6 models)
echo ================================================================================

set MODELS=lightgbm xgboost catboost random_forest gradient_boosting logistic_regression
set SUCCESS_COUNT=0
set FAIL_COUNT=0

for %%M in (%MODELS%) do (
    echo.
    echo [TRAINING] Model: %%M
    %PYTHON% src\models\train_hydra.py model=%%M

    if errorlevel 1 (
        echo [FAIL] Model %%M failed
        set /a FAIL_COUNT+=1
    ) else (
        echo [OK] Model %%M completed
        set /a SUCCESS_COUNT+=1
    )
)

echo.
echo Training Summary: !SUCCESS_COUNT! succeeded, !FAIL_COUNT! failed
echo.

REM Stage 3-6: Continue with evaluation, selection, report, notify
%PYTHON% src\pipelines\evaluate_models.py
%PYTHON% src\pipelines\select_best_model.py
%PYTHON% src\pipelines\generate_report.py
%PYTHON% src\pipelines\notify.py

:success
echo.
echo ================================================================================
echo SUCCESS! PIPELINE COMPLETED
echo ================================================================================
echo Models trained: !SUCCESS_COUNT! / 6
echo.
goto end

:error
echo ERROR! PIPELINE FAILED
:end
pause
