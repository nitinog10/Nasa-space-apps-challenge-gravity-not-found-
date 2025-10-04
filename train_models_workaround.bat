@echo off
echo ======================================================================
echo    TRAINING MODELS (Python 3.13 Workaround)
echo ======================================================================
echo.
echo This script uses a workaround for Python 3.13 numpy issues...
echo.

REM Set environment variables to avoid numpy crashes
set OPENBLAS_NUM_THREADS=1
set MKL_NUM_THREADS=1
set NUMEXPR_NUM_THREADS=1

echo Running safe training script...
python train_models_safe.py

echo.
pause
