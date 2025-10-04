@echo off
echo ======================================================================
echo    Installing Essential Packages for Model Training
echo ======================================================================
echo.
echo This will install the core packages needed to run the pipeline...
echo.

REM Upgrade pip first
python -m pip install --upgrade pip

REM Install core ML packages one by one
echo Installing numpy...
python -m pip install "numpy>=1.24.0,<2.0.0"

echo Installing pandas...
python -m pip install "pandas>=2.0.0,<3.0.0"

echo Installing scikit-learn...
python -m pip install "scikit-learn>=1.3.0"

echo Installing xgboost...
python -m pip install "xgboost>=2.0.0"

echo Installing lightgbm...
python -m pip install "lightgbm>=4.0.0"

echo Installing other essential packages...
python -m pip install matplotlib seaborn plotly joblib tqdm pyyaml requests

echo.
echo ======================================================================
echo    ✅ Essential packages installed!
echo ======================================================================
echo.
echo Now you can continue running the pipeline...
echo.
pause
