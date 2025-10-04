# Fix Python Version Issue

## The Problem
Python 3.13 is too new and has compatibility issues with numpy and other scientific packages on Windows.

## Solution: Install Python 3.11

1. Download Python 3.11 from: https://www.python.org/downloads/release/python-3119/
   - Choose "Windows installer (64-bit)"

2. During installation:
   - ✅ Check "Add Python 3.11 to PATH"
   - ✅ Check "Install for all users" (optional)

3. After installation, create a virtual environment with Python 3.11:
   ```powershell
   cd "D:\Ml model Nasa"
   py -3.11 -m venv venv
   .\venv\Scripts\Activate
   ```

4. Install packages in the virtual environment:
   ```powershell
   python -m pip install --upgrade pip
   python -m pip install scikit-learn xgboost lightgbm joblib matplotlib seaborn plotly pandas numpy
   ```

5. Continue with the pipeline:
   ```powershell
   python run_pipeline.py --skip-data --skip-features --run-evaluation
   ```
