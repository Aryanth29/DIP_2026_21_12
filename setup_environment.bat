@echo off
echo ============================================================
echo ECE501 DIP Project - Environment Setup
echo ============================================================
echo.

:: Check if Python is installed
python --version >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo [ERROR] Python is not found in PATH!
    echo.
    echo Please install Python 3.10 or 3.11 from:
    echo https://www.python.org/downloads/
    echo.
    echo IMPORTANT: Make sure to check the box:
    echo "Add python.exe to PATH" during installation!
    echo.
    pause
    exit /b 1
)

echo [1/4] Found Python:
python --version
echo.

echo [2/4] Creating virtual environment (.venv)...
python -m venv .venv
if %ERRORLEVEL% neq 0 (
    echo [ERROR] Failed to create virtual environment!
    pause
    exit /b 1
)

echo [3/4] Activating virtual environment and installing dependencies...
call .venv\Scripts\activate.bat
python -m pip install --upgrade pip
pip install -r requirements.txt
if %ERRORLEVEL% neq 0 (
    echo [ERROR] Failed to install requirements!
    pause
    exit /b 1
)

echo [4/4] Registering Jupyter kernel...
python -m ipykernel install --user --name dip_env --display-name "Python (DIP ECE501)"

echo.
echo ============================================================
echo Setup Complete!
echo.
echo To launch the Jupyter notebook:
echo   1. Run: call .venv\Scripts\activate.bat
echo   2. Run: jupyter notebook
echo   3. Open Code\ECE501.ipynb in the browser
echo ============================================================
echo.
pause
