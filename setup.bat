@echo off
REM CWMAI Windows Setup Script

echo ====================================================================
echo             CWMAI - Windows Setup Script
echo ====================================================================
echo.

REM Check Python installation
echo Checking Python installation...
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python not found. Please install Python 3.9 or higher.
    echo Download from: https://www.python.org/downloads/
    pause
    exit /b 1
)

REM Get Python version
for /f "tokens=2" %%i in ('python --version 2^>^&1') do set PYTHON_VERSION=%%i
echo Python %PYTHON_VERSION% detected

REM Create virtual environment
echo.
echo Creating virtual environment...
if exist venv (
    echo Removing existing virtual environment...
    rmdir /s /q venv
)
python -m venv venv

REM Activate virtual environment
echo Activating virtual environment...
call venv\Scripts\activate.bat

REM Upgrade pip
echo.
echo Upgrading pip and build tools...
python -m pip install --upgrade pip setuptools wheel

REM Install dependencies
echo.
echo Installing dependencies...
pip install -r requirements.txt

if errorlevel 1 (
    echo.
    echo WARNING: Some packages failed to install.
    echo Trying to install core packages individually...
    pip install psutil
    pip install cryptography
    pip install numpy
    pip install pandas
    pip install -r requirements.txt
)

REM Setup NLTK data
echo.
echo Setting up NLTK data...
python -c "import nltk; nltk.download('punkt', quiet=True); nltk.download('stopwords', quiet=True); nltk.download('wordnet', quiet=True); print('NLTK data downloaded')"

REM Setup environment file
echo.
echo Setting up environment configuration...
if not exist .env (
    if exist .env.local (
        copy .env.local .env
        echo Copied .env.local to .env
    ) else (
        echo WARNING: No .env file found. Please create one from .env.example
    )
) else (
    echo .env file already exists
)

REM Create batch file for easy activation
echo.
echo Creating activation script...
echo @echo off > activate.bat
echo call venv\Scripts\activate.bat >> activate.bat
echo echo CWMAI environment activated >> activate.bat
echo echo Python: %%VIRTUAL_ENV%%\Scripts\python.exe >> activate.bat

REM Test installation
echo.
echo Testing installation...
python -c "import sys; print(f'Python: {sys.version}')"
python -c "import redis; print('Redis module OK')" 2>nul || echo Redis module failed
python -c "import psutil; print('Psutil module OK')" 2>nul || echo Psutil module failed
python -c "import numpy; print('Numpy module OK')" 2>nul || echo Numpy module failed

echo.
echo ====================================================================
echo                     Setup Complete!
echo ====================================================================
echo.
echo To use CWMAI:
echo 1. Activate the environment: activate.bat
echo 2. Run CWMAI: python cwmai
echo 3. Or run continuous AI: python run_continuous_ai.py
echo.
pause