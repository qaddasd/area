@echo off
echo.
echo ===================================================
echo   Area - Windows Setup
echo ===================================================
echo.

:: Remember where we started
set "AREA_DIR=%~dp0"

:: Check Python
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python not found. Download from https://python.org
    echo         Make sure to check "Add Python to PATH" during install.
    pause
    exit /b 1
)
echo [OK] Python found

:: Check Git
git --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Git not found. Download from https://git-scm.com
    pause
    exit /b 1
)
echo [OK] Git found

:: Create venv
cd /d "%AREA_DIR%"
if not exist "venv" (
    echo [SETUP] Creating virtual environment...
    python -m venv venv
)
call venv\Scripts\activate.bat
echo [OK] Virtual environment activated

:: Install dependencies
echo.
echo [SETUP] Installing Python dependencies ^(this takes a few minutes^)...
python -m pip install --upgrade pip -q
python -m pip install -r requirements.txt -q
echo [OK] Dependencies installed

:: Area-3R runtime check
echo.
if exist "%AREA_DIR%area_3r_runtime" (
    echo [OK] Area-3R runtime found
) else (
    echo [INFO] Area-3R runtime not found at area_3r_runtime\
    echo [INFO] Place the Area-3R runtime there to enable geometric matching.
)

:: Return to Area directory
cd /d "%AREA_DIR%"

:: Verify bundled model weights (no internet download required)
echo.
echo [SETUP] Verifying bundled model weights...
if exist "%AREA_DIR%Area-loc.safetensors" (echo [OK] Area-loc weights found) else (echo [WARN] Area-loc.safetensors missing)
if exist "%AREA_DIR%Area-3R\model.safetensors" (echo [OK] Area-3R weights found) else (echo [WARN] Area-3R\model.safetensors missing)

:: Create data dirs
cd /d "%AREA_DIR%"
mkdir area_data\area_parts 2>nul
mkdir area_data\index 2>nul
echo [OK] Data directories created

:: Done
echo.
echo ===================================================
echo   Setup complete!
echo.
echo   To run Area:
echo     Double-click run.bat
echo   Or:
echo     venv\Scripts\activate
echo     python test_super.py
echo ===================================================
pause
