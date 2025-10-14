@echo off
REM Air Quality Forecasting System - Windows Setup Launcher
REM ======================================================
REM This batch file will automatically setup the Air Quality Forecasting System
REM on Windows systems. It will run the Python setup script and handle common issues.

title Air Quality Forecasting System - Setup

echo.
echo =====================================================
echo   AIR QUALITY FORECASTING SYSTEM - SETUP
echo =====================================================
echo.
echo 🌍 Setting up Air Quality Forecasting for Delhi
echo 📊 This will install dependencies and configure the system
echo.

REM Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Python is not installed or not in PATH
    echo.
    echo Please install Python 3.8+ from https://python.org
    echo Make sure to check "Add Python to PATH" during installation
    echo.
    pause
    exit /b 1
)

echo ✅ Python found
python --version

REM Check if we're in the right directory
if not exist "setup.py" (
    echo.
    echo ❌ setup.py not found in current directory
    echo Please navigate to the air_quality_forecast project directory
    echo.
    pause
    exit /b 1
)

echo ✅ Project files found

REM Run the Python setup script
echo.
echo 🚀 Running Python setup script...
echo.
python setup.py

REM Check if setup was successful
if errorlevel 1 (
    echo.
    echo ⚠️  Setup completed with some issues
    echo Check the output above for details
    echo.
    echo You can also try:
    echo - python test_setup.py (to diagnose issues)
    echo - python setup.py (to run setup again)
    echo.
) else (
    echo.
    echo ✅ Setup completed successfully!
    echo.
    echo You can now run:
    echo - run_dashboard.bat (for web dashboard)
    echo - run_demo.bat (for complete demo)
    echo - python launcher.py (for interactive menu)
    echo.
)

echo.
echo Press any key to continue...
pause >nul
