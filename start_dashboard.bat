@echo off
echo.
echo ====================================================
echo    AIR QUALITY FORECASTING DASHBOARD LAUNCHER
echo ====================================================
echo.
echo Starting the Delhi Air Quality Forecasting Dashboard...
echo This will open in your default browser automatically.
echo.
echo To stop the server, close this window or press Ctrl+C
echo ====================================================
echo.

REM Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python and try again.
    pause
    exit /b 1
)

REM Navigate to the script directory
cd /d "%~dp0"

REM Run the dashboard launcher
python start_dashboard.py

REM Keep window open if there's an error
if errorlevel 1 (
    echo.
    echo An error occurred. Please check the output above.
    pause
)
