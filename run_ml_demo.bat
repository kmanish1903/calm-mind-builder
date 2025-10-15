@echo off
echo Starting ML Demonstration...
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo Python is not installed or not in PATH
    pause
    exit /b 1
)

REM Install required packages if needed
echo Installing required packages...
pip install numpy torch scikit-learn transformers >nul 2>&1

REM Run the ML demonstration
echo.
echo Running ML Demo for Professor...
echo.
python ml_demo.py

echo.
echo Demo completed. Press any key to exit...
pause >nul