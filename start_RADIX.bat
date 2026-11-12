@echo off
:: ============================================================
::  RADIX - Launch the segmentation GUI
::  Double-click this file (or a desktop shortcut to it) to start.
::  The browser will open automatically at http://localhost:8501
:: ============================================================

set REPO_DIR=%~dp0
set ENV_NAME=radix

:: -- Find conda --------------------------------------------------
set CONDA_BAT=
if exist "%USERPROFILE%\miniconda3\Scripts\activate.bat" set CONDA_BAT=%USERPROFILE%\miniconda3\Scripts\activate.bat
if exist "%USERPROFILE%\anaconda3\Scripts\activate.bat"  set CONDA_BAT=%USERPROFILE%\anaconda3\Scripts\activate.bat
if exist "C:\ProgramData\miniconda3\Scripts\activate.bat" set CONDA_BAT=C:\ProgramData\miniconda3\Scripts\activate.bat
if exist "C:\ProgramData\anaconda3\Scripts\activate.bat"  set CONDA_BAT=C:\ProgramData\anaconda3\Scripts\activate.bat

if "%CONDA_BAT%"=="" (
    echo  ERROR: conda not found.
    echo  Run install.bat first, or install Miniconda from:
    echo  https://docs.conda.io/en/latest/miniconda.html
    pause
    exit /b 1
)

:: -- Activate environment and launch -------------------------------
echo  Starting RADIX...
call "%CONDA_BAT%" %ENV_NAME%

if errorlevel 1 (
    echo.
    echo  ERROR: Could not activate environment "%ENV_NAME%".
    echo  Run install.bat first.
    pause
    exit /b 1
)

cd /d "%REPO_DIR%"

streamlit run GUI.py --browser.gatherUsageStats false

:: Keep window open if Streamlit exits with an error
if errorlevel 1 (
    echo.
    echo  RADIX exited with an error. See output above.
    pause
)
