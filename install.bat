@echo off
:: ============================================================
::  RADIX - one-time environment setup
::  Run this once on a new machine, then use start_RADIX.bat
:: ============================================================

set REPO_DIR=%~dp0
set ENV_NAME=radix

echo.
echo  RADIX - Installing environment...
echo  This will take a few minutes on first run.
echo.

:: -- Find conda --------------------------------------------------
set CONDA_EXE=
if exist "%USERPROFILE%\miniconda3\Scripts\conda.exe" set CONDA_EXE=%USERPROFILE%\miniconda3\Scripts\conda.exe
if exist "%USERPROFILE%\anaconda3\Scripts\conda.exe"  set CONDA_EXE=%USERPROFILE%\anaconda3\Scripts\conda.exe
if exist "C:\ProgramData\miniconda3\Scripts\conda.exe" set CONDA_EXE=C:\ProgramData\miniconda3\Scripts\conda.exe
if exist "C:\ProgramData\anaconda3\Scripts\conda.exe"  set CONDA_EXE=C:\ProgramData\anaconda3\Scripts\conda.exe

if "%CONDA_EXE%"=="" (
    echo  ERROR: conda not found. Please install Miniconda first:
    echo  https://www.anaconda.com/download/success
    pause
    exit /b 1
)

echo  Found conda: %CONDA_EXE%

:: -- Accept Anaconda's Terms of Service for the default channels --
:: -- Newer conda versions refuse to install anything from the -----
:: -- defaults channels until this is accepted; without it, the ----
:: -- next step fails with a ToS/certificate-looking error. Older ---
:: -- conda versions don't have the "tos" command at all, so -------
:: -- failures here are silently ignored (best-effort, non-fatal). --
call "%CONDA_EXE%" tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main   >nul 2>nul
call "%CONDA_EXE%" tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r      >nul 2>nul
call "%CONDA_EXE%" tos accept --override-channels --channel https://repo.anaconda.com/pkgs/msys2  >nul 2>nul

:: -- Create environment (skip if it already exists) ---------------
:: -- Pinned to the exact Python version this was tested against. --
set PYTHON_VERSION=3.11.15
call "%CONDA_EXE%" env list | findstr /R /C:"^%ENV_NAME% " >nul
if errorlevel 1 (
    :: --override-channels -c conda-forge avoids Anaconda's defaults
    :: channels entirely, so the ToS gate never comes up regardless of
    :: whether the "tos accept" calls above worked on this conda version.
    call "%CONDA_EXE%" create -n %ENV_NAME% --override-channels -c conda-forge python=%PYTHON_VERSION% -y
    if errorlevel 1 goto :error
) else (
    echo  Environment "%ENV_NAME%" already exists - reusing it.
)

:: -- Install PyTorch: CUDA build if an NVIDIA GPU is present, -----
:: -- CPU build otherwise. Both pinned to the exact versions this --
:: -- was tested against, so torchvision/timm/transformers etc. ----
:: -- (installed afterward) resolve against a known-compatible -----
:: -- torch build instead of whatever is newest that day. ----------
:: -- Installed before requirements.txt so the numpy-under-2.0 -----
:: -- pin there re-asserts itself afterward. ------------------------
set TORCH_VERSION=2.5.1
set TORCHVISION_VERSION=0.20.1
set TORCHAUDIO_VERSION=2.5.1
where nvidia-smi >nul 2>nul
if errorlevel 1 (
    echo  No NVIDIA GPU detected - installing CPU-only PyTorch.
    call "%CONDA_EXE%" run -n %ENV_NAME% pip install torch==%TORCH_VERSION% torchvision==%TORCHVISION_VERSION% torchaudio==%TORCHAUDIO_VERSION%
    if errorlevel 1 goto :error
) else (
    echo  NVIDIA GPU detected - installing CUDA-enabled PyTorch.
    call "%CONDA_EXE%" run -n %ENV_NAME% pip install torch==%TORCH_VERSION% torchvision==%TORCHVISION_VERSION% torchaudio==%TORCHAUDIO_VERSION% --index-url https://download.pytorch.org/whl/cu121
    if errorlevel 1 (
        echo  CUDA PyTorch install failed - falling back to CPU-only PyTorch.
        call "%CONDA_EXE%" run -n %ENV_NAME% pip install torch==%TORCH_VERSION% torchvision==%TORCHVISION_VERSION% torchaudio==%TORCHAUDIO_VERSION%
        if errorlevel 1 goto :error
    )
)

:: -- Install remaining dependencies --------------------------------
call "%CONDA_EXE%" run -n %ENV_NAME% pip install -r "%REPO_DIR%requirements.txt"
if errorlevel 1 goto :error

echo.
echo  ============================================================
echo   Installation complete!
echo   Double-click start_RADIX.bat (or the desktop shortcut)
echo   to launch the application.
echo  ============================================================
echo.
pause
exit /b 0

:error
echo.
echo  ERROR: Installation failed. Check the output above.
pause
exit /b 1
