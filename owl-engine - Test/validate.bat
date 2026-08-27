@echo off
REM  Validate a real GGUF embedder end-to-end.
REM      validate.bat "embedding model\bge-m3-Q6_K.gguf"
REM
REM  Every Tier 1 number OWL reports is meaningless until this passes.
setlocal enabledelayedexpansion
cd /d "%~dp0"
call "%~dp0_common.bat" "%~f0"

if not exist "%VPY%" (
    echo  [FAIL] No .venv found. Run install.bat first.
    if defined DBLCLICK pause
    exit /b 1
)

REM  Comparison takes two model paths, so it bypasses the single-model
REM  discovery below entirely.
REM  Calibrate every model in a folder. Sidecars do not survive a clean
REM  checkout, and a stale one silently reinstates a fixed bug.
if /i "%~1"=="--calibrate-all" (
    set "DIR=%~2"
    if "%~2"=="" set "DIR=embedding model"
    "%VPY%" bench\validate_embedder.py --calibrate-all "!DIR!"
    set "RC=!ERRORLEVEL!"
    if defined DBLCLICK pause
    endlocal & exit /b %RC%
)

if /i "%~1"=="--compare" (
    "%VPY%" bench\validate_embedder.py --compare "%~2" "%~3"
    set "RC=!ERRORLEVEL!"
    if defined DBLCLICK pause
    endlocal & exit /b %RC%
)

set "MODEL=%~1"
if "%MODEL%"=="" (
    REM Nothing given - look for a single .gguf so double-clicking works.
    for /f "delims=" %%F in ('dir /b /s "%~dp0*.gguf" 2^>nul') do (
        if not defined MODEL set "MODEL=%%F"
    )
    if defined MODEL (
        echo  No model given; found one and using it:
        echo      !MODEL!
        echo.
    )
)

if "%MODEL%"=="" (
    echo  Usage:  validate.bat "path\to\model.gguf"
    echo.
    echo  Examples:
    echo      validate.bat "embedding model\bge-m3-Q6_K.gguf"
    echo      validate.bat "embedding model\bge-m3-Q6_K.gguf" --calibrate
    echo      validate.bat --compare "model-a.gguf" "model-b.gguf"
    if defined DBLCLICK pause
    exit /b 2
)

"%VPY%" bench\validate_embedder.py "%MODEL%" %2 %3
set "RC=%ERRORLEVEL%"
if defined DBLCLICK pause
endlocal & exit /b %RC%
