@echo off
REM  Run the correctness suite. Must stay under a few seconds, no GPU,
REM  no network. Pass extra pytest args straight through:
REM      run_tests.bat -k defence -v
setlocal
cd /d "%~dp0"
call "%~dp0_common.bat" "%~f0"

if not exist "%VPY%" (
    echo  [FAIL] No .venv found. Run install.bat first.
    if defined DBLCLICK pause
    exit /b 1
)
"%VPY%" -m pytest tests -q %*
set "RC=%ERRORLEVEL%"
if defined DBLCLICK pause
endlocal & exit /b %RC%
