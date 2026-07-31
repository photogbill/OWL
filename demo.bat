@echo off
REM  Run every example in order. No model, no GPU required.
setlocal
cd /d "%~dp0"
call "%~dp0_common.bat" "%~f0"

if not exist "%VPY%" (
    echo  [FAIL] No .venv found. Run install.bat first.
    if defined DBLCLICK pause
    exit /b 1
)

for %%E in (
    00_tier0_field_notes
    01_theory_of_mind
    02_handover
    03_semantic
    04_forward_direction
    05_trust_loop
) do (
    echo.
    echo  ###################################################################
    echo  #  examples\%%E.py
    echo  ###################################################################
    "%VPY%" "examples\%%E.py"
    if errorlevel 1 (
        echo  [FAIL] examples\%%E.py exited non-zero.
        if defined DBLCLICK pause
        exit /b 1
    )
)
echo.
echo  All examples completed.
if defined DBLCLICK pause
endlocal
