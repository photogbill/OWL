@echo off
REM  Open a command prompt with the venv already active, so `python`,
REM  `pytest` and `owl` all resolve to the isolated environment.
cd /d "%~dp0"
if not exist ".venv\Scripts\activate.bat" (
    echo  [FAIL] No .venv found. Run install.bat first.
    pause
    exit /b 1
)
echo.
echo   O.W.L. venv active. Try:
echo       owl check
echo       pytest tests -q
echo       python examples\04_forward_direction.py
echo.
cmd /k ".venv\Scripts\activate.bat"
