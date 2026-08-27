@echo off
REM ===================================================================
REM  O.W.L. - standalone install for Windows
REM
REM  Creates an isolated .venv, installs OWL editable, and tries to get
REM  a PREBUILT llama-cpp-python wheel. It never compiles from source:
REM  that needs CMake + Visual Studio Build Tools and usually fails.
REM
REM  OWL's core has zero dependencies. If the llama wheel cannot be
REM  found, Tier 0 still works completely and the script says so.
REM ===================================================================
setlocal enabledelayedexpansion
cd /d "%~dp0"
echo %cmdcmdline% | find /i "%~f0" >nul
if not errorlevel 1 set "DBLCLICK=1"

echo.
echo  ===================================================================
echo    O.W.L.  -  Observation ^& Wisdom Ledger
echo    standalone install
echo  ===================================================================
echo.

REM ---- 1. locate a usable python ------------------------------------
set "PY="
py -3 --version >nul 2>&1 && set "PY=py -3"
if not defined PY (
    python --version >nul 2>&1 && set "PY=python"
)
if not defined PY (
    python3 --version >nul 2>&1 && set "PY=python3"
)
if not defined PY (
    echo  [FAIL] Python not found on PATH.
    echo         Install Python 3.10 - 3.12 from python.org, tick
    echo         "Add python.exe to PATH", then re-run this script.
    goto :fail
)

for /f "tokens=2" %%V in ('%PY% --version 2^>^&1') do set "PYVER=%%V"
echo  [1/6]  Python !PYVER!   ^(via %PY%^)

%PY% -c "import sys;raise SystemExit(0 if (3,10)<=sys.version_info[:2] else 1)"
if errorlevel 1 (
    echo  [FAIL] Python 3.10 or newer is required. Found !PYVER!.
    goto :fail
)
%PY% -c "import sys;raise SystemExit(1 if sys.version_info[:2]>=(3,13) else 0)"
if errorlevel 1 (
    echo         NOTE: Python 3.13+ often has no prebuilt llama-cpp-python
    echo               wheel yet. 3.10 - 3.12 is the safe range. Core OWL
    echo               works on any of them.
)

REM ---- 2. venv ------------------------------------------------------
if exist ".venv\Scripts\python.exe" (
    echo  [2/6]  venv already present   ^(.venv^)
) else (
    echo  [2/6]  Creating venv   ^(.venv^) ...
    %PY% -m venv .venv
    if errorlevel 1 (
        echo  [FAIL] Could not create the venv.
        goto :fail
    )
)
set "VPY=%~dp0.venv\Scripts\python.exe"
if not exist "%VPY%" (
    echo  [FAIL] venv looks broken - "%VPY%" is missing.
    echo         Delete the .venv folder and run this script again.
    goto :fail
)

REM ---- 3. pip -------------------------------------------------------
echo  [3/6]  Upgrading pip ...
"%VPY%" -m pip install --upgrade pip --quiet --disable-pip-version-check
if errorlevel 1 (
    echo         [WARN] pip upgrade failed; continuing with the bundled pip.
)

REM ---- 4. core ------------------------------------------------------
echo  [4/6]  Installing owl-engine ^(editable^) + test tools ...
"%VPY%" -m pip install -e ".[dev]" --quiet --disable-pip-version-check
if errorlevel 1 (
    echo  [FAIL] Core install failed. Nothing else can work; stopping.
    echo         Re-run without --quiet to see pip's output:
    echo             .venv\Scripts\python.exe -m pip install -e ".[dev]"
    goto :fail
)

REM ---- 5. optional inference backend --------------------------------
REM  All of this lives in Python. cmd's `for /f` with backquotes mangles
REM  quoted paths containing spaces - a folder named "owl-engine - Test"
REM  broke the loop AND silently reported nothing even though the wheel
REM  had installed. Path quoting is not something to hand-roll in batch.
echo  [5/6]  Detecting inference backend ...
"%VPY%" scripts\detect_backend.py --explain
"%VPY%" scripts\install_backend.py

REM ---- 6. verify ----------------------------------------------------
echo  [6/6]  Verifying ...
echo.
"%VPY%" -m pytest tests -q
if errorlevel 1 (
    echo.
    echo  [FAIL] The test suite did not pass. Do not proceed - something
    echo         is wrong with this install.
    goto :fail
)
echo.
"%VPY%" -m owl check

echo.
echo  ===================================================================
echo    Ready.
echo.
echo      run_tests.bat        run the suite
echo      demo.bat             run every example
echo      bench.bat            run the benchmarks
echo      shell.bat            open a prompt inside the venv
echo      clean.bat            remove .venv and caches
echo.
echo      validate.bat "embedding model\bge-m3-Q6_K.gguf"
echo                           validate a real embedder  ^(needs Tier 1^)
echo  ===================================================================
echo.
if defined DBLCLICK pause
endlocal
exit /b 0

:fail
echo.
if defined DBLCLICK pause
endlocal
exit /b 1
