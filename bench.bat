@echo off
REM  Run the benchmark harnesses. These measure claims made in the
REM  README; if a number moves, the README is wrong until it is updated.
setlocal
cd /d "%~dp0"
call "%~dp0_common.bat" "%~f0"

if not exist "%VPY%" (
    echo  [FAIL] No .venv found. Run install.bat first.
    if defined DBLCLICK pause
    exit /b 1
)

echo.
echo  ###################################################################
echo  #  EPISTEMIC SCOREBOARD
echo  #  The field measures whether you can FIND it.
echo  #  This measures whether you should BELIEVE it.
echo  ###################################################################
"%VPY%" bench\scoreboard.py

echo.
echo  ###################################################################
echo  #  interference resistance  ^(MemoryLLM `nuc` methodology^)
echo  ###################################################################
"%VPY%" bench\interference.py

echo.
echo  -------------------------------------------------------------------
echo   For the embedder benchmark you need a .gguf model:
echo       validate.bat "embedding model\bge-m3-Q6_K.gguf"
echo  -------------------------------------------------------------------
if defined DBLCLICK pause
endlocal
