@echo off
REM  Remove the venv, caches and any scratch stores. Destructive but
REM  only to generated files - source and the model folder are untouched.
setlocal
cd /d "%~dp0"
echo %cmdcmdline% | find /i "%~f0" >nul
if not errorlevel 1 set "DBLCLICK=1"

echo.
echo  This will delete:
echo      .venv\           the isolated environment
echo      .pytest_cache\   test cache
echo      __pycache__\     compiled bytecode ^(recursive^)
echo      *.owl *.owl-wal *.owl-shm   scratch stores in this folder
echo.
echo  It will NOT touch your source, your .gguf models, or any store
echo  outside this folder.
echo.
set /p "OK=Proceed? [y/N] "
if /i not "%OK%"=="y" (
    echo  Cancelled.
    if defined DBLCLICK pause
    exit /b 0
)

if exist ".venv"         rmdir /s /q ".venv"
if exist ".pytest_cache" rmdir /s /q ".pytest_cache"
for /d /r %%D in (__pycache__) do if exist "%%D" rmdir /s /q "%%D"
del /q *.owl *.owl-wal *.owl-shm 2>nul

echo  Done. Run install.bat to rebuild.
if defined DBLCLICK pause
endlocal
