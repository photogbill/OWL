@echo off
REM ===================================================================
REM  Shared prelude for the O.W.L. batch scripts. Not run directly.
REM
REM    call "%~dp0_common.bat" "%~f0"
REM                             ^-- the CALLER's path, required
REM
REM  Sets:  VPY        full path to the venv python
REM         DBLCLICK   1 if this window was opened from Explorer
REM
REM  The caller's path must be passed in: %~f0 inside this file resolves
REM  to _common.bat itself, which never appears on the command line, so
REM  the double-click check would silently never fire.
REM
REM  That check matters. A .bat launched from Explorer closes the instant
REM  it finishes, so an error message is visible for about a tenth of a
REM  second. Scripts pause when double-clicked and do not when run from a
REM  prompt or from CI.
REM ===================================================================
set "VPY=%~dp0.venv\Scripts\python.exe"
if not "%~1"=="" (
    echo %cmdcmdline% | find /i "%~1" >nul
    if not errorlevel 1 set "DBLCLICK=1"
)
