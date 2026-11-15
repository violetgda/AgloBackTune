@echo off
setlocal
cd /d "%~dp0"
set PYTHONIOENCODING=utf-8
where py >nul 2>nul
if errorlevel 1 goto usepython
py -3 -m ticklab_algo %*
goto done
:usepython
python -m ticklab_algo %*
:done
if errorlevel 1 pause
