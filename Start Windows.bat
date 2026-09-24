@echo off
cd /d "%~dp0"
where py >nul 2>nul
if errorlevel 1 (
    echo Install Python 3.13 from https://www.python.org/downloads/ and try again.
) else (
    py -3.13 launch.py
)
pause
