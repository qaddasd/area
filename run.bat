@echo off
cd /d "%~dp0"
call venv\Scripts\activate.bat
venv\Scripts\python.exe test_super.py
pause
