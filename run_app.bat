@echo off
chcp 65001 >nul
title Instagram Photo Processor

echo ========================================
echo   Instagram Photo Processor
echo ========================================
echo.

cd /d "%~dp0"

set PYTHON=C:\Users\369\AppData\Local\Programs\Python\Python313\python.exe
set PIP=%PYTHON% -m pip
set STREAMLIT=%PYTHON% -m streamlit

echo Installing dependencies...
%PIP% install -r requirements.txt --quiet --disable-pip-version-check 2>nul

echo.
echo Starting web interface...
echo.
echo If browser does not open, go to: http://localhost:8501
echo.

start "" "%PYTHON%" -m streamlit run app.py --server.headless=true --browser.gatherUsageStats=false

timeout /t 3 /nobreak >nul

echo Done! Press Ctrl+C to stop server.
pause
