@echo off
chcp 65001 >nul
title Instagram Photo Processor

echo ========================================
echo   Instagram Photo Processor
echo ========================================
echo.

cd /d "%~dp0"

echo Installing dependencies...
python -m pip install -r requirements.txt --quiet

echo.
echo Starting web interface...
echo Open http://localhost:8501 in your browser
echo.

streamlit run app.py

pause
