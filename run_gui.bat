@echo off
REM Lanzador de la GUI Neuromorphic Lab para Windows (doble clic).
setlocal
cd /d "%~dp0"
python run_app.py
if errorlevel 1 (
    echo.
    echo [ERROR] No se pudo iniciar Neuromorphic Lab. Asegurate de tener Python
    echo         instalado y las dependencias:  pip install -r requirements.txt
    pause
)