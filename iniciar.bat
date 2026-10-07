@echo off
setlocal EnableDelayedExpansion
cd /d "%~dp0"
title LongToShort

set "PORT=8765"
for /f "usebackq tokens=2 delims=:, " %%p in (`findstr /c:"\"port\"" config.json`) do set "PORT=%%p"

rem --- Si ya esta corriendo, solo abrir el navegador ---
curl -s -o nul "http://127.0.0.1:%PORT%/api/health" >nul 2>nul
if not errorlevel 1 (
    echo LongToShort ya esta corriendo. Abriendo el navegador...
    start "" "http://127.0.0.1:%PORT%"
    exit /b 0
)

rem --- Entorno virtual propio de LongToShort ---
if not exist ".venv\Scripts\python.exe" (
    echo [LongToShort] Creando el entorno virtual .venv ...
    python -m venv .venv
    if errorlevel 1 (
        echo [ERROR] No se encontro Python 3.10 o superior. Instalalo desde https://www.python.org/
        pause
        exit /b 1
    )
)
call ".venv\Scripts\activate.bat"

rem --- Dependencias ---
python -c "import fastapi, uvicorn, yt_dlp, faster_whisper, fastembed, cv2, numpy" >nul 2>nul
if errorlevel 1 (
    echo [LongToShort] Instalando dependencias, puede tardar unos minutos...
    python -m pip install --upgrade pip
    python -m pip install -r requirements.txt
    if errorlevel 1 (
        echo [ERROR] No se pudieron instalar las dependencias. Revisa tu conexion a internet.
        pause
        exit /b 1
    )
) else (
    rem YouTube cambia seguido: mantener yt-dlp al dia evita errores de descarga.
    echo [LongToShort] Verificando actualizaciones de yt-dlp...
    python -m pip install -q --disable-pip-version-check -U "yt-dlp[default]" >nul 2>nul
)

rem --- GPU NVIDIA opcional (librerias CUDA para la transcripcion) ---
where nvidia-smi >nul 2>nul
if not errorlevel 1 (
    if not exist ".venv\Lib\site-packages\nvidia\cublas" (
        echo [LongToShort] GPU NVIDIA detectada: instalando librerias CUDA ^(una sola vez, ~1 GB^)...
        python -m pip install -r requirements-gpu.txt
    )
)

rem --- FFmpeg ---
where ffmpeg >nul 2>nul
if errorlevel 1 (
    echo.
    echo [AVISO] No se encontro FFmpeg en el PATH. Instalalo con:  winget install Gyan.FFmpeg
    echo         y volve a abrir este script. Ver README.md
    echo.
)

echo [LongToShort] Iniciando servidor en http://127.0.0.1:%PORT%
python -m app.main --open-browser
echo.
echo LongToShort se detuvo.
pause
