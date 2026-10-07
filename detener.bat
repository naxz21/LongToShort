@echo off
setlocal EnableDelayedExpansion
cd /d "%~dp0"

set "PORT=8765"
if exist "temp\server.port" set /p PORT=<"temp\server.port"
set "STOPPED="

if exist "temp\server.pid" (
    set /p PID=<"temp\server.pid"
    taskkill /PID !PID! /T /F >nul 2>nul && set "STOPPED=1"
    del "temp\server.pid" >nul 2>nul
)

rem Respaldo: cualquier proceso escuchando en el puerto de LongToShort.
for /f "tokens=5" %%p in ('netstat -ano ^| findstr /r /c:"127.0.0.1:%PORT% .*LISTENING"') do (
    taskkill /PID %%p /T /F >nul 2>nul && set "STOPPED=1"
)

if defined STOPPED (
    echo LongToShort detenido.
) else (
    echo LongToShort no estaba corriendo.
)
ping -n 3 127.0.0.1 >nul
