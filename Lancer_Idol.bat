@echo off
chcp 65001 >nul
title Idol
cd /d "%~dp0"
where py >nul 2>nul
if %errorlevel%==0 (
    py -3 launcher.py %*
) else (
    where python >nul 2>nul
    if %errorlevel%==0 (
        python launcher.py %*
    ) else (
        echo Python introuvable. Installez Python 3.12 depuis https://www.python.org/downloads/
        echo et cochez "Add python.exe to PATH" pendant l'installation.
    )
)
if errorlevel 1 (
    echo.
    echo Une erreur est survenue. Voir les messages ci-dessus.
    pause
)
