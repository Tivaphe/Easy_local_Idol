#!/usr/bin/env bash
# Lancement d'Idol sous Linux : ./lancer_idol.sh
cd "$(dirname "$0")" || exit 1
if ! command -v python3 >/dev/null 2>&1; then
  echo "python3 introuvable. Installez-le (ex. sudo apt install python3 python3-venv)."
  exit 1
fi
if ! python3 -c "import venv, ensurepip" >/dev/null 2>&1; then
  echo "Module venv absent. Installez-le : sudo apt install python3-venv"
  exit 1
fi
exec python3 launcher.py "$@"
