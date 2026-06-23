#!/usr/bin/env bash
# Lanza JupyterLab con el entorno aislado del pipeline UCFatigue.
# Uso: ./start_jupyter.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="$SCRIPT_DIR/.venv"

if [ ! -d "$VENV" ]; then
    echo "ERROR: entorno virtual no encontrado en $VENV"
    echo "Crea el entorno con: python3.11 -m venv $VENV"
    exit 1
fi

echo "Activando entorno: $VENV"
source "$VENV/bin/activate"

echo "Lanzando JupyterLab en: $SCRIPT_DIR/UCFatigue/pipeline"
jupyter lab --notebook-dir="$SCRIPT_DIR/UCFatigue/pipeline"
