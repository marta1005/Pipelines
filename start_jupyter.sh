#!/usr/bin/env bash
# Lanza JupyterLab con el entorno aislado del pipeline UCFatigue.
# Uso:
#   ./start_jupyter.sh          → sólo JupyterLab
#   ./start_jupyter.sh --mlflow → JupyterLab + MLflow UI en paralelo

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="$SCRIPT_DIR/.venv"
MLRUNS="$SCRIPT_DIR/UCFatigue/pipeline/mlruns"

if [ ! -d "$VENV" ]; then
    echo "ERROR: entorno virtual no encontrado en $VENV"
    echo "Crea el entorno con: python3.11 -m venv $VENV"
    exit 1
fi

source "$VENV/bin/activate"

# macOS marca los .pth que empiezan por __ como "hidden" (flag UF_HIDDEN),
# y Homebrew Python los salta silenciosamente. Lo limpiamos siempre,
# pero también forzamos PYTHONPATH como solución definitiva.
chflags -R nohidden "$VENV/lib/" 2>/dev/null || true

# PYTHONPATH añade el src/ al path de Python sin depender de los .pth.
# Esto es la solución robusta contra el bug de macOS UF_HIDDEN.
export PYTHONPATH="$SCRIPT_DIR/src:$PYTHONPATH"
export MLFLOW_ALLOW_FILE_STORE=true

# Crear las carpetas de datos si no existen (se borran por estar en .gitignore)
mkdir -p "$SCRIPT_DIR/UCFatigue/pipeline/data/artifacts"
mkdir -p "$SCRIPT_DIR/UCFatigue/pipeline/metadata"

# Verificar que surrogate_factory se puede importar
if ! python -c "import surrogate_factory" 2>/dev/null; then
    echo "ERROR: surrogate_factory no se puede importar."
    echo "  PYTHONPATH: $PYTHONPATH"
    echo "  Python:     $(which python)"
    exit 1
fi
echo "surrogate_factory OK"

if [[ "$1" == "--mlflow" ]]; then
    echo "Lanzando MLflow UI en http://localhost:5000  (tracking: $MLRUNS)"
    mlflow ui --backend-store-uri "file://$MLRUNS" --port 5000 &
    MLFLOW_PID=$!
    echo "MLflow PID: $MLFLOW_PID"
fi

echo "Lanzando JupyterLab en: $SCRIPT_DIR/UCFatigue/pipeline"
jupyter lab --notebook-dir="$SCRIPT_DIR/UCFatigue/pipeline"

# Cuando se cierra JupyterLab, para también el MLflow UI
if [[ -n "$MLFLOW_PID" ]]; then
    kill "$MLFLOW_PID" 2>/dev/null
fi
