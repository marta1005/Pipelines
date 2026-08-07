#!/usr/bin/env bash
# Lanza JupyterLab para cualquier use case del repositorio.
# Uso:
#   ./start_jupyter.sh                  → abre en la raíz del repo
#   ./start_jupyter.sh UCHardLanding    → abre en UCHardLanding/pipeline
#   ./start_jupyter.sh UCFatigue        → abre en UCFatigue/pipeline
#   ./start_jupyter.sh --mlflow         → JupyterLab + MLflow UI (UCFatigue por defecto)
#   ./start_jupyter.sh UCLoads --mlflow → JupyterLab + MLflow UI para UCLoads

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="$SCRIPT_DIR/.venv"

# Parsear argumentos
UC=""
MLFLOW_FLAG=""
for arg in "$@"; do
    if [[ "$arg" == "--mlflow" ]]; then
        MLFLOW_FLAG="yes"
    elif [[ -z "$UC" ]]; then
        UC="$arg"
    fi
done

# Carpeta de notebooks
if [[ -n "$UC" ]]; then
    NOTEBOOK_DIR="$SCRIPT_DIR/$UC/pipeline"
else
    NOTEBOOK_DIR="$SCRIPT_DIR"
fi

if [ ! -d "$VENV" ]; then
    echo "ERROR: entorno virtual no encontrado en $VENV"
    echo "Crea el entorno con: python3.11 -m venv $VENV"
    exit 1
fi

source "$VENV/bin/activate"

# macOS marca los .pth que empiezan por __ como "hidden" (flag UF_HIDDEN),
# y Homebrew Python los salta silenciosamente.
chflags -R nohidden "$VENV/lib/" 2>/dev/null || true

# PYTHONPATH añade el src/ al path de Python sin depender de los .pth.
export PYTHONPATH="$SCRIPT_DIR/src:$PYTHONPATH"
export MLFLOW_ALLOW_FILE_STORE=true

# Crear carpetas de datos para todos los use cases
for uc_name in UCFatigue UCLoads UCAirfoils UCHardLanding UCCpHTP; do
    mkdir -p "$SCRIPT_DIR/$uc_name/pipeline/data/artifacts"
    mkdir -p "$SCRIPT_DIR/$uc_name/pipeline/metadata"
done

# Verificar que surrogate_factory se puede importar
if ! python -c "import surrogate_factory" 2>/dev/null; then
    echo "ERROR: surrogate_factory no se puede importar."
    echo "  PYTHONPATH: $PYTHONPATH"
    echo "  Python:     $(which python)"
    exit 1
fi
echo "surrogate_factory OK"

if [[ -n "$MLFLOW_FLAG" ]]; then
    MLRUNS_UC="${UC:-UCFatigue}"
    MLRUNS="$SCRIPT_DIR/$MLRUNS_UC/pipeline/mlruns"
    echo "Lanzando MLflow UI en http://localhost:5000  (tracking: $MLRUNS)"
    mlflow ui --backend-store-uri "file://$MLRUNS" --port 5000 &
    MLFLOW_PID=$!
    echo "MLflow PID: $MLFLOW_PID"
fi

echo "Lanzando JupyterLab en: $NOTEBOOK_DIR"
jupyter lab --notebook-dir="$NOTEBOOK_DIR"

# Cuando se cierra JupyterLab, para también el MLflow UI
if [[ -n "$MLFLOW_PID" ]]; then
    kill "$MLFLOW_PID" 2>/dev/null
fi
