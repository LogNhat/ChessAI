#!/usr/bin/env bash
#
# run_api.sh — Run Chess AI REST API Service
# Usage: bash run_api.sh [port]
#

PORT=${1:-3210}
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  ♛  Chess AI REST API Service"
echo "  Fast, lightweight, developer-friendly backend"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# --- Check and free port if in use ---
echo "  Checking port $PORT..."
PIDS=$(lsof -t -i:"$PORT")

if [ -n "$PIDS" ]; then
    echo "  [!] Port $PORT is occupied by PID(s): $PIDS"
    echo "  [+] Stopping existing processes..."
    for PID in $PIDS; do
        kill -9 "$PID" 2>/dev/null
    done
    sleep 2
    echo "  [✓] Port $PORT has been successfully freed."
else
    echo "  [✓] Port $PORT is ready."
fi
# -------------------------------------

echo ""
echo "  Models  : Auto-discovering all *.pth in checkpoints/"
echo "  Port    : http://localhost:$PORT"
echo "  Access  : CORS enabled for local applications"
echo "  Docs    : http://localhost:$PORT/docs (OpenAPI Swagger UI)"
echo "  Press Ctrl+C to stop the API server"
echo ""

# Activate virtual environment
if [ -d "$SCRIPT_DIR/venv" ]; then
    source "$SCRIPT_DIR/venv/bin/activate"
elif [ -d "$SCRIPT_DIR/env" ] && [ -f "$SCRIPT_DIR/env/bin/activate" ]; then
    # In case the python environment is named env instead of venv
    source "$SCRIPT_DIR/env/bin/activate"
else
    echo "  [!] Virtual environment directory not found. Running with global python packages."
fi

# Run Uvicorn pointing to the api module
uvicorn api.main:app \
  --host 0.0.0.0 \
  --port "$PORT" \
  --reload \
  --reload-dir api \
  --log-level info
