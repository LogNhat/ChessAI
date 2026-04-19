#!/usr/bin/env bash
#
# Chạy Chess AI Web Server
# Usage: bash run_web.sh [port]
#

PORT=${1:-8000}
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  ♛  Chess AI Web Server"
echo "  SE-ResNet 12×192 · 20M moves"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "  Model   : checkpoints/best_model.pth"
echo "  Port    : http://localhost:$PORT"
echo "  Ctrl+C để dừng server"
echo ""

# Activate venv
source "$SCRIPT_DIR/venv/bin/activate"

# Run FastAPI
uvicorn web.main:app \
  --host 0.0.0.0 \
  --port "$PORT" \
  --reload \
  --reload-dir web \
  --log-level info
