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

# --- Phần mới: Kiểm tra và giải phóng cổng (Bản sửa lỗi đa tiến trình) ---
echo "  Đang kiểm tra cổng $PORT..."
PIDS=$(lsof -t -i:"$PORT")

if [ -n "$PIDS" ]; then
    echo "  [!] Cổng $PORT đang bị chiếm bởi các PID: $PIDS"
    echo "  [+] Đang dừng các tiến trình cũ..."
    # Dùng vòng lặp để kill từng PID một cách an toàn
    for PID in $PIDS; do
        kill -9 "$PID" 2>/dev/null
    done
    sleep 2 # Tăng thời gian chờ lên 2 giây để giải phóng hẳn
    echo "  [✓] Đã giải phóng cổng $PORT."
else
    echo "  [✓] Cổng $PORT đã sẵn sàng."
fi
# -----------------------------------------------------------------------

echo ""
echo "  Model   : checkpoints/best_model.pth"
echo "  Port    : http://localhost:$PORT"
echo "  Ctrl+C để dừng server"
echo ""

# Activate venv
if [ -d "$SCRIPT_DIR/venv" ]; then
    source "$SCRIPT_DIR/venv/bin/activate"
else
    echo "  [!] Không tìm thấy thư mục venv tại $SCRIPT_DIR/venv"
fi

# Run FastAPI
uvicorn web.main:app \
  --host 0.0.0.0 \
  --port "$PORT" \
  --reload \
  --reload-dir web \
  --log-level info