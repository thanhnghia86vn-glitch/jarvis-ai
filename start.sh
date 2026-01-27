#!/bin/bash

# 1. Chạy AI Brain (main.py) dưới dạng tiến trình ngầm (Background Process)
# Dấu '&' ở cuối giúp nó chạy ẩn và không chặn dòng lệnh tiếp theo
echo "🧠 Đang khởi động AI Brain..."
python main.py &

# 2. Chạy API Server (api_server.py) ở chế độ ưu tiên (Foreground)
# Lưu ý: Render sẽ cấp biến môi trường $PORT, ta phải dùng nó.
echo "🚀 Đang khởi động API Server..."
uvicorn api_server:app --host 0.0.0.0 --port $PORT