# 1. Chọn hệ điều hành nền (Python 3.10 ổn định)
FROM python:3.10-slim

# 2. Cài đặt công cụ hệ thống (FFmpeg cho âm thanh, Git, Curl)
RUN apt-get update && apt-get install -y \
    build-essential \
    ffmpeg \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# 3. Thiết lập thư mục làm việc
WORKDIR /app

# 4. Copy và cài đặt thư viện Python trước (để tận dụng Cache)
COPY requirements.txt .
RUN pip install --no-cache-dir --root-user-action=ignore -r requirements.txt
# 5. Copy toàn bộ mã nguồn dự án vào
COPY . .

# 6. Tạo các thư mục dữ liệu và cấp quyền ghi (Tránh lỗi Permission denied)
RUN mkdir -p uploads projects db_knowledge backups \
    && chmod -R 777 uploads projects db_knowledge backups

# --- PHẦN QUAN TRỌNG NHẤT: TẠO SCRIPT KHỞI CHẠY TRỰC TIẾP ---
# Kỹ thuật này giúp tránh lỗi xuống dòng (CRLF) của Windows 100%
RUN echo '#!/bin/bash' > start.sh \
    && echo 'echo "🧠 KHOI DONG AI BRAIN (Background)..."' >> start.sh \
    && echo 'python main.py &' >> start.sh \
    && echo 'echo "🚀 KHOI DONG API SERVER (Foreground)..."' >> start.sh \
    && echo 'uvicorn api_server:app --host 0.0.0.0 --port ${PORT:-8080}' >> start.sh \
    && chmod +x start.sh

# 7. Khai báo cổng (Render sẽ tự map, nhưng khai báo cho chuẩn)
ENV PORT=8080
EXPOSE 8080

# 8. Lệnh kích hoạt hệ thống (Chạy file script vừa tạo ở trên)
CMD ["./start.sh"]

