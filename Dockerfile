# 1. Chọn hệ điều hành nền (Khuyên dùng Python 3.11 cho AI ổn định nhất)
# Nếu bạn bắt buộc dùng 3.13 thì giữ nguyên, nhưng 3.11 sẽ ít lỗi vặt hơn.
FROM python:3.13-slim

# 2. Cài đặt công cụ hệ thống
# - build-essential: Để biên dịch các thư viện C++ (quan trọng cho AI)
# - ffmpeg: Xử lý âm thanh/video
# - ca-certificates: BẮT BUỘC để DuckDuckGo/Requests không bị lỗi SSL
# - libxml2-dev & libxslt-dev: Hỗ trợ mạnh cho BeautifulSoup4 đọc web
RUN apt-get update && apt-get install -y \
    build-essential \
    ffmpeg \
    git \
    curl \
    ca-certificates \
    libxml2-dev \
    libxslt-dev \
    && rm -rf /var/lib/apt/lists/*

# 3. Thiết lập thư mục làm việc
WORKDIR /app

# 4. Copy và cài đặt thư viện Python
COPY requirements.txt .

# Nâng cấp pip trước để tránh lỗi cài đặt các gói mới
RUN pip install --upgrade pip && \
    pip install --no-cache-dir --root-user-action=ignore -r requirements.txt

# 5. Copy toàn bộ mã nguồn dự án vào
COPY . .

# 6. Tạo các thư mục dữ liệu và cấp quyền ghi
# Thêm folder 'logs' để chứa file brain.log
RUN mkdir -p uploads projects db_knowledge backups logs \
    && chmod -R 777 uploads projects db_knowledge backups logs

# --- TẠO SCRIPT KHỞI CHẠY (Fix lỗi dòng Windows CRLF) ---
RUN echo '#!/bin/bash' > start.sh \
    && echo 'echo "🧠 KHOI DONG AI BRAIN TRONG NEN..."' >> start.sh \
    # Chuyển log vào thư mục logs cho gọn
    && echo 'python main.py > logs/brain.log 2>&1 &' >> start.sh \
    && echo 'echo "🚀 KHOI DONG API SERVER TREN CONG $PORT..."' >> start.sh \
    && echo 'uvicorn api_server:app --host 0.0.0.0 --port $PORT' >> start.sh \
    && chmod +x start.sh

# 7. Khai báo cổng
ENV PORT=8080
EXPOSE 8080

# 8. Kích hoạt
CMD ["./start.sh"]
