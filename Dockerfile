# 1. Chọn hệ điều hành nền (Python 3.10 nhẹ và ổn định)
FROM python:3.10-slim

# 2. Cài đặt các công cụ hệ thống cần thiết (FFmpeg cho âm thanh, GCC cho ChromaDB)
RUN apt-get update && apt-get install -y \
    build-essential \
    ffmpeg \
    git \
    && rm -rf /var/lib/apt/lists/*

# 3. Thiết lập thư mục làm việc
WORKDIR /app

# 4. Copy file thư viện và cài đặt
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 5. Copy toàn bộ code vào trong hộp
COPY . .

# 6. Tạo các thư mục cần thiết để tránh lỗi Permission
RUN mkdir -p uploads projects db_knowledge && chmod 777 uploads projects db_knowledge

# 7. Mở cổng 8080 (Cổng giao tiếp với thế giới bên ngoài)
ENV PORT=8080
EXPOSE 8080

# 8. Lệnh kích hoạt J.A.R.V.I.S
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8080"]