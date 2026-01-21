import os
import logging
from openai import OpenAI

# Cấu hình Log để dễ debug trên Render
logger = logging.getLogger("VOICE_ENGINE")

# --- 1. KHỞI TẠO CLIENT AN TOÀN (CHỐNG SẬP SERVER) ---
client = None
try:
    api_key = os.getenv("OPENAI_API_KEY")
    if api_key:
        client = OpenAI(api_key=api_key)
        logger.info("✅ VOICE ENGINE: Đã kết nối OpenAI TTS.")
    else:
        logger.warning("⚠️ VOICE ENGINE: Không tìm thấy OPENAI_API_KEY. Chức năng nói sẽ TẮT.")
except Exception as e:
    logger.error(f"❌ Lỗi khởi tạo Voice Client: {e}")

# --- 2. HÀM CHUYỂN ĐỔI ---
def text_to_speech_file(text, filename="output.mp3"):
    """
    Chuyển văn bản thành giọng nói (OpenAI TTS-1)
    """
    # Nếu client chưa khởi tạo được (do thiếu key), trả về None ngay
    if not client:
        return None

    try:
        # Gọi API OpenAI (Alloy là giọng khá mượt cho tiếng Việt)
        response = client.audio.speech.create(
            model="tts-1",
            voice="alloy",
            input=text
        )

        # Định nghĩa đường dẫn lưu file (Trong thư mục static để Web truy cập được)
        # Lưu ý: Trên Render, thư mục này sẽ bị reset khi deploy lại (nhưng không sao với file cache)
        save_path = os.path.join("static", "audio", filename)
        
        # Đảm bảo thư mục tồn tại
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        # [CẬP NHẬT] Sử dụng cách ghi file Binary chuẩn (stream_to_file sắp bị khai tử)
        with open(save_path, "wb") as f:
            f.write(response.content)

        # Trả về đường dẫn tương đối cho Frontend phát
        return f"/static/audio/{filename}"
        
    except Exception as e:
        logger.error(f"❌ Lỗi tạo giọng nói: {e}")
        return None
