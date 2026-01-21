import os
from openai import OpenAI

# Khởi tạo Client (Nó sẽ tự lấy key từ biến môi trường OPENAI_API_KEY)
client = OpenAI()

def text_to_speech_file(text, filename="output.mp3"):
    """
    Chuyển văn bản thành giọng nói chuẩn người thật bằng OpenAI
    """
    try:
        # OpenAI có các giọng: alloy, echo, fable, onyx, nova, shimmer
        # 'alloy' và 'shimmer' thường đọc tiếng Việt khá mượt
        response = client.audio.speech.create(
            model="tts-1",          # Model chuẩn (nhanh). Muốn hay hơn nữa thì dùng 'tts-1-hd'
            voice="alloy",          # Chọn giọng đọc (Nam/Nữ/Trầm/Bổng)
            input=text
        )

        # Lưu file vào thư mục tĩnh để web có thể phát được
        # Trên Cloud Run nên lưu vào /tmp nếu không cần lưu trữ lâu dài
        # Nhưng để web truy cập được, ta lưu vào static/audio
        
        save_path = os.path.join("static", "audio", filename)
        
        # Đảm bảo thư mục tồn tại
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        response.stream_to_file(save_path)
        return f"/static/audio/{filename}"
        
    except Exception as e:
        print(f"Lỗi tạo giọng nói: {e}")
        return None