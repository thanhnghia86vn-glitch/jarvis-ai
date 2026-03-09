import os
import asyncio
import logging
import edge_tts
from io import BytesIO

logger = logging.getLogger("VOICE_ENGINE_FREE")

# --- 1. HÀM NÓI MIỄN PHÍ (Edge-TTS) ---
# Giọng 'vi-VN-HoaiMyNeural' hoặc 'vi-VN-NamMinhNeural' cực kỳ tự nhiên
async def generate_voice_free(text, voice="vi-VN-HoaiMyNeural"):
    """
    Tạo giọng nói AI chất lượng cao miễn phí (Neural Voice)
    """
    if not text: return None
    try:
        communicate = edge_tts.Communicate(text, voice)
        audio_data = b""
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                audio_data += chunk["data"]
        
        logger.info(f"✅ JARVIS đã cất lời (Free Neural): {len(audio_data)} bytes")
        return audio_data
    except Exception as e:
        logger.error(f"❌ Lỗi giọng nói miễn phí: {e}")
        return None

# --- 2. HÀM NGHE MIỄN PHÍ (Google Speech Recognition) ---
def transcribe_audio_free(audio_bytes):
    """
    Dùng thư viện SpeechRecognition (miễn phí) để dịch giọng Commander
    """
    import speech_recognition as sr
    
    r = sr.Recognizer()
    audio_file = BytesIO(audio_bytes)
    
    # Chuyển đổi bytes sang định dạng mà SpeechRecognition hiểu được
    # (Lưu ý: Cần cài thêm 'pydub' và 'ffmpeg' trên Cloud để xử lý)
    try:
        with sr.AudioFile(audio_file) as source:
            audio = r.record(source)
            text = r.recognize_google(audio, language="vi-VN")
            return text
    except Exception as e:
        logger.error(f"❌ JARVIS không nghe rõ: {e}")
        return None
