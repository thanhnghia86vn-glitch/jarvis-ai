import asyncio
import websockets
import speech_recognition as sr
import threading
import queue
import sys
import os
import time
import requests
import pygame # Dùng để phát file MP3 từ Google
from termcolor import colored

# ==========================================
# CẤU HÌNH HỆ THỐNG
# ==========================================
SERVER_URL = "https://jarvis-ai-qklx.onrender.com/"      # Để gọi API giọng nói
SERVER_WS_URL = "wss://jarvis-ai-qklx.onrender.com/ws/nexus" # Để chat nhanh
EXIT_COMMANDS = ["tắt máy", "ngủ đi", "thoát", "exit", "dừng lại"]
TEMP_AUDIO_FILE = "temp_reply.mp3"

# --- CẤU HÌNH ĐÁNH THỨC ---
WAKE_WORDS = ["trợ lý", "jarvis", "ê", "dậy đi", "alo"] # Từ khóa đánh thức
IDLE_TIMEOUT = 30 # Thời gian tự ngủ lại (giây) nếu không nói gì

is_awake = False
last_interaction_time = 0
msg_queue = queue.Queue()

# Khởi tạo Loa
try:
    pygame.mixer.init()
except:
    print("❌ Lỗi khởi tạo Pygame Mixer")

# ==========================================
# 1. MODULE NÓI (GỌI SERVER GOOGLE TTS)
# ==========================================
def speak_now(text):
    """
    Gửi văn bản lên Server để lấy file MP3 (Google) về phát.
    """
    if not text: return

    def _run_speak():
        try:
            # Gọi API TTS của Server (Server sẽ dùng gTTS tạo file)
            response = requests.post(
                f"{SERVER_URL}/api/tts",
                json={"text": text},
                timeout=10
            )
            
            if response.status_code == 200:
                # Lưu file MP3
                with open(TEMP_AUDIO_FILE, "wb") as f:
                    f.write(response.content)
                
                # Phát file MP3
                if pygame.mixer.music.get_busy():
                    pygame.mixer.music.stop()
                    
                pygame.mixer.music.load(TEMP_AUDIO_FILE)
                pygame.mixer.music.play()
                
                # Chờ đọc xong
                while pygame.mixer.music.get_busy():
                    pygame.time.Clock().tick(10)
                    
                pygame.mixer.music.unload()
            else:
                print(colored("⚠️ Server TTS lỗi.", "red"))

        except Exception as e:
            print(colored(f"⚠️ Lỗi loa: {e}", "red"))

    # Khởi động luồng nói
    t = threading.Thread(target=_run_speak)
    t.start()

# ==========================================
# 2. MODULE NGHE (MIC LISTENER)
# ==========================================
def microphone_listener():
    global is_awake, last_interaction_time
    
    recognizer = sr.Recognizer()
    microphone = sr.Microphone()
    
    # Tăng độ nhạy để bắt từ khóa tốt hơn
    recognizer.energy_threshold = 300
    recognizer.dynamic_energy_threshold = True
    
    with microphone as source:
        recognizer.adjust_for_ambient_noise(source, duration=1)
        print(colored("\n🎧 HỆ THỐNG ĐÃ SẴN SÀNG.", "cyan", attrs=["bold"]))
        print(colored(f"👉 Hãy gọi: {', '.join(WAKE_WORDS)} để kích hoạt.", "white"))
        
        while True:
            try:
                # Kiểm tra xem có nên đi ngủ không
                if is_awake and (time.time() - last_interaction_time > IDLE_TIMEOUT):
                    is_awake = False
                    print(colored("\n💤 Hết thời gian chờ. Đang ngủ đông...", "blue"))
                    # (Optional) speak_now("Tôi đi ngủ đây.")

                # Hiển thị trạng thái
                if is_awake:
                    print(colored("\r🟢 [ON AIR] Đang nghe lệnh...", "green", attrs=["bold"]), end="", flush=True)
                else:
                    print(colored("\r⚫ [SLEEP] Chờ gọi tên...", "grey"), end="", flush=True)

                # Nghe
                audio = recognizer.listen(source, timeout=None, phrase_time_limit=5)
                
                try:
                    text = recognizer.recognize_google(audio, language="vi-VN").lower()
                    
                    if text:
                        # --- LOGIC ĐÁNH THỨC ---
                        if not is_awake:
                            # Kiểm tra xem câu nói có chứa từ khóa không
                            if any(w in text for w in WAKE_WORDS):
                                is_awake = True
                                last_interaction_time = time.time()
                                print(colored(f"\n⚡ ĐÃ NGHE: '{text}' -> KÍCH HOẠT!", "yellow"))
                                speak_now("Dạ, tôi đây.") # Phản hồi để CEO biết
                            else:
                                # Nếu đang ngủ mà nói linh tinh thì lờ đi
                                pass
                        
                        # --- LOGIC HỘI THOẠI ---
                        else:
                            # Đang thức -> Gửi lệnh đi xử lý
                            print(colored(f"\n🗣️ [CEO]: {text}", "green"))
                            last_interaction_time = time.time() # Reset đồng hồ đếm ngược
                            msg_queue.put(text)
                            
                            # Kiểm tra lệnh tắt
                            if any(cmd in text for cmd in EXIT_COMMANDS):
                                print(colored("🛑 Đang tắt...", "red"))
                                is_awake = False # Hoặc break nếu muốn thoát hẳn

                except sr.UnknownValueError: pass
                except sr.RequestError: pass
                
            except Exception as e:
                print(f"\n⚠️ Lỗi Mic: {e}")
                # Reset lại mic nếu lỗi driver
                recognizer = sr.Recognizer() 
                time.sleep(1)

# ==========================================
# 3. MODULE GIAO TIẾP (WEBSOCKETS)
# ==========================================
async def nexus_communicator():
    print(colored(f"🚀 Đang kết nối Server...", "yellow"))
    
    try:
        async with websockets.connect(SERVER_WS_URL) as websocket:
            print(colored("✅ KẾT NỐI THÀNH CÔNG!", "green", attrs=["bold"]))
            
            # --- LUỒNG NHẬN ---
            async def receive_messages():
                try:
                    async for message in websocket:
                        print(colored(f"\n🤖 [J.A.R.V.I.S]: {message}", "magenta", attrs=["bold"]))
                        
                        # PHÁT RA LOA (Gọi hàm speak_now ở trên)
                        speak_now(message)
                        global last_interaction_time
                        last_interaction_time = time.time()
                        print(colored("🎤...", "cyan"))
                except websockets.exceptions.ConnectionClosed:
                    print(colored("❌ Mất kết nối Server.", "red"))

            # --- LUỒNG GỬI ---
            async def send_messages():
                while True:
                    if not msg_queue.empty():
                        msg = msg_queue.get()
                        await websocket.send(msg)
                        # if any(cmd in msg.lower() for cmd in EXIT_COMMANDS):
                        #     await websocket.close()
                        #     sys.exit(0)
                    await asyncio.sleep(0.1)

            await asyncio.gather(receive_messages(), send_messages())

    except Exception as e:
        print(colored(f"❌ Không thể kết nối Server: {e}", "red"))

# ==========================================
# 4. MAIN ENTRY
# ==========================================
if __name__ == "__main__":
    os.system('cls' if os.name == 'nt' else 'clear')
    mic_thread = threading.Thread(target=microphone_listener, daemon=True)
    mic_thread.start()
    
    try:
        asyncio.run(nexus_communicator())
    except KeyboardInterrupt:
        print("\n👋 Tạm biệt CEO.")
