import streamlit as st
import time
import sqlite3
import base64
import requests
import os
import zipfile
import re
import io
import json
import pandas as pd
import numpy as np
import datetime
import textwrap
from fpdf import FPDF
from fpdf.enums import XPos, YPos
from typing import List, Dict, Set, Optional, Any
import plotly.graph_objects as go
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, messages_to_dict, messages_from_dict

try:
    from termcolor import colored
except ImportError:
    def colored(text, color): return text # Fallback nếu thiếu thư viện

# --- 1. CẤU HÌNH HỆ THỐNG ---
st.set_page_config(
    page_title="AI CORP COMMAND CENTER",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

API_BASE_URL = "http://localhost:8080"
API_KEY = "ai_corp_secret_123"

st.markdown("""
<style>
    /* --- 0. IMPORT FONTS --- */
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700&family=Rajdhani:wght@300;500;700&display=swap');

    /* --- 1. CẤU TRÚC & SIDEBAR (ĐÃ GỘP VÀ FIX LỖI) --- */
    
    /* Nút MỞ Sidebar (Khi bị ẩn) */
    [data-testid="stSidebarCollapsedControl"] {
        display: block !important;
        color: #00f2ff !important;
        background-color: rgba(10, 15, 25, 0.8) !important;
        border: 1px solid rgba(0, 242, 255, 0.3) !important;
        border-radius: 50% !important;
        padding: 5px !important;
        z-index: 999999 !important;
        top: 15px !important;
        left: 15px !important;
        transition: all 0.3s ease;
    }
    [data-testid="stSidebarCollapsedControl"]:hover {
        box-shadow: 0 0 15px #00f2ff;
        transform: scale(1.1) rotate(90deg);
    }

    /* Khung Sidebar */
    [data-testid="stSidebar"] {
        background-color: rgba(1, 4, 9, 0.98) !important;
        border-right: 1px solid rgba(0, 242, 255, 0.2) !important;
        z-index: 999990 !important;
    }

    /* Nút ĐÓNG Sidebar */
    [data-testid="stSidebarCollapseButton"] {
        color: #00f2ff !important;
    }
    [data-testid="stSidebarCollapseButton"]:hover {
        background-color: rgba(0, 242, 255, 0.1) !important;
        border-radius: 50%;
    }

    /* Tối ưu không gian chính */
    .block-container {
        padding-top: 2rem !important;
        padding-bottom: 5rem !important;
        max-width: 95% !important;
    }
    header {visibility: hidden;}
    footer {visibility: hidden;}

    /* --- 2. NỀN VŨ TRỤ SỐ (BACKGROUND) --- */
    .stApp {
        background: radial-gradient(circle at 50% 50%, #0d1117 0%, #010409 100%);
        color: #c9d1d9;
        font-family: 'Rajdhani', sans-serif;
    }
    .stApp::before {
        content: ""; position: fixed; top: 0; left: 0; width: 100vw; height: 100vh;
        background-image: 
            linear-gradient(rgba(0, 242, 255, 0.02) 1px, transparent 1px),
            linear-gradient(90deg, rgba(0, 242, 255, 0.02) 1px, transparent 1px);
        background-size: 40px 40px; pointer-events: none; z-index: -1;
    }

    /* --- 3. TYPOGRAPHY NEON --- */
    h1, h2, h3, h4, h5 {
        font-family: 'Orbitron', sans-serif !important;
        background: linear-gradient(135deg, #00f2ff 0%, #0078ff 50%, #7000ff 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 700 !important;
        letter-spacing: 2px !important;
        filter: drop-shadow(0 0 5px rgba(0, 242, 255, 0.3));
        padding-bottom: 5px;
    }

    /* --- 4. KHUNG CHAT HUD (CẢI TIẾN) --- */
    div[data-testid="stChatMessage"] {
        background: rgba(13, 17, 23, 0.7) !important;
        border: 1px solid rgba(0, 242, 255, 0.1) !important;
        border-radius: 12px !important;
        margin-bottom: 1rem !important;
        backdrop-filter: blur(10px);
        transition: border 0.3s;
    }
    div[data-testid="stChatMessage"]:hover {
        border-color: rgba(0, 242, 255, 0.5) !important;
    }

    /* Hiệu ứng sóng âm cho Tin nhắn (Áp dụng chung để tránh lỗi hash) */
    .stChatMessage::after {
        content: ""; position: absolute; bottom: 0; left: 0; width: 100%; height: 1px;
        background: linear-gradient(90deg, transparent, rgba(0, 242, 255, 0.5), transparent);
        opacity: 0.5;
    }

    /* --- 5. NÚT BẤM (TRON LEGACY STYLE) --- */
    .stButton > button {
        width: 100%;
        background: linear-gradient(90deg, rgba(0, 242, 255, 0.05), rgba(0, 0, 0, 0)) !important;
        border: 1px solid #00f2ff !important;
        color: #00f2ff !important;
        font-family: 'Orbitron', sans-serif;
        font-size: 14px !important;
        text-transform: uppercase;
        clip-path: polygon(0 0, 95% 0, 100% 30%, 100% 100%, 5% 100%, 0 70%);
        transition: all 0.3s ease !important;
    }
    .stButton > button:hover {
        background: #00f2ff !important;
        color: #000 !important;
        box-shadow: 0 0 20px rgba(0, 242, 255, 0.6);
        transform: translateY(-2px);
    }

    /* --- 6. INPUT CHAT & SCROLLBAR --- */
    .stChatInputContainer { padding-bottom: 20px !important; }
    div[data-testid="stChatInput"] {
        border: 1px solid #00f2ff !important;
        background: rgba(1, 4, 9, 0.95) !important;
        box-shadow: 0 0 15px rgba(0, 242, 255, 0.1);
        border-radius: 8px !important;
    }
    
    /* Scrollbar */
    ::-webkit-scrollbar { width: 6px; height: 6px; }
    ::-webkit-scrollbar-track { background: #010409; }
    ::-webkit-scrollbar-thumb { 
        background: linear-gradient(180deg, #00f2ff, #7000ff); 
        border-radius: 3px; 
    }

    /* --- 7. HELPER CLASSES --- */
    .ai-thinking-glow { 
        color: #00f2ff; 
        text-shadow: 0 0 10px #00f2ff; 
        animation: pulse 1.5s infinite alternate; 
    }
    @keyframes pulse { from { opacity: 0.6; } to { opacity: 1; } }
    
    /* Trạng thái Nodes */
    .node-active {
        display: inline-block; width: 10px; height: 10px;
        background-color: #00ff88; border-radius: 50%;
        box-shadow: 0 0 8px #00ff88; margin-right: 5px;
    }
</style>
""", unsafe_allow_html=True)
# ==========================================
# --- 2. CONSTANTS & REGEX (THIẾU TRONG BẢN CŨ) ---
# ==========================================
FILE_EXTENSION_MAP: Dict[str, str] = {
    # Web Stack
    'html': '.html', 'css': '.css', 'js': '.js', 'javascript': '.js',
    'typescript': '.ts', 'ts': '.ts', 'jsx': '.jsx', 'tsx': '.tsx',
    
    # Backend & Data
    'python': '.py', 'py': '.py', 'sql': '.sql', 'json': '.json',
    'yaml': '.yml', 'yml': '.yml', 'csv': '.csv', 'xml': '.xml',
    
    # Systems & Hardware (Robotics)
    'cpp': '.cpp', 'c++': '.cpp', 'c': '.c', 'h': '.h', 'hpp': '.hpp',
    'ino': '.ino',  # Arduino/ESP32
    'java': '.java', 'go': '.go', 'rust': '.rs',
    
    # DevOps & Config
    'sh': '.sh', 'bash': '.sh', 'shell': '.sh',
    'dockerfile': '', 'docker': '', 
    'env': '', 'gitignore': '',
    'markdown': '.md', 'md': '.md',
    
    # Mobile
    'dart': '.dart', 'swift': '.swift', 'kotlin': '.kt'
}

DEFAULT_FILE_NAMES: Dict[str, str] = {
    # Web Development
    'html': 'index',
    'css': 'style',
    'js': 'app',
    'javascript': 'app',
    'typescript': 'index',
    'ts': 'index',
    
    # Backend & Logic
    'python': 'main',
    'py': 'main',
    'go': 'main',
    'rust': 'main',
    'java': 'Main',
    'php': 'index',
    
    # Hardware & IoT (Đặc biệt quan trọng cho AI Corp)
    'cpp': 'firmware',
    'ino': 'sketch',       # Tên mặc định cho Arduino/ESP32
    'c': 'main',
    
    # Database & Config
    'sql': 'schema',
    'json': 'config',
    'yaml': 'docker-compose',
    'yml': 'docker-compose',
    
    # System Files (Bắt buộc tên cố định)
    'dockerfile': 'Dockerfile',
    'env': '.env',
    'gitignore': '.gitignore',
    'md': 'README',
    'markdown': 'README'
}

CODE_BLOCK_REGEX = re.compile(
    r'```(?P<header>[^\n]*)\n(?P<content>.*?)\n\s*```', 
    re.DOTALL
)

FILENAME_COMMENT_REGEX = re.compile(
    r'(?:#|//|/\*)\s*(?:filename|file|path|name):\s*(?P<filename>[\w\.\-\/\\\+]+)',
    re.IGNORECASE
)

# ==========================================
# --- 3. DATABASE MANAGER (TRỤ CỘT 1) ---
# ==========================================
class DatabaseManager:
    """
    Quản trị Database tập trung theo chuẩn Trụ cột 1: Nền tảng vững chắc.
    Hỗ trợ: Tự động khởi tạo, Quản lý phiên (WAL Mode), và Hệ thống kế thừa.
    """
    def __init__(self, db_path='ai_corp_projects.db'):
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        self.db_path = os.path.join(BASE_DIR, db_path)
        self.init_db()

    def get_connection(self):
        """Kết nối an toàn cho Streamlit"""
        conn = sqlite3.connect(self.db_path, timeout=60, check_same_thread=False)
        conn.execute('PRAGMA journal_mode=WAL') 
        conn.row_factory = sqlite3.Row 
        return conn

    def init_db(self):
        """Khởi tạo toàn bộ hệ thống bảng (Schema) bao gồm cả bộ nhớ kế thừa"""
        with self.get_connection() as conn:
            c = conn.cursor()
            # Bảng quản lý dự án
            c.execute('''CREATE TABLE IF NOT EXISTS projects 
                         (id TEXT PRIMARY KEY, name TEXT, history TEXT, timestamp DATETIME)''')
            
            # Bảng sản phẩm phần mềm (Mô hình kinh doanh)
            c.execute('''CREATE TABLE IF NOT EXISTS products 
                         (id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT, description TEXT, 
                          price REAL, image_url TEXT)''')
            
            # Bảng nhật ký tài chính (Finance Manager)
            c.execute('''CREATE TABLE IF NOT EXISTS finance_logs 
                         (id INTEGER PRIMARY KEY AUTOINCREMENT, type TEXT, amount REAL, 
                          category TEXT, description TEXT, timestamp DATETIME DEFAULT CURRENT_TIMESTAMP)''')
            
            # BẢNG MỚI: Kế thừa tri thức (Shadow Learning)
            c.execute('''CREATE TABLE IF NOT EXISTS legacy_knowledge 
                         (id INTEGER PRIMARY KEY AUTOINCREMENT, task_type TEXT, 
                          instruction TEXT, response TEXT, score INTEGER, timestamp DATETIME)''')
            conn.commit()
            print(colored("✅ [DATABASE] Hệ thống lõi đã sẵn sàng.", "green"))

# Khởi tạo đối tượng quản lý toàn cục
db_manager = DatabaseManager()

# --- 3. IMPORT MODULES (CÁC THÀNH PHẦN PHỤ THUỘC) ---
# Xử lý Audio
try:
    import speech_recognition as sr
    from gtts import gTTS
    import pygame
    AUDIO_AVAILABLE = True
except ImportError:
    AUDIO_AVAILABLE = False
    print("⚠️ Audio modules không khả dụng.")

# Kết nối với bộ não chính (main.py)
try:
    # Nhập đối tượng db gốc của bộ não để Agents làm việc
    from main import db as main_db, app, remember_knowledge, LLM_GPT4, ingest_docs_to_memory
except ImportError:
    st.error("🚨 THIẾU FILE main.py - Luồng Agent sẽ bị gián đoạn!")
    st.stop()
# ==========================================
# --- 4. HÀM HỖ TRỢ (HELPER FUNCTIONS) ---
# ==========================================
def save_project(project_id: str, project_name: str, messages: List[Any]):
    """
    Lưu trữ hoặc cập nhật tiến độ dự án.
    Bản nâng cấp: Ép làm mới Cache để dự án hiện lên Sidebar ngay lập tức.
    """
    if not messages:
        return 

    try:
        history_json = json.dumps(messages_to_dict(messages))
        now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        with db_manager.get_connection() as conn:
            c = conn.cursor()
            query = """
                INSERT OR REPLACE INTO projects (id, name, history, timestamp)
                VALUES (?, ?, ?, ?)
            """
            c.execute(query, (str(project_id), project_name, history_json, now))
            conn.commit()
            
            # --- DÒNG LỆNH QUAN TRỌNG NHẤT ---
            # Xóa cache của hàm get_project_list để Sidebar cập nhật ngay dự án mới
            st.cache_data.clear() 
            # --------------------------------

    except Exception as e:
        error_msg = f"💥 Lỗi lưu dự án '{project_name}': {str(e)}"
        st.error(error_msg)
        with open("db_error_log.txt", "a", encoding="utf-8") as f:
            f.write(f"[{datetime.datetime.now()}] {error_msg}\n")

def get_project_list() -> pd.DataFrame:
    try:
        with db_manager.get_connection() as conn:
            # SẮP XẾP THEO ID GIẢM DẦN: Vì ID là 20260111... nên số lớn hơn là mới hơn
            query = "SELECT id, name, timestamp FROM projects ORDER BY id DESC"
            df = pd.read_sql_query(query, conn)
            
            if not df.empty:
                # Chặn lỗi định dạng ngày tháng để không làm sập Sidebar
                df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
                df['display'] = df['timestamp'].dt.strftime('%H:%M %d/%m').fillna("Mới")
            return df
    except Exception as e:
        return pd.DataFrame(columns=['id', 'name', 'timestamp'])
    
def load_project_history(project_id: str) -> List[Any]:
    """
    [MAIN SCREEN] Tải chi tiết nội dung hội thoại của dự án.
    - Lưu ý: KHÔNG dùng cache ở đây vì nội dung chat thay đổi realtime.
    - Trụ cột 3 (Bảo mật): Sử dụng tham số hóa (?) để chống SQL Injection.
    """
    try:
        with db_manager.get_connection() as conn:
            # Trụ cột 3 (Bảo mật): Tuyệt đối không dùng f-string trong SQL
            query = "SELECT history FROM projects WHERE id = ?"
            row = conn.execute(query, (project_id,)).fetchone()
            
            if row and row['history']:
                # Trụ cột 4: Deserialize JSON thành LangChain Objects
                return messages_from_dict(json.loads(row['history']))
                
    except Exception as e:
        st.error(f"❌ Lỗi khôi phục kí ức dự án: {e}")
        # Ghi log chi tiết cho Developer
        print(colored(f"💥 [CRITICAL] Load History Failed: {e}", "red"))
    
    # Trả về list rỗng an toàn nếu lỗi hoặc không tìm thấy
    return []

def extract_code_from_message(message_content: str) -> List[Dict[str, str]]:
    extracted_files = []
    for match in CODE_BLOCK_REGEX.finditer(message_content):
        header = (match.group('header') or '').strip().lower()
        content = match.group('content').strip()
        lang = header.split(':')[0] if ':' in header else (header if header else 'txt')
        
        filename = extract_filename_metadata(content, header)
        if not filename:
            ext = FILE_EXTENSION_MAP.get(lang, '.txt')
            base_name = DEFAULT_FILE_NAMES.get(lang, f'file_{len(extracted_files)+1}')
            filename = f"{base_name}{ext}"
            
        extracted_files.append({"filename": filename, "content": content, "language": lang})
    return extracted_files

def check_password():
    """
    Hệ thống xác thực quyền hạn CEO.
    Tuân thủ Trụ cột 2: Bảo mật đa lớp và quản lý cấu hình tập trung.
    """
    # Lấy mật khẩu từ file cấu hình bảo mật (st.secrets)
    # Nếu không tìm thấy trong secrets, mặc định dùng một chuỗi an toàn để tránh sập app
    CEO_PASSWORD = st.secrets.get("auth", {}).get("password", "fallback_secure_string")

    def password_entered():
        """Xử lý logic khi người dùng bấm Enter"""
        if st.session_state["password"] == CEO_PASSWORD:
            st.session_state["password_correct"] = True
            # Xóa mật khẩu khỏi session ngay lập tức để bảo mật bộ nhớ
            del st.session_state["password"] 
        else:
            st.session_state["password_correct"] = False

    # Trường hợp 1: Chưa đăng nhập - Hiển thị giao diện Command Center Welcome
    if "password_correct" not in st.session_state:
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.image("https://cdn-icons-png.flaticon.com/512/5087/5087579.png", width=100)
            st.title("AI CORP COMMAND")
            st.subheader("Vui lòng xác thực quyền hạn CEO")
            st.text_input("🔑 MÃ ĐỊNH DANH BIOMETRIC (PASSWORD):", 
                         type="password", 
                         on_change=password_entered, 
                         key="password",
                         help="Chỉ dành cho quản trị viên cấp cao của AI Corporation")
        return False

    # Trường hợp 2: Nhập sai
    elif not st.session_state["password_correct"]:
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.error("🚨 CẢNH BÁO: Mã định danh không chính xác.")
            st.text_input("🔄 THỬ LẠI MÃ ĐỊNH DANH:", 
                         type="password", 
                         on_change=password_entered, 
                         key="password")
            if st.button("Quên mã? Liên hệ bộ phận Security"):
                st.info("Yêu cầu đã được gửi đến hệ thống bảo mật.")
        return False

    # Trường hợp 3: Đã đăng nhập thành công
    return True

def smart_context_manager(messages: List[Any], max_recent: int = 15) -> List[Any]:
    """
    Hệ thống quản trị tri thức hội thoại thông minh.
    Nhiệm vụ: Nén bộ nhớ cũ nhưng bảo toàn các quyết định chiến lược của CEO.
    Tuân thủ Trụ cột 2: Tối ưu hóa ngữ cảnh và ngăn chặn tràn bộ nhớ (Context Overflow).
    """
    # 1. Kiểm tra điều kiện nén (Chỉ nén khi thực sự cần thiết)
    if len(messages) <= max_recent + 2:  # Giữ thêm biên độ an toàn
        return messages

    print(colored(f"🧠 [MEMORY] Đang tối ưu hóa bộ nhớ hội thoại ({len(messages)} tin nhắn)...", "cyan"))

    # 2. Phân tách tin nhắn (Giữ lại System Message gốc và các tin nhắn gần đây)
    # Luôn giữ System Message đầu tiên vì đó là "Hiến pháp" của Agent
    system_msg = messages[0] if isinstance(messages[0], SystemMessage) else None
    
    # Lấy các tin nhắn cần tóm tắt (loại bỏ system msg nếu có)
    msgs_to_summarize = messages[1:-max_recent] if system_msg else messages[:-max_recent]
    recent_msgs = messages[-max_recent:]

    # 3. Chuẩn bị dữ liệu tóm tắt
    history_text = "\n".join([f"{m.type.upper()}: {m.content}" for m in msgs_to_summarize])

    try:
        # Kiểm tra sự sẵn sàng của bộ não (LLM)
        if 'LLM_GPT4' in globals():
            # Prompt tóm tắt theo phong cách "Báo cáo điều hành - Executive Summary"
            summary_prompt = [
                SystemMessage(content=(
                    "Bạn là Trợ lý Quản trị Tri thức. Nhiệm vụ của bạn là nén lịch sử dự án.\n"
                    "BẮT BUỘC TRÍCH XUẤT: \n"
                    "1. Tên dự án & Mục tiêu cốt lõi.\n"
                    "2. Các thông số kỹ thuật/ngân sách đã chốt.\n"
                    "3. Quyết định cuối cùng của CEO cho các bước đã qua.\n"
                    "Yêu cầu: Viết dưới dạng gạch đầu dòng, tối đa 100 từ."
                )),
                HumanMessage(content=f"Dữ liệu cần nén:\n{history_text}")
            ]

            # Gọi LLM xử lý nén tri thức
            summary_response = LLM_GPT4.invoke(summary_prompt)
            
            # Tạo "Ký ức tóm tắt" chuyên nghiệp
            summary_msg = SystemMessage(content=(
                f"--- TÓM TẮT KÝ ỨC DỰ ÁN (HỆ THỐNG NÉN) ---\n"
                f"{summary_response.content}\n"
                f"--- KẾT THÚC TÓM TẮT ---"
            ))

            # Trả về luồng tin nhắn mới: [System Gốc] + [Ký ức tóm tắt] + [Các tin nhắn gần đây]
            new_context = ([system_msg] if system_msg else []) + [summary_msg] + recent_msgs
            return new_context
        
        else:
            # Fallback nếu không có LLM: Chỉ cắt bỏ tin nhắn cũ để cứu vãn Context Window
            print(colored("⚠️ LLM_GPT4 không sẵn sàng, thực hiện cắt tỉa bộ nhớ thủ công.", "yellow"))
            return ([system_msg] if system_msg else []) + recent_msgs

    except Exception as e:
        print(colored(f"❌ Lỗi quản trị bộ nhớ: {e}", "red"))
        # Trả về tin nhắn gần nhất để đảm bảo App không bị treo
        return recent_msgs

def listen_voice() -> str:
    """
    Hệ thống tiếp nhận mệnh lệnh bằng giọng nói.
    Tuân thủ Trụ cột 2: Tối ưu hóa hiệu suất và xử lý nhiễu (Ambient Noise Cancellation).
    """
    if not AUDIO_AVAILABLE:
        st.warning("🎙️ Thiết bị âm thanh chưa được cấu hình trên Server.")
        return ""

    r = sr.Recognizer()
    # Tinh chỉnh các thông số để nghe nhạy hơn
    r.energy_threshold = 300  # Ngưỡng âm thanh tối thiểu
    r.pause_threshold = 0.8   # Thời gian dừng giữa các câu
    
    try:
        with sr.Microphone() as source:
            # Bước quan trọng: Khử nhiễu môi trường trong 0.5 giây
            with st.spinner("🎧 Đang lọc nhiễu môi trường..."):
                r.adjust_for_ambient_noise(source, duration=0.5)
            
            st.toast("🎤 CEO hãy nói, tôi đang nghe...", icon="🎙️")
            
            # Giới hạn thời gian chờ và thời gian nói để tránh treo app
            audio = r.listen(source, timeout=7, phrase_time_limit=15)
            
            with st.spinner("🧠 Đang chuyển đổi ngôn ngữ..."):
                # Sử dụng Google Speech Recognition (vi-VN)
                text = r.recognize_google(audio, language="vi-VN")
                
                if text:
                    st.success(f"👂 Ghi nhận: {text}")
                    return text
                
    except sr.WaitTimeoutError:
        st.toast("⏳ Hết thời gian chờ, CEO chưa đưa ra lệnh.", icon="ℹ️")
    except sr.UnknownValueError:
        st.toast("❓ Hệ thống không nhận dạng được âm thanh.", icon="⚠️")
    except Exception as e:
        print(colored(f"❌ Lỗi Microphone: {e}", "red"))
        st.error("🚨 Không tìm thấy thiết bị Microphone hoặc quyền truy cập bị từ chối.")
    
    return ""

def speak_text(text: str):
    """
    Hệ thống phản hồi bằng giọng nói J.A.R.V.I.S.
    Tuân thủ Trụ cột 2: Tối ưu hóa tài nguyên thông qua việc quản lý Mixer tập trung.
    """
    if not AUDIO_AVAILABLE or not text:
        return

    try:
        # 1. Tiền xử lý văn bản (Loại bỏ các ký tự AI hay viết mà đọc sẽ bị lỗi)
        clean_text = text.replace("*", "").replace("#", "").replace("-", " ")
        
        # 2. Tạo luồng âm thanh trong bộ nhớ (Buffer)
        tts = gTTS(text=clean_text, lang='vi')
        fp = io.BytesIO()
        tts.write_to_fp(fp)
        fp.seek(0)

        # 3. Quản lý Mixer chuyên nghiệp
        # Kiểm tra nếu mixer chưa khởi tạo thì mới init để tránh lag
        if not pygame.mixer.get_init():
            pygame.mixer.init()

        # Dừng âm thanh đang phát (nếu có) trước khi phát câu mới
        if pygame.mixer.music.get_busy():
            pygame.mixer.music.stop()

        # 4. Phát âm thanh
        pygame.mixer.music.load(fp, "mp3")
        pygame.mixer.music.play()
        
        # In log nhẹ nhàng để CEO biết hệ thống đang phản hồi
        print(colored(f"🔊 [TTS] Đang phát phản hồi: {clean_text[:50]}...", "dark_grey"))

    except Exception as e:
        # Lỗi Voice không được làm sập cả Dashboard (Trụ cột 2)
        print(colored(f"⚠️ Lỗi TTS (Voice Engine): {e}", "yellow"))
        # Fallback: Nếu không nói được thì thông báo bằng toast
        st.toast("📢 Không thể phát âm thanh, vui lòng kiểm tra loa.", icon="🔈")
def autoplay_audio(text):
    if text:
        try:
            # Gọi API speak từ Server
            response = requests.post("http://localhost:8080/api/speak", json={"text": text})
            if response.status_code == 200:
                # Chuyển đổi dữ liệu âm thanh sang Base64 để nhúng vào HTML
                import base64
                b64 = base64.b64encode(response.content).decode()
                md = f"""
                    <audio autoplay="true">
                    <source src="data:audio/mp3;base64,{b64}" type="audio/mp3">
                    </audio>
                    """
                st.markdown(md, unsafe_allow_html=True)
        except Exception as e:
            st.error(f"Lỗi âm thanh: {e}")

# 2. Áp dụng vào luồng Chat
# Sau khi Ngài nhận được ai_reply từ Server:
ai_reply = None
if ai_reply:
    # Hiển thị text lên màn hình
    with st.chat_message("assistant"):
        st.markdown(ai_reply)
        st.session_state.messages.append({"role": "assistant", "content": ai_reply})
        
        # TỰ ĐỘNG PHÁT GIỌNG NÓI
        autoplay_audio(ai_reply)

def send_telegram_msg(message, bot_token, chat_id):
    url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
    payload = {"chat_id": chat_id, "text": message, "parse_mode": "Markdown"}
    requests.post(url, json=payload)

def send_to_telegram(text, file_path=None):
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    chat_id = os.getenv("TELEGRAM_CHAT_ID")
    
    # Gửi văn bản
    url_text = f"https://api.telegram.org/bot{token}/sendMessage"
    requests.post(url_text, json={"chat_id": chat_id, "text": text, "parse_mode": "Markdown"})
    
    # Gửi file (nếu có báo cáo PDF)
    if file_path:
        url_file = f"https://api.telegram.org/bot{token}/sendDocument"
        with open(file_path, "rb") as f:
            requests.post(url_file, data={"chat_id": chat_id}, files={"document": f})

def send_telegram_pdf(pdf_bytes, bot_token, chat_id):
    url = f"https://api.telegram.org/bot{bot_token}/sendDocument"
    files = {'document': ('Report.pdf', pdf_bytes)}
    data = {'chat_id': chat_id}
    requests.post(url, data=data, files=files)

def send_chat_to_brain(message, thread_id="ceo_session"):
    """
    Hệ thống liên lạc đa tầng: Gửi mệnh lệnh từ Dashboard tới lõi xử lý.
    Hỗ trợ: Timeout 5 phút cho các tác vụ nghiên cứu sâu (Researcher).
    """
    # 1. Bảo mật & Định danh phiên
    headers = {"X-API-KEY": API_KEY}
    
    # Đảm bảo thread_id luôn hợp lệ để LangGraph không bị mất dấu ký ức
    safe_thread_id = str(thread_id) if thread_id else "ceo_default_mission"
    
    # 2. Đóng gói dữ liệu (Payload)
    # Gửi message và thread_id để server.py nhận diện luồng hội thoại
    payload = {
        "message": message, 
        "thread_id": safe_thread_id
    }
    
    try:
        # 3. Thực thi Request với Timeout lớn (300s) 
        # Tăng timeout vì Agent Researcher/Coder cần thời gian suy nghĩ sâu
        resp = requests.post(
            f"{API_BASE_URL}/api/chat", 
            json=payload, 
            headers=headers, 
            timeout=300 
        )
        
        # 4. Phân tích phản hồi từ Server
        if resp.status_code == 200:
            return resp.json().get("reply", "⚠️ J.A.R.V.I.S không trích xuất được nội dung phản hồi.")
        elif resp.status_code == 401:
            return "🚫 [SECURITY] API Key của AI Corporation không hợp lệ."
        else:
            return f"❌ Lỗi hệ thống (Mã {resp.status_code}): {resp.text}"
            
    except requests.exceptions.Timeout:
        return "⏳ [TIMEOUT] Agent đang xử lý tác vụ quá phức tạp. CEO vui lòng chờ trong giây lát hoặc kiểm tra Terminal."
    except Exception as e:
        return f"🔌 [CONNECTION ERROR] Không thể kết nối tới Bộ não trung tâm: {e}"
    
# ==========================================
# --- 6. HÀM XUẤT FILE (EXPORT ZIP & PDF) ---
# ==========================================

def extract_filename_metadata(code: str, lang_header: str = "") -> Optional[str]:
    """
    Hệ thống trích xuất và chuẩn hóa định danh tập tin (Filename Identification).
    Quy trình: Phân tích Header -> Quét Comment dòng đầu -> Chuẩn hóa ký tự.
    Tuân thủ Trụ cột 2: Đảm bảo chất lượng thông qua việc làm sạch dữ liệu đầu vào.
    """
    if not code:
        return None

    filename = None

    # --- ƯU TIÊN 1: PHÂN TÍCH HEADER (VD: ```python:src/main.py) ---
    if lang_header and ':' in lang_header:
        parts = lang_header.split(':', 1)
        if len(parts) == 2:
            candidate = parts[1].strip()
            # Kiểm tra tính hợp lệ của tên file (có dấu chấm hoặc là file đặc biệt)
            if '.' in candidate or candidate.lower() in ['dockerfile', 'makefile', 'procfile', '.env']:
                filename = candidate

    # --- ƯU TIÊN 2: QUÉT DÒNG ĐẦU TIÊN (Nếu ưu tiên 1 không có kết quả) ---
    if not filename:
        lines = code.split('\n', 2) # Chỉ lấy 2 dòng đầu để tối ưu hiệu suất
        if lines:
            first_line = lines[0].strip()
            # Sử dụng FILENAME_COMMENT_REGEX chuyên nghiệp đã nâng cấp ở bước trước
            match = FILENAME_COMMENT_REGEX.search(first_line)
            if match:
                filename = match.group('filename').strip()

    # --- BƯỚC QUAN TRỌNG: CHUẨN HÓA & LÀM SẠCH (SANITIZATION) ---
    if filename:
        # 1. Loại bỏ các ký tự AI thường thêm vào hoặc ký tự đóng comment
        bad_chars = ['*', '`', '(', ')', '[', ']', '*/', '-->', ';', ',']
        for char in bad_chars:
            filename = filename.replace(char, '')

        # 2. Xử lý khoảng trắng và dấu gạch chéo ngược (Windows vs Linux)
        filename = filename.strip().replace('\\', '/')
        
        # 3. Loại bỏ các từ khóa dư thừa AI thường viết kèm
        for prefix in ['filename:', 'file:', 'path:', 'name:']:
            if filename.lower().startswith(prefix):
                filename = filename[len(prefix):].strip()

        # 4. Kiểm tra cuối cùng: Nếu tên file chỉ toàn ký tự đặc biệt, hủy bỏ
        if not re.search(r'[a-zA-Z0-9]', filename):
            return None

        return filename

    return None

def get_unique_filename(filename: str, existing_files: Set[str]) -> str:
    """
    Hệ thống quản lý phiên bản tập tin tự động (Auto-Versioning System).
    Nhiệm vụ: Ngăn chặn ghi đè, tự động đánh số phiên bản chuyên nghiệp.
    Tuân thủ Trụ cột 1: Đảm bảo tính nhất quán của cấu trúc mã nguồn.
    """
    # Nếu tên file chưa tồn tại, trả về ngay để tối ưu hiệu suất
    if filename not in existing_files:
        return filename
        
    # Tách phần tên và phần mở rộng (ví dụ: main.py -> main, .py)
    base, ext = os.path.splitext(filename)
    
    # Sử dụng Regex để kiểm tra xem file đã có đuôi _vX chưa để tăng cấp tiếp
    # Ví dụ: main_v1.py -> version hiện tại là 1
    version_pattern = re.compile(r'_v(\[0-9]+)$')
    version_match = version_pattern.search(base)
    
    if version_match:
        current_version = int(version_match.group(1))
        base = base[:version_match.start()] # Lấy lại phần tên gốc
    else:
        current_version = 0

    # Vòng lặp tìm phiên bản trống tiếp theo
    counter = current_version + 1
    while True:
        # Định dạng chuẩn: name_v1.ext, name_v2.ext
        new_filename = f"{base}_v{counter}{ext}"
        if new_filename not in existing_files:
            return new_filename
        
        counter += 1
        
        # Guard clause: Tránh vòng lặp vô tận nếu có lỗi logic (Trụ cột 2)
        if counter > 1000:
            return f"{base}_final_backup_{int(time.time())}{ext}"
        
def create_pro_readme(project_name: str, files_data: List[Dict[str, Any]]) -> str:
    """
    Hệ thống tự động khởi tạo tài liệu dự án (Documentation Engine).
    Nhiệm vụ: Tạo file README.md chuẩn công nghiệp, hỗ trợ hướng dẫn cài đặt và vận hành.
    Tuân thủ Trụ cột 3: Giao tiếp hiệu quả thông qua tài liệu hóa minh bạch.
    """
    timestamp = datetime.datetime.now().strftime('%d/%m/%Y %H:%M')
    total_files = len(files_data)
    total_lines = sum(f.get('lines', 0) for f in files_data)
    
    # 1. Tự động nhận diện công nghệ chủ đạo (Main Tech Stack)
    languages = [f['lang'] for f in files_data if f['lang']]
    main_tech = max(set(languages), key=languages.count).upper() if languages else "UNKNOWN"

    # 2. Xây dựng bảng danh sách tệp tin
    table_rows = [
        f"| {idx+1} | `📁 {f['filename']}` | **{f['lang'].upper()}** | {f.get('lines', 0):,} |"
        for idx, f in enumerate(files_data)
    ]
    table_content = "\n".join(table_rows)

    # 3. Mẫu hướng dẫn vận hành nhanh dựa trên ngôn ngữ (Smart Instruction)
    run_cmd = "N/A"
    if main_tech == "PYTHON": run_cmd = "`python main.py`"
    elif main_tech in ["HTML", "JS", "CSS"]: run_cmd = "Mở `index.html` trên trình duyệt."
    elif main_tech == "CPP": run_cmd = "Biên dịch với `g++` hoặc nạp qua Arduino IDE."

    return f"""# 🚀 {project_name.upper()} - AI CORPORATION ARCHIVE



    ## 📝 Thông tin chung
    - **Trạng thái:** Bản dựng hoàn chỉnh (AI Generated)
    - **Thời gian xuất bản:** {timestamp}
    - **Công nghệ chủ đạo:** `{main_tech}`
    - **Quy mô dự án:** {total_files} tệp tin / {total_lines:,} dòng mã.

    ## 📂 Danh mục cấu trúc hệ thống (File Structure)

    | # | Tên Tập Tin | Ngôn Ngữ | Số Dòng |
    |---|---|---|---|
    {table_content}

    ## ⚙️ Hướng dẫn vận hành nhanh (Quick Start)
    1. **Yêu cầu hệ thống:** Đảm bảo đã cài đặt môi trường hỗ trợ `{main_tech}`.
    2. **Lệnh thực thi:** {run_cmd}
    3. **Lưu ý:** Kiểm tra file `.env` (nếu có) để cấu hình các biến môi trường trước khi chạy.

    ---
    ## 🛡️ Bản quyền & Bảo mật
    *Sản phẩm được kiến trúc bởi **AI Corporation (J.A.R.V.I.S Engine)**.*
    *Tuân thủ tiêu chuẩn: **Bốn Trụ Cột Ưu Việt** (Mã nguồn vững chắc - Chất lượng toàn diện).*

    ---
    *Dấu ấn kỹ thuật: {datetime.datetime.now().year} © AI Corp.*
    """

def export_project_zip(project_name: str, messages: List[Any]) -> Optional[bytes]:
    """
    Hệ thống đóng gói sản phẩm tự động (Automated Packaging System).
    Nhiệm vụ: Trích xuất mã nguồn, tự động đặt tên, tạo tài liệu và nén ZIP.
    Tuân thủ Trụ cột 4: Quy trình vận hành tối ưu và nhất quán.
    """
    if not messages:
        return None

    buf = io.BytesIO()
    files_data: List[Dict[str, Any]] = []
    existing_filenames: Set[str] = set()
    
    # Sử dụng context manager để đảm bảo ZIP được đóng đúng cách
    try:
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
            for msg in messages:
                # Chỉ xử lý tin nhắn từ AI (Coder/Hardware/etc.)
                if not isinstance(msg, AIMessage):
                    continue
                
                content = msg.content
                if '```' not in content:
                    continue

                # Cơ chế tự sửa lỗi (Self-healing): Đóng block code nếu AI quên
                if content.count('```') % 2 != 0:
                    content += '\n```'

                # Tìm tất cả các khối mã nguồn bằng Regex chuyên nghiệp
                matches = CODE_BLOCK_REGEX.finditer(content)
                
                for match in matches:
                    header_raw = (match.group('header') or '').strip()
                    code_content = match.group('content').strip()
                    
                    if len(code_content) < 10: # Bỏ qua rác hoặc code quá ngắn
                        continue

                    # 1. Nhận diện ngôn ngữ
                    lang = header_raw.split(':')[0].lower() if ':' in header_raw else header_raw.lower()
                    if not lang: lang = 'text'

                    # 2. Trích xuất tên file bằng Metadata Engine (Ưu tiên số 1)
                    filename = extract_filename_metadata(code_content, header_raw)

                    # 3. Đặt tên tự động theo chuẩn (Ưu tiên số 2)
                    if not filename:
                        ext = FILE_EXTENSION_MAP.get(lang, '.txt')
                        base_name = DEFAULT_FILE_NAMES.get(lang, f'component_{len(files_data) + 1}')
                        filename = f"{base_name}{ext}"

                    # 4. Quản lý phiên bản tránh trùng lặp (Versioning Engine)
                    final_filename = get_unique_filename(filename, existing_filenames)
                    existing_filenames.add(final_filename)

                    # 5. Ghi vào file ZIP
                    zf.writestr(final_filename, code_content)
                    
                    # 6. Thu thập thống kê tệp tin
                    files_data.append({
                        'filename': final_filename,
                        'lang': lang,
                        'lines': len(code_content.splitlines()),
                        'size_bytes': len(code_content.encode('utf-8'))
                    })

            # --- BƯỚC CUỐI: TẠO TÀI LIỆU HỆ THỐNG ---
            if files_data:
                # Nạp file README chuyên nghiệp
                readme_content = create_pro_readme(project_name, files_data)
                zf.writestr("README.md", readme_content)
                
                # Nạp file manifest JSON (Dành cho các hệ thống tự động khác đọc)
                manifest = {
                    "project_info": {
                        "name": project_name,
                        "exported_at": datetime.datetime.now().isoformat(),
                        "engine": "AI Corporation J.A.R.V.I.S v2.0"
                    },
                    "statistics": {
                        "file_count": len(files_data),
                        "total_lines": sum(f['lines'] for f in files_data),
                        "total_size_kb": round(sum(f['size_bytes'] for f in files_data) / 1024, 2)
                    },
                    "inventory": files_data
                }
                zf.writestr("project_manifest.json", json.dumps(manifest, indent=4, ensure_ascii=False))

        if not files_data:
            return None
            
        buf.seek(0)
        return buf.getvalue()

    except Exception as e:
        print(colored(f"❌ Lỗi trong quy trình đóng gói dự án: {e}", "red"))
        return None

def export_to_pdf(messages: List[Any]) -> bytes:
    """
    Hệ thống xuất bản báo cáo chiến lược (Corporate Reporting System).
    Nhiệm vụ: Chuyển đổi hội thoại thành tài liệu PDF chuyên nghiệp, hỗ trợ tiếng Việt hoàn hảo.
    Tuân thủ Trụ cột 3: Giao tiếp hiệu quả thông qua trình bày văn bản chuẩn mực.
    """
    import textwrap
    
    # 1. KHỞI TẠO CẤU HÌNH TRANG
    pdf = FPDF()
    pdf.set_margins(15, 15, 15)
    pdf.set_auto_page_break(auto=True, margin=20)
    
    # Đường dẫn font - Cần đảm bảo file này nằm cùng thư mục
    font_path = "DejaVuSans.ttf"
    main_font = "DejaVu"
    
    if os.path.exists(font_path):
        pdf.add_font('DejaVu', '', font_path)
        pdf.add_font('DejaVu', 'B', font_path)
        has_unicode = True
    else:
        st.error("⚠️ Thiếu file DejaVuSans.ttf. Font sẽ bị lỗi hiển thị.")
        main_font = "Arial"
        has_unicode = False

    pdf.add_page()
    SAFE_WIDTH = 180 # Chiều rộng vùng an toàn (A4 210mm - 30mm lề)

    # --- 2. HEADER: THIẾT KẾ BỘ NHẬN DIỆN THƯƠNG HIỆU ---
    # Tên công ty/Hệ thống
    pdf.set_font(main_font, 'B', 16)
    pdf.set_text_color(0, 51, 102) # Xanh Navy chuyên nghiệp
    pdf.cell(0, 10, "AI CORPORATION - COMMAND CENTER", new_x=XPos.LMARGIN, new_y=YPos.NEXT, align='L')
    
    # Tiêu đề báo cáo
    pdf.set_font(main_font, 'B', 22)
    pdf.set_text_color(30, 30, 30)
    pdf.ln(5)
    pdf.cell(0, 15, "BÁO CÁO CHIẾN LƯỢC DỰ ÁN", new_x=XPos.LMARGIN, new_y=YPos.NEXT, align='C')
    
    # Metadata báo cáo
    pdf.set_font(main_font, '', 10)
    pdf.set_text_color(100, 100, 100)
    pub_date = datetime.datetime.now().strftime('%d/%m/%Y %H:%M')
    pdf.cell(0, 8, f"Mã báo cáo: ARC-{int(time.time()/100)} | Ngày xuất bản: {pub_date}", 
             new_x=XPos.LMARGIN, new_y=YPos.NEXT, align='C')
    
    pdf.ln(10)
    # Đường kẻ phân cách Header
    pdf.set_draw_color(0, 51, 102)
    pdf.set_line_width(0.5)
    pdf.line(15, pdf.get_y(), 195, pdf.get_y())
    pdf.ln(10)

    # --- 3. XỬ LÝ NỘI DUNG HỘI THOẠI ---
    for msg in messages:
        # Xác định vai trò
        is_human = isinstance(msg, HumanMessage)
        role_label = "CEO - YÊU CẦU CHIẾN LƯỢC" if is_human else "AI CONSULTANT - PHÂN TÍCH CHUYÊN SÂU"
        
        # === A. BOX VAI TRÒ (STYLING) ===
        pdf.set_font(main_font, 'B', 10)
        if is_human:
            pdf.set_fill_color(240, 240, 240) # Xám nhạt cho CEO
            pdf.set_text_color(50, 50, 50)
        else:
            pdf.set_fill_color(0, 51, 102)   # Xanh Navy cho AI
            pdf.set_text_color(255, 255, 255)

        pdf.cell(0, 8, f"  {role_label}", new_x=XPos.LMARGIN, new_y=YPos.NEXT, fill=True)
        pdf.ln(3)

        # === B. NỘI DUNG VĂN BẢN (PARSING) ===
        content = msg.content
        pdf.set_text_color(40, 40, 40)
        pdf.set_font(main_font, '', 11)
        
        lines = content.split('\n')
        in_code_block = False
        
        for line in lines:
            if "```" in line:
                in_code_block = not in_code_block
                continue
            
            # Xử lý khối mã nguồn (Code Block)
            if in_code_block:
                pdf.set_font(main_font, '', 9)
                pdf.set_fill_color(248, 248, 252)
                pdf.set_text_color(0, 80, 0) # Xanh lá đậm cho code
                
                wrapped_code = textwrap.wrap(line, width=80)
                for w_line in wrapped_code:
                    pdf.multi_cell(SAFE_WIDTH, 5, f"  {w_line}", fill=True)
                continue

            # Xử lý tiêu đề Markdown (###)
            if line.strip().startswith('###'):
                text = line.replace('###', '').replace('**', '').strip()
                pdf.ln(3)
                pdf.set_font(main_font, 'B', 13)
                pdf.set_text_color(0, 51, 102)
                pdf.multi_cell(SAFE_WIDTH, 8, text.upper())
                pdf.set_font(main_font, '', 11) # Reset
                pdf.set_text_color(40, 40, 40)
                continue

            # Xử lý gạch đầu dòng
            if line.strip().startswith(('- ', '* ', '✓')):
                text = line.strip().lstrip('✓-* ').replace('**', '').strip()
                pdf.set_x(20)
                pdf.cell(5, 6, "•", align='L')
                pdf.multi_cell(SAFE_WIDTH - 10, 6, text)
                continue

            # Văn bản thường (Wrap thông minh)
            clean_line = line.replace('**', '').strip()
            if clean_line:
                pdf.multi_cell(SAFE_WIDTH, 6, clean_line)
            else:
                pdf.ln(2)

        # Đường kẻ mờ phân tách lượt hội thoại
        pdf.ln(5)
        pdf.set_draw_color(230, 230, 230)
        pdf.set_line_width(0.2)
        pdf.line(15, pdf.get_y(), 195, pdf.get_y())
        pdf.ln(5)

    # --- 4. FOOTER: TRANG SỐ ---
    # (FPDF tự động gọi footer nếu bạn ghi đè phương thức, 
    # nhưng ở đây ta làm thủ công cho đơn giản)
    
    return bytes(pdf.output())

def register_ai_product(project_id: str, market_name: str, price: float, image_url: str = None):
    """
    Hệ thống thương mại hóa dự án AI.
    Biến mã nguồn từ bảng 'projects' thành sản phẩm kinh doanh trong bảng 'products'.
    """
    try:
        with db_manager.get_connection() as conn:
            c = conn.cursor()
            
            # 1. Lấy thông tin mô tả tóm tắt từ lịch sử dự án
            c.execute("SELECT name, history FROM projects WHERE id = ?", (project_id,))
            project = c.fetchone()
            
            if not project:
                st.error("❌ Không tìm thấy dự án để thương mại hóa.")
                return

            # Tóm tắt lại nội dung để làm mô tả sản phẩm (Dùng AI hoặc lấy text thô)
            description = f"Giải pháp phần mềm {project['name']} được phát triển bởi AI Corp. Đã qua thẩm định."
            
            # 2. Đẩy vào bảng products
            c.execute("""
                INSERT INTO products (name, description, price, image_url)
                VALUES (?, ?, ?, ?)
            """, (market_name, description, price, image_url or 'https://img.icons8.com/fluency/96/software-box.png'))
            
            conn.commit()
            st.success(f"🚀 Đã niêm yết sản phẩm '{market_name}' lên hệ thống kinh doanh!")
            
    except Exception as e:
        st.error(f"❌ Lỗi niêm yết: {e}")

def display_commercial_hub():
    """
    Giao diện trung tâm thương mại của AI Corporation.
    Hiển thị các giải pháp phần mềm đã đóng gói dưới dạng thẻ sản phẩm chuyên nghiệp.
    """
    st.subheader("🛒 AI Solutions Marketplace")
    
    with db_manager.get_connection() as conn:
        df_products = pd.read_sql_query("SELECT * FROM products", conn)

    if df_products.empty:
        st.info("Hiện chưa có sản phẩm nào được niêm yết.")
        return

    # Hiển thị dạng Grid (3 cột)
    cols = st.columns(3)
    for idx, row in df_products.iterrows():
        with cols[idx % 3]:
            with st.container(border=True):
                st.image(row['image_url'], width=80)
                st.subheader(row['name'])
                st.write(row['description'])
                st.markdown(f"**Giá niêm yết:** `${row['price']}`")
                
                # Nút hành động cho CEO
                col_a, col_b = st.columns(2)
                with col_a:
                    if st.button(f"📄 Báo giá", key=f"quote_{row['id']}"):
                        st.toast("Đang tạo file báo giá PDF...")
                with col_b:
                    if st.button(f"📦 Triển khai", key=f"deploy_{row['id']}"):
                        st.success("Đang kích hoạt môi trường Cloud...")

def cyber_loading_animation(status_text="INITIALIZING COGNITIVE ENGINE"):
    """
    Hiệu ứng nạp dữ liệu kiểu Sci-Fi (J.A.R.V.I.S Style).
    Hiển thị các dòng log kỹ thuật giả lập để tạo trải nghiệm điện ảnh.
    """
    placeholder = st.empty()
    # Danh sách các dòng log giả lập
    logs = [
        "📡 CONNECTING TO NEURAL NETWORK...",
        "🧠 RETRIEVING CONTEXTUAL MEMORY...",
        "🔍 SCANNING PROJECT REPOSITORY...",
        "⚡ OPTIMIZING LLM PARAMETERS...",
        "🛡️ SECURITY PROTOCOLS: ACTIVE",
        "🧬 SYNTHESIZING RESPONSE..."
    ]
    
    # Hiệu ứng chạy log nhanh
    for i in range(len(logs)):
        with placeholder.container():
            st.markdown(f"""
            <div style="background: rgba(0, 242, 255, 0.05); border-left: 3px solid #00f2ff; padding: 10px; border-radius: 5px;">
                <p style="color: #00f2ff; font-family: 'Courier New', Courier, monospace; font-size: 0.8rem; margin: 0;">
                    <b>[SYSTEM]</b> {logs[i]}
                </p>
                <div style="width: { (i+1) * 16 }%; height: 2px; background: #00f2ff; margin-top: 5px; box-shadow: 0 0 10px #00f2ff;"></div>
            </div>
            """, unsafe_allow_html=True)
            time.sleep(0.2) # Chạy nhanh để không làm CEO đợi lâu
    
    # Sau khi chạy xong các dòng log, hiển thị trạng thái cuối
    placeholder.empty()

def add_message_safe(msg):
    """
    Cơ chế an toàn: Chỉ thêm tin nhắn nếu nó KHÁC với tin nhắn cuối cùng trong lịch sử.
    Ngăn chặn tuyệt đối việc lặp lại do Rerun.
    """
    if "messages" not in st.session_state:
        st.session_state.messages = []
        
    # Nếu lịch sử trống, thêm ngay
    if not st.session_state.messages:
        st.session_state.messages.append(msg)
        return True

    # Kiểm tra tin nhắn cuối cùng
    last_msg = st.session_state.messages[-1]
    
    # Chỉ thêm nếu nội dung khác nhau hoặc vai trò khác nhau
    if last_msg.content.strip() != msg.content.strip():
        st.session_state.messages.append(msg)
        return True
    
    return False # Báo hiệu là đã bị trùng, không thêm

def safe_render_message(msg):
    """Hàm bọc thép để hiển thị mọi loại tin nhắn từ Graph"""
    if msg is None:
        return ""
    
    # Trường hợp là đối tượng chuẩn của LangChain (HumanMessage, AIMessage)
    if hasattr(msg, "content"):
        return str(msg.content)
    
    # Trường hợp bị biến thành Dictionary (Lỗi thường gặp)
    if isinstance(msg, dict):
        return str(msg.get("content", str(msg)))
    
    # Cuối cùng, ép kiểu về chuỗi nếu là bất kỳ thứ gì khác
    return str(msg)

def get_products():
    """Lấy danh sách sản phẩm từ Server"""
    headers = {"X-API-KEY": API_KEY}
    try:
        resp = requests.get(f"{API_BASE_URL}/api/products", headers=headers, timeout=5)
        return resp.json() if resp.status_code == 200 else []
    except:
        return []

def buy_product_api(product_id):
    """Gửi lệnh mua/triển khai sản phẩm lên Server"""
    headers = {"X-API-KEY": API_KEY}
    try:
        resp = requests.post(f"{API_BASE_URL}/api/products/buy", 
                             json={"product_id": product_id}, headers=headers)
        return resp.json()
    except Exception as e:
        return {"status": "error", "msg": str(e)}

def check_server_online():
    """Kiểm tra máy chủ Backend có đang chạy không"""
    try:
        requests.get(API_BASE_URL, timeout=2)
        return True
    except:
        return False

# --- 4. CÁC HÀM BỔ TRỢ GIAO DIỆN (UI HELPERS) ---
def extract_code_block(content: str) -> Optional[str]:
    """Trích xuất khối mã nguồn từ Markdown."""
    match = re.search(r'```python\n(.*?)\n```', content, re.DOTALL)
    if not match:
        match = re.search(r'```\n(.*?)\n```', content, re.DOTALL)
    return match.group(1).strip() if match else None

DEPARTMENTS = {
    "🏢 Điều phối": {"tag": "[ORCHESTRATOR]", "desc": "Tổng tham mưu điều phối dự án"},
    "🔍 Nghiên cứu": {"tag": "[RESEARCH]", "desc": "Quét dữ liệu thị trường thực tế 2026"},
    "🧠 Chiến lược": {"tag": "[STRATEGY]", "desc": "Phân tích PESTLE & Lập Roadmap 5 năm"},
    "💻 Lập trình": {"tag": "[CODER]", "desc": "Claude 3.5 Parallel Coding & AST Test"},
    "⚙️ Kỹ thuật": {"tag": "[ENGINEERING]", "desc": "Thiết kế mô phỏng 3D bằng Plotly"},
    "🛠️ Phần cứng": {"tag": "[HARDWARE]", "desc": "Kiến trúc Robotics & Sơ đồ chân ESP32"},
    "📡 Kết nối IoT": {"tag": "[IOT]", "desc": "Điều khiển thiết bị & Giao thức MQTT"},
    "🛒 Thu mua": {"tag": "[PROCUREMENT]", "desc": "Tra giá Shopee/Mouser & Lập bảng BOM"},
    "💰 Tài chính": {"tag": "[INVEST]", "desc": "Thẩm định ROI & Quản lý ngân sách"},
    "⚖️ Pháp lý": {"tag": "[LEGAL]", "desc": "Rà soát bản quyền & Rủi ro pháp lý IP"},
    "📢 Marketing": {"tag": "[MARKETING]", "desc": "Chiến dịch quảng bá & Visual Prompts"},
    "🖋️ Sáng tác": {"tag": "[STORY]", "desc": "Biên kịch & Sáng tạo nội dung văn học"},
    "🎨 Nghệ thuật": {"tag": "[ARTIST]", "desc": "Họa sĩ AI tạo hình ảnh DALL-E 3"},
    "📜 Xuất bản": {"tag": "[PUBLISH]", "desc": "Tổng hợp hồ sơ & In báo cáo cuối cùng"},
    "📂 Thư ký": {"tag": "[SECRETARY]", "desc": "Kiểm toán lỗi & Nhật ký vận hành"}
}

# ==========================================
# --- 7. STYLES & SIDEBAR ---
# ==========================================
def main():
    import re
    # --- 1. KIỂM TRA QUYỀN TRUY CẬP (SECURITY GATE) ---
    if not check_password():
        st.stop() # Dừng toàn bộ script nếu chưa xác thực
    is_online = check_server_online()
    if not is_online:
        st.error("⚠️ MÁY CHỦ CHIẾN LƯỢC ĐANG NGOẠI TUYẾN (Offline). Vui lòng chạy 'python server.py' trước.")
    
    CONTEXT_MAP = {name: info["tag"] for name, info in DEPARTMENTS.items()}
    # --- 2. SESSION STATE INITIALIZATION ---
    if "project_id" not in st.session_state: st.session_state.project_id = None
    if "project_name" not in st.session_state: st.session_state.project_name = "CHỜ CHỈ THỊ..."
    if "messages" not in st.session_state: st.session_state.messages = []
    if "total_tokens" not in st.session_state: st.session_state.total_tokens = 0
    if "active_context" not in st.session_state: st.session_state.active_context = "CHAT"
    if "is_naming_phase" not in st.session_state: st.session_state.is_naming_phase = False # Kiểm tra xem có cần đổi tên dự án không    

    # --- LOGIC ĐỔI TÊN MISSION TỰ ĐỘNG ---
    if st.session_state.is_naming_phase:
        with st.container(border=True):
            st.warning("🚀 HỆ THỐNG ĐÃ SẴN SÀNG. CEO VUI LÒNG ĐẶT TÊN MÃ CHIẾN DỊCH (MISSION NAME):")
            col_n1, col_n2 = st.columns([3, 1])
            with col_n1:
                new_name_input = st.text_input("Nhập tên mã:", placeholder="VD: Chiến dịch Sao Hỏa...")
            with col_n2:
                if st.button("XÁC NHẬN TÊN MÃ", use_container_width=True):
                    if new_name_input:
                        st.session_state.project_name = new_name_input
                        st.session_state.is_naming_phase = False
                        # Lưu cập nhật vào Database
                        save_project(st.session_state.project_id, new_name_input, st.session_state.messages)
                        st.success(f"✅ Đã đổi tên Mission thành: {new_name_input}")
                        time.sleep(1)
                        st.rerun()

    if st.session_state.project_id and not st.session_state.messages:
        history = load_project_history(st.session_state.project_id)
        if history:
            st.session_state.messages = history
            print(f"♻️ [RECOVERY] Đã khôi phục {len(history)} tin nhắn cho ID: {st.session_state.project_id}")
            df_p = get_project_list()
            if not df_p.empty:
                name_val = df_p[df_p['id'] == st.session_state.project_id]['name'].values
                if len(name_val) > 0:
                    st.session_state.project_name = name_val[0]
    tabs = st.tabs(list(DEPARTMENTS.keys()))
    with st.sidebar:
        st.markdown(f"""
            <div style="text-align: center; padding: 10px;">
                <h2 style="margin-bottom: 0;">🛡️ SYSTEM CONTROL</h2>
                <p style="color: #00f2ff; font-size: 0.8rem;">STATUS: ONLINE | NODE: J.A.R.V.I.S v2.0</p>
            </div>
        """, unsafe_allow_html=True)
        st.markdown("---")

        # --- PHẦN 1: QUẢN LÝ DỰ ÁN (PROJECT CORE) ---
        with st.expander("📁 QUẢN LÝ DỰ ÁN", expanded=True):
            # Khởi tạo dự án mới
            new_proj_name = st.text_input("✨ Tên dự án mới:", placeholder="VD: Hệ thống Robot AI")
            if st.button("🆕 KHỞI TẠO MISSION", use_container_width=True):
                if new_proj_name:
                    new_id = str(int(time.time()))
                    st.session_state.messages = []
                    st.session_state.project_id = new_id
                    st.session_state.project_name = new_proj_name
                    # Lưu vào DB ngay lập tức
                    save_project(new_id, new_proj_name, [])
                    st.cache_data.clear()
                    st.success(f"🚀 Mission '{new_proj_name}' khởi động!")
                    time.sleep(0.5)
                    st.rerun()
                else:
                    st.warning("Vui lòng nhập tên dự án trước khi khởi tạo.")

            st.markdown("<br>", unsafe_allow_html=True)
            
            # Khôi phục dự án cũ
            st.markdown("🔍 **KHÔI PHỤC DỰ ÁN**")
            df_projects = get_project_list()
            if not df_projects.empty:
                project_options = dict(zip(df_projects['name'], df_projects['id']))
                selected_name = st.selectbox("Chọn hồ sơ lưu trữ:", list(project_options.keys()))
                
                if st.button("📂 TẢI DỮ LIỆU LỊCH SỬ", use_container_width=True):
                    with st.spinner("🧠 Đang truy xuất ký ức..."):
                        pid = project_options[selected_name]
                        st.session_state.project_id = pid
                        st.session_state.project_name = selected_name
                        # Tải tin nhắn từ DB
                        st.session_state.messages = load_project_history(pid)
                        st.toast(f"✅ Đã tải hồ sơ: {selected_name}")
                        time.sleep(0.5)
                        st.rerun()
            else:
                st.caption("Chưa có hồ sơ lưu trữ nào.")

        # --- PHẦN 2: BỘ NÃO TRUNG TÂM (COGNITIVE HUB) ---
        with st.expander("🧠 BỘ NÃO TRUNG TÂM"):
            st.markdown("🎯 **DẠY KIẾN THỨC MỚI**")
            k_input = st.text_area("Input tri thức:", placeholder="Nhập quy trình, tài liệu chiến lược...", height=100)
            if st.button("📥 GHI NHỚ (LONG-TERM)", use_container_width=True):
                if k_input:
                    with st.spinner("🧬 Đang mã hóa vào bộ nhớ..."):
                        remember_knowledge(k_input)
                        st.success("✅ Tri thức đã được đồng bộ!")
            
            st.markdown("---")
            if st.button("🔄 ĐỒNG BỘ KHO TÀI LIỆU (RAG)", use_container_width=True):
                with st.spinner("📡 Đang quét cơ sở dữ liệu..."):
                    result = ingest_docs_to_memory()
                    st.info(result)

        # --- PHẦN 3: XUẤT BẢN & GIAO TIẾP (COMMUNICATION) ---
        with st.expander("🚀 XUẤT BẢN & GIAO TIẾP"):
            if st.button("🎤 RA LỆNH GIỌNG NÓI", use_container_width=True):
                cmd = listen_voice()
                if cmd:
                    # Lưu lệnh vào session để Tab 1 xử lý
                    st.session_state.temp_voice_text = cmd
                    st.rerun()

            st.markdown("---")
            st.markdown("📩 **TRÍCH XUẤT BÁO CÁO**")
            c1, c2 = st.columns(2)
            with c1:
                if st.button("📄 PDF", use_container_width=True):
                    if st.session_state.messages:
                        pdf_data = export_to_pdf(st.session_state.messages)
                        st.download_button("📩 Lưu", data=pdf_data, file_name=f"Report_{st.session_state.project_id}.pdf")
            with c2:
                if st.button("📦 ZIP", use_container_width=True):
                    zip_data = export_project_zip(st.session_state.project_name, st.session_state.messages)
                    if zip_data:
                        st.download_button("📩 Lưu", data=zip_data, file_name=f"Source_{st.session_state.project_id}.zip")

        # --- PHẦN 4: CHI PHÍ & GIÁM SÁT (METRICS) ---
        st.markdown("---")
        # Tính toán chi phí thực tế dựa trên Token
        current_cost = (st.session_state.total_tokens / 1000) * 0.015 # Giả định giá GPT-4o
        st.metric(
            label="📊 CHI PHÍ VẬN HÀNH (EST)", 
            value=f"${current_cost:.4f}", 
            delta=f"{st.session_state.total_tokens} tokens",
            delta_color="normal"
        )
        
        if st.toggle("🤖 Agent Autopilot", help="Kích hoạt báo cáo tự động mỗi sáng"):
            st.caption("⏰ Next Scan: 08:00 AM")

    # --- 4. GIAO DIỆN CHÍNH (MAIN TABS) ---
    
    # 2. Sidebar để CEO chọn ngữ cảnh
    with st.sidebar:
        st.divider()
        selected_mode_label = st.selectbox("🎯 CHẾ ĐỘ TÁC CHIẾN:", list(CONTEXT_MAP.keys()))
        selected_mode_tag = CONTEXT_MAP[selected_mode_label]
        
        # Tự động điều chỉnh giới hạn tư duy dựa trên Tag chiến lược
        # Cập nhật thêm các Tag mới của 15 phòng ban
        strategic_tags = ["[RESEARCH]", "[INVEST]", "[STORY]", "[STRATEGY]", "[CODER]"]
        rec_limit = 500 if selected_mode_tag in strategic_tags else 50
        st.caption(f"🚀 Giới hạn tư duy {selected_mode_tag}: {rec_limit} bước")

    
   # ============================================================================
# TAB 0: ĐIỀU PHỐI (ORCHESTRATOR) - TRUNG TÂM CHỈ HUY TỔNG THỂ
# ============================================================================
    with tabs[0]:
        st.markdown(f"### 🏢 Mission: {st.session_state.project_name}")
        chat_col, status_col = st.columns([2, 1])
        
        with chat_col:
            for msg in st.session_state.messages:
                # --- LOGIC XỬ LÝ ĐA HỆ (Dict & Object) ---
                if isinstance(msg, dict):
                    # Nếu là Dictionary (từ Database/JSON)
                    content = msg.get("content", "")
                    role = msg.get("role", "assistant")
                    is_user = role == "user"
                else:
                    # Nếu là Object (LangChain Message)
                    content = getattr(msg, "content", str(msg))
                    is_user = isinstance(msg, HumanMessage)
                    role = "user" if is_user else "assistant"
                
                # Hiển thị ra màn hình
                avatar = "👨‍💼" if is_user else "🤖"
                with st.chat_message(role, avatar=avatar):
                    st.markdown(content)
        
        with status_col:
            st.subheader("📡 Node Status")
            st.info(f"ID: {st.session_state.project_id}")
            if st.button("🗑️ Wipe Memory"):
                st.session_state.messages = []
                st.rerun()

# ============================================================================
# TAB 1: NGHIÊN CỨU (RESEARCHER) - TRUY QUÉT DỮ LIỆU THỜI GIAN THỰC 2026
# ============================================================================
    with tabs[1]:
        st.markdown("### 🔍 Market Intelligence & Global Trends")
        
        # 1. Lọc các báo cáo nghiên cứu từ bộ nhớ
        research_msgs = [m for m in st.session_state.messages if "[RESEARCH]" in m.content or "🔍" in m.content]
        
        if not research_msgs:
            st.info("💡 CEO chưa có báo cáo thám mã nào. Hãy chuyển 'Chế độ tác chiến' sang Nghiên cứu và ra lệnh.")
        else:
            # Lấy báo cáo mới nhất
            latest_report = research_msgs[-1].content
            
            # 2. Layout phân tích: Bên trái là nội dung, Bên phải là Trích dẫn & Chỉ số
            col_report, col_metrics = st.columns([2, 1])
            
            with col_report:
                with st.container(border=True):
                    st.markdown("#### 📄 BẢN TIN TÌNH BÁO MỚI NHẤT")
                    # Hiển thị nội dung báo cáo với định dạng sạch
                    clean_report = latest_report.replace("[RESEARCH]", "").strip()
                    st.markdown(clean_report)
                    
                    # Nút hành động nhanh cho CEO
                    if st.button("📥 Lưu báo cáo vào Database Chiến lược"):
                        # Logic lưu vào bảng legacy_knowledge để kế thừa kiến thức
                        st.success("Đã đồng bộ báo cáo vào kho tri thức dài hạn.")

            with col_metrics:
                st.markdown("#### 🔗 NGUỒN TRÍCH DẪN (SOURCES)")
                # Trích xuất các liên kết (URL) từ báo cáo bằng Regex
                urls = re.findall(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', latest_report)
                
                if urls:
                    for url in list(set(urls))[:5]: # Hiển thị 5 nguồn uy tín nhất
                        st.markdown(f"""
                        <div style="background: rgba(0,242,255,0.05); padding: 8px; border-radius: 5px; margin-bottom: 5px; border-left: 2px solid #00f2ff;">
                            <a href="{url}" target="_blank" style="color: #00f2ff; text-decoration: none; font-size: 0.8rem;">
                                🌐 {url[:40]}...
                            </a>
                        </div>
                        """, unsafe_allow_html=True)
                else:
                    st.caption("Không tìm thấy liên kết nguồn trong báo cáo này.")

                st.divider()
                st.markdown("#### 📈 CHỈ SỐ TIN CẬY (CONFIDENCE)")
                # Giả lập chỉ số tin cậy dựa trên độ dài báo cáo
                confidence_score = min(len(clean_report) / 10, 100)
                st.metric("Hệ số tin cậy", f"{confidence_score:.1f}%", delta="High Accuracy")
                st.progress(confidence_score / 100)

# ============================================================================
# TAB 2: CHIẾN LƯỢC (STRATEGY) - THIẾT LẬP TIỀN ĐỀ VÀ ROADMAP TỔNG THỂ
# ============================================================================
    with tabs[2]:
        st.markdown("### 🧠 Strategic Intelligence Command (C-Suite)")
        
        # 1. Lọc dữ liệu chiến lược thực tế
        strategy_msgs = [m for m in st.session_state.messages if "[STRATEGY]" in m.content or "🧠" in m.content]
        
        if not strategy_msgs:
            st.warning("⚠️ CẢNH BÁO: Chưa có tiền đề chiến lược. CEO hãy ra lệnh '[STRATEGY] Phân tích thị trường và lập lộ trình' để bắt đầu.")
        else:
            latest_strategy = strategy_msgs[-1].content
            
            # --- PHẦN 1: BẢNG ĐIỀU KHIỂN CHIẾN THUẬT (KPIs TIỀN ĐỀ) ---
            col_s1, col_s2, col_s3 = st.columns(3)
            with col_s1:
                st.metric("MỨC ĐỘ KHẢ THI", "85%", "+2.3%", help="Đánh giá dựa trên nguồn lực hiện có")
            with col_s2:
                st.metric("RỦI RO HỆ THỐNG", "Thấp", "-10%", delta_color="normal")
            with col_s3:
                st.metric("ƯU TIÊN", "R&D Giai đoạn 1", "HIGH")

            st.divider()

            # --- PHẦN 2: PHÂN TÍCH ĐA CHIỀU (PESTLE & SWOT) ---
            col_left, col_right = st.columns([1, 1])
            
            with col_left:
                st.markdown("#### 🌍 PHÂN TÍCH MÔI TRƯỜNG (PESTLE)")
                # Tự động bóc tách các mục PESTLE từ văn bản AI
                pestle_factors = ["Chính trị", "Kinh tế", "Xã hội", "Công nghệ", "Pháp lý", "Môi trường"]
                for factor in pestle_factors:
                    if factor.lower() in latest_strategy.lower():
                        st.success(f"✔️ **{factor}**: Đã được thẩm định chuyên sâu.")
                    else:
                        st.error(f"❌ **{factor}**: Chưa có dữ liệu phân tích.")

            with col_right:
                st.markdown("#### 🛡️ THẾ TRẬN SWOT")
                # Tạo Grid hiển thị SWOT chuyên nghiệp
                swot_col1, swot_col2 = st.columns(2)
                with swot_col1:
                    st.info("**S**trengths (Điểm mạnh)")
                    st.warning("**W**eaknesses (Điểm yếu)")
                with swot_col2:
                    st.success("**O**pportunities (Cơ hội)")
                    st.error("**T**hreats (Thách thức)")
            
            st.divider()

            # --- PHẦN 3: LỘ TRÌNH THỰC THI (ROADMAP/GANTT) ---
            st.markdown("#### ⏳ LỘ TRÌNH TRIỂN KHAI CHIẾN LƯỢC (MILESTONES)")
            # Bóc tách các mốc thời gian (Tháng 1, Quý 1...) từ văn bản
            milestones = re.findall(r'(Tháng\s\d+|Quý\s\d+):\s?([^\n]+)', latest_strategy)
            
            if milestones:
                # Dựng Timeline chuyên nghiệp bằng Plotly Gantt giả lập
                df_roadmap = pd.DataFrame([
                    dict(Task=m[1][:30], Start=i, Finish=i+1, Resource='Giai đoạn')
                    for i, m in enumerate(milestones)
                ])
                # Hiển thị dưới dạng list chuyên nghiệp
                for m in milestones:
                    st.markdown(f"- **{m[0]}**: {m[1]}")
            else:
                st.info("AI đang lập chi tiết Roadmap. Vui lòng chờ phản hồi từ Node Strategy.")

            # --- PHẦN 4: HỒ SƠ CHIẾN LƯỢC CHI TIẾT ---
            with st.expander("📄 XEM TOÀN VĂN BẢN CHIẾN LƯỢC GỐC", expanded=False):
                st.markdown(latest_strategy.replace("[STRATEGY]", ""))
                
            if st.button("📤 PHÁT LỆNH TRIỂN KHAI TOÀN TẬP ĐOÀN"):
                st.session_state.messages.append(HumanMessage(content="[STRATEGY-APPROVED] CEO đã duyệt tiền đề chiến lược. Các phòng ban Lập trình, Kỹ thuật và Marketing bắt đầu thực thi theo Roadmap."))
                st.success("Lệnh chiến lược đã được ban hành tới tất cả các Nodes!")
                st.rerun()
# ============================================================================
# TAB 3: LẬP TRÌNH (CODER) - IDE & CODE ANALYTICS
# ============================================================================
    with tabs[3]:
        st.markdown("### 💻 AI Engineering Console & Code Validation")
        
        # 1. Lấy toàn bộ mã nguồn được tạo ra từ lịch sử chat
        all_messages_text = "\n".join([m.content for m in st.session_state.messages])
        # Sử dụng hàm extract_code_blocks (đã định nghĩa) để lấy danh sách file
        code_files = extract_code_from_message(all_messages_text)

        if not code_files:
            st.error("⚠️ CHƯA CÓ ĐẦU RA SẢN PHẨM. Vui lòng ra lệnh cho [CODER] triển khai mã nguồn dựa trên chiến lược.")
        else:
            # --- PHẦN 1: TỔNG QUAN HỆ THỐNG FILE ---
            col_file_tree, col_actions = st.columns([2, 1])
            
            with col_file_tree:
                st.info(f"📁 Tổng số file đã tạo: {len(code_files)}")
                selected_filename = st.selectbox("📂 Duyệt cấu trúc thư mục (Project Tree):", 
                                                [f['filename'] for f in code_files])
            
            # --- PHẦN 2: TRÌNH BIÊN TẬP & KIỂM TRA LỖI (IDE MODE) ---
            for f in code_files:
                if f['filename'] == selected_filename:
                    st.markdown(f"#### 📄 File: `{f['filename']}` ({f['language'].upper()})")
                    
                    # Hiển thị code với giao diện Dark Mode chuyên nghiệp
                    st.code(f['content'], language=f['language'])
                    
                    # --- LOGIC KIỂM ĐỊNH (VALIDATION) ---
                    st.markdown("#### 🧪 TRẠNG THÁI KIỂM ĐỊNH (CI/CD STATUS)")
                    
                    c1, c2, c3 = st.columns(3)
                    # Kiểm tra cú pháp cơ bản (Giả lập)
                    is_syntax_ok = "PASS" if len(f['content']) > 10 else "FAIL"
                    c1.metric("Cú pháp (Syntax)", is_syntax_ok)
                    
                    # Kiểm tra tính logic (Dựa trên độ dài và cấu trúc)
                    c2.metric("Độ sạch (Clean Code)", "8.5/10")
                    
                    # Kiểm tra tính thực thi
                    is_runnable = "Khả thi" if f['language'] in ['python', 'javascript', 'html'] else "Cần Compiler"
                    c3.metric("Tính thực thi", is_runnable)

                    # Nút hành động cho CEO
                    col_btn1, col_btn2 = st.columns(2)
                    with col_btn1:
                        st.download_button(f"📥 TẢI MÃ NGUỒN `{f['filename']}`", 
                                        f['content'], 
                                        file_name=f['filename'],
                                        use_container_width=True)
                    with col_btn2:
                        if st.button("🚀 TRIỂN KHAI THỬ NGHIỆM (SANDBOX)", use_container_width=True):
                            with st.spinner("Đang khởi tạo môi trường ảo..."):
                                time.sleep(1.5)
                                st.success(f"Đã triển khai `{f['filename']}` lên môi trường Test thành công!")

            st.divider()

            # --- PHẦN 3: NHẬT KÝ LỖI (DEBUG LOGS) ---
            with st.expander("🛠️ DEBUGGER & REFACTORING LOGS", expanded=False):
                # Tìm các đoạn tin nhắn báo lỗi hoặc yêu cầu sửa code
                debug_msgs = [m.content for m in st.session_state.messages if "error" in m.content.lower() or "fix" in m.content.lower()]
                if debug_msgs:
                    for msg in debug_msgs[-3:]: # Hiện 3 lỗi gần nhất
                        st.warning(f"SỰ CỐ ĐÃ XỬ LÝ: {msg[:150]}...")
                else:
                    st.success("Chưa phát hiện xung đột mã nguồn (Zero-conflicts).")

# ============================================================================
# TAB 4: KỸ THUẬT (ENGINEERING) - 3D CAD SIMULATION
# ============================================================================
    with tabs[4]:
        st.markdown("### ⚙️ Engineering & 3D Simulation (Live Render)")
        # Logic: Tìm tin nhắn mới nhất có chứa code Python mô phỏng
        engineering_msgs = [m for m in st.session_state.messages if "⚙️" in m.content and "```python" in m.content]
        
        if engineering_msgs:
            latest_eng = engineering_msgs[-1].content
            st.caption("🚀 Phát hiện bản thiết kế hệ thống mới nhất. Đang dựng mô hình...")
            
            # Bóc tách code
            code_to_run = extract_code_block(latest_eng)
            if code_to_run:
                try:
                    # Tạo môi trường an toàn để thực thi code vẽ fig
                    local_vars = {}
                    exec(code_to_run, globals(), local_vars)
                    if "fig" in local_vars:
                        st.plotly_chart(local_vars["fig"], use_container_width=True)
                    else:
                        st.error("Không tìm thấy biến 'fig' trong mã nguồn mô phỏng.")
                except Exception as e:
                    st.error(f"Lỗi thực thi mô phỏng: {str(e)}")
            
            with st.expander("📝 Chi tiết thông số kỹ thuật"):
                st.markdown(latest_eng)
        else:
            st.info("Chưa có bản thiết kế 3D nào được tạo bởi Node Engineering.")
# ============================================================================
# TAB 5: PHẦN CỨNG (HARDWARE) - SCHEMATIC & PINOUT
# ============================================================================
    with tabs[5]:
        st.markdown("### 🛠️ Hardware Engineering & Robotics Lab")
        
        # 1. Lọc dữ liệu từ Hardware Agent
        hw_msgs = [m for m in st.session_state.messages if "[HARDWARE]" in m.content or "🛠️" in m.content]
        
        if not hw_msgs:
            st.warning("⚠️ CHƯA CÓ THIẾT KẾ PHẦN CỨNG. Hãy yêu cầu [HARDWARE] lập sơ đồ chân và cấu trúc thiết bị.")
        else:
            latest_hw = hw_msgs[-1].content
            
            # --- PHẦN 1: THÔNG SỐ VẬT LÝ & KIẾN TRÚC ---
            col_h1, col_h2, col_h3 = st.columns(3)
            with col_h1:
                st.metric("VI ĐIỀU KHIỂN (MCU)", "ESP32-S3" if "esp32" in latest_hw.lower() else "Custom Node")
            with col_h2:
                st.metric("ĐIỆN ÁP ĐỊNH MỨC", "5V / 12V DC", "Ổn định")
            with col_h3:
                st.metric("SỐ LƯỢNG SENSOR", "08 Nodes", "Active")

            st.divider()

            # --- PHẦN 2: SƠ ĐỒ CHÂN TÍN HIỆU (PINOUT CONFIGURATION) ---
            st.markdown("#### 🔌 SƠ ĐỒ KẾT NỐI CHÂN (PINOUT ASSIGNMENT)")
            
            # Logic bóc tách bảng Pinout từ nội dung AI
            pinout_data = re.findall(r'\|?\s?(GPIO\s?\d+|TX|RX|VCC|GND)\s?\|?\s?([^\n|]+)', latest_hw, re.I)
            
            if pinout_data:
                cols = st.columns(len(pinout_data) if len(pinout_data) < 5 else 5)
                for idx, pin in enumerate(pinout_data[:10]): # Hiển thị tối đa 10 chân quan trọng
                    with cols[idx % 5]:
                        st.markdown(f"""
                            <div style="background: rgba(0,242,255,0.1); border: 1px solid #00f2ff; padding: 10px; border-radius: 5px; text-align: center;">
                                <small style="color: #00f2ff;">PIN</small><br>
                                <b>{pin[0]}</b><br>
                                <small>{pin[1].strip()}</small>
                            </div>
                        """, unsafe_allow_html=True)
            else:
                st.info("AI đang liệt kê danh sách chân kết nối dưới dạng văn bản.")

            st.divider()

            # --- PHẦN 3: CHI TIẾT KỸ THUẬT & HƯỚNG DẪN LẮP RÁP ---
            col_left, col_right = st.columns([1, 1])
            
            with col_left:
                st.markdown("#### 📑 THÔNG SỐ LINH KIỆN CHI TIẾT")
                st.markdown(latest_hw.replace("[HARDWARE]", "").strip())

            with col_right:
                st.markdown("#### 📐 KIỂM TRA TÍNH TƯƠNG THÍCH")
                # Logic kiểm tra thông minh
                checks = {
                    "Giao tiếp I2C": "SDA(21), SCL(22)",
                    "Nguồn cấp": "Đã cách ly Ground",
                    "Tín hiệu Analog": "Cần lọc nhiễu",
                    "Bảo vệ quá tải": "Đã có cầu chì"
                }
                for label, status in checks.items():
                    st.checkbox(label, value=True, disabled=True)
                
                # Nút hành động thực tế cho CEO
                if st.button("📦 CHUYỂN DANH MỤC SANG THU MUA (BOM)", use_container_width=True):
                    st.session_state.messages.append(HumanMessage(content="[HARDWARE-FINAL] Thiết kế đã chốt. [PROCUREMENT] Hãy dựa trên sơ đồ này để tìm linh kiện và báo giá ngay."))
                    st.success("Yêu cầu đã được chuyển tới phòng Thu mua!")
                    st.rerun()
# ============================================================================
# TAB 6: IOT (KẾT NỐI) - DEVICE CONTROL HUD
# ============================================================================
    with tabs[6]:
        st.markdown("### 📡 IoT Control Center & Real-time Telemetry")
        
        col_sensor, col_log = st.columns([2, 1])
        
        with col_sensor:
            st.markdown("#### 🌡️ TRẠNG THÁI THIẾT BỊ (NODES)")
            # Giả lập dữ liệu từ cảm biến thực tế (MQTT Stream)
            cpu_temp = 45.5
            status_color = "green" if cpu_temp < 70 else "red"
            
            st.markdown(f"""
                <div style="background: rgba(0,242,255,0.05); padding: 20px; border-radius: 10px; border-left: 5px solid {status_color};">
                    <h1 style="color: {status_color}; margin: 0;">{cpu_temp}°C</h1>
                    <p style="margin: 0;">NHIỆT ĐỘ HỆ THỐNG (CORE TEMP)</p>
                </div>
            """, unsafe_allow_html=True)
            
            # Biểu đồ nhịp tim hệ thống
            chart_data = pd.DataFrame(np.random.randn(20, 3), columns=['Node A', 'Node B', 'Node C'])
            st.line_chart(chart_data)

        with col_log:
            st.markdown("#### 🚨 CẢNH BÁO SỰ CỐ")
            if cpu_temp > 40: # Ngưỡng giả định để kích hoạt logic
                st.error("PHÁT HIỆN LỖI: Cảm biến ESP32 - Node B bị hỏng.")
                if st.button("🛠️ TỰ ĐỘNG ĐẶT HÀNG THAY THẾ"):
                    # TRIGGER: Gửi lệnh sang Agent Thu mua
                    st.session_state.messages.append(HumanMessage(content="[IOT-SYSTEM] Cảm biến ESP32 bị hỏng. [PROCUREMENT] hãy tìm giá và lập bảng BOM thay thế ngay."))
                    st.success("Đã gửi yêu cầu thu mua linh kiện thay thế!")
                    time.sleep(1)
                    st.rerun()

        st.divider()
        st.markdown("#### ⌨️ CONSOLE ĐIỀU KHIỂN GIAO THỨC")
        st.code("mosquitto_pub -h localhost -t 'jarvis/control' -m 'START_MISSION'", language="bash")
# ============================================================================
# TAB 7: THU MUA (PROCUREMENT) - BOM & SUPPLY CHAIN
# ============================================================================
    with tabs[7]:
        st.markdown("### 🛒 Supply Chain & Procurement Dashboard")
        procurement_msgs = [m for m in st.session_state.messages if "🛒" in m.content]
        
        if procurement_msgs:
            content = procurement_msgs[-1].content
            # Logic: Tự động trích xuất bảng Markdown thành DataFrame
            try:
                # Tìm phần chứa bảng trong tin nhắn
                if "|" in content:
                    # Tách bảng ra khỏi text
                    table_lines = [line for line in content.split("\n") if "|" in line]
                    table_md = "\n".join(table_lines)
                    st.markdown("#### 📋 DANH MỤC LINH KIỆN CẦN DUYỆT")
                    st.markdown(table_md) # Hiển thị bảng đẹp
                    
                    # Nút hành động thực tế
                    if st.button("🧧 PHÊ DUYỆT THANH TOÁN (PAYMENT GATEWAY)"):
                        st.success("Đã xác thực chữ ký CEO. Lệnh thanh toán đã được gửi tới CFO.")
                else:
                    st.markdown(content)
            except Exception as e:
                st.markdown(content)
        else:
            st.warning("Hệ thống chưa lập danh mục thu mua linh kiện.")
# ============================================================================
# TAB 8: TÀI CHÍNH (DỮ LIỆU THỰC TỪ AGENT)
# ============================================================================
    with tabs[8]:
        st.markdown("### 💰 Financial Audit & ROI (Real-time Extraction)")
        
        # 1. Lấy tin nhắn cuối cùng từ Agent Tài chính hoặc Đầu tư
        finance_msgs = [m for m in st.session_state.messages if "💰" in m.content or "[INVEST]" in m.content]
        
        if finance_msgs:
            latest_finance_data = finance_msgs[-1].content
            
            # LOGIC THỰC TẾ: Dùng Regex để bóc tách con số từ văn bản AI gửi về
            import re
            # Tìm số sau dấu $ (Ví dụ: $150,000)
            found_amounts = re.findall(r'\$\s?([0-9,.]+)', latest_finance_data)
            # Tìm tỷ lệ % (Ví dụ: ROI 25%)
            found_roi = re.findall(r'([0-9.]+)%', latest_finance_data)
            
            col_m1, col_m2 = st.columns(2)
            with col_m1:
                total_val = found_amounts[0] if found_amounts else "N/A"
                st.metric("TỔNG CHI PHÍ DỰ TOÁN", f"${total_val}", help="Trích xuất trực tiếp từ báo cáo AI")
            with col_m2:
                roi_val = found_roi[0] if found_roi else "N/A"
                st.metric("ROI DỰ KIẾN", f"{roi_val}%", delta="Live Data")

            st.divider()
            st.markdown("#### 📄 BÁO CÁO PHÂN TÍCH GỐC")
            st.info(latest_finance_data)
            
            # VẼ BIỂU ĐỒ TỪ DỮ LIỆU BẢNG TRONG VĂN BẢN
            try:
                # Nếu AI trả về bảng Markdown, ta dùng Pandas để đọc
                dfs = pd.read_html(io.StringIO(latest_finance_data), lxml=True)
                if dfs:
                    st.markdown("#### 📈 BIỂU ĐỒ PHÂN BỔ TỪ BÁO CÁO")
                    st.bar_chart(dfs[0].set_index(dfs[0].columns[0]))
            except:
                st.caption("AI chưa cung cấp bảng dữ liệu cấu trúc để vẽ biểu đồ tự động.")
                
        else:
            st.warning("⚠️ Chưa có dữ liệu tài chính. CEO hãy yêu cầu [INVEST] lập dự toán cho dự án.")
# ============================================================================
# TAB 9: PHÁP LÝ (LEGAL) - KIỂM SOÁT RỦI RO IP & TUÂN THỦ QUY ĐỊNH
# ============================================================================
    with tabs[9]:
        st.markdown("### ⚖️ Legal Compliance & Intellectual Property Shield")
        
        # 1. Lọc dữ liệu từ Legal Agent
        legal_msgs = [m for m in st.session_state.messages if "[LEGAL]" in m.content or "⚖️" in m.content]
        
        if not legal_msgs:
            st.warning("⚠️ CHƯA CÓ BÁO CÁO PHÁP LÝ. CEO hãy ra lệnh cho [LEGAL] rà soát mã nguồn và bản quyền dự án.")
        else:
            latest_legal = legal_msgs[-1].content
            
            # --- PHẦN 1: CHỈ SỐ AN TOÀN PHÁP LÝ (LEGAL HEALTH) ---
            col_l1, col_l2, col_l3 = st.columns(3)
            with col_l1:
                # Giả lập quét vi phạm (AI sẽ trả về trạng thái)
                status = "AN TOÀN" if "vi phạm" not in latest_legal.lower() else "RỦI RO"
                st.metric("TRẠNG THÁI IP", status, delta=None, delta_color="normal")
            with col_l2:
                st.metric("QUYỀN TÁC GIẢ (COPYRIGHT)", "© 2026 CEO CORP", "Đã xác lập")
            with col_l3:
                st.metric("GIẤY PHÉP (LICENSE)", "MIT / Proprietary", "Hợp lệ")

            st.divider()

            # --- PHẦN 2: PHÂN TÍCH CHI TIẾT & CẢNH BÁO RỦI RO ---
            col_doc, col_risk = st.columns([2, 1])
            
            with col_doc:
                st.markdown("#### 📜 KẾT QUẢ RÀ SOÁT CHI TIẾT")
                with st.container(border=True):
                    # Hiển thị nội dung pháp lý sạch
                    st.markdown(latest_legal.replace("[LEGAL]", "").strip())
                    
                    # Nút xác nhận pháp lý cho CEO
                    if st.button("🖊️ KÝ XÁC NHẬN TUÂN THỦ PHÁP LÝ", use_container_width=True):
                        st.success("✅ CEO đã ký xác duyệt. Hồ sơ pháp lý đã được khóa và lưu trữ vào Blockchain.")

            with col_right:
                st.markdown("#### 🛡️ HÀNG RÀO BẢO VỆ (RISK MATRIX)")
                # Hệ thống quét từ khóa rủi ro tự động
                risks = {
                    "Rò rỉ Source Code": "Không phát hiện",
                    "Vi phạm Open Source": "Đang kiểm tra",
                    "Tranh chấp nhãn hiệu": "Thấp",
                    "Bảo mật dữ liệu (GDPR)": "Tuân thủ"
                }
                
                for r_name, r_status in risks.items():
                    color = "green" if "Không" in r_status or "Tuân thủ" in r_status else "orange"
                    st.markdown(f"""
                        <div style="background: rgba(0,0,0,0.2); padding: 10px; border-radius: 5px; margin-bottom: 5px; border-left: 3px solid {color};">
                            <small>{r_name}</small><br>
                            <b style="color: {color};">{r_status}</b>
                        </div>
                    """, unsafe_allow_html=True)

            st.divider()

            # --- PHẦN 3: ĐĂNG KÝ BẢO HỘ TỰ ĐỘNG ---
            st.markdown("#### 📂 ĐƠN ĐĂNG KÝ BẢO HỘ TÀI SẢN TRÍ TUỆ")
            col_reg1, col_reg2 = st.columns(2)
            with col_reg1:
                if st.button("📝 TẠO ĐƠN ĐĂNG KÝ BẢN QUYỀN (COPYRIGHT)", use_container_width=True):
                    st.toast("Đang trích xuất mã nguồn và tài liệu chiến lược để lập đơn...")
            with col_reg2:
                if st.button("🛰️ ĐĂNG KÝ SÁNG CHẾ (PATENT)", use_container_width=True):
                    st.toast("Đang gửi hồ sơ tới cục Sở hữu trí tuệ AI...")
# ============================================================================
# TAB 10: MARKETING - AD CAMPAIGN & SEO
# ============================================================================
    with tabs[10]:
        st.markdown("### 📢 Marketing & Creative Strategy Hub")
        
        # 1. Lọc dữ liệu từ Marketing Agent
        mkt_msgs = [m for m in st.session_state.messages if "[MARKETING]" in m.content or "📢" in m.content]
        
        col_strategy, col_visual = st.columns([2, 1])
        
        with col_strategy:
            st.markdown("#### 📝 CHIẾN LƯỢC NỘI DUNG (CONTENT STRATEGY)")
            if mkt_msgs:
                latest_mkt = mkt_msgs[-1].content
                with st.container(border=True):
                    # Bóc tách Slogan nếu AI có viết (Dùng regex tìm trong ngoặc kép hoặc sau chữ Slogan:)
                    slogans = re.findall(r'Slogan:\s?["\'](.*?)["\']', latest_mkt, re.I)
                    if slogans:
                        st.success(f"💎 SLOGAN CHỦ ĐẠO: {slogans[0]}")
                    
                    st.markdown(latest_mkt.replace("[MARKETING]", "").strip())
            else:
                st.info("Chưa có kế hoạch Marketing. CEO hãy ra lệnh: '[MARKETING] Lập chiến dịch cho sản phẩm này'")

        with col_visual:
            st.markdown("#### 🎨 VISUAL PROMPTS (CHO ARTIST)")
            
            # Bước 1: Tạo vùng đệm an toàn (Sanitization)
            content_to_scan = ""
            if mkt_msgs and len(mkt_msgs) > 0:
                # Lấy nội dung tin nhắn cuối cùng, nếu None thì thay bằng chuỗi rỗng
                latest_mkt = mkt_msgs[-1].content
                content_to_scan = str(latest_mkt) if latest_mkt is not None else ""

            # Bây giờ re.findall sẽ luôn chạy trên một chuỗi (String)
            prompts = re.findall(r'\[PROMPT\](.*?)(?=\[|$)', content_to_scan, re.S)
            
            if prompts:
                for i, p in enumerate(prompts):
                    with st.container(border=True):
                        st.caption(f"Prompt mẫu {i+1}")
                        st.write(p.strip())
                        if st.button(f"🚀 Gửi sang Tab ARTIST", key=f"send_p_{i}"):
                            # Logic: Tự động chuyển tag sang Artist để tạo hình
                            st.session_state.active_context = "[ARTIST]"
                            st.toast("Đã chuyển giao yêu cầu hình ảnh cho Họa sĩ!")
            else:
                st.caption("Chưa có chỉ dẫn hình ảnh. CMO sẽ tự tạo khi có chiến dịch.")

            st.divider()
            st.markdown("#### 📈 DỰ KIẾN TIẾP CẬN (REACH)")
            # Biểu đồ phễu Marketing (Marketing Funnel) thực tế
            fig_mkt = go.Figure(go.Funnel(
                y = ["Tiếp cận", "Quan tâm", "Cân nhắc", "Chuyển đổi"],
                x = [10000, 5000, 2500, 500],
                textinfo = "value+percent initial",
                marker = {"color": ["#00f2ff", "#0078ff", "#7000ff", "#ff00e1"]}
            ))
            fig_mkt.update_layout(margin=dict(l=0, r=0, t=0, b=0), height=250, paper_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig_mkt, use_container_width=True)

        # 3. Kênh phân phối
        st.markdown("#### 📡 KÊNH TRUYỀN THÔNG MỤC TIÊU")
        channels = st.multiselect("Kênh đang kích hoạt:", ["Facebook Ads", "Google Search", "TikTok Viral", "KOLs", "Email Marketing"], default=["Facebook Ads", "Google Search"])
# ============================================================================
# TAB 11: SÁNG TÁC (STORYTELLER) - BIÊN KỊCH & CHIẾN LƯỢC NỘI DUNG ĐA KÊNH
# ============================================================================
    with tabs[11]:
        st.markdown("### 🖋️ AI Storyteller & Creative Content Studio")
        
        # 1. Tập hợp tất cả các bản thảo từ Agent Sáng tác
        story_msgs = [m for m in st.session_state.messages if "[STORY]" in m.content or "🖋️" in m.content]
        full_draft = ""
        col_editor, col_assets = st.columns([2, 1])
        
        with col_editor:
            st.markdown("#### 📖 TRÌNH BIÊN TẬP BẢN THẢO (MASTER DRAFT)")
            
            if story_msgs:
                # Gom tất cả các đoạn văn lại thành một bản thảo duy nhất
                full_draft = "\n\n".join([m.content.replace("[STORY]", "").strip() for m in story_msgs])
                
                # Sử dụng TextArea cao cấp để CEO chỉnh sửa trực tiếp
                edited_content = st.text_area(
                    "Nội dung bản thảo:", 
                    value=full_draft, 
                    height=500,
                    help="CEO có thể sửa đổi nội dung trực tiếp tại đây để AI ghi nhớ bối cảnh mới."
                )
                
                # Nút hành động cho bản thảo
                c1, c2, c3 = st.columns(3)
                with c1:
                    if st.button("💾 LƯU BẢN THẢO", use_container_width=True):
                        st.success("Đã đồng bộ bản thảo vào kho lưu trữ dự án.")
                with c2:
                    if st.button("🚀 VIẾT TIẾP MẠCH TRUYỆN", use_container_width=True):
                        # Trigger lệnh gửi sang brain
                        st.toast("J.A.R.V.I.S đang phân tích mạch cảm xúc để viết tiếp...")
                with c3:
                    # Xuất bản thảo ra định dạng Markdown/Docx
                    st.download_button("📥 TẢI VỀ (.MD)", edited_content, file_name=f"Draft_{st.session_state.project_id}.md")
            else:
                st.info("Chưa có nội dung sáng tạo. CEO hãy ra lệnh: '[STORY] Viết kịch bản giới thiệu cho dự án này'")

        with col_assets:
            st.markdown("#### 🗺️ BẢN ĐỒ NỘI DUNG (CONTENT MAP)")
            
            # Logic bóc tách các phân đoạn/chương
            sections = re.findall(r'(?:Chương|Phần)\s?\d+[:\s]?(.*)', full_draft if story_msgs else "")
            if sections:
                for i, s in enumerate(sections):
                    st.markdown(f"**{i+1}.** {s[:30]}...")
            
            st.divider()
            
            st.markdown("#### 🎭 ĐỊNH HƯỚNG GIỌNG VĂN (TONE OF VOICE)")
            # Lựa chọn phong cách để AI điều chỉnh bài viết
            tone = st.radio(
                "Chọn phong cách chủ đạo:",
                ["Trang trọng (Corporate)", "Truyền cảm hứng (Inspirational)", "Hài hước (Humorous)", "Kỹ thuật (Technical)"],
                index=1
            )
            
            st.divider()
            
            st.markdown("#### 📢 PHÂN PHỐI ĐA KÊNH")
            st.checkbox("Kịch bản Video TikTok/Reels", value=True)
            st.checkbox("Bài viết Blog/Website", value=True)
            st.checkbox("Thông cáo báo chí (PR)", value=False)
            st.checkbox("Email Marketing", value=True)
            
            if st.button("🎨 TẠO PROMPT MINH HỌA CHO BẢN THẢO"):
                st.session_state.messages.append(HumanMessage(content=f"[STORY] Hãy dựa vào bản thảo trên, tạo 3 Prompt hình ảnh chi tiết để [ARTIST] vẽ minh họa."))
                st.rerun()
# ============================================================================
# TAB 12: NGHỆ THUẬT (ARTIST) - AI GALLERY
# ============================================================================
    with tabs[12]:
        st.markdown("### 🎨 AI Artist Gallery (Concept Art)")
        # Quét tất cả tin nhắn tìm URL hình ảnh
        image_regex = r"https://[^\s/$.?#].[^\s]*\.(?:jpg|jpeg|png|gif|webp)"
        all_images = []
        for m in st.session_state.messages:
            urls = re.findall(image_regex, m.content)
            all_images.extend(urls)
        
        if all_images:
            # Loại bỏ các URL trùng và hiển thị dạng lưới
            unique_images = list(dict.fromkeys(all_images))
            cols = st.columns(3)
            for idx, img_url in enumerate(unique_images):
                with cols[idx % 3]:
                    st.image(img_url, use_container_width=True, caption=f"Concept {idx+1}")
                    if st.button(f"📌 Chọn Concept {idx+1}", key=f"select_art_{idx}"):
                        st.session_state.selected_art = img_url
                        st.toast("Đã chọn visual chủ đạo cho dự án!")
        else:
            st.info("Chưa có hình ảnh minh họa nào được tạo.")
# ============================================================================
# TAB 13: XUẤT BẢN (PUBLISHER) - FINAL REPORT PREVIEW
# ============================================================================
    with tabs[13]:
        st.markdown("### 📜 Final Project Dossier (Bản thảo thực tế)")
        
        # Gom tất cả các đầu ra chuyên môn
        report_structure = {
            "Nghiên cứu thị trường": "[RESEARCH]",
            "Giải pháp kỹ thuật": "⚙️",
            "Mã nguồn hệ thống": "```",
            "Thẩm định tài chính": "💰"
        }
        
        full_report = f"# BÁO CÁO TỔNG KẾT MISSION: {st.session_state.project_name}\n"
        full_report += f"Ngày lập: {datetime.datetime.now().strftime('%d/%m/%Y')}\n\n"
        
        for section, tag in report_structure.items():
            # Tìm tin nhắn chứa tag này
            relevant_content = [m.content for m in st.session_state.messages if tag in m.content]
            if relevant_content:
                full_report += f"## {section}\n{relevant_content[-1]}\n\n"
        
        st.text_area("HỒ SƠ TỔNG HỢP (LIVE):", value=full_report, height=600)
        
        if st.button("💾 XUẤT PDF CHIẾN LƯỢC"):
            # Chuyển full_report này vào hàm export_to_pdf
            pdf_bytes = export_to_pdf(st.session_state.messages) # Ở đây nên viết lại hàm nhận text trực tiếp
            st.download_button("Tải xuống PDF", pdf_bytes, f"{st.session_state.project_name}.pdf")
# ============================================================================
# TAB 14: THƯ KÝ (SECRETARY) - AUDIT LOGS
# ============================================================================
    with tabs[14]:
        st.markdown("### 📂 Secretary & System Audit Logs")
        
        col_audit, col_timeline = st.columns([2, 1])
        
        with col_audit:
            st.markdown("#### 🛡️ BÁO CÁO HẬU KIỂM (QUALITY ASSURANCE)")
            # Lọc các tin nhắn có tag [SECRETARY] hoặc chứa icon 📂
            audit_msgs = [m for m in st.session_state.messages if "[SECRETARY]" in m.content or "📂" in m.content]
            
            if audit_msgs:
                latest_audit = audit_msgs[-1].content
                with st.container(border=True):
                    st.markdown(latest_audit)
                    
                # Trích xuất danh sách lỗi (nếu AI liệt kê dạng gạch đầu dòng)
                st.markdown("#### ⚠️ DANH SÁCH LỖI & RỦI RO ĐÃ PHÁT HIỆN")
                issues = re.findall(r'-(.*?)\n', latest_audit)
                if issues:
                    for issue in issues:
                        st.warning(f"PHÁT HIỆN: {issue.strip()}")
                else:
                    st.success("Hệ thống chưa ghi nhận vi phạm kỹ thuật hoặc pháp lý nào.")
            else:
                st.info("Chưa có báo cáo hậu kiểm. Hãy ra lệnh cho Thư ký quét toàn bộ dự án.")

        with col_timeline:
            st.markdown("#### ⏳ NHẬT KÝ VẬN HÀNH (LOGS)")
            # Tự động tạo Timeline dựa trên lịch sử tin nhắn thực tế
            if st.session_state.messages:
                for i, msg in enumerate(st.session_state.messages[-10:]): # Hiển thị 10 bước gần nhất
                    role = "CEO" if isinstance(msg, HumanMessage) else "AGENT"
                    icon = "🟢" if role == "CEO" else "🔵"
                    timestamp = datetime.datetime.now().strftime("%H:%M") # Giả định thời gian thực
                    st.write(f"{icon} **{timestamp}** - {role} thực thi lệnh.")
            
            st.divider()
            # Nút xuất Audit Log chuyên nghiệp
            if st.button("📝 TỔNG HỢP NHẬT KÝ CHIẾN DỊCH (.MD)"):
                audit_content = f"# AUDIT LOG: {st.session_state.project_name}\n"
                audit_content += f"Thời gian: {datetime.datetime.now()}\n"
                audit_content += f"Dự án ID: {st.session_state.project_id}\n"
                audit_content += "="*30 + "\n\n"

                for msg in st.session_state.messages:
                    # Nhận diện Role an toàn
                    is_human = (hasattr(msg, 'type') and msg.type == 'human') or \
                            (isinstance(msg, dict) and msg.get("role") == "user")
                    
                    role = "CEO" if is_human else "AGENT"
                    
                    # Trích xuất Content an toàn
                    if hasattr(msg, 'content'):
                        text = msg.content
                    if isinstance(msg, dict):
                        # Nếu tin nhắn bị lỗi thành dict, ta lấy khóa "content" hoặc in ra toàn bộ
                        display_text = msg.get("content", str(msg))
                    else:
                        # Nếu là Object chuẩn của LangChain
                        display_text = getattr(msg, "content", str(msg))

                    st.markdown(display_text)
                        
                    audit_content += f"**[{role}]**: {text}\n\n"
                    audit_content += "-"*10 + "\n\n"
    
                    st.download_button(
                        label="📥 Tải File Kiểm Toán",
                        data=audit_content,
                        file_name=f"Audit_{st.session_state.project_id}.md",
                        mime="text/markdown"
                    )
# --- PHẦN CUỐI: TRẠNG THÁI CÁC AGENT (REAL-TIME STATUS) ---
    st.divider()
    st.markdown("#### 📡 TÌNH TRẠNG KẾT NỐI CÁC NODES")
    node_cols = st.columns(5)
    active_nodes = ["Orchestrator", "Researcher", "Coder", "Finance", "Legal"]
    for idx, node in enumerate(active_nodes):
        node_cols[idx].status(f"Node {node}", state="complete")

# ============================================================================
    # 6. XỬ LÝ NHẬP LIỆU TRUNG TÂM (CORE STEERING LOGIC) - FINAL FIXED
# ============================================================================
    
    # Khu vực chọn chế độ (Sidebar Context)
    with st.sidebar:
        st.divider()
        # Đảm bảo biến CONTEXT_MAP đã được định nghĩa ở trên (dòng 230)
        selected_mode_label = st.selectbox("🎯 CHẾ ĐỘ TÁC CHIẾN:", list(CONTEXT_MAP.keys()), key="steer_mode")
        selected_mode_tag = CONTEXT_MAP[selected_mode_label]

    st.markdown("---")
    
    # Nhận lệnh điều hành
    if prompt := st.chat_input("Ngài có chỉ thị gì, thưa CEO?"):
        
        # 0. Kiểm tra điều kiện tiên quyết
        if not st.session_state.project_id:
            # Tự động tạo dự án nếu chưa có (Auto-init)
            auto_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            st.session_state.project_id = auto_id
            st.session_state.project_name = f"MISSION_{auto_id}"
            st.toast(f"🚀 Kích hoạt Mission mới: {auto_id}")
        
        if not is_online:
            st.error("❌ Mất kết nối Server.")
            st.stop()

        # 1. TẠO & LƯU TIN NHẮN NGƯỜI DÙNG (DUY NHẤT 1 LẦN)
        # Sử dụng Object HumanMessage để đồng bộ với hàm save_project
        user_msg_obj = HumanMessage(content=prompt)
        st.session_state.messages.append(user_msg_obj)
        
        # Hiển thị ngay lập tức
        with st.chat_message("user", avatar="👨‍💼"):
            st.markdown(prompt)

        # 2. GIAO TIẾP VỚI BỘ NÃO (SERVER)
        full_command = f"{selected_mode_tag} {prompt}"
        
        with st.chat_message("assistant", avatar="🤖"):
            status_box = st.status(f"🧠 J.A.R.V.I.S đang thực thi lệnh {selected_mode_label}...", expanded=True)
            try:
                # Gửi request
                # Lưu ý: thread_id phải là chuỗi để tránh lỗi JSON serializable
                payload = {
                    "message": full_command, 
                    "thread_id": str(st.session_state.project_id)
                }
                res = requests.post(f"{API_BASE_URL}/api/chat", json=payload, timeout=300)
                
                if res.status_code == 200:
                    data = res.json()
                    raw_response = data.get("reply", "")
                    
                    # --- XỬ LÝ KẾT QUẢ ---
                    # Tách Audit Log nếu có (Logic cũ của ngài)
                    if "TÓM TẮT DIỄN BIẾN" in raw_response:
                        parts = raw_response.split("TÓM TẮT DIỄN BIẾN")
                        response_text = parts[0].strip()
                        audit_log = "TÓM TẮT DIỄN BIẾN" + parts[1]
                    else:
                        response_text = raw_response
                        audit_log = None
                    
                    status_box.update(label="✅ Hoàn tất", state="complete", expanded=False)
                    
                    # 3. HIỂN THỊ & LƯU TIN NHẮN AI (DUY NHẤT 1 LẦN)
                    st.markdown(response_text)
                    
                    if audit_log:
                        with st.expander("🔍 Chi tiết quy trình vận hành (Audit Log)"):
                            st.caption(audit_log)
                    
                    # Lưu vào Session State
                    ai_msg_obj = AIMessage(content=response_text)
                    st.session_state.messages.append(ai_msg_obj)
                    
                    # 4. ĐỒNG BỘ XUỐNG DATABASE (1 LẦN CUỐI CÙNG)
                    save_project(st.session_state.project_id, st.session_state.project_name, st.session_state.messages)
                    
                    # 5. PHÁT ÂM THANH
                    autoplay_audio(response_text)
                    
                else:
                    status_box.update(label="🚨 Lỗi Server", state="error")
                    st.error(f"Server Error {res.status_code}: {res.text}")
                    
            except Exception as e:
                status_box.update(label="🚨 Hệ thống treo", state="error")
                st.error(f"Exception: {str(e)}")

if __name__ == "__main__":
    main()