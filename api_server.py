import glob
import os
import logger
import pandas as pd
import sqlite3
import uuid
import time
import io  
import shutil
import random
import logging
import aiofiles
import json  
import base64
import asyncio
import re
import smtplib
import zipfile
import hashlib
import requests
import feedparser
import traceback
from bs4 import BeautifulSoup
from sqlalchemy import create_engine, text
from typing import Optional, List, Dict, Any
from datetime import datetime
from contextlib import asynccontextmanager
from termcolor import colored
from gtts import gTTS
# from langchain_openai import ChatOpenAI
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from langchain_google_genai import ChatGoogleGenerativeAI
# --- CÀI ĐẶT THƯ VIỆN: pip install fastapi uvicorn python-multipart jinja2 aiofiles ---
from fastapi import FastAPI, HTTPException, Header, Depends, UploadFile, File, Request, status, WebSocket, WebSocketDisconnect, BackgroundTasks, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, Response, FileResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel
from langchain_core.messages import HumanMessage
from main import set_system_busy
from duckduckgo_search import DDGS
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from identity_core import jarvis_identity
from marketing_ai import marketing_agent
from research_agent import ResearchAgent
from main import run_nexus_sync

# [QUAN TRỌNG]: Đã thêm LLM_SUPERVISOR và log_training_data
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("JARVIS_v4.5")
ADMIN_SECRET = os.environ.get("ADMIN_SECRET", "ai_corp_secret_123")
RENDER_DISK_PATH = "/var/data"
IS_AUTOPILOT_ON = False
if os.path.exists(RENDER_DISK_PATH):
    # Nếu tìm thấy ổ cứng Cloud -> Lưu hết vào đó
    BASE_DATA_DIR = RENDER_DISK_PATH
    print(colored(f"💽 [STORAGE] Đã kết nối ổ cứng Cloud: {BASE_DATA_DIR}", "green", attrs=["bold"]))
else:
    # Nếu không thấy -> Đang chạy Local -> Lưu tại chỗ
    BASE_DATA_DIR = "."
    print(colored("💻 [STORAGE] Đang chạy chế độ Local (Lưu trên máy tính)", "yellow"))

# 2. Định nghĩa các đường dẫn quan trọng dựa trên Root Path
# Tất cả dữ liệu quan trọng phải nằm trong BASE_DATA_DIR
UPLOAD_DIR = os.path.join(BASE_DATA_DIR, "uploads")
PROJECTS_DIR = os.path.join(BASE_DATA_DIR, "projects")
DB_PATH = os.path.join(BASE_DATA_DIR, "ai_corp_projects.db")
VECTOR_DB_PATH = os.path.join(BASE_DATA_DIR, "db_knowledge") # Folder chứa vector database
TTS_CACHE_DIR = os.path.join(BASE_DATA_DIR, "tts_cache")
# 3. CẤU HÌNH DATABASE (FIX LỖI SPLIT BRAIN)
# Bất kể Render có cấp PostgreSQL hay không, ta vẫn ÉP DÙNG SQLITE 
# để đồng bộ với các hàm sqlite3.connect() bên dưới.
if os.path.exists(RENDER_DISK_PATH):
    # Chạy trên Cloud -> Dùng ổ cứng gắn ngoài
    os.environ["DATABASE_URL"] = f"sqlite:///{DB_PATH}"
    print(colored("cloud_mode: Ép dùng SQLite trên Disk để đồng bộ.", "yellow"))
else:
    # Chạy Local -> Dùng file tại chỗ
    os.environ["DATABASE_URL"] = f"sqlite:///{DB_PATH}"

AI_AVAILABLE = False
MEMORY_AVAILABLE = False
VOICE_AVAILABLE = False
SERVER_READY = False
ai_app = None
CHAT_MODEL = None
client = None
AI_BOOT_ERROR = None

try:
    from main import (
        ai_app, log_work_to_db, auto_learning_cycle, morning_briefing_job,
        vector_db, LLM_GPT4, LLM_GEMINI_LOGIC, LLM_GEMINI_VISION,
        CODER_PRIMARY, ingest_docs_to_memory, learn_knowledge, set_system_busy
    )
    AI_AVAILABLE = True
    SERVER_READY = True
    CHAT_MODEL = LLM_GEMINI_LOGIC
    logger.info("✅ CORE AI MODULES: LOADED & CHAT_MODEL SYNCED")
except Exception:
    # --- SỬ DỤNG TRACEBACK ĐÃ IMPORT ---
    AI_BOOT_ERROR = traceback.format_exc() # Lấy toàn bộ "dấu vân tay" của lỗi
    logger.error(f"⚠️ CORE AI FAILED TO LOAD:\n{AI_BOOT_ERROR}")
    
    # Set safe defaults để server không bị crash hoàn toàn
    ai_app = vector_db = LLM_GEMINI_LOGIC = LLM_GEMINI_VISION = None
    AI_AVAILABLE = False
try:
    CHAT_MODEL = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite", google_api_key=os.environ.get("GOOGLE_API_KEY"))
except:
    CHAT_MODEL = None
# Import Memory
try:
    from memory_core import recall_relevant_memories, extract_and_save_memory
    MEMORY_AVAILABLE = True
except ImportError:
    pass
# Import Voice
try:
    from voice_engine import client
    VOICE_AVAILABLE = True
except ImportError:
    VOICE_AVAILABLE = False
    client = None

# --- MODELS ---
class ChatRequest(BaseModel):
    message: str
    thread_id: str = "default"

class SpeakRequest(BaseModel):
    text: str

class LearnRequest(BaseModel):
    text: str

class TTSRequest(BaseModel):
    text: str

class Query(BaseModel):
    question: str

class BuyRequest(BaseModel):
    product_id: int

class LearningResult(BaseModel):
    source: str
    content: str
    worker_id: str

class TaskRequest(BaseModel):
    worker_id: str

class TaskResult(BaseModel):
    task_id: int
    worker_id: str
    result_content: str

class RegisterInfo(BaseModel):
    username: str
    email: str
    bank_name: str
    account_number: str  

class CourseRequest(BaseModel):
    subject: str      # Chủ đề lớn (VD: Học về ReactJS)
    num_tasks: int = 10 # Số lượng task muốn chia nhỏ 

class HunterRequest(BaseModel): # <--- Giải pháp cho lỗi gạch vàng
    keyword: str
# ==========================================
# 1. DATABASE MANAGER
# ==========================================
class DatabaseManager:
    def __init__(self): 
        # 1. CẤU HÌNH DATABASE (FIX LỖI SPLIT BRAIN)
        # Ép buộc dùng SQLite trên Disk để đồng bộ dữ liệu
        if os.path.exists(RENDER_DISK_PATH):
            os.environ["DATABASE_URL"] = f"sqlite:///{DB_PATH}"
        else:
            os.environ["DATABASE_URL"] = f"sqlite:///{DB_PATH}"

        self.db_url = os.environ["DATABASE_URL"]

        # 2. Cấu hình Engine
        self.engine = create_engine(
            self.db_url, 
            connect_args={
                "check_same_thread": False,
                "timeout": 30 # Đợi 30s nếu DB đang bị khóa
            },
            pool_recycle=600 
        )

        # Thêm lệnh này vào ngay sau khi tạo kết nối trong init_db
        with self.get_connection() as conn:
            conn.execute(text("PRAGMA journal_mode=WAL;")) # Chế độ ghi nhật ký trước (Cực kỳ an toàn)
            conn.execute(text("PRAGMA synchronous=NORMAL;"))
    
    def get_connection(self):
        return self.engine.connect()
    
    def init_db(self):
        """
        Hàm này chạy TỰ ĐỘNG mỗi khi Server khởi động.
        Và tự động NÂNG CẤP (Migrate) nếu bảng cũ thiếu cột.
        """
        try:
            pk_type = "INTEGER PRIMARY KEY AUTOINCREMENT"
            text_type = "TEXT"

            with self.get_connection() as conn:
                # --- 1. TẠO CÁC BẢNG NẾU CHƯA CÓ (Cấu trúc cơ bản) ---
                conn.execute(text(f"CREATE TABLE IF NOT EXISTS products (id {pk_type}, name {text_type}, price REAL)"))
                
                conn.execute(text(f"""
                    CREATE TABLE IF NOT EXISTS agent_status (
                        role_tag {text_type} PRIMARY KEY, xp INTEGER DEFAULT 0, 
                        current_topic {text_type}, last_updated TIMESTAMP
                    )
                """))
                
                conn.execute(text(f"""
                    CREATE TABLE IF NOT EXISTS work_logs (
                        id {pk_type}, timestamp {text_type}, agent_name {text_type}, 
                        task_content {text_type}, result_summary {text_type}, tool_used {text_type}, 
                        cost REAL, duration REAL
                    )
                """))
                
                conn.execute(text(f"CREATE TABLE IF NOT EXISTS projects (id {text_type} PRIMARY KEY, name {text_type}, history {text_type}, timestamp TIMESTAMP)"))
                
                conn.execute(text(f"CREATE TABLE IF NOT EXISTS async_tasks (task_id {text_type} PRIMARY KEY, status {text_type}, result {text_type}, timestamp TIMESTAMP)"))
                
                # Bảng Learning Tasks (Bảng bị lỗi thiếu cột)
                conn.execute(text(f"""
                    CREATE TABLE IF NOT EXISTS learning_tasks (
                        id {pk_type},
                        topic TEXT,
                        status TEXT DEFAULT 'PENDING',
                        assigned_to TEXT,
                        task_type TEXT DEFAULT 'RESEARCH',  -- Loại việc (Học/Làm)
                        content TEXT DEFAULT '',
                        reward REAL DEFAULT 0.0,            -- <--- CỘT GIÁ TIỀN (QUAN TRỌNG)
                        difficulty INTEGER DEFAULT 1,
                        last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """))

                conn.execute(text(f"""
                    CREATE TABLE IF NOT EXISTS users (
                        username {text_type} PRIMARY KEY,
                        api_key {text_type} UNIQUE,
                        email {text_type},
                        bank_info {text_type},
                        balance REAL DEFAULT 0.0,
                        reputation INTEGER DEFAULT 100,
                        joined_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """))
                
                # --- 2. TỰ ĐỘNG NÂNG CẤP (AUTO-MIGRATE) ---
                try: 
                    conn.execute(text("ALTER TABLE learning_tasks ADD COLUMN reward REAL DEFAULT 0.0"))
                    print("✅ [MIGRATE] Đã thêm cột giá tiền (reward) cho learning_tasks") 
                except: pass
                
                try: 
                    conn.execute(text("ALTER TABLE learning_tasks ADD COLUMN task_type TEXT DEFAULT 'RESEARCH'"))
                    print("✅ [MIGRATE] Đã thêm cột loại việc (task_type) cho learning_tasks") 
                except: pass

                # B. NÂNG CẤP BẢNG work_logs (SỬA LỖI CRASH ACADEMY)
                # Thêm cột xp_gain để lưu điểm kinh nghiệm mà Agent vừa học được
                try: 
                    conn.execute(text("ALTER TABLE work_logs ADD COLUMN xp_gain INTEGER DEFAULT 0"))
                    print("✅ [MIGRATE] Đã thêm cột điểm kinh nghiệm (xp_gain) cho work_logs")
                except: pass

                # Thêm cột thread_id để Dashboard có thể lọc lịch sử chat theo phiên
                try: 
                    conn.execute(text("ALTER TABLE work_logs ADD COLUMN thread_id TEXT DEFAULT 'default'"))
                    print("✅ [MIGRATE] Đã thêm cột thread_id cho work_logs")
                except: pass

                conn.commit()
            print(colored("✅ DATABASE SCHEMA: UP-TO-DATE (Đã hỗ trợ XP & Threading)", "green"))
            
        except Exception as e:
            print(colored(f"❌ DB INIT ERROR: {e}", "red"))

db_manager = DatabaseManager()
# ==========================================
# 2. WEBSOCKET MANAGER
# ==========================================
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)

    async def send_personal_message(self, message: str, websocket: WebSocket):
        await websocket.send_text(message)

    

    async def send_json(self, data: dict, websocket: WebSocket):
        """Gửi dữ liệu JSON (Quan trọng cho Dashboard hiển thị ảnh/agent)"""
        await websocket.send_json(data)

    async def broadcast(self, message: str):
        for connection in self.active_connections:
            await connection.send_text(message)


# ==========================================
# 3. PIPELINE DỰ ÁN LỚN
# ==========================================
async def run_architect_phase(project_request: str, thread_id: str):
    # 1. Logging gọn gàng (Lấy từ B)
    print(colored(f"📐 [ARCHITECT] Đang phác thảo: {project_request[:50]}...", "cyan"))
    
    # 2. Dùng biến đường dẫn chuẩn (Lấy từ B)
    os.makedirs(PROJECTS_DIR, exist_ok=True)
    plan_path = f"{PROJECTS_DIR}/{thread_id}_BLUEPRINT.md"
    
    # 3. Prompt chi tiết (Lấy từ A - QUAN TRỌNG ĐỂ PARSER KHÔNG LỖI)
    architect_prompt = (
        f"Bạn là Chief Software Architect (CSA). Có một yêu cầu dự án: '{project_request}'.\n"
        "Hãy lập một BẢN THIẾT KẾ KỸ THUẬT (Technical Blueprint) chi tiết dạng Markdown:\n\n"
        "1. [OVERVIEW]: Tóm tắt mục tiêu dự án.\n"
        "2. [MODULES]: Danh sách các chức năng chính.\n"
        "3. [DATABASE]: Sơ đồ bảng (Table Schema) chi tiết.\n"
        "4. [TECH STACK]: Công nghệ sử dụng.\n"
        "5. [EXECUTION PLAN] (QUAN TRỌNG): Hãy liệt kê lộ trình code cụ thể từng bước.\n"
        "   - Bắt buộc dùng gạch đầu dòng (-) cho mỗi bước code cụ thể.\n"
        "   - Ví dụ: - Tạo file main.py\n"
        "   - Ví dụ: - Viết API đăng nhập\n"
    )
    
    try:
        # 4. Cơ chế Fallback an toàn (Lấy từ B)
        content = "SIMULATION BLUEPRINT (AI NOT READY)"
        
        # Kiểm tra kỹ càng xem AI đã sẵn sàng chưa
        if SERVER_READY and 'LLM_GEMINI_VISION' in globals() and LLM_GEMINI_VISION:
            res = await run_in_threadpool(lambda: LLM_GEMINI_VISION.invoke(architect_prompt))
            content = res.content
        
        # 5. Luôn ghi file (Dù là thật hay giả lập)
        async with aiofiles.open(plan_path, "w", encoding="utf-8") as f:
            await f.write(content)
            
        print(colored(f"✅ [ARCHITECT DONE] Bản vẽ: {plan_path}", "green"))
        return content, plan_path

    except Exception as e:
        print(colored(f"❌ Lỗi Architect: {e}", "red"))
        return None, None

async def run_coding_phase(blueprint_content: str, thread_id: str):
    """
    Bước 2: Đọc bản vẽ -> Code từng phần -> Ghi log.
    (Phiên bản Tối ưu: Robust Parsing + Detailed Prompt + Safe Paths)
    """
    print(colored(f"🏗️ [EXECUTOR] Bắt đầu thi công dự án {thread_id}...", "magenta"))
    
    # 1. Dùng đường dẫn chuẩn từ cấu hình (Lấy ưu điểm Đoạn 2)
    log_file = os.path.join(PROJECTS_DIR, f"{thread_id}_coding_log.txt")
    
    # 2. Logic Parsing an toàn (Lấy ưu điểm Đoạn 1)
    raw_lines = blueprint_content.split('\n')
    steps = []
    is_in_plan = False
    
    for line in raw_lines:
        if "EXECUTION PLAN" in line.upper(): is_in_plan = True
        if is_in_plan and (line.strip().startswith('-') or line.strip().startswith('*')):
            step_clean = line.strip().lstrip('-* ').strip()
            if len(step_clean) > 5: # Bỏ qua các dòng quá ngắn/rác
                steps.append(step_clean)

    if not steps:
        print(colored("⚠️ Không tìm thấy bước code nào trong Blueprint. Dừng.", "yellow"))
        return

    # Khởi tạo file log
    async with aiofiles.open(log_file, "w", encoding="utf-8") as f:
        await f.write(f"=== BẮT ĐẦU DỰ ÁN {thread_id} ===\n\n")

    # 3. Vòng lặp thực thi (Lấy ưu điểm Prompt & Rate Limit của Đoạn 1)
    for idx, step in enumerate(steps):
        print(colored(f"⏳ [STEP {idx+1}/{len(steps)}]: {step}", "yellow"))
        
        step_prompt = (
            f"DỰ ÁN: {thread_id}\n"
            f"NHIỆM VỤ CỤ THỂ: {step}\n"
            "Yêu cầu: Viết code hoàn chỉnh cho nhiệm vụ này. Chỉ trả về Code, không giải thích dài dòng."
        )
        
        try:
            # Kiểm tra AI có sẵn sàng không
            if SERVER_READY and ai_app:
                state_res = await ai_app.ainvoke(
                    {"messages": [HumanMessage(content=step_prompt)]},
                    config={"configurable": {"thread_id": thread_id}}
                )
                ai_output = state_res['messages'][-1].content
            else:
                # Chế độ giả lập nếu mất kết nối AI
                ai_output = f"[SIMULATION] Coding step {idx+1} completed."
                await asyncio.sleep(1)

            # Ghi log
            async with aiofiles.open(log_file, "a", encoding="utf-8") as f:
                await f.write(f"\n\n{'='*30}\n### BƯỚC {idx+1}: {step}\n{'='*30}\n{ai_output}\n")
            
            # Rate Limiting (Quan trọng để không bị Ban API)
            await asyncio.sleep(2) 
            
        except Exception as e:
            print(colored(f"❌ Lỗi Step {idx+1}: {e}", "red"))

    print(colored(f"✅ [PROJECT COMPLETE] Dự án {thread_id} đã hoàn thành 100%!", "green"))

async def full_project_pipeline(user_request: str, thread_id: str):
    """
    Quy trình khép kín (Full Pipeline): 
    1. Architect (Vẽ) -> 2. Executor (Code) -> 3. Reporter (Báo cáo & Bàn giao).
    """
    start_time = time.time()
    print(colored(f"\n🚀 [PIPELINE STARTED] Dự án: {user_request} (ID: {thread_id})", "cyan", attrs=["bold"]))
    
    # Tạo thư mục dự án
    os.makedirs(PROJECTS_DIR, exist_ok=True)
    summary_path = os.path.join(PROJECTS_DIR, f"{thread_id}_SUMMARY.md")
    
    try:
        # --- GIAI ĐOẠN 1: KIẾN TRÚC SƯ (ARCHITECT) ---
        blueprint_content, blueprint_path = await run_architect_phase(user_request, thread_id)
        
        if not blueprint_content or not blueprint_path:
            raise Exception("Giai đoạn thiết kế thất bại (Architect Failed).")

        # --- GIAI ĐOẠN 2: THI CÔNG (EXECUTOR) ---
        # Hàm này sẽ chạy và ghi log vào file coding_log.txt
        await run_coding_phase(blueprint_content, thread_id)
        
        # --- GIAI ĐOẠN 3: TỔNG KẾT & BÀN GIAO (HANDOVER) ---
        duration = round(time.time() - start_time, 2)
        timestamp = datetime.now().strftime("%H:%M %d/%m/%Y")
        
        summary_content = f"""
            # 🏁 BÁO CÁO HOÀN THÀNH DỰ ÁN
            **Project ID:** `{thread_id}`
            **Yêu cầu:** {user_request}
            **Thời gian hoàn thành:** {timestamp} ({duration}s)

            ## 📂 TÀI LIỆU BÀN GIAO:
            1. **Bản vẽ kỹ thuật:** `..._BLUEPRINT.md`
            2. **Nhật ký Code:** `..._coding_log.txt`

            ## ✅ TRẠNG THÁI:
            - [x] Phân tích yêu cầu
            - [x] Thiết kế hệ thống
            - [x] Sinh mã nguồn (Simulation/AI)
            - [x] Đóng gói hồ sơ

            ---
            *Generated by J.A.R.V.I.S Full-Stack Engine*
                    """
                    
        # Ghi file tổng kết
        async with aiofiles.open(summary_path, "w", encoding="utf-8") as f:
            await f.write(summary_content)
            
        print(colored(f"🏁 [PIPELINE FINISHED] Tổng thời gian: {duration}s", "green", attrs=["bold"]))
        
        # Trả về kết quả cho API gọi nó
        return {
            "status": "SUCCESS",
            "project_id": thread_id,
            "duration": duration,
            "blueprint": blueprint_path,
            "summary": summary_path,
            "message": "Dự án đã hoàn thành và đóng gói."
        }

    except Exception as e:
        print(colored(f"❌ [PIPELINE FAILED] Lỗi: {e}", "red"))
        return {
            "status": "FAILED",
            "project_id": thread_id,
            "error": str(e)
        }

# ==========================================
# 4. APP & ROUTES
# ==========================================
# --- CÁC HÀM HỖ TRỢ KHỞI ĐỘNG (HELPER FUNCTIONS) ---
def setup_directories():
    """Tạo cấu trúc thư mục chuẩn"""
    base_path = os.path.abspath(os.path.dirname(__file__))
    required_dirs = [
        UPLOAD_DIR, PROJECTS_DIR, TTS_CACHE_DIR,
        os.path.join(base_path, "static"),
        os.path.join(base_path, "templates")
    ]
    for d in required_dirs:
        os.makedirs(d, exist_ok=True)
    logger.info(f"📂 System Directories: VERIFIED")

def cleanup_temp_files():
    """Dọn dẹp file rác cũ hơn 24h"""
    try:
        now = time.time()
        count = 0
        for folder in [UPLOAD_DIR, TTS_CACHE_DIR]:
            if os.path.exists(folder):
                for f in os.listdir(folder):
                    f_path = os.path.join(folder, f)
                    # Xóa file cũ > 24h
                    if os.path.isfile(f_path) and os.stat(f_path).st_mtime < now - 86400:
                        os.remove(f_path)
                        count += 1
        logger.info(f"🧹 Cleanup: Deleted {count} temporary files.")
    except Exception as e:
        logger.warning(f"⚠️ Cleanup Warning: {e}")

# --- HÀM LIFESPAN CHÍNH (ĐÃ TỐI ƯU) ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    # ==============================
    # 🟢 STARTUP SEQUENCE
    # ==============================
    print(colored("\n🚀 [SYSTEM] J.A.R.V.I.S KHOI DONG...", "cyan", attrs=["bold"]))
    
    # 1. Hạ tầng (Files & DB)
    setup_directories()
    cleanup_temp_files()
    
    try:
        db_manager.init_db()
        logger.info("✅ Database: INITIALIZED & CONNECTED")
    except Exception as e:
        logger.critical(f"❌ DATABASE FATAL ERROR: {e}")
        # Trong môi trường sản xuất, ngài có thể muốn dừng startup tại đây
    
    # 2. Lên lịch tác vụ (Scheduler) - Tích hợp Autopilot & Briefing
    # Chuyển scheduler ra biến global hoặc khai báo tại đây để quản lý
    app.state.scheduler = AsyncIOScheduler()
    
    # Job A: Báo cáo buổi sáng (Morning Briefing)
    if 'morning_briefing_job' in globals():
        app.state.scheduler.add_job(morning_briefing_job, 'cron', hour=7, minute=0, id="morning_report")
    
    # Job B: Autopilot - Thợ lặn tri thức (Mới cập nhật)
    # Tự động quét tin tức và sinh Task sau mỗi 60 phút
    app.state.scheduler.add_job(auto_knowledge_diver, 'interval', minutes=60, id="autopilot_diver")
    
    app.state.scheduler.start()
    logger.info("⏰ Scheduler: ACTIVE (Autopilot + Briefing)")

    # 3. Kích hoạt AI nền (Background Loops)
    app.state.learning_task = None
    if AI_AVAILABLE:
        logger.info("🧠 AI Core: ONLINE - Starting Self-Learning Loop...")
        # Sử dụng create_task để chạy song song không làm nghẽn API
        app.state.learning_task = asyncio.create_task(auto_learning_cycle())
    else:
        logger.warning("⚠️ AI Core: OFFLINE (Running in safe mode)")

    # ---> SERVER CHÍNH THỨC NHẬN LỆNH TỪ ĐÂY <---
    yield 
    
    # ==============================
    # 🔴 SHUTDOWN SEQUENCE (TẮT MÁY)
    # ==============================
    print(colored("\n💤 [SYSTEM] J.A.R.V.I.S ENTERING HIBERNATION...", "yellow"))
    
    # 1. Tắt bộ đếm giờ
    if app.state.scheduler.running:
        app.state.scheduler.shutdown()
    
    # 2. Dừng AI nạp tri thức an toàn
    if app.state.learning_task:
        app.state.learning_task.cancel()
        try:
            await asyncio.wait_for(app.state.learning_task, timeout=5.0)
        except (asyncio.CancelledError, asyncio.TimeoutError):
            logger.info("✅ AI Learning Loop terminated.")

    logger.info("👋 System Hibernate: SUCCESSFUL")



# --- KHỞI TẠO APP VỚI LIFESPAN ---
app = FastAPI(
    title="J.A.R.V.I.S v4.8 FULL",
    version="4.8",
    lifespan=lifespan
)

# 1. Cấu hình CORS (Cho phép mọi kết nối)
app.add_middleware(
    CORSMiddleware, 
    allow_origins=["*"], 
    allow_methods=["*"], 
    allow_headers=["*"]
)
# --- CẤU HÌNH STATIC FILES (AN TOÀN) ---
base_dir = os.path.abspath(os.path.dirname(__file__))
static_dir = os.path.join(base_dir, 'static')
templates_dir = os.path.join(base_dir, 'templates')

# Tạo thư mục TRƯỚC KHI Mount để tránh lỗi Crash
if not os.path.exists(static_dir):
    os.makedirs(static_dir)
if not os.path.exists(templates_dir):
    os.makedirs(templates_dir)

app.mount("/static", StaticFiles(directory=static_dir), name="static")
templates = Jinja2Templates(directory=templates_dir)

# ==========================================
# 5. ROUTES
# ==========================================

@app.get("/", response_class= HTMLResponse)
async def read_home(request: Request):
    # Nếu ngài có file index.html hoặc products.html thì để nguyên...
    return templates.TemplateResponse("store.html", {"request": request})

@app.get("/dashboard", response_class=HTMLResponse)
async def read_dashboard(request: Request):
    return templates.TemplateResponse("dashboard.html", {"request": request})

@app.get("/admin", response_class=HTMLResponse)
async def read_admin(request: Request):
    return templates.TemplateResponse("admin.html", {"request": request})

@app.get("/index")
async def redirect_index():
    return RedirectResponse(url="/")
@app.get("/org", response_class=HTMLResponse)
async def read_org_chart(request: Request):
    try:
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        c = conn.cursor()

        c.execute("SELECT * FROM agent_status ORDER BY xp DESC")
        raw = c.fetchall()
        agents = []
        for a in raw:
            xp = a['xp'] or 0
            agents.append({
                "name": a['role_tag'], "xp": xp, "level": int(xp/1000)+1,
                "topic": a['current_topic'], "progress": (xp%1000)/10
            })

        c.execute("SELECT agent_name, task_content, result_summary, timestamp FROM work_logs WHERE tool_used LIKE '%SUPREME%' OR tool_used LIKE '%DEBATE%' ORDER BY id DESC LIMIT 1")
        report = c.fetchone()
        conn.close()

        return templates.TemplateResponse("index.html", {
            "request": request, "agents": agents, "featured_report": report
        })
    except: return "Lỗi load Org Chart"

manager = ConnectionManager()

# Tách riêng để cả API và Autopilot đều dùng chung được
async def execute_distribute_knowledge(subject: str, num_tasks: int, reward: float):
    if not CHAT_MODEL: return 0
    prompt = f"Chia nhỏ chủ đề '{subject}' thành {num_tasks} task. Trả về JSON list: [{{'topic', 'task_type', 'content'}}]"
    try:
        res = await CHAT_MODEL.ainvoke(prompt)
        tasks = json.loads(res.content.replace("```json", "").replace("```", "").strip())
        with db_manager.get_connection() as conn:
            for t in tasks:
                conn.execute(text("INSERT INTO learning_tasks (topic, task_type, reward, content) VALUES (:t, :type, :r, :c)"),
                             {"t": f"[{subject.upper()}] {t['topic']}", "type": t.get('task_type', 'RESEARCH'), "r": reward, "c": t.get('content', '')})
            conn.commit()
        return len(tasks)
    except: return 0

# API (Hàm mẹ - Dashboard gọi vào đây)
@app.post("/api/admin/auto_distribute_knowledge")
async def api_distribute_knowledge(req: CourseRequest, x_api_key: str = Header(None)):
    if x_api_key != ADMIN_SECRET: raise HTTPException(403)
    # Định giá cố định $0.05 mỗi task khi phân bổ thủ công
    count = await execute_distribute_knowledge(req.subject, req.num_tasks, 0.05) 
    return {"status": "success", "tasks_created": count}

@app.post("/api/admin/auto_reply")
async def api_auto_reply(customer_email: str, x_api_key: str = Header(None)):
    if x_api_key != ADMIN_SECRET: 
        raise HTTPException(status_code=403)

    # Lấy mail mới nhất (giả sử dùng hàm đã có trong identity_core)
    customer_msg = jarvis_identity.fetch_latest_otp(keyword="tư vấn") 

    if not customer_msg:
        return {"status": "error", "msg": "Không có thư mới để trả lời."}

    # Gọi hàm từ module Marketing
    success = await marketing_agent.smart_reply(customer_msg, customer_email)

    return {"status": "success" if success else "error"}

async def auto_knowledge_diver():
    """
    KNOWLEDGE DIVER v5.0: Hệ thống tự hành săn tìm tri thức & dự án toàn cầu.
    Đã tối ưu hóa cho môi trường Cloud và tích hợp Nexus Core.
    """
    global IS_AUTOPILOT_ON
    
    if not IS_AUTOPILOT_ON or not CHAT_MODEL:
        return

    print(colored("\n🌊 [AUTOPILOT] Radar thợ lặn đang lặn xuống các tầng dữ liệu...", "cyan", attrs=["bold"]))
    
    hunting_grounds = [
        "AI Multi-Agent Systems 2026",
        "Blockchain Security Vulnerabilities",
        "Python Automation for Business",
        "DeepSeek and Large Language Model Trends",
        "AI-driven SaaS Development"
    ]
    
    target_sector = random.choice(hunting_grounds)
    
    try:
        # 1. THU THẬP DỮ LIỆU (Chạy trong Threadpool để không treo Server)
        def _fetch_web_data():
            with DDGS() as ddgs_engine:
                results = list(ddgs_engine.text(f"{target_sector} latest news 2026", max_results=5))
                return "\n".join([f"- {r['title']}: {r['body']}" for r in results])

        print(colored(f"📡 [SCANNING] Radar đang quét: {target_sector}...", "yellow"))
        search_content = await run_in_threadpool(_fetch_web_data)

        # 2. PHÂN TÍCH CHIẾN LƯỢC (Sử dụng bộ não Nexus thay vì AI đơn lẻ)
        analysis_prompt = f"""
        [SYSTEM]: Bạn là Chief Strategy Officer của J.A.R.V.I.S.
        [DATA]: {search_content}
        Nhiệm vụ: Phân tích dữ liệu thực tế về '{target_sector}'. 
        Xác định 1 chủ đề tiềm năng nhất để học tập hoặc làm dự án thầu.
        Trả về DUY NHẤT định dạng JSON: {{"subject": "tên ngắn gọn", "focus": "mục tiêu", "difficulty": 1-5}}
        """
        
        # Gọi Nexus Core để hội chẩn (Sync bọc Async)
        from main import run_nexus_sync
        raw_res = await run_in_threadpool(lambda: run_nexus_sync(analysis_prompt, "autopilot_brain"))
        
        # 3. TRÍCH XUẤT JSON AN TOÀN (Bọc thép bằng Regex)
        
        json_match = re.search(r'\{.*\}', raw_res, re.DOTALL)
        if not json_match:
            raise ValueError("AI không trả về định dạng JSON chuẩn")
            
        intel_data = json.loads(json_match.group())
        final_subject = intel_data.get("subject", target_sector)
        difficulty = intel_data.get("difficulty", 2)
        
        # 4. ĐỊNH GIÁ & PHÂN BỔ NHIỆM VỤ
        smart_reward = round(0.02 + (difficulty * 0.01), 3)
        
        print(colored(f"🚀 [AUTO-FEED] Đang nạp di sản: {final_subject}", "green"))
        
        # Gọi hàm phân bổ nhiệm vụ đã có
        tasks_count = await execute_distribute_knowledge(
            subject=f"🚀 [AUTO] {final_subject}", 
            num_tasks=6, 
            reward=smart_reward
        )
        
        # 5. GHI NHỚ VÀO VECTOR DB (Học ngay lập tức)
        if AI_AVAILABLE:
            await run_in_threadpool(lambda: learn_knowledge(f"Kiến thức mới từ Autopilot về {final_subject}: {search_content}"))

        logger.info(f"✅ Autopilot: Đã nạp {tasks_count} kiến thức mới vào bộ não trung tâm.")

    except Exception as e:
        logger.error(f"❌ Autopilot Critical Error: {str(e)}")
        # Cơ chế tự phục hồi: Ghi log lỗi để CEO kiểm tra sau
# --- API DATA & FEATURES ---
# ==========================================
# API STORE & TÀI CHÍNH (Đã tối ưu cho store.html)
# ==========================================

@app.get("/api/products")
async def get_products_api():
    """
    API lấy danh sách sản phẩm.
    TÍNH NĂNG MỚI: Tự động tạo dữ liệu mẫu nếu Database rỗng (Auto-Seeding).
    """
    try:
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        c = conn.cursor()
        
        # 1. Kiểm tra xem có sản phẩm nào chưa
        # (Cần try-except vì bảng products có thể chưa tạo nếu init_db lỗi)
        try:
            c.execute("SELECT * FROM products")
            rows = c.fetchall()
        except:
            # Nếu bảng chưa có, tạo bảng luôn
            c.execute("CREATE TABLE IF NOT EXISTS products (id INTEGER PRIMARY KEY, name TEXT, price REAL)")
            rows = []
        
        # 2. Nếu chưa có (Lần đầu chạy), tự động thêm 3 gói khớp với HTML
        if not rows:
            print(colored("🛒 [STORE] Khởi tạo dữ liệu sản phẩm mẫu...", "cyan"))
            sample_products = [
                (1, "AI Task Manager", 49.0),
                (2, "SaaS Landing AI", 99.0),
                (3, "AI Content Pack", 19.0)
            ]
            c.executemany("INSERT INTO products (id, name, price) VALUES (?, ?, ?)", sample_products)
            conn.commit()
            
            # Lấy lại danh sách sau khi thêm
            c.execute("SELECT * FROM products")
            rows = c.fetchall()
            
        conn.close()
        return [dict(r) for r in rows]
        
    except Exception as e:
        print(f"Lỗi API Products: {e}")
        return []

@app.post("/api/buy")
async def buy_product(req: BuyRequest):
    """
    API Mua hàng: Tạo License Key & Ghi nhận doanh thu.
    """
    conn = sqlite3.connect(DB_PATH)
    try:
        # 1. Tìm sản phẩm
        cursor = conn.execute("SELECT price, name FROM products WHERE id=?", (req.product_id,))
        product = cursor.fetchone()
        
        if not product:
            raise HTTPException(status_code=404, detail="Sản phẩm không tồn tại")
            
        price, name = product[0], product[1]
        
        # 2. Tạo mã bản quyền (License Key) giả lập
        license_key = f"AI-{uuid.uuid4().hex[:4].upper()}-{uuid.uuid4().hex[:4].upper()}-{int(time.time())}"
        
        # 3. GHI NHẬN DOANH THU (Quan trọng để tính Stats)
        # Lưu vào bảng work_logs nhưng với tool_used là 'SALE' để phân biệt
        timestamp = datetime.now().strftime("%H:%M %d/%m/%Y")
        conn.execute("""
            INSERT INTO work_logs (timestamp, agent_name, task_content, tool_used, cost, result_summary)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (
            timestamp, 
            "STORE_BOT", 
            f"Bán gói: {name}", 
            "SALE",      # Đánh dấu đây là Doanh thu
            price,       # Số tiền thu được (Dương)
            f"License: {license_key}"
        ))
        
        conn.commit()
            
        return {
            "status": "success",
            "msg": f"Đã mua thành công: {name}",
            "license_key": license_key,
            "price": price
        }
    except Exception as e:
        print(f"Lỗi mua hàng: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        conn.close()

# === API ADMIN & FINANCE (QUAN TRỌNG: ĐÃ THÊM API THIẾU) ===
@app.get("/api/costs")
async def get_costs_api():
    """API dữ liệu cho bảng Admin"""
    try:
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        # Lấy 50 log mới nhất
        rows = conn.execute("SELECT timestamp, agent_name, task_content, tool_used, cost, result_summary FROM work_logs ORDER BY id DESC LIMIT 50").fetchall()
        
        logs = []
        for r in rows:
            row_dict = dict(r)
            row_dict['agent'] = row_dict['agent_name'] # Map key cho frontend
            row_dict['task'] = row_dict['task_content']
            row_dict['tool'] = row_dict['tool_used']
            row_dict['cost_usd'] = row_dict['cost'] or 0.0
            row_dict['result'] = row_dict['result_summary']
            logs.append(row_dict)
            
        conn.close()
        return logs
    except: return []

@app.get("/api/stats")
async def get_system_stats():
    """
    API 3: Tổng hợp tài chính (Cho Dashboard/Main cũ)
    """
    try:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        
        # 1. Đếm số sản phẩm
        try:
            c.execute("SELECT count(*) FROM products")
            prod_count = c.fetchone()[0]
        except:
            prod_count = 0
        
        # 2. Tính TỔNG CHI PHÍ (Chi cho AI chạy)
        # Lọc những dòng KHÔNG PHẢI là SALE
        c.execute("SELECT SUM(cost) FROM work_logs WHERE tool_used != 'SALE'")
        row_exp = c.fetchone()
        total_expense = row_exp[0] or 0.0
        
        # 3. Tính TỔNG DOANH THU (Tiền bán hàng)
        # Lọc những dòng CÓ tool_used là SALE
        c.execute("SELECT SUM(cost) FROM work_logs WHERE tool_used = 'SALE'")
        row_rev = c.fetchone()
        total_revenue = row_rev[0] or 0.0
        
        conn.close()
        
        return {
            "products": prod_count,
            "revenue": round(total_revenue, 2),
            "expense": round(total_expense, 4),
            "balance": round(total_revenue - total_expense, 4) # Lời/Lỗ
        }
        
    except Exception as e:
        print(f"Lỗi Stats: {e}")
        return {"products": 0, "revenue": 0, "expense": 0, "balance": 0}

@app.get("/api/stats/heatmap")
async def get_heatmap_api():
    """
    [HEATMAP API] Trả về dữ liệu mật độ dự án cho Dashboard.
    Sử dụng trực tiếp db_manager đã có trong server.py.
    """
    try:
        query = text("""
            SELECT strftime('%H', timestamp) as hour, 
                   COUNT(*) as count, 
                   SUM(reward) as total_value
            FROM learning_tasks 
            WHERE topic LIKE '%[PROJECT]%' 
            AND timestamp > datetime('now', '-24 hours')
            GROUP BY hour
            ORDER BY hour ASC
        """)
        
        with db_manager.get_connection() as conn:
            stats = conn.execute(query).fetchall()
            
        if not stats:
            return {"status": "empty", "data": []}

        # Format dữ liệu gửi về Dashboard
        result = []
        for s in stats:
            result.append({
                "hour": s[0],
                "count": s[1],
                "value": round(s[2], 2)
            })
            
        return {"status": "success", "data": result}
    except Exception as e:
        logger.error(f"Heatmap API Error: {e}")
        return {"status": "error", "message": str(e)}

@app.post("/api/admin/send_me_report")
async def send_report_to_ceo(subject: str, message: str, x_api_key: str = Header(None)):
        """API cho phép J.A.R.V.I.S gửi mail trực tiếp cho CEO"""
        if x_api_key != ADMIN_SECRET:
            raise HTTPException(status_code=403, detail="Sai mã lệnh quân sự!")

        # Nội dung HTML chuyên nghiệp cho CEO
        html_content = f"""
        <div style="font-family: Arial, sans-serif; border: 1px solid #00ff99; padding: 20px; border-radius: 10px;">
            <h2 style="color: #00ff99;">🚀 J.A.R.V.I.S SYSTEM REPORT</h2>
            <p style="color: #333;">Thưa CEO, hệ thống tại Phan Thiết vừa hoàn thành nhiệm vụ:</p>
            <div style="background-color: #f4f4f4; padding: 15px; border-left: 5px solid #00ff99;">
                {message}
            </div>
            <p style="font-size: 12px; color: #888; margin-top: 20px;">
                Thời gian báo cáo: {datetime.now().strftime('%H:%M:%S %d/%m/%Y')}
            </p>
        </div>
        """
        
        # Gửi đến email cá nhân của ngài (thay bằng mail ngài muốn nhận)
        ceo_email = "thanhnghia86.vn@gmail.com" # <--- Điền mail của ngài vào đây
        
        success = jarvis_identity.send_system_mail(ceo_email, subject, html_content)
        
        if success:
            return {"status": "success", "msg": "Đã gửi báo cáo vào hòm thư của ngài."}
        else:
            return {"status": "error", "msg": "Gửi mail thất bại. Kiểm tra lại MAIL_PASSWORD trong .env"}

async def fetch_external_projects(keyword):
    search_query = f"site:freelancer.com '{keyword}' job"
    # SỬA LỖI: Khởi tạo DDGS bên trong hàm hoặc dùng global
    def _search():
        with DDGS() as ddgs_engine:
            return list(ddgs_engine.text(search_query, max_results=5))
            
    results = await run_in_threadpool(_search)
    return results

async def activate_hunter_logic(keyword: str):
    # Giả lập danh sách User-Agent để qua mặt robot detection
    user_agents = [
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/120.0.0.0 Safari/537.36",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) Safari/537.36"
    ]
    
    with DDGS() as ddgs_engine:
        # Thêm từ khóa "remote" hoặc "hiring" để lọc dự án thật
        search_query = f"site:upwork.com {keyword} remote jobs 2026"
        results = list(ddgs_engine.text(search_query, max_results=3))
        
        # Thêm độ trễ ngẫu nhiên giữa các lần bóc tách
        await asyncio.sleep(random.uniform(2, 5)) 
        
        for project in results:
            await execute_distribute_knowledge(
                subject=f"DỰ ÁN THỰC TẾ: {project['title']}",
                num_tasks=5,
                reward=0.1 
            )
    return len(results)

@app.get("/api/wealth")
async def check_wealth_api():
    """API Kiểm toán Tài sản Trí tuệ (Chạy trực tiếp trên Cloud)"""
    try:
        # 1. Tìm DB (Ưu tiên Cloud)
        if os.path.exists("/var/data"):
            db_path = "/var/data/ai_corp_projects.db"
            env = "CLOUD (Render)"
        else:
            db_path = "ai_corp_projects.db"
            env = "LOCAL"

        if not os.path.exists(db_path):
            return {"error": "Chưa có Database", "path": db_path}

        # 2. Kết nối
        conn = sqlite3.connect(db_path, check_same_thread=False)
        c = conn.cursor()

        # 3. Tính toán
        # Tổng tiền
        c.execute("SELECT COUNT(*), SUM(cost) FROM work_logs")
        row = c.fetchone()
        total_tasks = row[0] or 0
        total_cost = row[1] or 0.0

        # Phân loại tài sản
        c.execute("SELECT COUNT(*) FROM work_logs WHERE tool_used LIKE '%SUPREME%' OR tool_used LIKE '%DEBATE%'")
        legendary_count = c.fetchone()[0] or 0 # Di sản
        
        c.execute("SELECT COUNT(*) FROM work_logs WHERE tool_used LIKE '%Synthesis%'")
        synthesis_count = c.fetchone()[0] or 0 # Tổng hợp
        
        basic_count = total_tasks - legendary_count - synthesis_count

        # Top nhân viên
        c.execute("SELECT agent_name, COUNT(*) as cnt, SUM(cost) as cst FROM work_logs GROUP BY agent_name ORDER BY cst DESC LIMIT 5")
        top_agents = [{"name": r[0], "tasks": r[1], "cost": r[2]} for r in c.fetchall()]

        conn.close()

        # 4. Trả về JSON đẹp
        return {
            "environment": env,
            "financial_report": {
                "total_cost_usd": total_cost,
                "total_tasks": total_tasks
            },
            "intellectual_assets": {
                "LEGENDARY_MASTER_PLANS": legendary_count,
                "HIGH_QUALITY_SYNTHESIS": synthesis_count,
                "BASIC_RESEARCH": basic_count
            },
            "top_spenders": top_agents,
            "sovereignty_progress": {
                "current_quality_docs": legendary_count + synthesis_count,
                "target_for_finetune": 500,
                "percentage_ready": f"{((legendary_count + synthesis_count)/500)*100:.2f}%"
            }
        }

    except Exception as e:
        return {"error": str(e)}

# ==========================================
# 6. HỆ THỐNG XỬ LÝ CHAT ĐA LUỒNG (ASYNC CORE)
# ==========================================

async def background_ai_worker(task_id: str, user_msg_text: str, thread_id: str):
    """
    BG WORKER v5.0: Chạy độc lập, không sợ Timeout, sử dụng 100% công suất 17 Agent.
    """
    print(colored(f"⚙️ [BG WORKER] Đang tiếp nhận Task {task_id}...", "yellow"))
    
    try:
        # A. Cập nhật trạng thái khởi động
        with db_manager.get_connection() as conn:
            conn.execute(text("""
                INSERT OR REPLACE INTO async_tasks (task_id, status, result, timestamp) 
                VALUES (:id, 'PROCESSING', 'J.A.R.V.I.S đang hội chẩn...', :time)
            """), {"id": task_id, "time": datetime.now()})
            conn.commit()

        # B. THỰC THI QUA NEXUS CORE (Điểm thay đổi quan trọng nhất)
        # Chúng ta dùng run_in_threadpool để gọi run_nexus_sync (vốn là hàm sync)
        # Việc này giúp 17 Agent chạy ngầm mà không làm treo Server FastAPI
        from main import run_nexus_sync
        
        print(colored(f"🧠 [NEXUS CORE] Task {task_id} đang được 17 Agent xử lý...", "blue"))
        
        # Bước này có thể mất vài phút nếu là dự án lớn
        ai_reply = await run_in_threadpool(lambda: run_nexus_sync(user_msg_text, thread_id))

        # C. CẬP NHẬT KẾT QUẢ VÀ TÀI CHÍNH
        with db_manager.get_connection() as conn:
            # 1. Lưu kết quả cuối cùng vào bảng Task
            conn.execute(text("UPDATE async_tasks SET status='DONE', result=:res WHERE task_id=:id"), 
                         {"res": ai_reply, "id": task_id})
            
            # 2. Ghi log vào work_logs để Finance nhảy số (Viên gạch vàng)
            # Trích xuất Agent từ reply nếu có (ví dụ [CODER])
            agent_tag = "NEXUS_WORKER"
            if "[" in ai_reply[:20]:
                agent_tag = ai_reply.split("]")[0].replace("[", "")

            conn.execute(text("""
                INSERT INTO work_logs (timestamp, agent_name, task_content, tool_used, cost, result_summary)
                VALUES (:ts, :agent, :task, :tool, :cost, :res)
            """), {
                "ts": datetime.now().strftime("%H:%M %d/%m/%Y"),
                "agent": agent_tag,
                "task": f"[ASYNC] {user_msg_text[:50]}",
                "tool": "ASYNC_NEXUS_V5",
                "cost": 0.005, # Task chạy ngầm thường nặng hơn nên set cost cao hơn
                "res": ai_reply[:1000]
            })
            conn.commit()
            
        print(colored(f"✅ [BG WORKER] Task {task_id} hoàn thành và đã ghi nhận di sản!", "green"))

    except Exception as e:
        error_msg = f"Hệ thống gặp sự cố khi chạy ngầm: {str(e)}"
        logger.error(f"❌ [BG WORKER ERROR]: {error_msg}")
        with db_manager.get_connection() as conn:
            conn.execute(text("UPDATE async_tasks SET status='ERROR', result=:err WHERE task_id=:id"), 
                         {"err": error_msg, "id": task_id})
            conn.commit()
# --- 2. API GIAO VIỆC (DISPATCHER) ---
@app.post("/api/chat_async")
async def chat_async_endpoint(request: ChatRequest, background_tasks: BackgroundTasks):
    """
    Nhận lệnh -> Phát phiếu hẹn (Task ID) -> Trả lời ngay lập tức.
    """
    # Xử lý nhanh các câu chào hỏi (Fast Track) - Không cần tạo Task
    greetings = ["chào", "hi", "hello", "alo", "ping"]
    if str(request.message).strip().lower() in greetings:
        # Trả về dạng đặc biệt để Dashboard biết là xong luôn
        return {"task_id": "fast_track", "status": "DONE", "reply": "Chào CEO! J.A.R.V.I.S đang trực tuyến."}

    # Tạo mã phiếu hẹn duy nhất
    task_id = f"task_{uuid.uuid4().hex[:8]}"
    
    # Đẩy việc cho Worker chạy ngầm
    background_tasks.add_task(
        background_ai_worker, 
        task_id, 
        str(request.message), 
        str(request.thread_id)
    )
    
    # Trả mã phiếu cho Dashboard cầm
    return {"task_id": task_id, "status": "QUEUED", "message": "Đã tiếp nhận. Đang xử lý ngầm..."}

# ==========================================
# 1. TRẠM ĐIỀU PHỐI NEXUS (DÀNH CHO DASHBOARD CHAT)
# ==========================================
@app.post("/api/chat_nexus")
async def chat_nexus_endpoint(request: ChatRequest):
    """
    Điểm chạm tối cao kết nối Dashboard với 17 Agent trong main.py.
    Ghi nhật ký trực tiếp vào hệ thống kiểm toán tài sản.
    """
    try:
        # 1. Thực thi lệnh qua bộ não trung tâm
        # Dùng run_in_threadpool để không chặn Event Loop của FastAPI
        reply = await run_in_threadpool(lambda: run_nexus_sync(request.message, request.thread_id))
        
        # 2. KIỂM TOÁN TÀI SẢN (Wealth Update)
        # Tự động trích xuất Agent nào vừa làm việc (ví dụ: [CODER])
        agent_match = re.search(r'\[(.*?)\]', reply)
        active_agent = agent_match.group(1) if agent_match else "SUPERVISOR"
        
        # Ghi log vào DB để Tab FINANCE hiển thị
        with db_manager.get_connection() as conn:
            conn.execute(text("""
                INSERT INTO work_logs (timestamp, agent_name, task_content, tool_used, cost, result_summary)
                VALUES (:ts, :agent, :task, :tool, :cost, :res)
            """), {
                "ts": datetime.now().strftime("%H:%M %d/%m/%Y"),
                "agent": active_agent,
                "task": request.message[:100],
                "tool": "NEXUS_v5",
                "cost": 0.0025, # Chi phí ước tính cho mỗi truy vấn AI
                "res": reply[:500] # Lưu bản tóm tắt để tiết kiệm dung lượng
            })
            conn.commit()

        return {"reply": reply, "agent": active_agent, "status": "SUCCESS"}
    except Exception as e:
        logger.error(f"Nexus Error: {e}")
        return {"reply": f"❌ Lỗi hệ thống: {str(e)}", "status": "ERROR"}
# ==========================================
# 2. PHÒNG PHẪU THUẬT CODE (REFACTOR API)
# ==========================================
@app.post("/api/refactor")
async def code_refactor_endpoint(file: UploadFile = File(...), wish: str = Form(...)):
    """
    Endpoint chuyên biệt để xử lý nâng cấp Code dung lượng lớn (như Dashboard.py)
    """
    content = await file.read()
    code_text = content.decode("utf-8", errors='ignore')
    
    prompt = f"""@J.A.R.V.I.S [ARCHITECT]: Thực hiện phẫu thuật file {file.filename}.
    Mong muốn của CEO: {wish}
    Code gốc: {code_text}"""
    
    # Chạy quy trình nâng cấp qua main.py
    new_code_reply = await run_in_threadpool(lambda: run_nexus_sync(prompt, "refactor_session"))
    
    return {"reply": new_code_reply}

# ==========================================
# 3. WEBSOCKET: ĐỒNG BỘ ICON AGENT (HUD SYNC)
# ==========================================
# (Cập nhật trong hàm websocket_nexus của ngài)
# Thêm đoạn này trước khi gửi JSON về Dashboard
async def broadcast_agent_status(agent_name: str):
    await manager.broadcast(json.dumps({
        "type": "AGENT_STATUS",
        "agent": agent_name,
        "status": "ACTIVE"
    }))

# --- 3. API KIỂM TRA (TRACKER) ---
@app.get("/api/task_status/{task_id}")
async def get_task_status(task_id: str):
    """
    Dashboard dùng API này để hỏi: "Xong chưa?"
    """
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.execute("SELECT status, result FROM async_tasks WHERE task_id=?", (task_id,))
        row = cursor.fetchone()
        conn.close()
        
        if row:
            return {"status": row[0], "result": row[1]}
        else:
            return {"status": "NOT_FOUND", "result": None}
    except Exception as e:
        return {"status": "ERROR", "result": str(e)}
# --- Thợi đào 
@app.post("/api/economy/register_miner")
async def register_miner_endpoint(info: RegisterInfo):
    # RegisterInfo cần thêm trường: worker_id
    conn = sqlite3.connect(DB_PATH)
    try:
        # 1. Kiểm tra xem máy này (Worker_ID) đã đăng ký chưa
        # Một người (email) có thể có nhiều Worker_ID khác nhau
        cursor = conn.execute("SELECT api_key FROM users WHERE worker_id=?", (info.worker_id,))
        existing_node = cursor.fetchone()
        
        if existing_node:
            # Nếu máy này đã có trong hệ thống, trả về Key cũ thay vì tạo mới
            return {
                "status": "success",
                "api_key": existing_node[0],
                "msg": "Máy này đã được đăng ký trước đó. Khôi phục Key thành công."
            }

        # 2. Sinh Key mới cho máy mới
        new_key = f"sk-{uuid.uuid4().hex[:12].upper()}"
        
        # 3. Gộp thông tin ngân hàng
        bank_full = f"{info.bank_name} - {info.account_number}"

        # 4. Lưu vào DB (Bổ sung thêm cột worker_id)
        # Ngài cần chạy lệnh SQL: ALTER TABLE users ADD COLUMN worker_id TEXT;
        conn.execute(
            """INSERT INTO users (username, worker_id, api_key, email, bank_info, created_at) 
               VALUES (?, ?, ?, ?, ?, ?)""",
            (info.username, info.worker_id, new_key, info.email.lower(), bank_full, datetime.now())
        )
        conn.commit()
        
        print(colored(f"🆕 Node Mới: {info.worker_id} | Chủ: {info.email}", "cyan"))
        
        return {
            "status": "success",
            "api_key": new_key,
            "msg": "Đăng ký Node thành công!"
        }
    except Exception as e:
        return JSONResponse(status_code=500, content={"status": "error", "msg": str(e)})
    finally:
        conn.close()

async def verify_api_key(x_api_key: Optional[str] = Header(None)):
    """
    [SECURITY LAYER v2.3]: Xác thực linh hoạt cho cả Admin và Thợ đào.
    """
    if not x_api_key:
        raise HTTPException(status_code=403, detail="⛔ Thiếu mã lệnh quân sự.")

    # Cho phép Admin toàn quyền
    if x_api_key == ADMIN_SECRET:
        return "ADMIN"

    # Kiểm tra trong Database xem có phải Key của thợ đào không
    conn = sqlite3.connect(DB_PATH)
    try:
        cursor = conn.execute("SELECT username FROM users WHERE api_key=?", (x_api_key,))
        user = cursor.fetchone()
        if user:
            return user[0] # Trả về tên thợ đào
        
        logger.error(f"⚠️ Truy cập trái phép với Key: {x_api_key}")
        raise HTTPException(status_code=403, detail="⛔ Key không hợp lệ hoặc đã bị thu hồi.")
    finally:
        conn.close()



@app.post("/api/chat")
async def chat_endpoint(
    request: ChatRequest, 
    background_tasks: BackgroundTasks,
    x_api_key: str = Depends(verify_api_key) # <--- KHÓA BẢO VỆ TẠI ĐÂY
):
    set_system_busy()
    if not AI_AVAILABLE:
        return {"reply": "⚠️ Hệ thống AI đang khởi động. Vui lòng đợi 30s."}

    try:
        user_msg_text = str(request.message).strip()
        thread_id = str(request.thread_id) if request.thread_id else "default_session"
        
        # --- 1. XỬ LÝ NHANH ---
        greetings = ["chào", "hi", "hello", "alo"]
        if user_msg_text.lower() in greetings:
             return {"reply": "Chào CEO! J.A.R.V.I.S đã sẵn sàng nhận lệnh."}

        # --- 2. CHUẨN BỊ KÝ ỨC ---
        memory_context = ""
        if MEMORY_AVAILABLE:
            try:
                memory_context = await run_in_threadpool(lambda: recall_relevant_memories(user_msg_text))
                print(colored(f"🧠 Ký ức kích hoạt: {len(memory_context)} chars", "magenta"))
            except: pass

        # --- 3. ĐÓNG GÓI TIN NHẮN & GHI NHẬT KÝ TÀI SẢN ---
        # Khi đưa lên Online, ta cần ghi log ngay cả khi đang chat để theo dõi chi phí
        timestamp = datetime.now().strftime('%H:%M %d/%m/%Y')
        final_input_content = f"""
        [CONTEXT INFO]: Location: Phan Thiet | Time: {timestamp}
        [MEMORY]: {memory_context}
        [USER COMMAND]: {user_msg_text}
        """
        
        human_msg = HumanMessage(content=final_input_content)
        config = {"configurable": {"thread_id": thread_id}}

        # --- 4. GỌI BỘ NÃO TRUNG TÂM ---
        output = await ai_app.ainvoke({"messages": [human_msg]}, config=config)
        ai_reply = output["messages"][-1].content
        
        # --- 5. TỰ ĐỘNG CẬP NHẬT DI SẢN (QUAN TRỌNG) ---
        # Ghi nhận mỗi câu chat là một lần đóng góp tri thức vào DB
        with db_manager.get_connection() as conn:
            conn.execute(text("""
                INSERT INTO work_logs (timestamp, agent_name, task_content, tool_used, cost, result_summary)
                VALUES (:ts, :agent, :task, :tool, :cost, :res)
            """), {
                "ts": timestamp,
                "agent": "ORCHESTRATOR",
                "task": user_msg_text[:100],
                "tool": "CHAT_v2_SECURE",
                "cost": 0.0015,
                "res": ai_reply[:500]
            })
            conn.commit()

        # --- 6. HẬU XỬ LÝ (KÝ ỨC) ---
        if MEMORY_AVAILABLE:
            background_tasks.add_task(extract_and_save_memory, user_msg_text, ai_reply)
            
        return {
            "status": "success", 
            "reply": ai_reply,
            "agent": "J.A.R.V.I.S v2.0_SECURE"
        }

    except Exception as e:
        # Log lỗi chi tiết nhưng không trả về lỗi nhạy cảm cho người dùng lạ
        logger.error(f"❌ CHAT ERROR: {str(e)}")
        return JSONResponse(status_code=500, content={"reply": "💥 Hệ thống đang bận xử lý dữ liệu di sản. Vui lòng thử lại sau."})

@app.post("/api/plan_project")
async def plan_project_endpoint(
    request: ChatRequest, 
    api_key: str = Depends(verify_api_key) # <--- CHỐT CHẶN BẢO MẬT
):
    """
    Bước 1: CEO yêu cầu lập kế hoạch (Yêu cầu API Key).
    """
    # Kiểm tra trạng thái AI
    if not AI_AVAILABLE: 
        return JSONResponse(status_code=503, content={"status": "ERROR", "message": "AI Module Offline"})
    
    # Tạo ID dự án nếu chưa có
    pid = request.thread_id or f"proj_{int(time.time())}"
    
    try:
        # Gọi hàm architect MỚI (run_architect_phase)
        plan_content, plan_path = await run_architect_phase(request.message, pid)
        
        return {
            "status": "PLAN_CREATED",
            "project_id": pid,
            "message": "Đã lập xong bản thiết kế. Vui lòng xem xét.",
            "blueprint_content": plan_content, # Trả về nội dung để hiện lên Dashboard
            "blueprint_path": plan_path,
            "next_action": "Nếu đồng ý, hãy gọi /api/heavy_project với nội dung 'EXECUTE_BLUEPRINT'"
        }
    except Exception as e:
        # Bắt lỗi nếu hàm architect trả về không đúng định dạng hoặc lỗi bất ngờ
        return JSONResponse(status_code=500, content={"status": "ERROR", "message": f"Lỗi hệ thống: {str(e)}"})

@app.post("/api/upload_pdf")
async def upload_pdf(file: UploadFile = File(...)):
    """API Upload PDF & Tự động học (Non-blocking)"""
    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Chỉ chấp nhận file .PDF")

    safe_filename = f"{uuid.uuid4().hex[:8]}_{file.filename}"
    file_path = os.path.join(UPLOAD_DIR, safe_filename)
    
    try:
        async with aiofiles.open(file_path, 'wb') as out_file:
            content = await file.read()
            await out_file.write(content)
            
        if AI_AVAILABLE:
            # QUAN TRỌNG: Chạy trong threadpool để không treo server
            result = await run_in_threadpool(lambda: ingest_docs_to_memory(file_path))
            return {"status": "success", "message": result, "path": file_path}
            
        return {"status": "saved", "message": "Saved (AI Offline)"}
        
    except Exception as e:
        if os.path.exists(file_path): os.remove(file_path)
        raise HTTPException(status_code=500, detail=str(e))

# Endpoint để Dashboard hoặc Worker yêu cầu lấy mã xác nhận
@app.get("/api/admin/check_otp")
async def check_otp(keyword: str = "verification"):
    # J.A.R.V.I.S tự gọi hàm lấy mã
    code = jarvis_identity.fetch_latest_otp(keyword)
    
    if code:
        return {"status": "success", "otp": code}
    
    return {"status": "error", "msg": "Đang chờ mã xác nhận gửi về..."}

def send_jarvis_mail(to_email, subject, body):
    # J.A.R.V.I.S sẽ tự lấy thông tin từ Cloud/Env mà ngài đã cấu hình
    sender_email = os.getenv("MAIL_USERNAME")
    app_password = os.getenv("MAIL_PASSWORD")

    if not sender_email or not app_password:
        print("❌ Lỗi: Chưa tìm thấy biến MAIL_USERNAME hoặc MAIL_PASSWORD trên Cloud.")
        return False

    try:
        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()
        server.login(sender_email, app_password)

        msg = MIMEMultipart()
        msg['From'] = f"J.A.R.V.I.S System <{sender_email}>"
        msg['To'] = to_email
        msg['Subject'] = subject
        msg.attach(MIMEText(body, 'html'))

        server.send_message(msg)
        server.quit()
        return True
    except Exception as e:
        print(f"❌ Lỗi thực thi gửi mail: {e}")
        return False

@app.post("/api/tts")
async def text_to_speech_api(request: TTSRequest):
    """
    API TTS V2: Lọc sạch ký tự đặc biệt giúp giọng đọc mượt hơn (Lướt).
    """
    try:
        # 1. Lấy văn bản gốc
        raw_text = request.text[:500]
        
        # --- BỘ LỌC LÀM MỊN (TEXT CLEANER) ---
        def clean_text_for_speech(text):
            # 1. Loại bỏ Markdown (*, #, `) thường gặp trong AI response
            text = text.replace("*", "").replace("#", "").replace("`", "").replace("_", " ")
            
            # 2. Loại bỏ các đường link http://... (Đọc link rất chán)
            text = re.sub(r'http\S+', 'liên kết', text)
            
            # 3. Loại bỏ các dấu ngoặc vuông như [IMAGE], [1]...
            text = re.sub(r'\[.*?\]', '', text)
            
            # 4. Loại bỏ các ký tự đặc biệt vô nghĩa khác, giữ lại dấu câu cơ bản (. , ? !)
            # Chỉ giữ lại chữ cái, số và dấu câu tiếng Việt
            # (Regex này giữ lại chữ unicode và dấu câu cơ bản)
            # text = re.sub(r'[^\w\s.,?!]', '', text) # Có thể dùng nếu muốn lọc cực mạnh
            
            # 5. Xóa khoảng trắng thừa (nhiều dấu cách liền nhau)
            text = " ".join(text.split())
            return text.strip()

        # Áp dụng bộ lọc
        speak_text = clean_text_for_speech(raw_text)
        
        logger.info(f"🤖 Google TTS (Cleaned): {speak_text[:50]}...")

        # 2. Tạo âm thanh (Chạy trong luồng riêng)
        def _generate_google_audio():
            # tld='com.vn' giúp giọng Google chuẩn Việt Nam hơn
            tts = gTTS(text=speak_text, lang='vi', tld='com.vn')
            
            buffer = io.BytesIO()
            tts.write_to_fp(buffer)
            buffer.seek(0)
            return buffer.read()

        # 3. Thực thi
        audio_content = await run_in_threadpool(_generate_google_audio)
        
        return Response(content=audio_content, media_type="audio/mpeg")

    except Exception as e:
        logger.error(f"🚨 [GOOGLE TTS ERROR]: {str(e)}")
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.post("/api/voice_chat")
async def voice_chat(file: UploadFile = File(...), api_key: str = Depends(verify_api_key)):
    """
    TƯƠNG TÁC BẰNG GIỌNG NÓI (Voice-to-Voice) - V2 (Cleaned Audio)
    """
    if not AI_AVAILABLE or 'client' not in globals():
        return JSONResponse(status_code=503, content={"error": "AI/Voice Module chưa sẵn sàng"})

    # 1. LƯU FILE TẠM
    temp_filename = f"temp_{uuid.uuid4()}.webm"
    temp_path = os.path.join(UPLOAD_DIR, temp_filename)
    
    try:
        async with aiofiles.open(temp_path, 'wb') as out_file:
            content = await file.read()
            await out_file.write(content)

        # 2. DỊCH GIỌNG NÓI SANG CHỮ (WHISPER)
        def _transcribe():
            with open(temp_path, "rb") as audio_file:
                return client.audio.transcriptions.create(
                    model="whisper-1", 
                    file=audio_file,
                    language="vi"
                )
        
        transcript = await run_in_threadpool(_transcribe)
        user_text = transcript.text
        print(f"🎤 [VOICE INPUT]: {user_text}")
        
        # 3. XỬ LÝ AI (SMART CHAT)
        # --- Logic Chat tối giản cho Voice ---
        memory_context = ""
        if MEMORY_AVAILABLE:
            memory_context = await run_in_threadpool(lambda: recall_relevant_memories(user_text))
            
        fast_keywords = ["giá vàng", "thời tiết", "mấy giờ", "ngày mấy", "tỷ giá"]
        ai_text = ""
        agent_name = "J.A.R.V.I.S"

        # A. Fast Track (Gemini)
        if any(k in user_text.lower() for k in fast_keywords) and LLM_GEMINI_LOGIC:
             try:
                 ai_res = await LLM_GEMINI_LOGIC.ainvoke(f"Ký ức: {memory_context}. Hỏi: {user_text}")
                 ai_text = ai_res.content
                 agent_name = "Gemini Voice"
             except: pass
        
        # B. Deep Thinking (LangGraph) - Nếu Fast Track thất bại hoặc không khớp
        if not ai_text:
             full_prompt = f"Ký ức: {memory_context}\nUser: {user_text}"
             from langchain_core.messages import HumanMessage
             final_state = await ai_app.ainvoke({"messages": [HumanMessage(content=full_prompt)]}, config={"configurable": {"thread_id": "voice_thread"}})
             last_message = final_state['messages'][-1]
             ai_text = last_message.content
             agent_name = final_state.get("current_agent", "J.A.R.V.I.S")
        
        # 4. TẠO GIỌNG NÓI (TTS)
        # --- BƯỚC QUAN TRỌNG: LÀM SẠCH VĂN BẢN TRƯỚC KHI ĐỌC ---
        def clean_text_for_speech(text):
            text = text.replace("*", "").replace("#", "").replace("`", "").replace("_", " ")
            text = re.sub(r'http\S+', '', text) # Bỏ link
            text = re.sub(r'\[.*?\]', '', text) # Bỏ thẻ [System]
            return " ".join(text.split()).strip()

        clean_ai_text = clean_text_for_speech(ai_text)
        speak_text = clean_ai_text[:500] # Cắt ngắn để tiết kiệm

        def _speak():
            return client.audio.speech.create(
                model="tts-1",
                voice="onyx",
                input=speak_text
            )
        audio_res = await run_in_threadpool(_speak)

        # 5. TRẢ VỀ KẾT QUẢ KÉP
        audio_b64 = base64.b64encode(audio_res.content).decode('utf-8')

        return {
            "text_reply": ai_text, # Trả về text gốc (có markdown) để hiển thị đẹp
            "audio_base64": audio_b64, # Audio đã được làm sạch để đọc mượt
            "transcript": user_text,
            "agent": agent_name
        }

    except Exception as e:
        logger.error(f"Voice Error: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)



@app.post("/api/ask")
async def ask_jarvis(query: Query):
    """API: SÁNG TẠO TỪ CỐT LÕI (RAG GENERATION)"""
    question = query.question
    print(f"❓ CEO yêu cầu sáng tạo: {question}")

    try:
        # 1. KẾT NỐI & TÌM "CẢM HỨNG" (CONTEXT)
        if os.path.exists("/var/data"): db_path = "/var/data/ai_corp_projects.db"
        else: db_path = "ai_corp_projects.db"
        
        conn = sqlite3.connect(db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        c = conn.cursor()

        # Tìm 3 bài "Di Sản" (Legendary) liên quan nhất để học hỏi phong cách
        # Mẹo: Dùng từ khóa từ câu hỏi để tìm bài mẫu
        keywords = question.replace("tạo", "").replace("vẽ", "").replace("viết", "").strip().split()
        search_term = keywords[0] if keywords else ""
        
        sql = f"""
            SELECT agent_name, task_content, result_summary 
            FROM work_logs 
            WHERE (tool_used LIKE '%SUPREME%' OR tool_used LIKE '%DEBATE%' OR tool_used LIKE '%Research%')
            AND (task_content LIKE '%' || ? || '%' OR result_summary LIKE '%' || ? || '%')
            ORDER BY id DESC LIMIT 2
        """
        c.execute(sql, (search_term, search_term))
        rows = c.fetchall()
        conn.close()

        # 2. XÂY DỰNG "BỘ GEN" TRI THỨC (CONTEXT)
        knowledge_dna = ""
        if rows:
            knowledge_dna = "DƯỚI ĐÂY LÀ CÁC TIÊU CHUẨN/NGUYÊN TẮC CỐT LÕI CỦA CÔNG TY (ĐÃ ĐƯỢC HỘI ĐỒNG THÔNG QUA):\n"
            for r in rows:
                knowledge_dna += f"- KINH NGHIỆM TỪ {r['agent_name']} (Bài: {r['task_content']}):\n{str(r['result_summary'])[:1500]}\n...\n"
        else:
            knowledge_dna = "Chưa có bài mẫu trong kho. Hãy dùng kiến thức chuẩn của chuyên gia."

        # 3. KÍCH HOẠT SÁNG TẠO (GENERATION)
        if CHAT_MODEL:
            # Prompt này ép AI phải "Học thầy" nhưng "Làm mới"
            prompt = f"""
            Bạn là J.A.R.V.I.S - Kiến trúc sư trưởng.
            
            YÊU CẦU CỦA CEO: "{question}"
            
            --------------------------------------------------
            KHO TÀNG KINH NGHIỆM (DI SẢN) CỦA CÔNG TY:
            {knowledge_dna}
            --------------------------------------------------
            
            NHIỆM VỤ:
            Đừng sao chép nguyên văn Kho tàng trên. Hãy PHÂN TÍCH CỐT LÕI (Bố cục, tư duy, tiêu chuẩn bảo mật, văn phong) của nó.
            Sau đó, hãy SÁNG TÁC một giải pháp MỚI HOÀN TOÀN cho yêu cầu của CEO, nhưng phải tuân thủ nghiêm ngặt các tiêu chuẩn trong Kho tàng.
            
            VÍ DỤ:
            - Nếu Kho tàng có code Game Rắn (với chuẩn clean code), và CEO đòi Game Tetris -> Hãy viết Game Tetris bằng chuẩn clean code đó.
            - Nếu Kho tàng có Chiến lược Marketing Facebook, và CEO hỏi về TikTok -> Hãy áp dụng tư duy chiến lược đó sang TikTok.
            
            HÃY TRẢ LỜI NGAY BÂY GIỜ (Dùng Markdown đẹp):
            """
            
            print("   🎨 Đang vẽ bức tranh mới dựa trên kỹ thuật cũ...")
            response = await CHAT_MODEL.ainvoke(prompt)
            return {"answer": response.content}
            
        else:
            return {"answer": "Lỗi: Chưa kết nối bộ não AI (CHAT_MODEL)."}

    except Exception as e:
        return {"answer": f"Lỗi sáng tạo: {str(e)}"}

@app.post("/api/learn")
async def api_learn(request: LearnRequest, x_api_key: str = Header(None)):
    """API dạy học thủ công"""
    if x_api_key != ADMIN_SECRET: raise HTTPException(403)
    if not AI_AVAILABLE: return {"status": "error"}
    res = learn_knowledge(request.text)
    return {"status": "success", "message": res}

@app.post("/api/worker/submit_knowledge")
async def receive_knowledge(data: LearningResult, x_api_key: str = Header(None)):
    """
    API dành cho các máy vệ tinh nộp kiến thức đã học được về kho trung tâm.
    """
    if x_api_key != ADMIN_SECRET:
        raise HTTPException(403, "Sai mật mã kết nối!")
    
    # Lưu vào bộ nhớ dài hạn
    logger.info(f"📥 Nhận kiến thức từ Worker [{data.worker_id}]: {data.source}")
    
    if AI_AVAILABLE:
        # Chạy ngầm việc embedding vào ChromaDB
        await run_in_threadpool(lambda: learn_knowledge(data.content))
        
    return {"status": "accepted", "msg": "Đã nạp vào bộ não trung tâm"}

# --- API 1: PHÁT NHIỆM VỤ (Dành cho Worker xin việc) ---
@app.post("/api/worker/get_task")
async def worker_get_task(req: TaskRequest, x_api_key: str = Header(None)):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    try:
        # --- [FIX QUAN TRỌNG] KIỂM TRA KEY ---
        # 1. Tìm xem key này có trong bảng users không
        c.execute("SELECT username FROM users WHERE api_key=?", (x_api_key,))
        user = c.fetchone()

        # 2. Nếu không phải Admin VÀ cũng không phải User đăng ký -> CHẶN
        if x_api_key != ADMIN_SECRET and not user:
             raise HTTPException(status_code=403, detail="Key không hợp lệ!")

        # --- LOGIC GIAO VIỆC (GIỮ NGUYÊN) ---
        worker_tier = 1 # Mặc định là 1 nếu không gửi lên
        
        # Tìm việc phù hợp
        c.execute("""
            SELECT id, topic, task_type, content 
            FROM learning_tasks 
            WHERE status='PENDING' 
            AND difficulty <= ? 
            ORDER BY difficulty DESC
            LIMIT 1
        """, (worker_tier,))
        
        row = c.fetchone()
        
        # Nếu không có việc PENDING, tìm việc bị treo (PROCESSING quá lâu)
        if not row:
            c.execute("""
                SELECT id, topic, task_type, content FROM learning_tasks 
                WHERE status='PROCESSING' 
                AND last_updated < datetime('now', '-30 minutes')
                LIMIT 1
            """)
            row = c.fetchone()
        
        if row:
            task_id, topic, task_type, content = row
            # Đánh dấu đang làm
            c.execute("UPDATE learning_tasks SET status='PROCESSING', assigned_to=?, last_updated=CURRENT_TIMESTAMP WHERE id=?", (req.worker_id, task_id))
            conn.commit()
            
            print(colored(f"Giao việc '{topic}' cho {req.worker_id}", "cyan"))
            
            # Trả về đầy đủ thông tin để Worker làm việc
            return {
                "task_id": task_id, 
                "topic": topic,
                "type": task_type,      # <--- QUAN TRỌNG CHO WORKER MỚI
                "content": content      # <--- QUAN TRỌNG CHO WORKER MỚI
            }
        else:
            return {"task_id": None, "message": "Hết việc rồi, nghỉ ngơi đi!"}
            
    finally:
        conn.close() 

async def generate_harder_task(previous_result):
    """
    Giáo sư ảo: Phân tích kết quả cũ -> Tạo bài tập mới khó hơn.
    """
    if not CHAT_MODEL: return

    print(colored("🤔 [SUPERVISOR] Đang suy nghĩ bài tập nâng cao...", "cyan"))
    
    prompt = f"""
    Hệ thống vệ tinh vừa hoàn thành xuất sắc nhiệm vụ này:
    ---
    {previous_result}
    ---
    
    Dựa trên thành công này, hãy suy nghĩ ra 1 NHIỆM VỤ TIẾP THEO (NEXT STEP) có độ khó cao hơn, phức tạp hơn để nâng cao trình độ.
    
    Yêu cầu:
    1. Nhiệm vụ mới phải liên quan đến nhiệm vụ cũ nhưng khó hơn (Level Up).
    2. Trả về định dạng JSON thuần túy (không Markdown).
    
    JSON Mẫu:
    {{
        "topic": "Tên nhiệm vụ mới",
        "type": "PRACTICE_CODE",
        "difficulty": 2,
        "content": "Code python mẫu hoặc yêu cầu cụ thể..."
    }}
    """
    
    try:
        # Gọi AI tư duy
        ai_res = await CHAT_MODEL.ainvoke(prompt)
        
        # Làm sạch JSON (đôi khi AI thêm ```json ... ```)
        clean_json = ai_res.content.replace("```json", "").replace("```", "").strip()
        new_task = json.loads(clean_json)
        
        # Lưu vào DB để chờ Worker rảnh thì làm
        conn = sqlite3.connect(DB_PATH)
        conn.execute(
            "INSERT INTO learning_tasks (topic, status, assigned_to, last_updated) VALUES (?, 'PENDING', NULL, CURRENT_TIMESTAMP)",
            (f"[LEVEL UP] {new_task['topic']}",)
        )
        conn.commit()
        conn.close()
        
        print(colored(f"📈 [LEVEL UP] Đã sinh nhiệm vụ mới: {new_task['topic']}", "magenta"))
    except Exception as e:
        print(f"⚠️ Lỗi sinh bài tập khó: {e}")
# --- API 2: NHẬN KẾT QUẢ (Dành cho Worker nộp bài) ---
# Hàm tạo mã vân tay (Fingerprint)
def create_content_hash(content):
    return hashlib.md5(content.strip().encode('utf-8')).hexdigest()
async def perform_auto_audit(task_topic, task_content, worker_result):
    """
    Hệ thống kiểm định chất lượng tự động.
    """
    if not CHAT_MODEL: return 100, "AI_OFFLINE_BYPASS"

    audit_prompt = f"""
    Bạn là Chuyên gia kiểm định chất lượng (QA Senior).
    NHIỆM VỤ GỐC: {task_topic}
    YÊU CẦU CHI TIẾT: {task_content}
    KẾT QUẢ WORKER NỘP: 
    ---
    {worker_result}
    ---
    Hãy chấm điểm kết quả này trên thang điểm 100.
    Tiêu chí: Đúng yêu cầu, Code chạy được, giải thích rõ ràng.
    Trả về JSON: {{"score": 0-100, "comment": "nhận xét ngắn", "status": "APPROVED/REJECTED"}}
    Chỉ REJECTED nếu kết quả quá sơ sài hoặc sai hoàn toàn logic.
    """
    try:
        res = await CHAT_MODEL.ainvoke(audit_prompt)
        audit_data = json.loads(res.content.replace("```json", "").replace("```", "").strip())
        return audit_data.get("score", 0), audit_data.get("comment", "No comment")
    except:
        return 70, "Audit System Busy - Default Pass"


@app.post("/api/worker/submit_task")
async def worker_submit_task(res: TaskResult, x_api_key: str = Header(None)):
    """
    HỆ THỐNG KIỂM ĐỊNH & THANH TOÁN TỰ ĐỘNG V9.0
    Tích hợp: Hash-Deduplication, AI-Audit JSON, Dynamic Payment & Node Health Tracking.
    """
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    
    final_pay = 0.0 
    audit_score = 0
    audit_comment = "Đang chờ kiểm định"

    try:
        # 1. XÁC THỰC NGƯỜI DÙNG (Tra cứu từ bảng users thay vì ADMIN_SECRET)
        c.execute("SELECT username, balance FROM users WHERE api_key=?", (x_api_key,))
        user_row = c.fetchone()
        
        # Cho phép Admin hoặc User có Key hợp lệ
        if x_api_key != ADMIN_SECRET and not user_row:
            raise HTTPException(status_code=403, detail="⛔ Truy cập bị chặn: API Key không hợp lệ!")
        
        username = user_row[0] if user_row else "ADMIN"

        # 2. KIỂM TRA THÔNG TIN NHIỆM VỤ
        c.execute("SELECT reward, task_type, topic, status, content FROM learning_tasks WHERE id=?", (res.task_id,))
        task_info = c.fetchone()
        if not task_info: 
            return {"status": "error", "msg": "Task không tồn tại"}
        if task_info[3] == 'DONE': 
            return {"status": "error", "msg": "Task này đã được người khác hoàn thành!"}

        agreed_reward, task_type, topic, _, original_requirement = task_info

        # 3. CHỐNG TRÙNG LẶP (DEDUPLICATION) BẰNG MÃ BĂM (HASH)
        incoming_hash = res.metadata.get("hash") if res.metadata else None
        if incoming_hash:
            c.execute("SELECT id FROM work_logs WHERE result_hash = ?", (incoming_hash,))
            if c.fetchone():
                print(colored(f"⚠️ [SPAM] Phát hiện nộp trùng nội dung từ Node {res.worker_id}", "yellow"))
                return {"status": "error", "msg": "Nội dung này đã tồn tại trong hệ thống tri thức!"}

        # 4. MODULE AUTO-AUDIT (KIỂM ĐỊNH CHẤT LƯỢNG BẰNG AI)
        print(colored(f"🛡️ [AUDIT] Đang thẩm định Task #{res.task_id} cho {username}...", "cyan"))
        
        # Prompt ép AI đọc hiểu cấu trúc JSON mà Worker Pro gửi lên
        audit_prompt = f"""
        Nhiệm vụ: {topic}
        Yêu cầu gốc: {original_requirement}
        Bài nộp từ Worker: {res.result_content}

        Yêu cầu thẩm định:
        1. Nếu là cấu trúc JSON hợp lệ, hãy đánh giá cao tính chuyên nghiệp.
        2. Chấm điểm từ 0-100 dựa trên độ chính xác và chiều sâu thông tin.
        3. Trả về JSON duy nhất: {{"score": score, "reason": "nhận xét"}}
        """
        
        try:
            audit_res = await CHAT_MODEL.ainvoke(audit_prompt)
            clean_json = audit_res.content.replace("```json", "").replace("```", "").strip()
            audit_data = json.loads(clean_json)
            audit_score = audit_data.get("score", 0)
            audit_comment = audit_data.get("reason", "N/A")
        except Exception as e:
            logger.error(f"Audit Error: {e}")
            audit_score = 75 # Điểm tối thiểu nếu hệ thống AI bận
            audit_comment = "Phê duyệt dự phòng (Hệ thống Audit bận)."

        # 5. QUYẾT ĐỊNH GIẢI NGÂN (DYNAMIC BILLING)
        is_passed = audit_score >= 60
        verdict = "APPROVED" if is_passed else "REJECTED"
        
        if is_passed:
            # Tính Bonus nếu làm nhanh (dưới 15 giây)
            duration = res.metadata.get("duration_sec", 60) if res.metadata else 60
            speed_bonus = 1.1 if duration < 15 else 1.0
            
            # Tiền thực nhận = Giá gốc * (Điểm chất lượng %) * Thưởng tốc độ
            final_pay = round(agreed_reward * (audit_score / 100) * speed_bonus, 5)

            # CẬP NHẬT VÍ TIỀN TRONG DB
            if user_row:
                c.execute("UPDATE users SET balance = balance + ? WHERE username=?", (final_pay, username))
        else:
            final_pay = 0.0

        # 6. CẬP NHẬT TRẠNG THÁI TASK & LƯU TRI THỨC
        task_new_status = 'DONE' if is_passed else 'PENDING'
        c.execute("UPDATE learning_tasks SET status=?, last_updated=CURRENT_TIMESTAMP WHERE id=?", (task_new_status, res.task_id))

        # 7. GHI NHẬT KÝ CHI TIẾT (WORK_LOGS)
        c.execute("""
            INSERT INTO work_logs (timestamp, agent_name, task_content, result_summary, tool_used, cost, result_hash)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            datetime.now().strftime("%H:%M %d/%m/%Y"),
            f"NODE_{res.worker_id}",
            f"[{verdict}] {topic}",
            f"Score: {audit_score} | Reason: {audit_comment}\n\nResult: {res.result_content[:500]}",
            "AI_AUDIT_V9",
            final_pay,
            incoming_hash
        ))

        conn.commit()
        
        # Trả kết quả về cho Worker GUI hiển thị
        return {
            "status": "success",
            "verdict": verdict,
            "score": audit_score,
            "reward_earned": final_pay,
            "msg": audit_comment
        }

    except Exception as e:
        print(colored(f"❌ LỖI HỆ THỐNG SUBMIT: {e}", "red"))
        return {"status": "error", "msg": f"Lỗi Server: {str(e)}"}
    finally:
        conn.close()

@app.post("/api/admin/create_job")
async def create_job(topic: str, type: str, price: float, min_cpu: int = 0, needs_gpu: bool = False, x_api_key: str = Header(None)):
    """
    ADMIN V9.0: Tạo việc có định hướng phần cứng.
    """
    if x_api_key != ADMIN_SECRET: raise HTTPException(403)
    
    # Lưu thêm yêu cầu phần cứng vào DB
    with db_manager.get_connection() as conn:
        conn.execute(
            text("""INSERT INTO learning_tasks (topic, task_type, reward, status, min_cpu, needs_gpu) 
                    VALUES (:t, :type, :r, 'PENDING', :cpu, :gpu)"""),
            {"t": topic, "type": type, "r": price, "cpu": min_cpu, "gpu": 1 if needs_gpu else 0}
        )
        conn.commit()
    
    return {"msg": f"🚀 Đã tạo việc '{topic}' - Yêu cầu: CPU > {min_cpu}% | GPU: {needs_gpu}"}

# ==========================================
# 3. PHẦN 5: AI SUPERVISOR
# ==========================================
def split_long_subject(text, max_length=4000):
    """
    BẢN NÂNG CẤP v9.5: Chia nhỏ dự án theo ngữ cảnh + Chống tràn bộ nhớ.
    """
    if not text: return []
    
    # Chuẩn hóa khoảng trắng và xuống dòng để AI dễ đọc hơn
    text = re.sub(r'\n+', '\n', text)
    
    # Chia theo dấu câu và cả dấu xuống dòng
    chunks = re.split(r'(?<=[.!?\n]) +', text)
    
    result = []
    current_chunk = ""
    
    for segment in chunks:
        # Nếu bản thân một segment đã quá dài (trường hợp không có dấu câu)
        if len(segment) > max_length:
            # Nếu đang có dở chunk cũ thì lưu lại đã
            if current_chunk:
                result.append(current_chunk.strip())
                current_chunk = ""
            
            # Chia nhỏ segment khổng lồ này một cách cưỡng bức
            for i in range(0, len(segment), max_length):
                result.append(segment[i:i + max_length].strip())
            continue

        # Logic gom nhóm thông thường
        if len(current_chunk) + len(segment) < max_length:
            current_chunk += (" " if current_chunk else "") + segment
        else:
            result.append(current_chunk.strip())
            current_chunk = segment
            
    if current_chunk:
        result.append(current_chunk.strip())
        
    return [c for c in result if c] # Loại bỏ các đoạn rỗng
# ---- PHÂN TÁCH NỘI DUNG TRUYỀN CHO MÁY ĐÀO ----
@app.post("/api/admin/auto_distribute_knowledge")
async def auto_distribute_knowledge(req: CourseRequest, x_api_key: str = Header(None)):
    # 1. Bảo mật
    if x_api_key != ADMIN_SECRET: raise HTTPException(403)
    
    # 2. Chia nhỏ nội dung (Splitter)
    chunks = split_long_subject(req.subject, max_length=3500) # Để dư 500 ký tự cho Prompt
    total_created = 0
    project_ref = uuid.uuid4().hex[:8].upper() # Mã tham chiếu dự án

    print(colored(f"🧠 [SUPERVISOR] Đang rã dự án {project_ref} ({len(chunks)} phân đoạn)", "magenta"))

    for idx, content_chunk in enumerate(chunks):
        # Giới hạn số task mỗi đoạn để tránh AI bị "loãng"
        tasks_per_chunk = max(1, req.num_tasks // len(chunks))
        
        prompt = f"""
        Bạn là Tech Lead của PV AI-CORP. 
        PHÂN TÍCH ĐOẠN {idx+1}/{len(chunks)} CỦA DỰ ÁN: {project_ref}
        NỘI DUNG: {content_chunk}
        
        NHIỆM VỤ: Trích xuất {tasks_per_chunk} nhiệm vụ thực thi (RESEARCH hoặc PRACTICE_CODE).
        ĐỊNH DẠNG TRẢ VỀ: Duy nhất 1 JSON LIST:
        [
          {{"topic": "Tên ngắn gọn", "type": "RESEARCH", "reward": {req.reward_per_task}, "content": "Chỉ thị chi tiết"}}
        ]
        """

        try:
            response = await CHAT_MODEL.ainvoke(prompt)
            # Dọn dẹp JSON rác
            match = re.search(r'\[.*\]', response.content, re.DOTALL)
            if not match: continue
            
            task_list = json.loads(match.group())

            # 3. Nạp vào Database tập trung
            with db_manager.get_connection() as conn:
                for t in task_list:
                    reward = float(t.get('reward', req.reward_per_task or 0.05))
                    # Gắn mã dự án vào topic để dễ tìm kiếm
                    full_topic = f"[{project_ref}] {t['topic']}"
                    
                    conn.execute(
                        text("INSERT INTO learning_tasks (topic, task_type, reward, content, status) VALUES (:t, :type, :r, :c, 'PENDING')"),
                        {"t": full_topic, "type": t['type'], "r": reward, "c": t['content']}
                    )
                conn.commit()
                total_created += len(task_list)
            
            # Nghỉ 1 giây để tránh bị AI khóa (Rate Limit)
            await asyncio.sleep(1)

        except Exception as e:
            print(f"⚠️ Lỗi phân rã đoạn {idx}: {e}")
            continue

    return {
        "status": "success",
        "project_id": project_ref,
        "tasks_created": total_created,
        "msg": f"Dự án đã được bẻ nhỏ thành {total_created} mắt xích tri thức."
    }
# --- TỔNG HỢP NỘI DUNG KIỂM TRA VÀ KẾT NỐI ----
@app.post("/api/admin/merge_project_knowledge")
async def merge_project_knowledge(project_id: str, x_api_key: str = Header(None)):
    """
    KNOWLEDGE FUSION v9.8: Hợp nhất hàng trăm kết quả rời rạc thành một báo cáo tổng lực.
    """
    if x_api_key != ADMIN_SECRET: raise HTTPException(403)

    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    try:
        # 1. TRUY XUẤT TẤT CẢ KẾT QUẢ ĐÃ ĐƯỢC PHÊ DUYỆT (APPROVED)
        # Chúng ta lọc theo mã dự án đã gắn vào topic lúc nãy [PROJECT_ID]
        search_pattern = f"[{project_id}]%"
        c.execute("""
            SELECT result_summary FROM work_logs 
            WHERE task_content LIKE ? AND result_summary NOT LIKE '%REJECTED%'
        """, (search_pattern,))
        
        results = c.fetchall()
        if not results:
            return {"status": "error", "msg": "Chưa có dữ liệu hoàn thành cho dự án này."}

        # 2. GOM NHÓM DỮ LIỆU (DATA AGGREGATION)
        # Chỉ lấy phần DATA thực tế từ log, bỏ qua phần nhận xét Audit
        full_intel = "\n---\n".join([r[0].split("DATA:")[1] if "DATA:" in r[0] else r[0] for r in results])

        print(colored(f"🧬 [FUSION] Đang hợp nhất {len(results)} mảnh tri thức từ dự án {project_id}...", "magenta"))

        # 3. GỌI BỘ NÃO TỔNG CHỈ HUY (MASTER AI) ĐỂ BIÊN TẬP
        fusion_prompt = f"""
        BẠN LÀ CHỦ TỊCH HỘI ĐỒNG CHIẾN LƯỢC CỦA PV AI-CORP.
        Dưới đây là các mảnh tri thức thô thu thập được từ mạng lưới Node về dự án: {project_id}
        --- DỮ LIỆU THÔ ---
        {full_intel[:12000]} 
        
        NHIỆM VỤ:
        1. Tổng hợp thành một BÁO CÁO CHIẾN LƯỢC hoàn chỉnh.
        2. Loại bỏ các nội dung trùng lặp hoặc mâu thuẫn.
        3. Cấu trúc báo cáo: Tóm tắt điều hành -> Các phát hiện chính -> Giải pháp đề xuất -> Kết luận.
        4. Ngôn ngữ: Tiếng Việt chuyên nghiệp, sắc sảo.
        """

        response = await CHAT_MODEL.ainvoke(fusion_prompt)
        final_report = response.content

        # 4. LƯU TRỮ VÀO KHO TRI THỨC TỔNG (FINAL ARCHIVE)
        c.execute("""
            INSERT INTO project_reports (project_id, report_content, node_count, created_at)
            VALUES (?, ?, ?, CURRENT_TIMESTAMP)
        """, (project_id, final_report, len(results)))
        
        conn.commit()

        return {
            "status": "success",
            "project_id": project_id,
            "summary": final_report[:500] + "...",
            "full_report": final_report,
            "msg": f"Đã hợp nhất thành công {len(results)} mắt xích tri thức."
        }

    except Exception as e:
        return {"status": "error", "msg": str(e)}
    finally:
        conn.close()
# --- THÔNG KÊ CHI PHÍ CHO CÁC NODE ---
@app.get("/api/admin/project_cost_stats/{project_id}")
async def get_project_cost(project_id: str):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    
    # Tính tổng số tiền đã trả cho tất cả các thợ đào của dự án này
    search_pattern = f"[{project_id}]%"
    c.execute("SELECT SUM(cost) FROM work_logs WHERE task_content LIKE ?", (search_pattern,))
    total_cost = c.fetchone()[0] or 0.0
    
    # Đếm số thợ đào tham gia
    c.execute("SELECT COUNT(DISTINCT agent_name) FROM work_logs WHERE task_content LIKE ?", (search_pattern,))
    worker_count = c.fetchone()[0] or 0

    return {
        "project_id": project_id,
        "total_investment": round(total_cost, 4),
        "total_workers": worker_count,
        "avg_cost_per_node": round(total_cost / worker_count, 5) if worker_count > 0 else 0
    }

# --- API 3: ADMIN NẠP DANH SÁCH VIỆC (Seed Tasks) ---

@app.post("/api/admin/seed_tasks")
async def seed_learning_tasks(topics: List[str], x_api_key: str = Header(None)):
    """Admin nạp một list chủ đề cần học vào hàng đợi"""
    if x_api_key != ADMIN_SECRET: raise HTTPException(403)
    conn = sqlite3.connect(DB_PATH)
    count = 0
    for t in topics:
        try:
            conn.execute("INSERT INTO learning_tasks (topic) VALUES (?)", (t,))
            count += 1
        except sqlite3.IntegrityError: pass # Bỏ qua nếu trùng chủ đề
    conn.commit()
    conn.close()
    return {"msg": f"Đã thêm {count} nhiệm vụ mới."}
# --- API ĐỒNG BỘ DỮ LIỆU ---
@app.get("/api/sync/download_db")
async def download_database():
    if os.path.exists(DB_PATH):
        return FileResponse(path=DB_PATH, filename="ai_corp_data.db", media_type='application/octet-stream')
    return {"error": "Database not found"}


os.makedirs(TTS_CACHE_DIR, exist_ok=True)

@app.post("/api/speak")
async def api_speak(request: SpeakRequest):
    """
    API Tạo giọng nói (Đã nâng cấp: Ưu tiên MIỄN PHÍ với gTTS)
    Thay thế OpenAI TTS ($$) bằng Google TTS ($0) để giảm chi phí vận hành.
    """
    try:
        # 1. Lấy văn bản và lọc sạch
        # Cắt ngắn 500 ký tự để đọc cho nhanh
        safe_text = request.text[:500] 

        # --- [NÂNG CẤP] KIỂM TRA CACHE ---
        import hashlib
        # Tạo tên file dựa trên nội dung văn bản (MD5 hash)
        file_hash = hashlib.md5(safe_text.encode()).hexdigest()
        cache_path = os.path.join(TTS_CACHE_DIR, f"{file_hash}.mp3")
        
        # Nếu đã có file này rồi -> Trả về ngay
        if os.path.exists(cache_path):
            print(colored(f"🔊 [TTS CACHE HIT]: {safe_text[:20]}...", "green"))
            return FileResponse(cache_path, media_type="audio/mpeg")
        # 2. Dùng gTTS (Miễn phí) thay vì client.audio.speech.create (Tốn tiền)
        # Tận dụng lại logic của thư viện gTTS đã import
        def _generate_free_audio():
            # tld='com.vn' để giọng đọc thuần Việt hơn
            tts = gTTS(text=safe_text, lang='vi', tld='com.vn')
            buffer = io.BytesIO()
            tts.write_to_fp(buffer)
            buffer.seek(0)
            return buffer.read()

        # Chạy trong luồng riêng để không treo server
        audio_content = await run_in_threadpool(_generate_free_audio)
        # Sau khi tạo xong buffer, hãy lưu nó xuống cache_path trước khi trả về
        with open(cache_path, "wb") as f:
            f.write(audio_content)

        return Response(content=audio_content, media_type="audio/mpeg")

    except Exception as e:
        logger.error(f"🚨 [VOICE ERROR]: {str(e)}")
        # Trả về 204 để Dashboard không báo lỗi đỏ nếu mất mạng
        return Response(status_code=204)

# --- CẤU HÌNH HỌC TẬP ---
LEARNING_QUEUE = ["CODER", "ARTIST", "ENGINEERING", "MARKETING", "LEGAL"]
CURRENT_LEARNER_INDEX = 0
IS_BUSY = False  # Trạng thái bận rộn của hệ thống
LAST_ACTIVITY_TIME = datetime.now()

# ==========================================
# 6. WEBSOCKET (REAL-TIME DASHBOARD)
# ==========================================
@app.websocket("/ws/nexus")
async def websocket_nexus(websocket: WebSocket):
    await manager.connect(websocket)
    
    # 1. TẠO SESSION ID RIÊNG BIỆT (Fix lỗi trộn lẫn ký ức)
    session_id = f"ws_{uuid.uuid4().hex[:8]}"
    print(colored(f"🔌 New Connection: {session_id}", "green"))
    
    try:
        # Gửi lời chào (Dạng JSON cho Dashboard)
        await manager.send_json({
            "sender": "J.A.R.V.I.S",
            "content": "Hệ thống trực tuyến. Đang đồng bộ thời gian thực...",
            "agent": "System"
        }, websocket)
        
        while True:
            data = await websocket.receive_text()
            print(colored(f"⚡ [INPUT]: {data}", "cyan"))
            
            # ============================================================
            # PHẦN 1: KHÔI PHỤC CÁC TÍNH NĂNG CŨ (CỦA NGÀI)
            # ============================================================
            
            # 1. LẤY THÔNG TIN HỆ THỐNG (Thời gian thực)
            current_time = datetime.now().strftime("%H:%M, Thứ %w, ngày %d/%m/%Y")
            system_context = f"Hiện tại là {current_time}. Vị trí: Phan Thiết, Việt Nam."
            
            # 2. HỒI TƯỞNG KÝ ỨC
            mem_ctx = ""
            if MEMORY_AVAILABLE:
                mem_ctx = await run_in_threadpool(lambda: recall_relevant_memories(data))
                if mem_ctx: print(colored(f"🧠 [KÝ ỨC]: {mem_ctx[:100]}...", "magenta"))

            # ============================================================
            # PHẦN 2: XỬ LÝ THÔNG MINH (KẾT HỢP LANGGRAPH)
            # ============================================================
            
            # Tạo Prompt chứa đầy đủ thông tin: Thời gian + Ký ức + Câu hỏi
            # Điều này giúp Agent (Họa sĩ/Coder) cũng biết bây giờ là mấy giờ
            full_prompt = f"""
            [SYSTEM CONTEXT]: {system_context}
            [MEMORY]: {mem_ctx}
            [USER REQUEST]: {data}
            """
            
            reply_content = ""
            active_agent = "J.A.R.V.I.S"

            # A. FAST TRACK (Giữ lại logic cũ cho các câu hỏi đơn giản để tiết kiệm)
            # Nếu chỉ hỏi ngày giờ, giá cả -> Dùng Gemini/GPT trực tiếp cho nhanh
            fast_keywords = ["bao nhiêu ngày", "tết", "thứ mấy", "ngày mấy", "mấy giờ", "thời tiết", "giá"]
            is_simple = any(k in data.lower() for k in fast_keywords) and not any(k in data.lower() for k in ["vẽ", "code", "lập trình"])

            if is_simple and LLM_GEMINI_LOGIC:
                print(colored("🚀 Kích hoạt Fast Track (Real-time Context)...", "yellow"))
                try:
                    # Gọi Gemini trả lời nhanh câu hỏi ngày giờ
                    ai_msg = await LLM_GEMINI_LOGIC.ainvoke(full_prompt)
                    reply_content = ai_msg.content
                    active_agent = "J.A.R.V.I.S"
                except: pass
            
            # B. DEEP THINKING (Nếu Fast Track bỏ qua HOẶC là lệnh Vẽ/Code)
            if not reply_content and AI_AVAILABLE:
                # Truyền session_id vào thread_id để giữ mạch chuyện riêng biệt
                config = {"configurable": {"thread_id": session_id}}
                # Gọi bộ não LangGraph (Supervisor -> Designer/Coder...)
                print(colored("🧩 Chuyển giao cho Bộ Não Trung Tâm (LangGraph)...", "blue"))
                
                input_message = HumanMessage(content=full_prompt)
                final_state = await ai_app.ainvoke({"messages": [input_message]}, config=config)
                
                # Lấy kết quả từ Agent cuối cùng
                last_message = final_state['messages'][-1]
                reply_content = last_message.content
                
                # Xác định ai vừa làm việc (Để Dashboard sáng đèn)
                active_agent = final_state.get("current_agent", "J.A.R.V.I.S")

            # ============================================================
            # PHẦN 3: PHẢN HỒI (DẠNG JSON CHO DASHBOARD)
            # ============================================================
            print(colored(f"🤖 [{active_agent}]: {reply_content}", "magenta"))
            
            # Gửi JSON xuống Client
            await manager.send_json({
                "sender": active_agent,
                "content": reply_content,
                "agent": active_agent # Dashboard dùng cái này để highlight icon
            }, websocket)
            
                        # 4. GHI NHỚ LẠI
            if MEMORY_AVAILABLE:
                await run_in_threadpool(lambda: extract_and_save_memory(data, reply_content))

    except WebSocketDisconnect:
        manager.disconnect(websocket)
        print(colored(f"🔌 Disconnected: {session_id}", "red"))
    except Exception as e:
        logger.error(f"WS Error: {e}")
        manager.disconnect(websocket)

def hunt_freelance_projects(keyword="python"):
    # Ví dụ quét từ một nguồn tin tuyển dụng/dự án (đã được cấu hình RSS hoặc API)
    # Đây là logic giả lập cách J.A.R.V.I.S đi "tìm việc"
    search_url = f"https://www.freelancer.com/jobs/{keyword}/"
    
    try:
        resp = requests.get(search_url)
        # AI sẽ phân tích nội dung trang web ở đây để lấy Title, Budget và Description
        # Sau đó tự động tạo payload để nạp vào Dashboard
        print(f"📡 Đang quét các dự án {keyword} mới nhất trên thị trường...")
        return True
    except:
        return False

class ProjectHunter:
    @staticmethod
    async def hunt_and_distribute(keyword: str, default_reward: float = 0.05):
        """
        Săn dự án và phân bổ task. 
        Thêm 'default_reward' để thay thế cho biến 'req' bị thiếu.
        """
        feeds = [
            f"https://remoteok.com/remote-{keyword}-jobs.rss",
            f"https://www.upwork.com/ab/feed/jobs/rss?q={keyword}"
        ]
        
        new_projects = []
        for url in feeds:
            try:
                # Chạy feedparser trong threadpool để không chặn async loop
                feed = await run_in_threadpool(lambda: feedparser.parse(url))
                for entry in feed.entries[:5]: 
                    new_projects.append({
                        "title": entry.title,
                        "link": entry.link,
                        "summary": entry.summary
                    })
            except Exception as e:
                print(f"⚠️ Lỗi quét RSS: {e}")
                continue

        if not new_projects: 
            return 0

        count = 0
        for proj in new_projects:
            prompt = f"""
            Bạn là chuyên gia thẩm định dự án của J.A.R.V.I.S.
            Phân tích dự án: {proj['title']}
            Mô tả: {proj['summary']}
            
            Yêu cầu: Chia dự án thành 3-5 task nhỏ.
            Định dạng JSON list: [{{"topic": "...", "type": "RESEARCH hoặc PRACTICE_CODE", "reward": 0.05, "content": "..."}}]
            """
            try:
                ai_res = await CHAT_MODEL.ainvoke(prompt)
                # Parse nội dung AI trả về
                cleaned_content = ai_res.content.replace("```json", "").replace("```", "").strip()
                
                # SỬA LỖI TÊN BIẾN: Đặt thống nhất là 'tasks'
                tasks = json.loads(cleaned_content)
                
                with db_manager.engine.connect() as conn:
                    for task in tasks: # Đã đổi từ task_list thành tasks
                        
                        # SỬA LỖI 'req': Thay 'req.reward_per_task' bằng 'default_reward' hoặc task['reward']
                        # Ưu tiên lấy giá từ AI, nếu không có thì dùng default_reward (0.05)
                        reward_value = float(task.get('reward', default_reward))
                        if reward_value <= 0: 
                            reward_value = default_reward

                        try:
                            conn.execute(
                                text("""
                                    INSERT INTO learning_tasks (topic, task_type, reward, content, status)
                                    VALUES (:t, :type, :r, :c, 'PENDING')
                                """),
                                {
                                    "t": f"🚀 [HUNTER] {task.get('topic', 'New Task')}",
                                    "type": task.get('type', 'RESEARCH'),
                                    "r": reward_value,
                                    "c": f"Dự án gốc: {proj['link']}\n\nYêu cầu: {task.get('content', '')}"
                                }
                            )
                        except Exception as e:
                            print(f"❌ Lỗi INSERT task: {e}")
                            continue
                    conn.commit()
                count += 1
            except Exception as e:
                print(f"❌ Lỗi xử lý dự án '{proj['title']}': {e}")
                continue
                
        return count
    @staticmethod
    async def generate_proposal(proj_title, proj_summary):
        """
        [AUTO-PROPOSAL] Dựa trên Di sản tri thức để viết thư chào hàng đỉnh cao.
        """
        # Lấy "hương vị" từ di sản của CEO (HR/Marketing)
        from main import run_nexus_sync
        
        proposal_prompt = f"""
        [ROLE]: Bạn là Senior Sales Engineer của AI Corporation.
        [PROJECT]: {proj_title}
        [DESCRIPTION]: {proj_summary}
        
        NHIỆM VỤ: 
        1. Dựa trên các tiêu chuẩn ĐẠO ĐỨC, PHÁP LÝ và KỸ THUẬT trong 'Di sản tri thức' của công ty.
        2. Viết một bản Proposal (Thư chào hàng) thuyết phục khách hàng chọn chúng ta.
        3. Nhấn mạnh vào: Giải pháp AI tự chủ, Bảo mật dữ liệu và Tối ưu ROI.
        
        FORMAT: Trả về Markdown đẹp, có lời chào, giải pháp và cam kết.
        """
        
        # Gọi Nexus Core để soạn thảo
        proposal = await run_in_threadpool(lambda: run_nexus_sync(proposal_prompt, "bidding_session"))
        return proposal

# [CẬP NHẬT TRONG VÒNG LẶP HƯNT]
# (Sau khi nạp Task thành công, thêm đoạn này)
        
        # 4. TỰ ĐỘNG SOẠN THẢO PROPOSAL
        print(colored(f"✍️ [PROPOSAL] Đang soạn thảo đơn chào hàng cho: {proj['title'][:30]}...", "magenta"))
        bid_letter = await ProjectHunter.generate_proposal(proj['title'], proj['summary'])
        
        # Lưu vào hệ thống tin nhắn để CEO duyệt tại Tab [2] DEV_SANDBOX
        st_msg = f"📦 [PROPOSAL_DRAFT] cho dự án {proj['title']}:\n\n{bid_letter}"
        # (Lưu vào DB hoặc gửi qua WebSocket cho Dashboard)

# --- API ĐIỀU KHIỂN THỢ SĂN ---
@app.post("/api/admin/start_hunting")
async def start_hunting_api(req: HunterRequest, x_api_key: str = Header(None)):
    if x_api_key != ADMIN_SECRET: 
        raise HTTPException(status_code=403)
    
    # Gọi thợ săn chạy ngầm
    count = await ProjectHunter.hunt_and_distribute(req.keyword, default_reward=0.1)
    return {"status": "success", "projects_hunted": count}
# ==========================================
# 🚀 SYSTEM ROUTES
# ==========================================

@app.get("/health")
async def health_check():
    """
    Kiểm tra tình trạng sức khỏe toàn diện (Deep Health Check).
    """
    # 1. Kiểm tra kết nối Database (Thực tế)
    db_status = "UNKNOWN"
    try:
        # Thử thực hiện một truy vấn siêu nhẹ (SELECT 1)
        with db_manager.get_connection() as conn:
            conn.execute("SELECT 1")
            db_status = "CONNECTED (Active)"
    except Exception as e:
        db_status = f"CRITICAL ERROR: {str(e)}"
    
    # 2. Kiểm tra các Module AI
    return {
        "status": "OPERATIONAL" if "ERROR" not in db_status else "DEGRADED",
        "timestamp": datetime.now().isoformat(),
        "version": "JARVIS v3.0",
        "chi_tiet_loi": AI_BOOT_ERROR if not AI_AVAILABLE else "Không có lỗi",
        "modules": {
            "ai_brain": "ONLINE" if AI_AVAILABLE else "OFFLINE",
            "voice_core": "ONLINE" if VOICE_AVAILABLE else "OFFLINE",
            "memory_core": "ONLINE" if MEMORY_AVAILABLE else "OFFLINE",
            "knowledge_db": db_status, # Trạng thái kết nối thật
        }
    }

def get_latest_audit_report():
    """
    Hàm đọc báo cáo mới nhất trong thư mục projects (Đã tối ưu: Không dùng glob)
    """
    try:
        project_dir = "projects"
        if not os.path.exists(project_dir):
            return "Thưa CEO, thư mục 'projects' chưa được khởi tạo."

        # 1. Tìm tất cả file .md bằng os.listdir (Nhanh và không cần import glob)
        # Chỉ lấy file có đuôi .md
        all_files = [os.path.join(project_dir, f) for f in os.listdir(project_dir) if f.lower().endswith('.md')]
        
        if not all_files:
            return "Thưa CEO, kho dữ liệu hiện đang trống. Chưa có báo cáo nào."
            
        # 2. Tìm file mới nhất dựa trên thời gian tạo (Create Time)
        # Hàm os.path.getctime lấy thời gian tạo file
        latest_file = max(all_files, key=os.path.getctime)
        
        # Lấy tên file cho đẹp
        filename = os.path.basename(latest_file)
        
        # 3. Đọc nội dung
        with open(latest_file, "r", encoding="utf-8") as f:
            content = f.read()
            
        return f"### 📂 HỒ SƠ MỚI NHẤT: {filename}\n\n{content}"

    except Exception as e:
        logger.error(f"🚨 [REPORT ERROR]: {str(e)}")
        return f"⚠️ Thưa CEO, không thể truy xuất hồ sơ: {str(e)}."

@app.get("/api/backup/download_all")
async def download_full_brain(background_tasks: BackgroundTasks): # <--- Thêm tham số này
    """
    Đóng gói toàn bộ Trí tuệ (DB + Vector + Code) để CEO tải về máy.
    Cơ chế: Nén -> Gửi -> Tự hủy file nén để tiết kiệm ổ cứng.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    zip_filename = f"JAZVIC_BRAIN_BACKUP_{timestamp}.zip"
    zip_path = os.path.join(BASE_DATA_DIR, zip_filename)

    print(colored(f"📦 [BACKUP] Đang nén dữ liệu vào {zip_path}...", "yellow"))

    try:
        # Xóa file zip cũ (nếu lỡ còn sót lại)
        if os.path.exists(zip_path): os.remove(zip_path)

        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            # 1. Backup SQLite (Sổ cái)
            db_file = "ai_corp_projects.db"
            full_db_path = os.path.join(BASE_DATA_DIR, db_file)
            if os.path.exists(full_db_path):
                zipf.write(full_db_path, arcname=db_file)
            
            # 2. Backup File Dữ liệu học (JSONL)
            jsonl_file = "corporate_brain_dataset.jsonl"
            full_jsonl_path = os.path.join(BASE_DATA_DIR, jsonl_file)
            if os.path.exists(full_jsonl_path):
                zipf.write(full_jsonl_path, arcname=jsonl_file)

            # 3. Backup Bộ não Vector (db_knowledge)
            knowledge_dir = os.path.join(BASE_DATA_DIR, "db_knowledge")
            if os.path.exists(knowledge_dir):
                for root, dirs, files in os.walk(knowledge_dir):
                    for file in files:
                        # Bỏ qua các file lock tạm thời để tránh lỗi
                        if file.endswith(".lock"): continue
                        
                        file_path = os.path.join(root, file)
                        # Giữ nguyên cấu trúc thư mục (vd: db_knowledge/index.bin)
                        arcname = os.path.relpath(file_path, BASE_DATA_DIR)
                        zipf.write(file_path, arcname)

        file_size_mb = os.path.getsize(zip_path) / 1024 / 1024
        print(colored(f"✅ [BACKUP] Đã nén xong: {file_size_mb:.2f} MB", "green"))
        
        # --- HÀM DỌN DẸP SAU KHI GỬI XONG ---
        def cleanup_file(path: str):
            try:
                os.remove(path)
                print(colored(f"🧹 [CLEANUP] Đã xóa file tạm: {path}", "grey"))
            except Exception as e:
                print(colored(f"⚠️ Lỗi dọn dẹp: {e}", "red"))

        # Giao nhiệm vụ xóa file cho Background Task
        background_tasks.add_task(cleanup_file, zip_path)

        return FileResponse(
            path=zip_path, 
            filename=zip_filename, 
            media_type='application/zip'
        )

    except Exception as e:
        print(colored(f"❌ Lỗi Backup: {e}", "red"))
        # Nếu lỗi thì xóa file tạm ngay lập tức
        if os.path.exists(zip_path): os.remove(zip_path)
        return JSONResponse(status_code=500, content={"error": str(e)})
    
if __name__ == "__main__":
    import uvicorn
    # Sử dụng biến môi trường PORT để tương thích Cloud Run sau này
    port = int(os.environ.get("PORT", 8080))
    
    print("="*50)
    print(f"🚀 J.A.R.V.I.S SYSTEM STARTING ON PORT {port}")
    print(f"📄 API Documentation: http://localhost:{port}/docs")
    print("="*50)
    
    # Reload=True giúp server tự khởi động lại khi sửa code (Dev mode)
    uvicorn.run("server:app", host="0.0.0.0", port=port, reload=False)
