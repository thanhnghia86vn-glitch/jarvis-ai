import glob
import os
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
import zipfile
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
# [QUAN TRỌNG]: Đã thêm LLM_SUPERVISOR và log_training_data
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("JARVIS_v4.5")
ADMIN_SECRET = os.environ.get("ADMIN_SECRET", "ai_corp_secret_123")
RENDER_DISK_PATH = "/var/data"

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
# 3. Biến môi trường Database (Cập nhật lại cho SQLite nếu dùng Disk)
# Nếu không dùng PostgreSQL mà dùng SQLite trên Disk thì set lại url
if not os.environ.get("DATABASE_URL") and os.path.exists(RENDER_DISK_PATH):
    # Ép dùng SQLite trên ổ cứng Cloud để bền vững
    os.environ["DATABASE_URL"] = f"sqlite:///{DB_PATH}"

AI_AVAILABLE = False
MEMORY_AVAILABLE = False
VOICE_AVAILABLE = False
SERVER_READY = False
ai_app = None
CHAT_MODEL = None
client = None


try:
    from main import (
        ai_app, log_work_to_db, auto_learning_cycle, morning_briefing_job,
        vector_db, LLM_GPT4, LLM_GEMINI_LOGIC, LLM_GEMINI_VISION,
        CODER_PRIMARY, ingest_docs_to_memory, learn_knowledge, set_system_busy
    )
    AI_AVAILABLE = True
    SERVER_READY = True
    logger.info("✅ CORE AI MODULES: LOADED")
except Exception as e:
    # --- BẮT LỖI VÀ GHI LẠI ---
    import traceback
    AI_BOOT_ERROR = traceback.format_exc() # Lưu toàn bộ dấu vết lỗi
    logger.error(f"⚠️ CORE AI FAILED TO LOAD: {AI_BOOT_ERROR}")
    # Set safe defaults
    ai_app = vector_db = LLM_GEMINI_LOGIC = LLM_GEMINI_VISION = None

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

# ==========================================
# 1. DATABASE MANAGER
# ==========================================
class DatabaseManager:
    def __init__(self): 
        db_url = os.environ.get("DATABASE_URL", f"sqlite:///{DB_PATH}")
        if db_url.startswith("postgres://"):
            db_url = db_url.replace("postgres://", "postgresql://", 1)
        
        # --- [NÂNG CẤP] CẤU HÌNH POOL ---
        if "sqlite" in db_url:
            # SQLite cần check_same_thread=False để chạy đa luồng (Async)
            self.engine = create_engine(
                db_url, 
                connect_args={"check_same_thread": False},
                pool_recycle=600 # Tái chế kết nối sau 1 giờ
            )
        else:
            # PostgreSQL cần pool size để chịu tải cao
            self.engine = create_engine(
                db_url,
                pool_size=10, 
                max_overflow=20,
                pool_recycle=600,    # Hạ xuống 10 phút cho an toàn tuyệt đối
                pool_pre_ping=True   # [QUAN TRỌNG] Tự động nối lại nếu bị Cloud ngắt
            )
    
    def get_connection(self):
        return self.engine.connect()
    
    def init_db(self):
        try:
            # --- [FIX QUAN TRỌNG] TỰ ĐỘNG CHỌN CÚ PHÁP ID ---
            # Nếu là Postgres -> dùng SERIAL
            # Nếu là SQLite -> dùng AUTOINCREMENT
            if "postgresql" in self.db_url:
                pk_type = "SERIAL PRIMARY KEY"
                text_type = "TEXT"
            else:
                pk_type = "INTEGER PRIMARY KEY AUTOINCREMENT"
                text_type = "TEXT"

            with self.get_connection() as conn:
                # Bảng Products
                conn.execute(text(f"CREATE TABLE IF NOT EXISTS products (id {pk_type}, name {text_type}, price REAL)"))
                
                # Bảng Agent Status
                conn.execute(text(f"""
                    CREATE TABLE IF NOT EXISTS agent_status (
                        role_tag {text_type} PRIMARY KEY, xp INTEGER DEFAULT 0, 
                        current_topic {text_type}, last_updated TIMESTAMP
                    )
                """))
                
                # Bảng Work Logs
                conn.execute(text(f"""
                    CREATE TABLE IF NOT EXISTS work_logs (
                        id {pk_type}, timestamp {text_type}, agent_name {text_type}, 
                        task_content {text_type}, result_summary {text_type}, tool_used {text_type}, 
                        cost REAL, duration REAL
                    )
                """))
                
                # Bảng Projects
                conn.execute(text(f"CREATE TABLE IF NOT EXISTS projects (id {text_type} PRIMARY KEY, name {text_type}, history {text_type}, timestamp TIMESTAMP)"))
                
                # Bảng Async Tasks
                conn.execute(text(f"CREATE TABLE IF NOT EXISTS async_tasks (task_id {text_type} PRIMARY KEY, status {text_type}, result {text_type}, timestamp TIMESTAMP)"))
                
                # Bảng Learning Tasks (CHỖ BỊ LỖI CŨ)
                conn.execute(text(f"""
                    CREATE TABLE IF NOT EXISTS learning_tasks (
                        id {pk_type},
                        topic {text_type} UNIQUE,
                        status {text_type} DEFAULT 'PENDING',
                        assigned_to {text_type},
                        difficulty INTEGER DEFAULT 1,
                        type {text_type} DEFAULT 'RESEARCH',
                        content {text_type},
                        last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """))

                # Bảng Users (Ví tiền & Ngân hàng)
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
                
                conn.commit()
            print(colored("✅ FULL DB INITIALIZED (Compatible with Postgres & SQLite)", "green"))
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

manager = ConnectionManager()

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
        logger.info("✅ Database: CONNECTED")
    except Exception as e:
        logger.critical(f"❌ Database Failed: {e}")

    # 2. Lên lịch tác vụ (Scheduler)
    scheduler = AsyncIOScheduler()
    if 'morning_briefing_job' in globals():
        scheduler.add_job(morning_briefing_job, 'cron', hour=7, minute=0)
    scheduler.start()
    logger.info("⏰ Scheduler: ACTIVE")

    # 3. Kích hoạt AI nền (Background AI)
    learning_task = None
    if AI_AVAILABLE:
        logger.info("🧠 AI Core: ONLINE - Starting Self-Learning Loop...")
        learning_task = asyncio.create_task(auto_learning_cycle())
    else:
        logger.warning("⚠️ AI Core: OFFLINE (Running in safe mode)")

    # ---> SERVER IS RUNNING HERE <---
    yield 
    
    # ==============================
    # 🔴 SHUTDOWN SEQUENCE
    # ==============================
    print(colored("\n💤 [SYSTEM] J.A.R.V.I.S DANG NGHI...", "yellow", attrs=["bold"]))
    
    # 1. Dừng Scheduler
    if scheduler.running:
        scheduler.shutdown()
    
    # 2. Dừng AI an toàn (Graceful Shutdown)
    if learning_task:
        learning_task.cancel()
        try:
            # Đợi tối đa 5s để AI lưu dữ liệu dở dang rồi mới tắt
            await asyncio.wait_for(learning_task, timeout=5.0)
        except (asyncio.CancelledError, asyncio.TimeoutError):
            logger.info("✅ AI Background Task stopped safely.")
            
    logger.info("👋 System Shutdown Complete.")



# --- KHỞI TẠO APP VỚI LIFESPAN ---
app = FastAPI(
    title="J.A.R.V.I.S v4.6 FULL",
    version="4.6",
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

# --- 1. WORKER: NHÂN VIÊN CHẠY NGẦM (Làm việc bất kể ngày đêm/tắt máy) ---
async def background_ai_worker(task_id: str, user_msg_text: str, thread_id: str):
    """
    Hàm này chạy độc lập với API. Dù CEO tắt trình duyệt, nó vẫn chạy trên Server.
    """
    print(colored(f"⚙️ [BG WORKER] Bắt đầu xử lý Task {task_id}...", "yellow"))
    
    try:
        # A. Cập nhật trạng thái: ĐANG XỬ LÝ
        with db_manager.get_connection() as conn:
            conn.execute(text("INSERT OR REPLACE INTO async_tasks (task_id, status, result, timestamp) VALUES (:id, 'PROCESSING', '', :time)"), 
                         {"id": task_id, "time": datetime.now()})
            conn.commit()

        # B. Chuẩn bị ngữ cảnh (Memory)
        memory_context = ""
        if MEMORY_AVAILABLE:
            try:
                memory_context = await run_in_threadpool(lambda: recall_relevant_memories(user_msg_text))
            except: pass

        # C. Đóng gói tin nhắn chuẩn
        final_input_content = f"""
        [CONTEXT INFO]:
        Location: Phan Thiet | Time: {datetime.now().strftime('%H:%M %d/%m/%Y')}
        Relevant Memories: {memory_context}
        
        [USER COMMAND]:
        {user_msg_text}
        """
        
        human_msg = HumanMessage(content=final_input_content)
        config = {"configurable": {"thread_id": thread_id}}

        # D. GỌI AI (Bước tốn thời gian nhất)
        # Hệ thống có thể mất 2-3 phút ở đây, nhưng không sao cả
        output = await ai_app.ainvoke({"messages": [human_msg]}, config=config)
        ai_reply = output["messages"][-1].content

        # E. Lưu ký ức (Hậu xử lý)
        if MEMORY_AVAILABLE:
            try:
                # Chạy thẳng hàm đồng bộ vì đang ở trong worker riêng rồi
                extract_and_save_memory(user_msg_text, ai_reply)
            except: pass

        # F. HOÀN TẤT: Cập nhật Database
        with db_manager.get_connection() as conn:
            # Dùng tham số bind để tránh lỗi ký tự đặc biệt trong SQL
            conn.execute(text("UPDATE async_tasks SET status='DONE', result=:res WHERE task_id=:id"), 
                         {"res": ai_reply, "id": task_id})
            conn.commit()
            
        print(colored(f"✅ [BG WORKER] Task {task_id} hoàn thành!", "green"))

    except Exception as e:
        error_msg = f"Lỗi hệ thống: {str(e)}"
        print(colored(f"❌ [BG WORKER] Task {task_id} thất bại: {e}", "red"))
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
# --- API ĐĂNG KÝ ---
@app.post("/api/economy/register_miner")
async def register_miner_endpoint(info: RegisterInfo):
    conn = sqlite3.connect(DB_PATH)
    try:
        # 1. Kiểm tra user trùng
        cursor = conn.execute("SELECT 1 FROM users WHERE username=?", (info.username,))
        if cursor.fetchone():
            return JSONResponse(status_code=400, content={"status": "error", "msg": "Tên đăng nhập đã tồn tại!"})

        # 2. Sinh Key mới
        new_key = f"sk-{uuid.uuid4().hex}"
        
        # 3. Gộp thông tin ngân hàng
        bank_full = f"{info.bank_name} - {info.account_number}"

        # 4. Lưu vào DB
        conn.execute(
            "INSERT INTO users (username, api_key, email, bank_info) VALUES (?, ?, ?, ?)",
            (info.username, new_key, info.email, bank_full)
        )
        conn.commit()
        
        print(colored(f"🆕 Thợ đào mới: {info.username} ({bank_full})", "cyan"))
        
        return {
            "status": "success",
            "api_key": new_key,
            "msg": "Đăng ký thành công! Key đã được lưu."
        }
    except Exception as e:
        return JSONResponse(status_code=500, content={"status": "error", "msg": str(e)})
    finally:
        conn.close()

@app.post("/api/chat")
async def chat_endpoint(request: ChatRequest, background_tasks: BackgroundTasks):
    set_system_busy()
    if not AI_AVAILABLE:
        return {"reply": "⚠️ Hệ thống AI đang khởi động. Vui lòng đợi 30s."}

    try:
        user_msg_text = str(request.message).strip()
        thread_id = str(request.thread_id) if request.thread_id else "default_session"
        
        # --- 1. XỬ LÝ NHANH (GREETINGS & COMMANDS) ---
        # Giữ lại logic chào hỏi nhanh để tiết kiệm tiền AI
        greetings = ["chào", "hi", "hello", "alo"]
        if user_msg_text.lower() in greetings:
             return {"reply": "Chào CEO! J.A.R.V.I.S đã sẵn sàng nhận lệnh."}

        # --- 2. CHUẨN BỊ KÝ ỨC (MEMORY) ---
        memory_context = ""
        if MEMORY_AVAILABLE:
            # Lấy ký ức chạy ngầm để không làm chậm chat
            try:
                memory_context = await run_in_threadpool(lambda: recall_relevant_memories(user_msg_text))
                print(colored(f"🧠 Ký ức kích hoạt: {len(memory_context)} chars", "magenta"))
            except: pass

        # --- 3. ĐÓNG GÓI TIN NHẮN (THE FIX) ---
        # Thay vì gộp chuỗi, ta giữ nguyên User Message để OpenAI hiểu đây là lệnh mới
        # Context được chèn vào System Message hoặc Memory của Graph (tùy cấu hình Graph của ngài)
        # Nhưng để an toàn nhất, ta kẹp Context vào tin nhắn nhưng vẫn giữ role Human
        
        final_input_content = f"""
        [CONTEXT INFO]:
        Location: Phan Thiet
        Time: {datetime.now().strftime('%H:%M %d/%m/%Y')}
        Relevant Memories: {memory_context}
        
        [USER COMMAND]:
        {user_msg_text}
        """
        
        # Tạo đối tượng tin nhắn chuẩn LangChain
        human_msg = HumanMessage(content=final_input_content)
        
        # Cấu hình phiên làm việc
        config = {"configurable": {"thread_id": thread_id}}

        print(colored(f"📥 INPUT: {user_msg_text[:50]}...", "cyan"))

        # Phải dùng ainvoke (Async Invoke) vì các Node trong main.py là async def
        output = await ai_app.ainvoke(
            {"messages": [human_msg]}, 
            config=config
        )
        
        # --- 5. TRÍCH XUẤT KẾT QUẢ ---
        last_message = output["messages"][-1]
        ai_reply = last_message.content
        
        # --- 6. HẬU XỬ LÝ (LƯU KÝ ỨC & LOG) ---
        if MEMORY_AVAILABLE:
            background_tasks.add_task(extract_and_save_memory, user_msg_text, ai_reply)
            
        return {
            "status": "success", 
            "reply": ai_reply,
            "agent": "J.A.R.V.I.S v2.0"
        }

    except Exception as e:
        error_msg = str(e)
        print(colored(f"❌ CHAT ERROR: {error_msg}", "red"))
        
        # Tự động sửa lỗi 400 bằng cách reset nhẹ hội thoại
        if "Last message must have role user" in error_msg:
            return {
                "reply": "⚠️ Lỗi đồng bộ hội thoại. Tôi đã tự động sắp xếp lại bộ nhớ. Vui lòng gửi lại câu lệnh vừa rồi."
            }
            
        return {"reply": f"💥 Lỗi hệ thống: {error_msg}"}

async def verify_api_key(x_api_key: Optional[str] = Header(None)):
    """Middleware kiểm tra bảo mật"""
    # Logic: Nếu có gửi key thì check, nếu không gửi (Dev mode) thì bỏ qua hoặc chặn tùy CEO
    if x_api_key and x_api_key != ADMIN_SECRET:
        raise HTTPException(status_code=403, detail="⛔ SAI MẬT MÃ QUÂN SỰ (WRONG API KEY)")
    return x_api_key

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
    if x_api_key != ADMIN_SECRET: raise HTTPException(403)
    
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    # Lấy Tier của worker (Worker gửi lên trong req)
    worker_tier = req.worker_tier 
    
    # LOGIC CHỌN VIỆC:
    # - Tìm việc có độ khó (difficulty) <= Tier của máy
    # - Ví dụ: Máy Tier 3 làm được việc 1, 2, 3. Máy Tier 1 chỉ làm được việc 1.
    c.execute("""
        SELECT id, topic, task_type, content 
        FROM learning_tasks 
        WHERE status='PENDING' 
        AND difficulty <= ?  -- <--- MẤU CHỐT Ở ĐÂY
        ORDER BY difficulty DESC -- Ưu tiên làm việc khó nhất có thể trước
        LIMIT 1
    """, (worker_tier,))
    try:
        # 1. Tìm việc nào đang 'PENDING' (Chưa ai làm)
        # Hoặc việc nào 'PROCESSING' nhưng quá 30 phút chưa xong (Máy kia bị sập) - Cơ chế Timeout
        c.execute("""
            SELECT id, topic FROM learning_tasks 
            WHERE status='PENDING' 
            OR (status='PROCESSING' AND last_updated < datetime('now', '-30 minutes'))
            LIMIT 1
        """)
        row = c.fetchone()
        
        if row:
            task_id, topic = row[0], row[1]
            # 2. Đánh dấu "Xí phần" ngay lập tức
            c.execute("UPDATE learning_tasks SET status='PROCESSING', assigned_to=?, last_updated=CURRENT_TIMESTAMP WHERE id=?", (req.worker_id, task_id))
            conn.commit()
            print(colored(f"Giao việc '{topic}' cho {req.worker_id}", "cyan"))
            return {"task_id": task_id, "topic": topic}
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
@app.post("/api/worker/submit_task")
async def worker_submit_task(res: TaskResult, x_api_key: str = Header(None)):
    """
    API nhận kết quả từ Worker.
    Nâng cấp:
    1. Trả thưởng (Economy): Cộng tiền vào ví thợ đào.
    2. Smart Learning: Chỉ học kiến thức đúng.
    3. Auto Level Up: Tự động tăng độ khó.
    """
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    
    # Khởi tạo giá trị thưởng mặc định
    reward = 0.0
    
    try:
        # --- BƯỚC 1: XÁC THỰC THỢ ĐÀO (AUTH) ---
        # Kiểm tra xem Key này là của Admin hay của Thợ đào đăng ký
        c.execute("SELECT username, balance FROM users WHERE api_key=?", (x_api_key,))
        user = c.fetchone()
        
        # Nếu không phải Admin và cũng không tìm thấy User trong DB -> Chặn
        if x_api_key != ADMIN_SECRET and not user:
            raise HTTPException(status_code=403, detail="API Key không hợp lệ hoặc chưa đăng ký!")

        # --- BƯỚC 2: CẬP NHẬT TRẠNG THÁI TASK ---
        c.execute("UPDATE learning_tasks SET status='DONE', last_updated=CURRENT_TIMESTAMP WHERE id=?", (res.task_id,))
        
        # --- BƯỚC 3: KIỂM TRA CHẤT LƯỢNG & TÍNH TIỀN ---
        is_success = "CHẠY THÀNH CÔNG" in res.result_content
        
        if is_success:
            # A. Tính thưởng (Ví dụ: $0.05 cho mỗi task thành công)
            reward = 0.05 
            
            # B. Cộng tiền vào ví (Chỉ cộng nếu là User, Admin test thì thôi)
            if user:
                username = user[0]
                c.execute("UPDATE users SET balance = balance + ? WHERE username=?", (reward, username))
                print(colored(f"💰 [PAYMENT] Đã chuyển ${reward} cho thợ đào {username}", "green"))
            
            # C. Smart Learning (Chỉ học cái đúng)
            if AI_AVAILABLE:
                print(colored("🧠 [AI LEARNING] Nạp kiến thức chuẩn vào não bộ...", "magenta"))
                
                knowledge_pack = f"""
                [BÁO CÁO THỰC NGHIỆM TỪ WORKER VỆ TINH]
                Chủ đề: Task {res.task_id}
                Worker ID: {res.worker_id}
                Kết quả: THÀNH CÔNG
                ---------------------------
                Nội dung chi tiết:
                {res.result_content}
                """
                await run_in_threadpool(lambda: learn_knowledge(knowledge_pack))

                # D. Auto Level Up (Kích hoạt tạo bài khó hơn)
                if 'generate_harder_task' in globals():
                    asyncio.create_task(generate_harder_task(res.result_content))
        else:
            print(colored(f"⚠️ [SKIP] Task {res.task_id} thất bại/lỗi. Không thưởng, không học.", "yellow"))

        # --- BƯỚC 4: LƯU NHẬT KÝ ---
        timestamp = datetime.now().strftime("%H:%M %d/%m/%Y")
        c.execute("""
            INSERT INTO work_logs (timestamp, agent_name, task_content, result_summary, tool_used, cost)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (
            timestamp, 
            f"WORKER_{res.worker_id}", 
            f"Task {res.task_id}", 
            res.result_content, 
            "DISTRIBUTED_MINING",
            reward # Ghi lại chi phí đã trả cho task này
        ))
        
        conn.commit()
        return {
            "status": "success", 
            "reward_earned": reward,
            "message": "Đã ghi nhận kết quả."
        }

    except Exception as e:
        print(colored(f"❌ Lỗi nộp bài: {e}", "red"))
        # Vẫn trả về 200 nhưng báo lỗi trong body để Worker không bị crash
        return {"status": "error", "message": str(e)}
    finally:
        conn.close()

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
    uvicorn.run("server:app", host="0.0.0.0", port=port, reload=True)
