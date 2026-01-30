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
from sqlalchemy import create_engine, text
from typing import Optional, List, Dict, Any
from datetime import datetime
from contextlib import asynccontextmanager
from termcolor import colored
from gtts import gTTS
from apscheduler.schedulers.asyncio import AsyncIOScheduler
# --- CÀI ĐẶT THƯ VIỆN: pip install fastapi uvicorn python-multipart jinja2 aiofiles ---
from fastapi import FastAPI, HTTPException, Header, Depends, UploadFile, File, Request, status, WebSocket, WebSocketDisconnect, BackgroundTasks, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel
from langchain_core.messages import HumanMessage
from main import set_system_busy
# [QUAN TRỌNG]: Đã thêm LLM_SUPERVISOR và log_training_data
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("JARVIS_BACKEND")
# --- CẤU HÌNH HỆ THỐNG ---
ADMIN_SECRET = os.environ.get("ADMIN_SECRET", "ai_corp_secret_123")

# 1. Xác định đường dẫn gốc (Root Path)
# Kiểm tra xem thư mục /var/data (Mount path trên Render) có tồn tại không
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

# 3. Biến môi trường Database (Cập nhật lại cho SQLite nếu dùng Disk)
# Nếu không dùng PostgreSQL mà dùng SQLite trên Disk thì set lại url
if not os.environ.get("DATABASE_URL") and os.path.exists(RENDER_DISK_PATH):
    # Ép dùng SQLite trên ổ cứng Cloud để bền vững
    os.environ["DATABASE_URL"] = f"sqlite:///{DB_PATH}"

AI_AVAILABLE = False
MEMORY_AVAILABLE = False
VOICE_AVAILABLE = False
SERVER_READY = False

try:
    from main import (
        ai_app, 
        log_work_to_db,          # <--- Hàm chấm công chuẩn
        auto_learning_cycle,     # <--- Vòng lặp tự học mới
        morning_briefing_job,    # <--- Job đọc báo sáng mới
        vector_db, 
        LLM_GPT4, 
        LLM_PERPLEXITY, 
        LLM_GEMINI_LOGIC, 
        LLM_GEMINI_VISION,
        CODER_PRIMARY,
        ingest_docs_to_memory,
        learn_knowledge,
        set_system_busy
    )

    AI_AVAILABLE = True
    SERVER_READY = True
    logger.info("✅ CORE AI MODULES: LOADED")
except Exception as e:
    # --- BẮT LỖI VÀ GHI LẠI ---
    import traceback
    AI_BOOT_ERROR = traceback.format_exc() # Lưu toàn bộ dấu vết lỗi
    logger.error(f"⚠️ CORE AI FAILED TO LOAD: {AI_BOOT_ERROR}")
    
    # Set biến về None để không crash server
    AI_AVAILABLE = False
    ai_app = None
    vector_db = None
    LLM_GPT4 = None
    LLM_PERPLEXITY = None
    LLM_GEMINI_LOGIC = None
    LLM_GEMINI_VISION = None
    CODER_PRIMARY = None

# --- IMPORT MODULES NỘI BỘ KHÁC ---
try:
    from memory_core import recall_relevant_memories, extract_and_save_memory
    MEMORY_AVAILABLE = True
    logger.info("✅ MEMORY CORE: LOADED")
except ImportError:
    logger.warning("⚠️ memory_core.py not found. Memory features disabled.")
  
 
try:
    from voice_engine import client
    VOICE_AVAILABLE = True
except ImportError:
    VOICE_AVAILABLE = False
    client = None

# ==========================================
# 1. DATABASE MANAGER
# ==========================================
class DatabaseManager:
    def __init__(self):
        # 1. Lấy link DB (Ưu tiên từ biến môi trường, nếu không có thì dùng file Local)
        # Lưu ý: DB_PATH phải được định nghĩa ở trên đầu file server.py (vd: DB_PATH = "jarvis_memory.db")
        self.db_url = os.environ.get("DATABASE_URL")
        
        if self.db_url:
            # Fix lỗi tương thích: Render dùng 'postgres://' nhưng SQLAlchemy cần 'postgresql://'
            if self.db_url.startswith("postgres://"):
                self.db_url = self.db_url.replace("postgres://", "postgresql://", 1)
            
            # Tạo động cơ kết nối Cloud
            self.engine = create_engine(self.db_url)
            print(colored("🔌 KẾT NỐI DATABASE: CLOUD (POSTGRESQL)", "green"))
        else:
            # Tạo động cơ kết nối Local (SQLite) qua SQLAlchemy
            # Lưu ý: Dùng 3 dấu gạch chéo /// cho đường dẫn tương đối
            self.engine = create_engine(f"sqlite:///{DB_PATH}")
            print(colored("🔌 KẾT NỐI DATABASE: LOCAL (SQLITE)", "cyan"))

    def get_connection(self):
        """
        [FIX LỖI QUAN TRỌNG]
        Hàm này bây giờ trả về kết nối của SQLAlchemy chứ KHÔNG dùng sqlite3 trực tiếp nữa.
        """
        return self.engine.connect()
    
    def init_db(self):
        """Khởi tạo cấu trúc bảng & Dữ liệu mẫu (Chuẩn SQLAlchemy)"""
        try:
            with self.get_connection() as conn:
                # 1. TẠO CÁC BẢNG (Dùng cú pháp text() để an toàn)
                
                # Bảng Products (Sản phẩm bán)
                conn.execute(text("CREATE TABLE IF NOT EXISTS products (id INTEGER PRIMARY KEY, name TEXT, price REAL)"))
                
                # Bảng Finance Logs (Tổng quan thu chi)
                conn.execute(text("CREATE TABLE IF NOT EXISTS finance_logs (id INTEGER PRIMARY KEY, type TEXT, amount REAL)"))
                
                # Bảng Agent Status (Level và XP)
                conn.execute(text("""
                    CREATE TABLE IF NOT EXISTS agent_status (
                        role_tag TEXT PRIMARY KEY, 
                        xp INTEGER DEFAULT 0, 
                        current_topic TEXT, 
                        last_updated TIMESTAMP
                    )
                """))

                # --- [MỚI] BẢNG WORK LOGS (SỔ CÁI CHI TIẾT) ---
                # Đây là bảng quan trọng nhất để ngài soi chi phí và nội dung học
                conn.execute(text("""
                    CREATE TABLE IF NOT EXISTS work_logs (
                        id INTEGER PRIMARY KEY,  -- Tự tăng
                        timestamp TEXT,          -- Thời gian (Giờ/Ngày)
                        agent_name TEXT,         -- Tên nhân viên (Coder, Researcher...)
                        task_content TEXT,       -- Nội dung đề bài
                        result_summary TEXT,     -- Kết quả học được/làm được
                        tool_used TEXT,          -- Dùng súng gì (DeepSeek, GPT-4...)
                        cost REAL,               -- Tốn bao nhiêu tiền ($)
                        duration REAL            -- Mất bao nhiêu giây
                    )
                """))
                # Bảng nhật ký học tập (Meta-Cognition)
                conn.execute(text("""
                    CREATE TABLE IF NOT EXISTS learning_logs (
                        id INTEGER PRIMARY KEY,
                        event_type TEXT,
                        content TEXT,
                        agent_name TEXT,
                        timestamp TIMESTAMP
                    )
                """))
                # Bảng dự án (Lưu báo cáo)
                conn.execute(text("""
                    CREATE TABLE IF NOT EXISTS projects (
                        id TEXT PRIMARY KEY,
                        name TEXT,
                        history TEXT,
                        timestamp TIMESTAMP
                    )
                """))
                conn.commit()
                print(colored("✅ Database đã sẵn sàng.", "green"))

        except Exception as e:
            print(colored(f"❌ Lỗi khởi tạo DB: {e}", "red"))

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
    """
    Bước 1: Vẽ sơ đồ và kế hoạch thi công.
    Output: File BLUEPRINT.md chứa danh sách các bước (Steps).
    """
    print(colored(f"📐 [ARCHITECT] Đang phác thảo dự án: {project_request}", "cyan"))
    os.makedirs("projects", exist_ok=True)
    plan_path = f"projects/{thread_id}_BLUEPRINT.md"
    
    try:
        if not SERVER_READY: return "Simulation Plan", plan_path

        architect_prompt = (
            f"Bạn là Chief Software Architect (CSA). Có một yêu cầu dự án: '{project_request}'.\n"
            "Hãy lập một BẢN THIẾT KẾ KỸ THUẬT (Technical Blueprint) chi tiết dạng Markdown:\n\n"
            "1. [OVERVIEW]: Tóm tắt mục tiêu dự án.\n"
            "2. [MODULES]: Danh sách các chức năng chính.\n"
            "3. [DATABASE]: Sơ đồ bảng (Table Schema) chi tiết.\n"
            "4. [TECH STACK]: Công nghệ sử dụng.\n"
            "5. [EXECUTION PLAN] (QUAN TRỌNG): Hãy liệt kê lộ trình code cụ thể từng bước.\n"
            "   - Bắt buộc dùng gạch đầu dòng (-) cho mỗi bước.\n"
            "   - Ví dụ:\n"
            "   - Tạo môi trường ảo và file requirements.txt\n"
            "   - Thiết kế database models trong models.py\n"
            "   - Viết API đăng nhập\n"
        )
        
        plan_res = await run_in_threadpool(lambda: LLM_GEMINI_VISION.invoke(architect_prompt))
        content = plan_res.content
        
        async with aiofiles.open(plan_path, "w", encoding="utf-8") as f:
            await f.write(content)
            
        print(colored(f"✅ [ARCHITECT DONE] Bản vẽ đã xong: {plan_path}", "green"))
        return content, plan_path

    except Exception as e:
        print(colored(f"❌ Lỗi Architect: {e}", "red"))
        return None, None

async def run_coding_phase(blueprint_content: str, thread_id: str):
    """
    Bước 2: Đọc bản vẽ -> Code từng phần -> Ghi log.
    """
    print(colored(f"🏗️ [EXECUTOR] Bắt đầu thi công dự án {thread_id}...", "magenta"))
    log_file = f"projects/{thread_id}_coding_log.txt"
    
    raw_lines = blueprint_content.split('\n')
    steps = []
    is_in_plan = False
    
    # Parsing thông minh để tìm EXECUTION PLAN
    for line in raw_lines:
        if "EXECUTION PLAN" in line.upper(): is_in_plan = True
        if is_in_plan and (line.strip().startswith('-') or line.strip().startswith('*')):
            step_clean = line.strip().lstrip('-* ').strip()
            if len(step_clean) > 5:
                steps.append(step_clean)

    if not steps:
        print(colored("⚠️ Không tìm thấy bước code nào trong Blueprint. Dừng.", "yellow"))
        return

    async with aiofiles.open(log_file, "w", encoding="utf-8") as f:
        await f.write(f"=== BẮT ĐẦU DỰ ÁN {thread_id} ===\n\n")

    for idx, step in enumerate(steps):
        print(colored(f"⏳ [STEP {idx+1}/{len(steps)}]: {step}", "yellow"))
        
        step_prompt = (
            f"DỰ ÁN: {thread_id}\n"
            f"NHIỆM VỤ CỤ THỂ: {step}\n"
            "Yêu cầu: Viết code hoàn chỉnh cho nhiệm vụ này. Không giải thích dài dòng."
        )
        
        try:
            if SERVER_READY:
                state_res = await ai_app.ainvoke(
                    {"messages": [HumanMessage(content=step_prompt)]},
                    config={"configurable": {"thread_id": thread_id}}
                )
                ai_output = state_res['messages'][-1].content
            else:
                ai_output = f"[SIMULATION] Coding step {idx+1}..."
                await asyncio.sleep(1)

            async with aiofiles.open(log_file, "a", encoding="utf-8") as f:
                await f.write(f"\n\n{'='*30}\n### BƯỚC {idx+1}: {step}\n{'='*30}\n{ai_output}\n")
            
            await asyncio.sleep(2) # Nghỉ để tránh Rate Limit
            
        except Exception as e:
            print(colored(f"❌ Lỗi Step {idx+1}: {e}", "red"))

    print(colored(f"✅ [PROJECT COMPLETE] Dự án {thread_id} đã hoàn thành 100%!", "green"))

async def full_project_pipeline(user_request: str, thread_id: str):
    """
    Quy trình khép kín: Architect -> Blueprint -> Executor -> Code.
    """
    blueprint, path = await run_architect_phase(user_request, thread_id)
    if blueprint:
        await run_coding_phase(blueprint, thread_id)
    else:
        print("❌ Dự án bị hủy do lỗi thiết kế.")


# ==========================================
# 4. APP & ROUTES
# ==========================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    directories_to_create = [
        UPLOAD_DIR,      # /var/data/uploads
        PROJECTS_DIR,    # /var/data/projects
        "static",        # ./static (Code)
        "templates"      # ./templates (Code)
    ]
    
    for d in directories_to_create:
        if not os.path.exists(d): 
            os.makedirs(d)
            print(f"📁 Đã tạo thư mục: {d}")

    # 2. Khởi tạo Database
    db_manager.init_db()

      
    # Tạo thư mục cần thiết
    for d in [UPLOAD_DIR, "static", "templates", "projects"]:
        if not os.path.exists(d): os.makedirs(d)
        
    # 3. KÍCH HOẠT JOB BÁO CÁO SÁNG (Từ Main)
    scheduler = AsyncIOScheduler()
    scheduler.add_job(morning_briefing_job, 'cron', hour=7, minute=0)
    scheduler.start()
    
    # 4. KÍCH HOẠT CHẾ ĐỘ TỰ HỌC (Từ Main)
    print("🎓 [SYSTEM] Kích hoạt chế độ 'Adaptive Learning' (Học luân phiên)...")
    learning_task = asyncio.create_task(auto_learning_cycle())
    yield # Server chạy tại đây
    
    # --- SHUTDOWN ---
    scheduler.shutdown()
    print("💤 [SYSTEM] Đang giải tán lớp học...")
    learning_task.cancel()
    try: await learning_task
    except asyncio.CancelledError: pass
    logger.info("💤 SYSTEM SHUTDOWN.")
        
    

app = FastAPI(
    title="J.A.R.V.I.S Neural Backend",
    version="3.0",
    lifespan=lifespan
)

# 1. Cấu hình CORS (Cho phép mọi kết nối)
app.add_middleware(
    CORSMiddleware, 
    allow_origins=["*"], 
    allow_methods=["*"], 
    allow_headers=["*"]
)
# 2. Thiết lập đường dẫn tĩnh (Auto-Create Folder)
base_dir = os.path.abspath(os.path.dirname(__file__))
static_dir = os.path.join(base_dir, 'static')
templates_dir = os.path.join(base_dir, 'templates')

# --- QUAN TRỌNG: Tạo thư mục nếu chưa có (Fix lỗi Render) ---
if not os.path.exists(static_dir):
    os.makedirs(static_dir)
    print(colored("⚠️ Đã tự động tạo thư mục 'static'.", "yellow"))

if not os.path.exists(templates_dir):
    os.makedirs(templates_dir)
    print(colored("⚠️ Đã tự động tạo thư mục 'templates'.", "yellow"))

# 3. Mount Static & Templates
app.mount("/static", StaticFiles(directory=static_dir), name="static")
templates = Jinja2Templates(directory=templates_dir)

# --- DATA MODELS (Pydantic) ---
class ChatRequest(BaseModel):
    message: str
    thread_id: str = "ceo_session"

class SpeakRequest(BaseModel):
    text: str

class LearnRequest(BaseModel):
    text: str

class BuyRequest(BaseModel):
    product_id: int

class TTSRequest(BaseModel):
    text: str

# ==========================================
# 5. API ENDPOINTS
# ==========================================
async def verify_api_key(x_api_key: Optional[str] = Header(None)):
    """Middleware kiểm tra bảo mật"""
    # Logic: Nếu có gửi key thì check, nếu không gửi (Dev mode) thì bỏ qua hoặc chặn tùy CEO
    if x_api_key and x_api_key != ADMIN_SECRET:
        raise HTTPException(status_code=403, detail="⛔ SAI MẬT MÃ QUÂN SỰ (WRONG API KEY)")
    return x_api_key


@app.get("/admin")
async def admin_page(request: Request):
    # Truyền thêm biến api_key sang giao diện HTML
    return templates.TemplateResponse("admin.html", {
        "request": request, 
        "api_key": ADMIN_SECRET # <--- QUAN TRỌNG: Dòng này giúp hiển thị Key
    })


@app.get("/")
async def home_page(request: Request):
    # Nếu ngài có file index.html hoặc products.html thì để nguyên
    # Nếu muốn mặc định vào Dashboard thì đổi thành "dashboard.html"
    return templates.TemplateResponse("store.html", {"request": request}) 
    # Lưu ý: Đảm bảo file index.html này tồn tại trong thư mục templates

# 2. Trang Dashboard (Giao diện Chat & Vẽ tranh - J.A.R.V.I.S COMMAND CENTER)
@app.get("/dashboard")
async def dashboard_page(request: Request):
    return templates.TemplateResponse("dashboard.html", {"request": request})

@app.get("/index")
async def dashboard_page(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/api/agents")
async def get_agents_status():
    """
    API 1: Cung cấp dữ liệu cho cột TRÁI (Danh sách nhân viên)
    """
    try:
        with db_manager.get_connection() as conn:
            # Lấy thông tin từ bảng agent_status
            result = conn.execute(text("SELECT role_tag, xp, current_topic, last_updated FROM agent_status ORDER BY xp DESC"))
            agents = []
            for row in result:
                xp = row[1] if row[1] else 0
                level = int(xp / 100) + 1 # Công thức tính Level
                agents.append({
                    "role_tag": row[0],
                    "xp": xp,
                    "level": level,
                    "current_topic": row[2] or "Đang chờ lệnh",
                    "last_updated": str(row[3])
                })
            return agents
    except Exception as e:
        logger.error(f"Agents API Error: {e}")
        return []

@app.get("/api/costs")
async def get_costs_history():
    """
    API 2: Cung cấp dữ liệu cho cột PHẢI (Nhật ký làm việc & Tiền nong)
    """
    try:
        with db_manager.get_connection() as conn:
            # Lấy 50 dòng mới nhất từ Sổ Cái (work_logs)
            # Lưu ý: Cần khớp tên cột với lúc tạo bảng
            result = conn.execute(text("SELECT timestamp, agent_name, task_content, tool_used, cost, result_summary FROM work_logs ORDER BY id DESC LIMIT 50"))
            logs = []
            for row in result:
                logs.append({
                    "timestamp": row[0],
                    "agent": row[1],
                    "task": row[2],
                    "tool": row[3],
                    "cost_usd": row[4], # Dashboard JS tìm key 'cost_usd' này
                    "result": row[5]
                })
            return logs
    except Exception as e:
        logger.error(f"Costs API Error: {e}")
        return []

@app.get("/api/stats")
async def get_system_stats():
    """
    API 3: Tổng hợp tài chính (Cho trang Store/Main cũ)
    """
    try:
        with db_manager.get_connection() as conn:
            prod_count = conn.execute(text("SELECT count(*) FROM products")).fetchone()[0]
            
            # Tính tổng chi phí thực tế từ bảng work_logs
            # (Chính xác hơn cách tính nhân XP cũ)
            expense_query = conn.execute(text("SELECT SUM(cost) FROM work_logs"))
            total_expense = expense_query.fetchone()[0] or 0.0
            
            return {
                "products": prod_count,
                "revenue": 0,       # Chưa bán hàng
                "expense": round(total_expense, 4),
                "balance": round(0 - total_expense, 4)
            }
    except Exception as e:
        return {"products": 0, "revenue": 0, "expense": 0, "balance": 0}

@app.get("/api/products")
async def get_products_api():
    try:
        with db_manager.get_connection() as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute("SELECT * FROM products").fetchall()
            return [dict(r) for r in rows]
    except: return []

@app.post("/api/buy")
async def buy_product(req: BuyRequest):
    conn = sqlite3.connect(DB_PATH)
    try:
        product = conn.execute("SELECT price, name FROM products WHERE id=?", (req.product_id,)).fetchone()
        if not product:
            raise HTTPException(status_code=404, detail="Product not found")
            
        price, name = product[0], product[1]
        license_key = str(uuid.uuid4()).upper()[:19]
        
        # Xử lý tài chính (Dynamic Import để tránh lỗi vòng lặp)
        try:
            from finance_manager import process_order_revenue
            process_order_revenue(order_id=int(time.time()), total_amount=price)
        except ImportError:
            pass 
            
        return {
            "status": "success",
            "msg": f"Đã mua thành công: {name}",
            "license_key": license_key
        }
    finally:
        conn.close()

# --- API ĐỒNG BỘ DỮ LIỆU ---
@app.get("/api/sync/download_db")
async def download_database():
    if os.path.exists(DB_PATH):
        return FileResponse(path=DB_PATH, filename="ai_corp_data.db", media_type='application/octet-stream')
    return {"error": "Database not found"}


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

@app.post("/api/speak")
async def api_speak(request: SpeakRequest):
    """
    API Tạo giọng nói (Đã tối ưu hóa Non-blocking & Fail-safe).
    """
    # 1. Kiểm tra an toàn: Nếu module voice chưa load hoặc client chưa có -> Bỏ qua nhẹ nhàng
    if not VOICE_AVAILABLE or 'client' not in globals() or client is None:
        # Trả về 204 (No Content) để Dashboard biết mà im lặng, không báo lỗi đỏ
        return Response(status_code=204)
    
    try:
        # 2. Tối ưu chi phí & Tốc độ: Chỉ đọc 500 ký tự đầu
        # (J.A.R.V.I.S không nên đọc cả bài văn dài, tốn tiền và lâu)
        safe_text = request.text[:1000] 

        # 3. Kỹ thuật Non-blocking (QUAN TRỌNG NHẤT)
        # Đẩy việc gọi OpenAI sang luồng khác để Server vẫn nhận chat của người khác được
        def _generate_audio():
            return client.audio.speech.create(
                model="tts-1",
                voice="onyx", 
                input=safe_text
            )
        
        # Dùng await để đợi luồng phụ xử lý xong
        response = await run_in_threadpool(_generate_audio)
        return Response(content=response.content, media_type="audio/mpeg")

    except Exception as e:
        logger.error(f"🚨 [VOICE ERROR]: {str(e)}")
        # Nếu lỗi (hết tiền, mất mạng...), trả về 204 để Dashboard vẫn chạy tiếp mượt mà
        return Response(status_code=204)

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

# --- CẤU HÌNH HỌC TẬP ---
LEARNING_QUEUE = ["CODER", "ARTIST", "ENGINEERING", "MARKETING", "LEGAL"]
CURRENT_LEARNER_INDEX = 0
IS_BUSY = False  # Trạng thái bận rộn của hệ thống
LAST_ACTIVITY_TIME = datetime.now()

# --- TÍCH HỢP VÀO STARTUP ---
@app.on_event("startup")
async def start_scheduler():
    # Chạy loop này ở chế độ nền (không chặn API)
    asyncio.create_task(auto_learning_cycle())


@app.post("/api/learn")
async def api_learn(request: LearnRequest, x_api_key: str = Header(None)):
    if x_api_key != ADMIN_SECRET: raise HTTPException(403)
    if not AI_AVAILABLE: return {"status": "error", "message": "AI Offline"}
    
    res = learn_knowledge(request.text)
    return {"status": "success", "message": res}

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
    Hàm đọc báo cáo mới nhất trong thư mục projects
    """
    try:
        # 1. Trỏ đúng vào thư mục 'projects'
        # Tìm tất cả file .md (Bao gồm cả Project_Audit và Morning_Briefing)
        search_path = os.path.join("projects", "*.md") 
        list_of_files = glob.glob(search_path)
        
        if not list_of_files:
            return "Thưa CEO, kho dữ liệu (folder projects) hiện đang trống. Chưa có báo cáo nào được tạo."
            
        # 2. Tìm file mới nhất dựa trên thời gian tạo (Create Time)
        latest_file = max(list_of_files, key=os.path.getctime)
        
        # Lấy tên file cho đẹp
        filename = os.path.basename(latest_file)
        
        # 3. Đọc nội dung
        with open(latest_file, "r", encoding="utf-8") as f:
            content = f.read()
            
        return f"### 📂 HỒ SƠ MỚI NHẤT: {filename}\n\n{content}"

    except Exception as e:
        logger.error(f"🚨 [REPORT ERROR]: {str(e)}")
        return f"⚠️ Thưa CEO, không thể truy xuất hồ sơ: {str(e)}."

@app.get("/api/costs")
async def get_costs_api():
    """API lấy dữ liệu báo cáo tài chính (Đã đồng bộ đường dẫn DB)"""
    try:
        # 1. ÉP CỨNG ĐƯỜNG DẪN DB CLOUD (Giống main.py)
        if os.path.exists("/var/data"):
            db_path = "/var/data/ai_corp_projects.db"
        else:
            db_path = "ai_corp_projects.db"
            
        # Kiểm tra file có tồn tại không
        if not os.path.exists(db_path):
            return [] # Trả về mảng rỗng nếu chưa có DB

        # 2. KẾT NỐI TRỰC TIẾP (Bỏ qua db_manager để tránh cache cũ)
        # Dùng check_same_thread=False để tránh lỗi luồng
        conn = sqlite3.connect(db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row # Để lấy dữ liệu dạng Dictionary
        c = conn.cursor()
        
        # 3. TRUY VẤN DỮ LIỆU
        try:
            # Lấy 50 dòng mới nhất
            c.execute("""
                SELECT timestamp, agent_name, task_content, tool_used, cost, result_summary 
                FROM work_logs 
                ORDER BY id DESC LIMIT 50
            """)
            rows = c.fetchall()
            
            logs = []
            for row in rows:
                logs.append({
                    "timestamp": row["timestamp"],
                    "agent": row["agent_name"], # Map đúng tên cột
                    "task": row["task_content"],
                    "tool": row["tool_used"],
                    "cost_usd": row["cost"] if row["cost"] else 0.0, # Map sang cost_usd cho Frontend
                    "result": row["result_summary"]
                })
                
            conn.close()
            return logs
            
        except sqlite3.OperationalError:
            # Nếu bảng chưa có -> Trả về rỗng
            conn.close()
            return []

    except Exception as e:
        print(f"Lỗi API Costs: {e}")
        return []

# --- ENTRY POINT (CHẠY SERVER) ---
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
