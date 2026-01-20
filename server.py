import glob
import os
import pandas as pd
import sqlite3
import uuid
import time
import io  # <--- THÊM CÁI NÀY
import shutil
import random
import logging
import aiofiles
import json  # <--- ĐÃ THÊM
import base64
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
# [QUAN TRỌNG]: Đã thêm LLM_SUPERVISOR và log_training_data
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("JARVIS_BACKEND")
# --- CẤU HÌNH HỆ THỐNG ---
ADMIN_SECRET = os.environ.get("ADMIN_SECRET", "ai_corp_secret_123")
UPLOAD_DIR = "uploads"
DB_PATH = "ai_corp_projects.db"

AI_AVAILABLE = False
MEMORY_AVAILABLE = False
VOICE_AVAILABLE = False

try:
    from main import (
        ai_app,                 # Bộ não LangGraph (Graph đã compile)
        log_training_data,      # Hàm tự học
        learn_knowledge,        # Hàm học kiến thức mới
        ingest_docs_to_memory,  # Hàm đọc PDF
        vector_db,              #Database Vector (Cho Cronjob)
        LLM_GPT4,               # Model GPT-4
        LLM_PERPLEXITY,         # Model Search
        LLM_GEMINI,             # Model Google
        LLM_SUPERVISOR,          # [MỚI] Tổng quản để chia việc dự án lớn
        CODER_PRIMARY
    
    ) 

    AI_AVAILABLE = True
    logger.info("✅ CORE AI MODULES: LOADED")
except ImportError as e:
    logger.error(f"⚠️ CORE AI FAILED TO LOAD: {e}")
    ai_app = None
    vector_db = None
    LLM_GPT4 = None
    LLM_PERPLEXITY = None
    LLM_GEMINI = None
    LLM_SUPERVISOR = None
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
    logger.warning("⚠️ Voice Engine not found.")
    client = None

CURRICULUM = {
    "[FINANCE]": [
        "Phân tích xu hướng giá vàng SJC và thế giới hôm nay",
        "Dự báo tỷ giá USD/VND tuần này",
        "Biến động thị trường Crypto (Bitcoin/ETH) 24h qua",
        "Chỉ số VN-Index và tác động vĩ mô"
    ],
    "[CODER]": [
        "Top Python libraries for AI Agents ",
        "FastAPI advanced patterns and performance tuning",
        "LangChain vs LangGraph architecture comparison",
        "Optimizing Docker containers for Python apps"
    ],
    "[MARKETING]": [
        "Xu hướng TikTok viral tại Việt Nam tuần này",
        "Chiến lược SEO mới nhất của Google Update",
        "Content marketing trends for Tech products 2026",
        "Phân tích quảng cáo Facebook hiệu quả ngành công nghệ"
    ],
    "[LEGAL]": [
        "Luật Giao dịch điện tử mới nhất tại Việt Nam",
        "Quy định về bảo vệ dữ liệu cá nhân (Nghị định 13)",
        "Bản quyền trong kỷ nguyên AI (Intellectual Property & AI)"
    ],
    "[HARDWARE]": [
        "ESP32-S3 pinout and datasheet updates",
        "Các loại cảm biến IoT giá rẻ mới nhất trên thị trường",
        "Kỹ thuật thiết kế mạch PCB chống nhiễu (Anti-interference)"
    ],
    "[ARTIST]": [
        "Phong cách vẽ Digital Art đương đại",
        "Xu hướng màu sắc (Color Trends) năm 2026",
        "Kỹ thuật Prompting cho DALL-E 3 và Midjourney"
    ],
    "[IOT]": [
        "Giao thức MQTT và bảo mật thiết bị IoT",
        "Nhà thông minh (Smart Home) integration trends",
        "Zigbee vs WiFi vs LoRaWAN comparison"
    ]
}

# ==========================================
# 2. CLASS QUẢN LÝ KẾT NỐI (WEBSOCKET)
# ==========================================
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
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

class DatabaseManager:
    """Quản lý kết nối Database cho Server"""
    def __init__(self, db_path=DB_PATH):
        self.db_path = db_path

    def get_connection(self):
        """Tạo kết nối cho lệnh 'with'"""
        return sqlite3.connect(self.db_path, check_same_thread=False)

# Khởi tạo biến toàn cục (Sửa lỗi gạch chân chữ 'db_manager')
db_manager = DatabaseManager()
# ==========================================
# 3. KHỞI TẠO APP & DATABASE
# ==========================================

def init_db():
    conn = sqlite3.connect(DB_PATH)
    
    # 1. Các bảng cũ (Giữ nguyên)
    conn.execute("CREATE TABLE IF NOT EXISTS products (id INTEGER PRIMARY KEY, name TEXT, price REAL)")
    conn.execute("CREATE TABLE IF NOT EXISTS finance_logs (id INTEGER PRIMARY KEY, type TEXT, amount REAL)")
    
    # 2. --- BẢNG MỚI: TRẠNG THÁI NHÂN SỰ AI (AGENT STATUS) ---
    # role_tag: Mã định danh (VD: [ARTIST], [CODER]...) -> Làm khóa chính (Primary Key)
    # xp: Điểm kinh nghiệm tích lũy (Mặc định là 0)
    # current_topic: Chủ đề vừa học gần nhất
    # last_updated: Thời gian cập nhật
    conn.execute('''CREATE TABLE IF NOT EXISTS agent_status 
                    (role_tag TEXT PRIMARY KEY, 
                     xp INTEGER DEFAULT 0, 
                     current_topic TEXT, 
                     last_updated DATETIME)''')
    
    conn.commit() # Lưu các thay đổi cấu trúc bảng
    conn.close()

async def morning_briefing_job():
    """
    J.A.R.V.I.S tự động thức dậy lúc 7:00 sáng để học tin tức.
    """
    print(colored("\n⏰ [CRON JOB] Đang thực hiện quét tin tức buổi sáng...", "cyan", attrs=["bold"]))
    
    if not AI_AVAILABLE or not LLM_PERPLEXITY:
        print(colored("⚠️ Bỏ qua Cron Job vì AI Module chưa sẵn sàng.", "yellow"))
        return

    # 1. Các chủ đề cần tự học (CEO có thể tùy chỉnh)
    topics = [
        "Những xu hướng công nghệ mới nhất trong 24h qua",
        "Biến động thị trường tài chính, chứng khoán và crypto hôm nay",
        "Các thư viện Python mới nổi tuần này"
        
    ]
    
    report_buffer = []
    
    for topic in topics:
        try:
            # Tìm kiếm thông tin mới nhất
            print(colored(f"--> Đang tìm hiểu: {topic}...", "white"))
            res = await LLM_PERPLEXITY.ainvoke(topic)
            content = res.content
            
            # Tự động ghi nhớ vào đầu não
            if MEMORY_AVAILABLE:
                # Lưu vào ChromaDB với nhãn "Auto-Learning"
                await run_in_threadpool(lambda: vector_db.add_texts(
                    texts=[content],
                    metadatas=[{
                        "source": "Morning_Briefing", 
                        "topic": topic,
                        "timestamp": datetime.now().isoformat()
                    }]
                ))
            
            report_buffer.append(f"### {topic}\n{content[:500]}...") # Lưu tóm tắt
            
        except Exception as e:
            print(colored(f"❌ Lỗi tự học chủ đề '{topic}': {e}", "red"))

    # 2. Tạo báo cáo tóm tắt để CEO đọc khi thức dậy
    final_report = "\n".join(report_buffer)
    today = datetime.now().strftime("%Y-%m-%d")
    
    # Lưu thành file Markdown trong thư mục projects
    report_path = f"projects/Morning_Briefing_{today}.md"
    async with aiofiles.open(report_path, "w", encoding="utf-8") as f:
        await f.write(f"# 🌅 BẢN TIN SÁNG {today}\n\n{final_report}")
        
    print(colored(f"✅ [DONE] Đã hoàn thành tự học và lưu báo cáo tại: {report_path}", "green"))
# --- API MỚI: CHO PHÉP TẢI DỮ LIỆU VỀ MÁY ---


@asynccontextmanager
async def lifespan(app: FastAPI):
    # --- PHẦN STARTUP (Chạy khi Server bật) ---
    init_db()
    for d in [UPLOAD_DIR, "static", "templates","projects"]:
        if not os.path.exists(d): os.makedirs(d)
    logger.info(f"✅ J.A.R.V.I.S SYSTEM ONLINE. Database connected at {DB_PATH}")
    
    # --- KÍCH HOẠT LỊCH TRÌNH TỰ ĐỘNG ---
    scheduler = AsyncIOScheduler()
    # Chạy mỗi ngày vào lúc 7:00 sáng
    scheduler.add_job(morning_briefing_job, 'cron', hour=7, minute=0)
    # Hoặc chạy thử nghiệm: mỗi 1 tiếng chạy 1 lần
    # scheduler.add_job(morning_briefing_job, 'interval', hours=1) 
    
    scheduler.start()
    logger.info("⏰ SCHEDULER ACTIVATED: Chế độ tự học đã bật.")
    yield # Điểm phân cách giữa Bật và Tắt
    
    # --- PHẦN SHUTDOWN (Chạy khi Server tắt - Nếu cần dọn dẹp) ---
    scheduler.shutdown()
    logger.info("💤 J.A.R.V.I.S SYSTEM SHUTTING DOWN...")

# --- KHỞI TẠO APP ---
app = FastAPI(
    title="J.A.R.V.I.S Neural Backend",
    description="Hệ điều hành AI Corporation - Enterprise Edition",
    version="3.0", # Version updated
    lifespan=lifespan
)

# Cấu hình CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- THIẾT LẬP THƯ MỤC TĨNH ---
base_dir = os.path.abspath(os.path.dirname(__file__))
static_dir = os.path.join(base_dir, 'static')
templates_dir = os.path.join(base_dir, 'templates')

# Tự động tạo thư mục nếu thiếu (Tránh lỗi Crash)
for d in [UPLOAD_DIR, static_dir, templates_dir]:
    if not os.path.exists(d):
        os.makedirs(d)

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

# --- DEPENDENCIES ---
async def verify_api_key(x_api_key: Optional[str] = Header(None)):
    """Middleware kiểm tra bảo mật"""
    # Logic: Nếu có gửi key thì check, nếu không gửi (Dev mode) thì bỏ qua hoặc chặn tùy CEO
    if x_api_key and x_api_key != ADMIN_SECRET:
        raise HTTPException(status_code=403, detail="⛔ SAI MẬT MÃ QUÂN SỰ (WRONG API KEY)")
    return x_api_key

async def heavy_project_executor(project_request: str, thread_id: str):
    """
    Hàm xử lý dự án lớn (ERP, CRM) chạy nền hàng giờ đồng hồ.
    """
    import asyncio
    print(colored(f"🏗️ [HEAVY PROJECT] Bắt đầu: {project_request}", "magenta", attrs=["bold"]))
    
    log_file = f"projects/{thread_id}_log.txt"
    blueprint_path = f"projects/{thread_id}_BLUEPRINT.md"

    try:
        if not LLM_SUPERVISOR:
             raise Exception("LLM_SUPERVISOR chưa được khởi tạo. Kiểm tra lại main.py")
        # Giai đoạn 1: Lập kế hoạch (Dùng Supervisor)
        plan_prompt = f"Là kiến trúc sư phần mềm, hãy chia dự án '{project_request}' thành danh sách các bước (modules) kỹ thuật cụ thể để code lần lượt. Trả về dạng gạch đầu dòng."
        # Lưu ý: Cần import LLM_SUPERVISOR từ main
        plan_res = await run_in_threadpool(lambda: LLM_SUPERVISOR.invoke(plan_prompt))
        steps = [s.strip() for s in plan_res.content.split('\n') if '-' in s or '*' in s]
        
        with open(log_file, "w", encoding="utf-8") as f:
            f.write(f"PROJECT PLAN:\n{plan_res.content}\n{'='*50}\n")
            
        # Giai đoạn 2: Code từng phần (Loop)
        for idx, step in enumerate(steps):
            print(colored(f"⏳ Doing Step {idx+1}/{len(steps)}: {step}", "yellow"))
            
            # Gọi AI Brain (ai_app) để thực hiện bước này
            # Truyền ngữ cảnh là các file đã làm xong (đọc từ log hoặc vector db nếu cần)
            step_input = f"Thực hiện bước {idx+1}: {step}. Hãy viết code hoàn chỉnh cho module này."
            
            # Gọi vào LangGraph
            state_res = await ai_app.ainvoke(
                {"messages": [HumanMessage(content=step_input)]},
                config={"configurable": {"thread_id": thread_id}}
            )
            
            ai_output = state_res['messages'][-1].content
            
            # Lưu kết quả vào file
            with open(log_file, "a", encoding="utf-8") as f:
                f.write(f"\n\n--- STEP {idx+1}: {step} ---\n{ai_output}\n")
            
            # Nghỉ 2 giây để tránh spam API
            await asyncio.sleep(2)

        print(colored(f"✅ [DONE] Dự án {thread_id} đã hoàn tất!", "green"))

    except Exception as e:
        print(colored(f"❌ [FAILED] Dự án bị lỗi: {e}", "red"))
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(f"\n❌ ERROR: {str(e)}")

# 2. API Endpoint để kích hoạt
@app.post("/api/heavy_project")
async def start_heavy_project(request: ChatRequest, background_tasks: BackgroundTasks):
    """
    API để CEO kích hoạt chế độ làm dự án lớn.
    """
    # Tạo Thread ID riêng cho dự án nếu chưa có
    pid = request.thread_id or f"proj_{int(time.time())}"
    
    # Đẩy vào chạy nền (Fire and Forget)
    background_tasks.add_task(heavy_project_executor, request.message, pid)
    
    return {
        "status": "PROCESSING",
        "project_id": pid,
        "message": "Đã tiếp nhận dự án ERP. Hệ thống sẽ chạy ngầm.",
        "log_path": f"projects/{pid}_log.txt"
    }

@app.get("/api/sync/download_db")
async def download_database():
    db_path = "ai_corp_projects.db" # Tên file Database của ngài
    if os.path.exists(db_path):
        # Trả về file cho Client tải
        return FileResponse(path=db_path, filename="ai_corp_projects_cloud.db", media_type='application/octet-stream')
    return {"error": "Chưa có dữ liệu nào được tạo ra."}

# ==========================================
# 🚀 SYSTEM ROUTES
# ==========================================

@app.get("/health")
async def health_check():
    """
    Kiểm tra tình trạng sức khỏe toàn diện của hệ thống.
    """
    # Kiểm tra kết nối Database vật lý
    db_status = "CONNECTED" if os.path.exists(DB_PATH) else "MISSING"
    
    return {
        "status": "OPERATIONAL",
        "timestamp": datetime.now().isoformat(),
        "modules": {
            "ai_brain": "ONLINE" if AI_AVAILABLE else "OFFLINE",
            "voice_core": "ONLINE" if VOICE_AVAILABLE else "OFFLINE",
            "memory_core": "ONLINE" if MEMORY_AVAILABLE else "OFFLINE",
            "knowledge_db": db_status,
        }
    }

@app.get("/api/stats")
async def get_system_stats():
    conn = sqlite3.connect(DB_PATH)
    try:
        cursor = conn.cursor()
        # Dùng try-except để tránh lỗi nếu bảng chưa tồn tại
        try:
            prod_count = cursor.execute("SELECT count(*) FROM products").fetchone()[0]
            income = cursor.execute("SELECT SUM(amount) FROM finance_logs WHERE type='income'").fetchone()[0] or 0
            expense = cursor.execute("SELECT SUM(amount) FROM finance_logs WHERE type='expense'").fetchone()[0] or 0
        except sqlite3.OperationalError:
            return {"products": 0, "revenue": 0, "expense": 0, "balance": 0}
        return {
            "products": prod_count,
            "balance": income - expense,
            "revenue": income,
            "expense": expense
        }
    finally:
        conn.close()

# ==========================================
# 🤖 AI ROUTES (ASYNC MODE)
# ==========================================
def get_latest_audit_report():
    try:
        list_of_files = glob.glob('Project_Audit_*.md')
        if not list_of_files:
            return "Thưa CEO, hiện tại hệ thống chưa ghi nhận báo cáo nào. Ngài có muốn khởi động một dự án mới?"
        latest_file = max(list_of_files, key=os.path.getctime)
        with open(latest_file, "r", encoding="utf-8") as f:
            content = f.read()
        # Trả về định dạng tóm tắt để Gemini/J.A.R.V.I.S dễ "đọc"
        return f"### 📊 BÁO CÁO CẬP NHẬT TỪ {latest_file}\n\n{content}"

    except Exception as e:
        logger.error(f"🚨 [REPORT ERROR]: {str(e)}")
        # Trả về chuỗi thông báo lỗi thay vì Dict
        return f"⚠️ Thưa CEO, tôi gặp khó khăn khi truy xuất hồ sơ: {str(e)}. Có vẻ như tệp tin đang bị khóa hoặc đã bị di chuyển."
# --- CẬP NHẬT HÀM CHAT ---

@app.post("/api/chat")
async def chat(request: ChatRequest, background_tasks: BackgroundTasks):
    """
    SMART CHAT V3: MEMORY INTEGRATED + OPTIMIZED ROUTING
    """
    if not AI_AVAILABLE:
        return {"reply": "⚠️ Hệ thống AI đang khởi động lại. Vui lòng đợi."}
    
    try:
        user_msg = str(request.message).strip()
        user_msg_lower = user_msg.lower()
        
        # --- 1. INTERCEPTOR (Xã giao & Báo cáo - Giữ nguyên cho nhanh) ---
        greetings = ["chào", "hi", "hello", "alo", "có đó không"]
        if any(k == user_msg_lower for k in greetings) and len(user_msg.split()) < 5:
            hour = datetime.now().hour
            time_greet = "buổi sáng" if 5 <= hour < 12 else "buổi chiều" if 12 <= hour < 18 else "buổi tối"
            return {"reply": f"Chào CEO! Chúc ngài một {time_greet} tốt lành. Tôi đang chờ lệnh."}

        if any(k in user_msg_lower for k in ["tổng kết", "báo cáo audit", "kiểm toán"]):
            return {"reply": get_latest_audit_report()}

        # --- 2. MEMORY RECALL (HỒI TƯỞNG KÝ ỨC) ---
        memory_context = ""
        if MEMORY_AVAILABLE:
            print(colored(f"🧠 Đang lục lọi ký ức cho: '{user_msg}'...", "magenta"))
            # Chạy trong threadpool để không chặn
            memory_context = await run_in_threadpool(lambda: recall_relevant_memories(user_msg))
        # --- 3. FAST TRACK (Hỏi nhanh đáp gọn) ---
        fast_keywords = ["giá vàng", "thời tiết", "mấy giờ", "ngày mấy", "tỷ giá", "kết quả", "bóng đá", "ai là", "dân số", "giá coin"]
        is_fast_query = any(k in user_msg_lower for k in fast_keywords)

        if is_fast_query:
            # Ưu tiên 1: Perplexity (Nếu hỏi tin tức/dữ liệu realtime)
            if LLM_PERPLEXITY:
                try:
                    fast_response = await LLM_PERPLEXITY.ainvoke(user_msg)
                    return {"reply": fast_response.content, "agent": "⚡ Perplexity Search", "timestamp": datetime.now().isoformat()}
                except: pass 
            
            # Ưu tiên 2: Gemini (Nếu cần tốc độ suy luận nhanh)
            if LLM_GEMINI:
                try:
                    # Inject Memory nhẹ vào Fast Track để AI thông minh hơn (VD: Thời tiết -> nhớ vị trí Phan Thiết)
                    fast_prompt = f"Thông tin bổ trợ (Ký ức): {memory_context}\nCâu hỏi: {user_msg}"
                    direct_response = await LLM_GEMINI.ainvoke([
                        {"role": "system", "content": "Bạn là AI Search Engine. Trả lời Ngắn gọn, Chính xác. Không phân tích dài dòng."},
                        {"role": "user", "content": fast_prompt}
                    ])
                    return {"reply": direct_response.content, "agent": "⚡ Gemini Speed", "timestamp": datetime.now().isoformat()}
                except: pass

        # --- 4. DEEP THINKING (Gọi LangGraph - Bộ não chính) ---
        # Bơm Ký ức (Memory) vào ngữ cảnh hệ thống
        current_context = f"[SYSTEM INFO: Time={datetime.now().strftime('%H:%M')}, Location=Phan Thiet]"
        full_prompt = (
            f"{current_context}\n"
            f"[ACTIVE MEMORY - KÝ ỨC LIÊN QUAN]:\n{memory_context}\n\n"
            f"[USER REQUEST]: {user_msg}"
        )
        
        from langchain_core.messages import HumanMessage
        config = {"configurable": {"thread_id": request.thread_id}}
        
        # Gọi LangGraph xử lý
        final_state = await ai_app.ainvoke(
            {"messages": [HumanMessage(content=full_prompt)]}, 
            config=config
        )
        
        last_message = final_state['messages'][-1]
        ai_reply = last_message.content if hasattr(last_message, 'content') else str(last_message)
        current_agent = final_state.get("current_agent", "J.A.R.V.I.S")

        # --- 5. MEMORY SAVE (BACKGROUND TASK) ---
        # Lưu ký ức CHỦ ĐỘNG mà không bắt CEO phải chờ
        if MEMORY_AVAILABLE:
            background_tasks.add_task(extract_and_save_memory, user_msg, ai_reply)
        background_tasks.add_task(log_training_data, user_msg, ai_reply, success=True)
        
        return {
            "reply": ai_reply,
            "agent": current_agent,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Chat Error: {e}")
        return {"reply": f"💥 Lỗi xử lý logic: {str(e)}"}
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
    API TTS: SỬ DỤNG GOOGLE TRANSLATE (MIỄN PHÍ)
    """
    try:
        # 1. Lấy văn bản (Cắt ngắn để tránh lỗi Google nếu quá dài)
        speak_text = request.text[:500]
        logger.info(f"🤖 Google TTS Request: {speak_text[:50]}...")

        # 2. Hàm tạo âm thanh Google (Chạy trong luồng riêng để không chặn Server)
        def _generate_google_audio():
            # lang='vi': Tiếng Việt
            # tld='com.vn': Giọng Việt Nam chuẩn hơn
            tts = gTTS(text=speak_text, lang='vi', tld='com.vn')
            
            # Lưu vào bộ nhớ đệm (RAM) thay vì ổ cứng -> Nhanh hơn
            buffer = io.BytesIO()
            tts.write_to_fp(buffer)
            buffer.seek(0)
            return buffer.read()

        # 3. Thực thi
        audio_content = await run_in_threadpool(_generate_google_audio)
        
        # 4. Trả về file âm thanh
        return Response(content=audio_content, media_type="audio/mpeg")

    except Exception as e:
        logger.error(f"🚨 [GOOGLE TTS ERROR]: {str(e)}")
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.post("/api/voice_chat")
async def voice_chat(file: UploadFile = File(...), api_key: str = Depends(verify_api_key)):
    """
    TƯƠNG TÁC BẰNG GIỌNG NÓI (Voice-to-Voice)
    1. Nghe (Whisper) -> 2. Hiểu & Làm (Smart Chat) -> 3. Nói lại (TTS)
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

        # 2. DỊCH GIỌNG NÓI SANG CHỮ (WHISPER - TAI NGHE)
        # Chạy trong luồng phụ để không treo server
        def _transcribe():
            with open(temp_path, "rb") as audio_file:
                return client.audio.transcriptions.create(
                    model="whisper-1", 
                    file=audio_file,
                    language="vi" # Ưu tiên tiếng Việt
                )
        
        transcript = await run_in_threadpool(_transcribe)
        user_text = transcript.text
        print(f"🎤 [VOICE INPUT]: {user_text}")
        
        # 2. Smart Chat
        chat_req = ChatRequest(message=user_text, thread_id="voice_session")
        
        # --- Logic Chat tối giản cho Voice (Để tránh lỗi BackgroundTasks) ---
        memory_context = await run_in_threadpool(lambda: recall_relevant_memories(user_text)) if MEMORY_AVAILABLE else ""
        fast_keywords = ["giá vàng", "thời tiết", "mấy giờ", "ngày mấy"]
        if any(k in user_text.lower() for k in fast_keywords) and LLM_GEMINI:
             ai_res = await LLM_GEMINI.ainvoke(f"Ký ức: {memory_context}. Hỏi: {user_text}")
             ai_text = ai_res.content
             agent_name = "Gemini Voice"
        else:
             # Deep Thinking
             full_prompt = f"Ký ức: {memory_context}\nUser: {user_text}"
             from langchain_core.messages import HumanMessage
             final_state = await ai_app.ainvoke({"messages": [HumanMessage(content=full_prompt)]}, config={"configurable": {"thread_id": "voice"}})
             last_message = final_state['messages'][-1]
             ai_text = last_message.content
             agent_name = final_state.get("current_agent", "J.A.R.V.I.S")
        
        speak_text = ai_text[:500] 
        def _speak():
            return client.audio.speech.create(
                model="tts-1",
                voice="onyx",
                input=speak_text
            )
        audio_res = await run_in_threadpool(_speak)

        # 5. TRẢ VỀ KẾT QUẢ KÉP (TEXT + AUDIO BLOB)
        # Ta trả về JSON chứa text, còn Audio sẽ được Frontend xử lý riêng hoặc dùng base64
        
        audio_b64 = base64.b64encode(audio_res.content).decode('utf-8')

        return {
            "text_reply": ai_text,
            "audio_base64": audio_b64,
            "transcript": user_text,
            "agent": agent_name
        }

    except Exception as e:
        logger.error(f"Voice Error: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})
    finally:
        # Dọn dẹp file rác
        if os.path.exists(temp_path):
            os.remove(temp_path)

# Thêm hàm hỗ trợ tính Level từ XP
def calculate_level(xp: int) -> int:
    # Công thức đơn giản: Cứ 100 XP là lên 1 Level. Level khởi đầu là 1.
    return int(xp / 100) + 1

# Cập nhật hàm đào tạo
async def specialized_training_job(role_tag: str):
    print(colored(f"🎓 [TRAINING] Bắt đầu đào tạo chuyên sâu cho {role_tag}...", "cyan"))
    
    # 1. Lấy chủ đề (Như cũ)
    topics = CURRICULUM.get(role_tag, [])
    if not topics: return
    current_topic_learned = topics[0] # Lấy chủ đề đầu tiên làm ví dụ

    # ... (Phần code đi search và lưu kiến thức cũ giữ nguyên) ...
    # Giả sử ngài đã search và có nội dung trong biến 'full_knowledge'
    
    # --- ĐOẠN MỚI: CẬP NHẬT TRẠNG THÁI VÀO DB ---
    try:
        with db_manager.get_connection() as conn:
            c = conn.cursor()
            
            # a. Lấy XP hiện tại
            c.execute("SELECT xp FROM agent_status WHERE role_tag = ?", (role_tag,))
            row = c.fetchone()
            current_xp = row['xp'] if row else 0
            
            # b. Cộng thêm XP (VD: Mỗi lần học xong cộng 50 XP)
            new_xp = current_xp + 50
            
            # c. Lưu vào DB
            c.execute("""
                INSERT OR REPLACE INTO agent_status (role_tag, xp, current_topic, last_updated)
                VALUES (?, ?, ?, ?)
            """, (role_tag, new_xp, current_topic_learned, datetime.datetime.now()))
            conn.commit()
            
        new_level = calculate_level(new_xp)
        print(colored(f"✅ [UPGRADE] {role_tag} đã học xong '{current_topic_learned}'. XP: {new_xp} -> Level {new_level}", "green"))
        
    except Exception as e:
        print(colored(f"❌ Lỗi cập nhật trạng thái Agent: {e}", "red"))

# Trong server.py, phần khai báo API
@app.get("/api/agents/status")
async def get_agents_status_endpoint():
    """Trả về danh sách trạng thái của tất cả Agents"""
    try:
        with db_manager.get_connection() as conn:
             # Lấy dữ liệu và tính luôn Level
            df = pd.read_sql_query("SELECT *, (xp / 100) + 1 as level FROM agent_status", conn)
            return df.to_dict(orient="records")
    except Exception as e:
        return {"error": str(e)}
# ==========================================
# ⚡ WEBSOCKET REAL-TIME (THE NEXUS)
# ==========================================
async def architect_planner(project_request: str, thread_id: str):
    """
    Chỉ lập kế hoạch chi tiết, KHÔNG viết code.
    Mục tiêu: Để CEO duyệt trước logic.
    """
    print(colored(f"📐 [ARCHITECT] Đang phác thảo dự án: {project_request}", "cyan"))
    
    # Prompt chuyên dụng cho Kiến trúc sư
    architect_prompt = (
        f"Bạn là Chief Software Architect (CSA). Có một yêu cầu dự án ERP: '{project_request}'.\n"
        "Hãy lập một BẢN THIẾT KẾ KỸ THUẬT (Technical Blueprint) chi tiết gồm:\n"
        "1. [MODULES]: Danh sách các chức năng chi tiết.\n"
        "2. [DATABASE]: Sơ đồ bảng (Table Schema) cho SQLite/PostgreSQL.\n"
        "3. [TECH STACK]: Công nghệ sử dụng (Frontend/Backend/Libs).\n"
        "4. [FLOW]: Quy trình nghiệp vụ (Ví dụ: Nhập kho -> Cập nhật tồn -> Báo cáo).\n"
        "5. [FILE STRUCTURE]: Cấu trúc thư mục dự kiến.\n\n"
        "YÊU CẦU: Trình bày dạng Markdown rõ ràng để CEO duyệt."
    )
    
    # Dùng Supervisor (Gemini 1.5 Pro) vì context window lớn, tư duy tốt
    plan_res = await run_in_threadpool(lambda: LLM_SUPERVISOR.invoke(architect_prompt))
    
    # Lưu bản vẽ ra file để CEO xem
    plan_path = f"projects/{thread_id}_BLUEPRINT.md"
    with open(plan_path, "w", encoding="utf-8") as f:
        f.write(plan_res.content)
        
    print(colored(f"✅ [PLAN READY] Bản vẽ đã xong: {plan_path}", "green"))
    return plan_res.content, plan_path

@app.post("/api/plan_project")
async def plan_project_endpoint(request: ChatRequest):
    """
    Bước 1: CEO yêu cầu lập kế hoạch.
    """
    if not AI_AVAILABLE: return {"status": "ERROR", "message": "AI Offline"}
    
    pid = request.thread_id or f"proj_{int(time.time())}"
    
    # Gọi hàm architect (Chờ kết quả luôn để trả về cho CEO xem ngay)
    plan_content, plan_path = await architect_planner(request.message, pid)
    
    return {
        "status": "PLAN_CREATED",
        "project_id": pid,
        "message": "Đã lập xong bản thiết kế. Vui lòng xem xét.",
        "blueprint_content": plan_content, # Trả về nội dung để hiện lên Dashboard
        "blueprint_path": plan_path,
        "next_action": "Nếu đồng ý, hãy gọi /api/heavy_project với nội dung 'EXECUTE_BLUEPRINT'"
    }

@app.websocket("/ws/nexus")
async def websocket_nexus(websocket: WebSocket):
    await manager.connect(websocket)
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

            if is_simple and LLM_GEMINI:
                print(colored("🚀 Kích hoạt Fast Track (Real-time Context)...", "yellow"))
                try:
                    # Gọi Gemini trả lời nhanh câu hỏi ngày giờ
                    ai_msg = await LLM_GEMINI.ainvoke(full_prompt)
                    reply_content = ai_msg.content
                    active_agent = "J.A.R.V.I.S"
                except: pass
            
            # B. DEEP THINKING (Nếu Fast Track bỏ qua HOẶC là lệnh Vẽ/Code)
            if not reply_content and AI_AVAILABLE:
                # Gọi bộ não LangGraph (Supervisor -> Designer/Coder...)
                print(colored("🧩 Chuyển giao cho Bộ Não Trung Tâm (LangGraph)...", "blue"))
                
                input_message = HumanMessage(content=full_prompt)
                
                final_state = await ai_app.ainvoke(
                    {"messages": [input_message]},
                    config={"configurable": {"thread_id": "ws_live_session"}}
                )
                
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
            response_data = {
                "sender": active_agent,
                "content": reply_content,
                "agent": active_agent # Dashboard dùng cái này để highlight icon
            }
            
            await manager.send_json(response_data, websocket)

            # 4. GHI NHỚ LẠI
            if MEMORY_AVAILABLE:
                await run_in_threadpool(lambda: extract_and_save_memory(data, reply_content))

    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception as e:
        logger.error(f"WS Error: {e}")
        manager.disconnect(websocket)
        
@app.post("/api/learn")
async def api_learn(request: LearnRequest, api_key: str = Depends(verify_api_key)):
    if not AI_AVAILABLE: return {"status": "error", "message": "AI Module Offline"}
    result = learn_knowledge(request.text)
    return {"status": "success", "message": result}

@app.post("/api/upload_pdf")
async def upload_pdf(file: UploadFile = File(...)):
    safe_filename = f"{uuid.uuid4().hex[:8]}_{file.filename}"
    file_path = os.path.join(UPLOAD_DIR, safe_filename)
    
    try:
        # Ghi file bất đồng bộ (Non-blocking I/O)
        async with aiofiles.open(file_path, 'wb') as out_file:
            content = await file.read()
            await out_file.write(content)
            
        if AI_AVAILABLE:
            result = ingest_docs_to_memory(file_path)
            return {"status": "success", "message": result, "path": file_path}
        return {"status": "saved", "message": "File saved but AI ingestion skipped (Offline)."}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")

# ==========================================
# 🛒 STORE ROUTES
# ==========================================

@app.get("/api/products")
async def get_products():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        products = conn.execute("SELECT * FROM products").fetchall()
        return [dict(row) for row in products]
    except:
        return []
    finally:
        conn.close()

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

# ==========================================
# 🖥️ FRONTEND ROUTES
# ==========================================

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

# 3. Trang Admin (Quản trị & Nạp kiến thức)
@app.get("/admin")
async def admin_page(request: Request):
    # Truyền thêm biến api_key sang giao diện HTML
    return templates.TemplateResponse("admin.html", {
        "request": request, 
        "api_key": ADMIN_SECRET # <--- QUAN TRỌNG: Dòng này giúp hiển thị Key
    })
# --- ENTRY POINT (CHẠY SERVER) ---

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
