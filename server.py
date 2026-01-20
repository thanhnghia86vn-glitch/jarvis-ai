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
    ],
    "[ORCHESTRATOR]": [
        "Tin tức công nghệ thế giới 24h qua", "Quản lý dự án Agile hiệu quả"
    ],
    "[RESEARCH]": [
        "Báo cáo thị trường công nghệ Việt Nam 2026", "Xu hướng tiêu dùng Gen Z"
    ]

}

# ==========================================
# 1. DATABASE MANAGER
# ==========================================
class DatabaseManager:
    """Quản lý kết nối Database cho Server"""
    def __init__(self, db_path=DB_PATH):
        self.db_path = db_path

    def get_connection(self):
        """Tạo kết nối cho lệnh 'with'"""
        return sqlite3.connect(self.db_path, check_same_thread=False)
    
    def init_db(self):
        """Khởi tạo cấu trúc bảng"""
        with self.get_connection() as conn:
            # Bảng sản phẩm & tài chính cũ
            conn.execute("CREATE TABLE IF NOT EXISTS products (id INTEGER PRIMARY KEY, name TEXT, price REAL)")
            conn.execute("CREATE TABLE IF NOT EXISTS finance_logs (id INTEGER PRIMARY KEY, type TEXT, amount REAL)")
            
            # Bảng Agent Status (QUAN TRỌNG CHO ADMIN)
            conn.execute('''CREATE TABLE IF NOT EXISTS agent_status 
                            (role_tag TEXT PRIMARY KEY, 
                             xp INTEGER DEFAULT 0, 
                             current_topic TEXT, 
                             last_updated DATETIME)''')
            conn.commit()
        logger.info("✅ DATABASE INITIALIZED")

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
# 3. BACKGROUND JOBS (AI TRAINING & CRON)
# ==========================================
def calculate_level(xp: int) -> int:
    # Công thức đơn giản: Cứ 100 XP là lên 1 Level. Level khởi đầu là 1.
    return int(xp / 100) + 1

# Cập nhật hàm đào tạo
async def specialized_training_job(role_tag: str):
    """Hàm đào tạo chuyên sâu cho từng Agent"""
    print(colored(f"🎓 [TRAINING] Bắt đầu đào tạo cho {role_tag}...", "cyan"))
    
    topics = CURRICULUM.get(role_tag, [])
    if not topics: return
    
    # Chọn ngẫu nhiên 1 chủ đề để học cho đỡ tốn tài nguyên
    current_topic = random.choice(topics)
    
    try:
        # 1. Giả lập học (Hoặc gọi Perplexity thật nếu có)
        learned_content = f"Nội dung chi tiết về {current_topic} cập nhật lúc {datetime.now()}"
        if LLM_PERPLEXITY:
            try:
                res = await LLM_PERPLEXITY.ainvoke(current_topic)
                learned_content = res.content
            except: pass

        # 2. Lưu vào Vector DB (Ký ức dài hạn)
        if MEMORY_AVAILABLE and vector_db:
            await run_in_threadpool(lambda: vector_db.add_texts(
                texts=[learned_content],
                metadatas=[{"source": "Auto-Training", "agent": role_tag, "topic": current_topic}]
            ))

        # 3. Cập nhật XP và Level vào Database (Cho Admin Panel)
        with db_manager.get_connection() as conn:
            c = conn.cursor()
            # Lấy XP cũ
            row = c.execute("SELECT xp FROM agent_status WHERE role_tag = ?", (role_tag,)).fetchone()
            current_xp = row[0] if row else 0
            
            # Cộng XP (50 điểm mỗi lần học)
            new_xp = current_xp + 50
            
            # Lưu
            c.execute("""
                INSERT OR REPLACE INTO agent_status (role_tag, xp, current_topic, last_updated)
                VALUES (?, ?, ?, ?)
            """, (role_tag, new_xp, current_topic, datetime.now()))
            conn.commit()
            
        print(colored(f"✅ [UPGRADE] {role_tag} học xong '{current_topic}'. XP: {new_xp} (Lv.{calculate_level(new_xp)})", "green"))

    except Exception as e:
        print(colored(f"❌ Lỗi đào tạo {role_tag}: {e}", "red"))
# Trong server.py, phần khai báo API

async def morning_briefing_job():
    """
    PHIÊN BẢN 2.0: Tự động học tin tức + Cộng XP cho [ORCHESTRATOR] + Tạo file báo cáo
    """
    role_tag = "[ORCHESTRATOR]"
    print(colored(f"\n⏰ [CRON JOB] {role_tag} đang thực hiện quét tin tức buổi sáng...", "cyan", attrs=["bold"]))
    
    if not AI_AVAILABLE or not LLM_PERPLEXITY:
        print(colored("⚠️ Bỏ qua Cron Job vì AI Module chưa sẵn sàng.", "yellow"))
        return

    # Lấy chủ đề từ Giáo Trình chung
    topics = CURRICULUM.get(role_tag, ["Tin tức công nghệ nổi bật", "Thị trường tài chính"])
    report_buffer = []
    
    for topic in topics:
        try:
            print(colored(f"--> {role_tag} đang đọc: {topic}...", "white"))
            res = await LLM_PERPLEXITY.ainvoke(topic)
            content = res.content
            
            if MEMORY_AVAILABLE and vector_db:
                await run_in_threadpool(lambda: vector_db.add_texts(
                    texts=[content],
                    metadatas=[{"source": "Morning_Briefing", "agent": role_tag, "topic": topic}]
                ))
            report_buffer.append(f"### {topic}\n{content[:800]}...") 
        except: pass

    # Tạo báo cáo & Cộng XP
    if report_buffer:
        today = datetime.now().strftime("%Y-%m-%d")
        report_path = f"projects/Morning_Briefing_{today}.md"
        try:
            async with aiofiles.open(report_path, "w", encoding="utf-8") as f:
                await f.write(f"# 🌅 BẢN TIN SÁNG {today}\n\n" + "\n\n".join(report_buffer))
            print(colored(f"✅ [DONE] Đã lưu báo cáo: {report_path}", "green"))
            
            # Cộng 100 XP
            with db_manager.get_connection() as conn:
                c = conn.cursor()
                row = c.execute("SELECT xp FROM agent_status WHERE role_tag = ?", (role_tag,)).fetchone()
                new_xp = (row[0] if row else 0) + 100
                c.execute("INSERT OR REPLACE INTO agent_status (role_tag, xp, current_topic, last_updated) VALUES (?, ?, ?, ?)", 
                          (role_tag, new_xp, "Tổng hợp tin tức sáng", datetime.now()))
                conn.commit()
        except Exception as e:
            print(colored(f"❌ Lỗi Job Sáng: {e}", "red"))

@asynccontextmanager
async def lifespan(app: FastAPI):
    # --- STARTUP ---
    db_manager.init_db()
    
    # Tạo thư mục cần thiết
    for d in [UPLOAD_DIR, "static", "templates", "projects"]:
        if not os.path.exists(d): os.makedirs(d)
        
    # --- SCHEDULER SETUP (QUAN TRỌNG) ---
    scheduler = AsyncIOScheduler()
    
    # 1. Briefing sáng (7:00)
    scheduler.add_job(morning_briefing_job, 'cron', hour=7, minute=0)
    
    # 2. Lên lịch đào tạo cho từng Agent (Rải rác trong ngày để không nghẽn mạng)
    # Ví dụ: Mỗi 2-4 tiếng các Agent sẽ tự đi học 1 lần
    for idx, role in enumerate(CURRICULUM.keys()):
        # Hack nhỏ: Cộng thêm phút để các job không chạy cùng lúc
        scheduler.add_job(
            specialized_training_job, 
            'interval', 
            hours=4, 
            minutes=idx * 5, # Mỗi ông cách nhau 5 phút
            args=[role]
        )
        
    scheduler.start()
    logger.info(f"⏰ SCHEDULER ACTIVATED: Đã lên lịch đào tạo cho {len(CURRICULUM)} Agents.")
    
    yield # Server chạy tại đây
    
    # --- SHUTDOWN ---
    scheduler.shutdown()
    logger.info("💤 SYSTEM SHUTDOWN.")

app = FastAPI(
    title="J.A.R.V.I.S Neural Backend",
    version="3.0",
    lifespan=lifespan
)

# Cấu hình CORS & Static
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
base_dir = os.path.abspath(os.path.dirname(__file__))
app.mount("/static", StaticFiles(directory=os.path.join(base_dir, 'static')), name="static")
templates = Jinja2Templates(directory=os.path.join(base_dir, 'templates'))
# Cấu hình CORS

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


async def verify_api_key(x_api_key: Optional[str] = Header(None)):
    """Middleware kiểm tra bảo mật"""
    # Logic: Nếu có gửi key thì check, nếu không gửi (Dev mode) thì bỏ qua hoặc chặn tùy CEO
    if x_api_key and x_api_key != ADMIN_SECRET:
        raise HTTPException(status_code=403, detail="⛔ SAI MẬT MÃ QUÂN SỰ (WRONG API KEY)")
    return x_api_key

@app.get("/api/agents/status")
async def get_agents_status_endpoint():
    """API trả về Level & XP của Agent cho giao diện Admin"""
    try:
        with db_manager.get_connection() as conn:
            # Lấy dữ liệu và tính Level
            df = pd.read_sql_query("SELECT *, (xp / 100) + 1 as level FROM agent_status ORDER BY xp DESC", conn)
            # Chuyển đổi timestamp thành chuỗi để JSON không lỗi
            df['last_updated'] = df['last_updated'].astype(str)
            return df.to_dict(orient="records")
    except Exception as e:
        logger.error(f"Agent Status Error: {e}")
        return []

@app.get("/api/stats")
async def get_system_stats():
    """Thống kê tài chính"""
    try:
        with db_manager.get_connection() as conn:
            prod_count = conn.execute("SELECT count(*) FROM products").fetchone()[0]
            income = conn.execute("SELECT SUM(amount) FROM finance_logs WHERE type='income'").fetchone()[0] or 0
            expense = conn.execute("SELECT SUM(amount) FROM finance_logs WHERE type='expense'").fetchone()[0] or 0
            return {
                "products": prod_count,
                "revenue": income,
                "expense": expense,
                "balance": income - expense
            }
    except:
        return {"products": 0, "revenue": 0, "expense": 0}

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
            
        # B. Lưu vào dữ liệu huấn luyện (Training Data) - Kiểm tra an toàn
        if 'log_training_data' in globals() and log_training_data:
             background_tasks.add_task(log_training_data, request.message, ai_reply, success=True)
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
        if any(k in user_text.lower() for k in fast_keywords) and LLM_GEMINI:
             try:
                 ai_res = await LLM_GEMINI.ainvoke(f"Ký ức: {memory_context}. Hỏi: {user_text}")
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
        # Gọi hàm architect (Chờ kết quả luôn để trả về cho CEO xem ngay)
        # Hàm architect_planner đã được tối ưu ở bước trước
        plan_content, plan_path = await architect_planner(request.message, pid)
        
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
# 7. CHẠY DỰ ÁN LỚN (BACKGROUND)
# ==========================================
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
        plan_prompt = (
            f"Bạn là Kiến trúc sư phần mềm (Architect). Yêu cầu dự án: '{project_request}'.\n"
            "Hãy chia nhỏ dự án này thành các bước kỹ thuật (Coding Steps) cụ thể.\n"
            "QUAN TRỌNG: Chỉ trả về danh sách các bước, bắt đầu bằng dấu gạch ngang (-).\n"
            "Ví dụ:\n- Tạo file models.py\n- Viết API login"
        )
        plan_res = await run_in_threadpool(lambda: LLM_SUPERVISOR.invoke(plan_prompt))
        raw_steps = plan_res.content.split('\n')
        steps = []
        for s in raw_steps:
            s = s.strip()
            # Lọc các dòng là bullet point hoặc số thứ tự
            if s and (s.startswith('-') or s.startswith('*') or (s[0].isdigit() and s[1] in ['.', ')'])):
                steps.append(s)
        
        # Ghi log kế hoạch (Dùng aiofiles để không chặn Server)
        async with aiofiles.open(log_file, "w", encoding="utf-8") as f:
            await f.write(f"=== PROJECT PLAN: {project_request} ===\n{plan_res.content}\n{'='*50}\n")
            
        if not steps:
            print(colored("⚠️ Không tìm thấy bước nào trong kế hoạch. Dừng.", "red"))
            return
        # Giai đoạn 2: Code từng phần (Loop)
        for idx, step in enumerate(steps):
            print(colored(f"⏳ Doing Step {idx+1}/{len(steps)}: {step}", "yellow"))
            
            # Prompt nhắc lại ngữ cảnh (Context Injection)
            # Giúp AI nhớ nó đang làm dự án gì, tránh lạc đề
            step_input = (
                f"[DỰ ÁN TỔNG THỂ]: {project_request}\n"
                f"[NHIỆM VỤ HIỆN TẠI]: Bước {idx+1}: {step}.\n"
                "Hãy viết code hoàn chỉnh và chi tiết cho nhiệm vụ này."
            )
            
            # Gọi AI Brain (LangGraph đã là async nên dùng await trực tiếp)
            state_res = await ai_app.ainvoke(
                {"messages": [HumanMessage(content=step_input)]},
                config={"configurable": {"thread_id": thread_id}}
            )
            
            ai_output = state_res['messages'][-1].content
            
            # Ghi log kết quả (Async Write)
            async with aiofiles.open(log_file, "a", encoding="utf-8") as f:
                await f.write(f"\n\n--- KẾT QUẢ BƯỚC {idx+1}: {step} ---\n{ai_output}\n")
            
            # Nghỉ 2 giây để tránh spam API
            await asyncio.sleep(2)

        print(colored(f"✅ [DONE] Dự án {thread_id} đã hoàn tất!", "green"))

    except Exception as e:
        print(colored(f"❌ [FAILED] Dự án bị lỗi: {e}", "red"))
        # Ghi lỗi vào file log (Async Write)
        try:
            async with aiofiles.open(log_file, "a", encoding="utf-8") as f:
                await f.write(f"\n❌ SYSTEM ERROR: {str(e)}")
        except: pass

async def architect_planner(project_request: str, thread_id: str):
    """
    KẾN TRÚC SƯ TRƯỞNG: Lập bản vẽ kỹ thuật & Lộ trình thi công.
    (Phiên bản Async + Tối ưu Prompt cho Executor)
    """
    print(colored(f"📐 [ARCHITECT] Đang phác thảo dự án: {project_request}", "cyan"))
    
    # Tạo đường dẫn file trước
    plan_path = f"projects/{thread_id}_BLUEPRINT.md"

    try:
        if not LLM_SUPERVISOR:
             raise Exception("LLM_SUPERVISOR chưa được khởi tạo (AI Offline).")

        # --- NÂNG CẤP PROMPT ---
        # Thêm mục số 6 để tạo thuận lợi cho 'heavy_project_executor' đọc task
        architect_prompt = (
            f"Bạn là Chief Software Architect (CSA). Có một yêu cầu dự án: '{project_request}'.\n"
            "Hãy lập một BẢN THIẾT KẾ KỸ THUẬT (Technical Blueprint) chi tiết dạng Markdown:\n\n"
            "1. [OVERVIEW]: Tóm tắt mục tiêu dự án.\n"
            "2. [MODULES]: Danh sách các chức năng chính.\n"
            "3. [DATABASE]: Sơ đồ bảng (Table Schema) chi tiết.\n"
            "4. [TECH STACK]: Công nghệ sử dụng (Frontend/Backend/Libs).\n"
            "5. [FILE STRUCTURE]: Cấu trúc thư mục dự kiến.\n"
            "6. [EXECUTION PLAN] (QUAN TRỌNG): Hãy liệt kê lộ trình code cụ thể từng bước.\n"
            "   - Bắt buộc dùng gạch đầu dòng (-) cho mỗi bước.\n"
            "   - Ví dụ:\n"
            "   - Tạo môi trường ảo và file requirements.txt\n"
            "   - Thiết kế database models trong models.py\n"
            "   - Viết API đăng nhập\n"
        )
        
        # Gọi AI (Chạy trong Threadpool để không chặn Server)
        plan_res = await run_in_threadpool(lambda: LLM_SUPERVISOR.invoke(architect_prompt))
        
        # Ghi file bất đồng bộ (Non-blocking I/O)
        async with aiofiles.open(plan_path, "w", encoding="utf-8") as f:
            await f.write(plan_res.content)
            
        print(colored(f"✅ [PLAN READY] Bản vẽ đã xong: {plan_path}", "green"))
        
        # Trả về nội dung để hiển thị ngay lên Dashboard
        return plan_res.content, plan_path

    except Exception as e:
        error_msg = f"Lỗi lập kế hoạch: {str(e)}"
        print(colored(f"❌ {error_msg}", "red"))
        
        # Ghi file lỗi để debug
        try:
            async with aiofiles.open(plan_path, "w", encoding="utf-8") as f:
                await f.write(f"# ⚠️ PROJECT ERROR\n{error_msg}")
        except: pass
        
        return error_msg, plan_path
    
# 2. API Endpoint để kích hoạt
@app.post("/api/heavy_project")
async def start_heavy_project(
    request: ChatRequest, 
    background_tasks: BackgroundTasks,
    api_key: str = Depends(verify_api_key) # <--- THÊM DÒNG NÀY ĐỂ BẢO VỆ
):
    """
    API để CEO kích hoạt chế độ làm dự án lớn (Yêu cầu API Key).
    """
    # Tạo Thread ID riêng cho dự án nếu chưa có
    pid = request.thread_id or f"proj_{int(time.time())}"
    
    # Đẩy vào chạy nền (Fire and Forget)
    # Lưu ý: heavy_project_executor phải là hàm async (đã sửa ở bước trước)
    background_tasks.add_task(heavy_project_executor, request.message, pid)
    
    return {
        "status": "PROCESSING",
        "project_id": pid,
        "message": "Đã tiếp nhận dự án. Hệ thống đang xử lý ngầm...",
        "log_path": f"projects/{pid}_log.txt"
    }

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
        "version": "J.A.R.V.I.S v3.0",
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


# ==========================================
# ⚡ WEBSOCKET REAL-TIME (THE NEXUS)
# ==========================================


# ==========================================
# 🖥️ FRONTEND ROUTES
# ==========================================

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

