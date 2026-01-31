import glob
import os
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
from langchain_google_genai import ChatGoogleGenerativeAI
from fastapi import FastAPI, HTTPException, Header, Depends, UploadFile, File, Request, status, WebSocket, WebSocketDisconnect, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, Response, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel
from langchain_core.messages import HumanMessage

# === CONFIG ===
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("JARVIS_v4.5")
ADMIN_SECRET = os.environ.get("ADMIN_SECRET", "ai_corp_secret_123")
RENDER_DISK_PATH = "/var/data"

# Setup Paths
if os.path.exists(RENDER_DISK_PATH):
    BASE_DATA_DIR = RENDER_DISK_PATH
    print(colored(f"💽 [STORAGE] Cloud Disk: {BASE_DATA_DIR}", "green", attrs=["bold"]))
else:
    BASE_DATA_DIR = "."
    print(colored("💻 [STORAGE] Local Disk", "yellow"))

UPLOAD_DIR = os.path.join(BASE_DATA_DIR, "uploads")
PROJECTS_DIR = os.path.join(BASE_DATA_DIR, "projects")
DB_PATH = os.path.join(BASE_DATA_DIR, "ai_corp_projects.db")
VECTOR_DB_PATH = os.path.join(BASE_DATA_DIR, "db_knowledge")

if not os.environ.get("DATABASE_URL") and os.path.exists(RENDER_DISK_PATH):
    os.environ["DATABASE_URL"] = f"sqlite:///{DB_PATH}"

# === LOAD AI MODULES (Safe Loading) ===
AI_AVAILABLE = False
MEMORY_AVAILABLE = False
VOICE_AVAILABLE = False
SERVER_READY = False
ai_app = None
CHAT_MODEL = None
client = None

try:
    from main import ai_app, log_work_to_db, auto_learning_cycle, morning_briefing_job, LLM_GEMINI_VISION
    AI_AVAILABLE = True
    SERVER_READY = True
    logger.info("✅ AI CORE LOADED")
except Exception as e:
    logger.warning(f"⚠️ AI CORE OFFLINE: {e}")

try:
    CHAT_MODEL = ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=os.environ.get("GOOGLE_API_KEY"))
except: pass

try:
    from memory_core import recall_relevant_memories, extract_and_save_memory, ingest_docs_to_memory, learn_knowledge
    MEMORY_AVAILABLE = True
except: pass

try: 
    from voice_engine import client
    VOICE_AVAILABLE = True
except: client = None

# === MODELS ===
class ChatRequest(BaseModel):
    message: str
    thread_id: str = "default"
class BuyRequest(BaseModel):
    product_id: int
class TTSRequest(BaseModel):
    text: str
class LearnRequest(BaseModel):
    text: str
class Query(BaseModel):
    question: str

# === DATABASE MANAGER ===
class DatabaseManager:
    def __init__(self): 
        # Fix: Xử lý chuỗi kết nối cho SQLAlchemy
        db_url = os.environ.get("DATABASE_URL", f"sqlite:///{DB_PATH}")
        if db_url.startswith("postgres://"):
            db_url = db_url.replace("postgres://", "postgresql://", 1)
        self.engine = create_engine(db_url)
    
    def get_connection(self):
        return self.engine.connect()
    
    def init_db(self):
        try:
            with self.get_connection() as conn:
                conn.execute(text("CREATE TABLE IF NOT EXISTS products (id INTEGER PRIMARY KEY, name TEXT, price REAL)"))
                conn.execute(text("""
                    CREATE TABLE IF NOT EXISTS agent_status (
                        role_tag TEXT PRIMARY KEY, xp INTEGER DEFAULT 0, 
                        current_topic TEXT, last_updated TIMESTAMP
                    )
                """))
                conn.execute(text("""
                    CREATE TABLE IF NOT EXISTS work_logs (
                        id INTEGER PRIMARY KEY, timestamp TEXT, agent_name TEXT, 
                        task_content TEXT, result_summary TEXT, tool_used TEXT, 
                        cost REAL, duration REAL
                    )
                """))
                conn.execute(text("CREATE TABLE IF NOT EXISTS projects (id TEXT PRIMARY KEY, name TEXT, history TEXT, timestamp TIMESTAMP)"))
                conn.commit()
            print(colored("✅ FULL DB INITIALIZED", "green"))
        except Exception as e:
            print(colored(f"❌ DB INIT ERROR: {e}", "red"))

db_manager = DatabaseManager()

# === WEBSOCKET MANAGER ===
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []
    
    async def connect(self, ws: WebSocket):
        await ws.accept()
        self.active_connections.append(ws)
    
    def disconnect(self, ws: WebSocket):
        # [FIX]: Syntax Error cũ (if không được viết cùng dòng với def)
        if ws in self.active_connections:
            self.active_connections.remove(ws)
            
    async def send_json(self, data: dict, ws: WebSocket):
        await ws.send_json(data)
        
    async def broadcast(self, message: str): 
        for conn in self.active_connections:
            await conn.send_text(message)

manager = ConnectionManager()

# === PROJECT ARCHITECT ===
async def run_architect_phase(project_request: str, thread_id: str):
    print(colored(f"📐 ARCHITECT: {project_request[:50]}", "cyan"))
    os.makedirs(PROJECTS_DIR, exist_ok=True)
    plan_path = f"{PROJECTS_DIR}/{thread_id}_BLUEPRINT.md"
    
    prompt = f"""Bạn là Chief Software Architect. Yêu cầu: '{project_request}'.
    Lập TECHNICAL BLUEPRINT (Markdown):
    1. OVERVIEW
    2. MODULES
    3. DATABASE
    4. TECH STACK
    5. EXECUTION PLAN (Các bước code chi tiết dùng dấu -)"""
    
    try:
        content = "SIMULATION BLUEPRINT"
        if SERVER_READY and 'LLM_GEMINI_VISION' in globals() and LLM_GEMINI_VISION:
            res = await run_in_threadpool(lambda: LLM_GEMINI_VISION.invoke(prompt))
            content = res.content
        
        async with aiofiles.open(plan_path, "w", encoding="utf-8") as f:
            await f.write(content)
        return content, plan_path
    except Exception as e:
        print(colored(f"❌ ARCHITECT ERROR: {e}", "red"))
        return None, None

async def run_coding_phase(blueprint_content: str, thread_id: str):
    print(colored(f"🏗️ EXECUTOR: {thread_id}", "magenta"))
    log_file = f"{PROJECTS_DIR}/{thread_id}_coding_log.txt"
    
    steps = [line.strip().lstrip('-').strip() for line in blueprint_content.split('\n') if "EXECUTION PLAN" in line.upper() or (line.strip().startswith('-') and len(line) > 10)]
    steps = steps[:10] # Limit steps
    
    async with aiofiles.open(log_file, "w", encoding="utf-8") as f: 
        await f.write(f"BẮT ĐẦU DỰ ÁN {thread_id}\n")
    
    for idx, step in enumerate(steps):
        print(colored(f"⏳ STEP {idx+1}: {step[:50]}", "yellow"))
        try:
            output = f"[SIMULATED CODE] {step}"
            if ai_app:
                res = await ai_app.ainvoke(
                    {"messages": [HumanMessage(content=f"Code task: {step}")]}, 
                    config={"configurable": {"thread_id": thread_id}}
                )
                output = res["messages"][-1].content
            
            async with aiofiles.open(log_file, "a", encoding="utf-8") as f:
                await f.write(f"\n--- BƯỚC {idx+1} ---\n{step}\n{output}\n")
        except: pass

# === APP SETUP ===
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    for d in [UPLOAD_DIR, PROJECTS_DIR, "static", "templates"]:
        os.makedirs(d, exist_ok=True)
    
    db_manager.init_db()
    
    scheduler = AsyncIOScheduler()
    if 'morning_briefing_job' in globals():
        scheduler.add_job(morning_briefing_job, 'cron', hour=7, minute=0)
    scheduler.start()
    
    if AI_AVAILABLE:
        asyncio.create_task(auto_learning_cycle())
    
    yield
    # Shutdown
    scheduler.shutdown()

app = FastAPI(title="J.A.R.V.I.S v4.5 FULL", version="4.5", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

base_dir = os.path.abspath(os.path.dirname(__file__))
static_dir = os.path.join(base_dir, 'static')
templates_dir = os.path.join(base_dir, 'templates')

# 1. Tạo thư mục ngay lập tức (Không chờ lifespan)
if not os.path.exists(static_dir):
    os.makedirs(static_dir)
    print(colored("⚠️ Đã tự động tạo thư mục 'static' để tránh lỗi mount.", "yellow"))

if not os.path.exists(templates_dir):
    os.makedirs(templates_dir)
    print(colored("⚠️ Đã tự động tạo thư mục 'templates'.", "yellow"))

# 2. Bây giờ Mount mới an toàn
app.mount("/static", StaticFiles(directory=static_dir), name="static")
templates = Jinja2Templates(directory=templates_dir)
# === WEB ROUTES ===
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("store.html", {"request": request})

@app.get("/dashboard", response_class=HTMLResponse) 
async def dashboard(request: Request):
    return templates.TemplateResponse("dashboard.html", {"request": request})

@app.get("/admin", response_class=HTMLResponse)
async def admin(request: Request):
    return templates.TemplateResponse("admin.html", {"request": request})

@app.get("/org", response_class=HTMLResponse)
async def org_chart(request: Request):
    try:
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        c = conn.cursor()
        
        c.execute("SELECT * FROM agent_status ORDER BY xp DESC")
        agents = []
        for r in c.fetchall():
            xp = r['xp'] or 0
            agents.append({
                "name": r['role_tag'], "xp": xp, "level": int(xp/1000)+1,
                "topic": r['current_topic'], "progress": (xp%1000)/10
            })
            
        c.execute("SELECT * FROM work_logs WHERE tool_used LIKE '%SUPREME%' ORDER BY id DESC LIMIT 1")
        report = c.fetchone()
        conn.close()
        return templates.TemplateResponse("index.html", {"request": request, "agents": agents, "featured_report": report})
    except: return "Lỗi Org Chart"

# === API STORE ===
@app.get("/api/products")
async def products(): 
    try:
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        c = conn.cursor()
        
        c.execute("SELECT * FROM products")
        rows = c.fetchall()
        
        if not rows: # Auto-Seed
            sample = [(1,"AI Task Manager",49.0),(2,"SaaS Landing AI",99.0),(3,"AI Content Pack",19.0)]
            c.executemany("INSERT INTO products (id, name, price) VALUES (?,?,?)", sample)
            conn.commit()
            c.execute("SELECT * FROM products")
            rows = c.fetchall()
            
        conn.close()
        return [dict(r) for r in rows]
    except: return []

@app.post("/api/buy")
async def buy(req: BuyRequest):
    conn = sqlite3.connect(DB_PATH)
    try:
        cur = conn.execute("SELECT price, name FROM products WHERE id=?",(req.product_id,))
        prod = cur.fetchone()
        if not prod: raise HTTPException(404,"Sản phẩm không tồn tại")
        
        price, name = prod[0], prod[1]
        key = f"AI-{uuid.uuid4().hex[:8].upper()}-{int(time.time())}"
        
        conn.execute("INSERT INTO work_logs (timestamp,agent_name,task_content,tool_used,cost,result_summary) VALUES (?,?,?,?,?,?)",
                     (datetime.now().strftime("%H:%M %d/%m"), "STORE_BOT", f"Bán {name}", "SALE", price, f"License: {key}"))
        conn.commit()
        return {"status": "success", "msg": f"Mua {name} thành công!", "license_key": key}
    except Exception as e:
        raise HTTPException(500, str(e))
    finally: conn.close()

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
async def stats():
    try:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        prod_count = c.execute("SELECT count(*) FROM products").fetchone()[0]
        expense = c.execute("SELECT SUM(cost) FROM work_logs WHERE tool_used != 'SALE'").fetchone()[0] or 0.0
        revenue = c.execute("SELECT SUM(cost) FROM work_logs WHERE tool_used = 'SALE'").fetchone()[0] or 0.0
        conn.close()
        return {"products": prod_count, "revenue": round(revenue,2), "expense": round(expense,4), "balance": round(revenue-expense,4)}
    except: return {"products": 0, "revenue": 0, "expense": 0, "balance": 0}

@app.get("/api/wealth")
async def wealth():
    try:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        total_tasks = c.execute("SELECT COUNT(*) FROM work_logs").fetchone()[0] or 0
        total_cost = c.execute("SELECT SUM(cost) FROM work_logs").fetchone()[0] or 0.0
        legendary = c.execute("SELECT COUNT(*) FROM work_logs WHERE tool_used LIKE '%SUPREME%'").fetchone()[0] or 0
        conn.close()
        
        return {
            "financial_report": {"total_cost_usd": total_cost, "total_tasks": total_tasks},
            "intellectual_assets": {"LEGENDARY": legendary}
        }
    except: return {}

# === API AI ===
@app.post("/api/chat")
async def chat(request: ChatRequest, bg_tasks: BackgroundTasks):
    msg = request.message
    if msg.lower() in ['hi', 'chào']: return {"reply": "Chào bạn! Tôi có thể giúp gì?"}
    
    try:
        reply = "Hệ thống đang bận."
        if AI_AVAILABLE and ai_app:
            # LangGraph Processing
            res = await ai_app.ainvoke({"messages": [HumanMessage(content=msg)]}, config={"configurable": {"thread_id": request.thread_id}})
            reply = res["messages"][-1].content
        elif CHAT_MODEL:
            # Gemini Fallback
            res = await CHAT_MODEL.ainvoke(msg)
            reply = res.content
            
        if MEMORY_AVAILABLE: 
            bg_tasks.add_task(extract_and_save_memory, msg, reply)
            
        return {"reply": reply}
    except Exception as e: return {"reply": f"Lỗi: {e}"}

@app.post("/api/plan_project")
async def plan_project(request: ChatRequest):
    content, path = await run_architect_phase(request.message, request.thread_id or "proj_new")
    return {"status": "PLAN_CREATED", "content": content, "path": path}

@app.post("/api/upload_pdf")
async def upload_pdf(file: UploadFile = File(...)):
    if not file.filename.endswith('.pdf'): raise HTTPException(400)
    path = os.path.join(UPLOAD_DIR, f"{uuid.uuid4().hex}_{file.filename}")
    async with aiofiles.open(path, 'wb') as f: await f.write(await file.read())
    if AI_AVAILABLE: 
        await run_in_threadpool(lambda: ingest_docs_to_memory(path))
    return {"status": "success", "path": path}

@app.post("/api/tts")
async def tts(request: TTSRequest):
    def gen(): 
        text = request.text[:500]
        buf = io.BytesIO()
        gTTS(text, lang='vi', tld='com.vn').write_to_fp(buf)
        buf.seek(0)
        return buf.read()
    return Response(content=await run_in_threadpool(gen), media_type="audio/mpeg")

@app.post("/api/voice_chat")
async def voice_chat(file: UploadFile = File(...)):
    if not VOICE_AVAILABLE: return JSONResponse(503, {"error": "Voice Offline"})
    temp = os.path.join(UPLOAD_DIR, f"temp_{uuid.uuid4()}.webm")
    try:
        async with aiofiles.open(temp, 'wb') as f: await f.write(await file.read())
        def trans(): 
            with open(temp, "rb") as af: return client.audio.transcriptions.create(model="whisper-1", file=af, language="vi").text
        text = await run_in_threadpool(trans)
        
        reply = "..."
        if CHAT_MODEL: reply = (await CHAT_MODEL.ainvoke(text)).content
        return {"transcript": text, "reply": reply}
    except Exception as e: return JSONResponse(500, {"error": str(e)})
    finally: 
        # [FIX]: Syntax error cũ (finally không được if cùng dòng)
        if os.path.exists(temp): 
            os.remove(temp)

# === WEBSOCKET ===
@app.websocket("/ws/nexus")
async def nexus(websocket: WebSocket):
    await manager.connect(websocket)
    await manager.send_json({"sender": "J.A.R.V.I.S", "content": "Online & Ready!", "agent": "System"}, websocket)
    try:
        while True:
            data = await websocket.receive_text()
            reply = "Processing..."
            if CHAT_MODEL:
                # [FIX]: Cấu trúc await đúng
                res = await CHAT_MODEL.ainvoke(data)
                reply = res.content
            await manager.send_json({"sender": "J.A.R.V.I.S", "content": reply, "agent": "J.A.R.V.I.S"}, websocket)
    except WebSocketDisconnect: 
        manager.disconnect(websocket)

# === MAIN ===
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8080))
    print(colored("🚀 J.A.R.V.I.S v4.5 FULL FEATURE STARTING", "cyan"))
    print(colored(f"📊 API Docs: http://localhost:{port}/docs", "green"))
    uvicorn.run("server:app", host="0.0.0.0", port=port, reload=True)
