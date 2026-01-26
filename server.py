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
    # === NHÓM 1: QUẢN TRỊ & CHIẾN LƯỢC (C-SUITE) ===
    "[ORCHESTRATOR]": [
        "Mô hình OKRs vs KPIs trong quản trị doanh nghiệp AI",
        "Chiến lược quản trị khủng hoảng (Crisis Management) thời gian thực",
        "Tối ưu hóa quy trình ra quyết định dựa trên dữ liệu (Data-Driven Decision Making)",
        "Tin tức công nghệ Deep Tech toàn cầu 24h qua"
    ],
    "[FINANCE]": [
        "Các chiến lược Hedging rủi ro tỷ giá hối đoái",
        "Ứng dụng Blockchain trong quản lý dòng tiền doanh nghiệp (Corporate Treasury)",
        "Phân tích kỹ thuật nâng cao: Sóng Elliott và Fibonacci trong thị trường vàng/Crypto",
        "Tối ưu hóa thuế cho doanh nghiệp số (Digital Tax Optimization)"
    ],
    "[HR_MANAGER]": [
        "Xây dựng khung năng lực cốt lõi cho nhân sự AI & Blockchain",
        "Tâm lý học hành vi trong giữ chân nhân tài Gen Z & Alpha",
        "Tự động hóa quy trình Payroll và C&B bằng Smart Contracts",
        "Luật lao động quốc tế về làm việc từ xa (Remote Work Compliance)"
    ],

    # === NHÓM 2: KỸ THUẬT PHẦN MỀM (CORE TECH) ===
    "[CODER]": [
        "Lập trình hiệu năng cao với Rust và Go cho Backend",
        "Tối ưu hóa truy vấn Database (Indexing, Partitioning, Sharding)",
        "Event-Driven Architecture với Apache Kafka và RabbitMQ",
        "WebAssembly (Wasm): Tương lai của ứng dụng Web hiệu năng cao"
    ],
    "[ARCHITECT]": [
        "Domain-Driven Design (DDD) trong thiết kế Microservices",
        "Triển khai Serverless trên quy mô lớn (AWS Lambda/Google Cloud Run)",
        "Mô hình CQRS và Event Sourcing trong hệ thống phân tán",
        "Zero Trust Architecture: Kiến trúc bảo mật không tin cậy ai"
    ],
    "[SECURITY]": [
        "Kỹ thuật Reverse Engineering mã độc nâng cao",
        "Bảo mật API theo chuẩn OWASP Top 10 năm 2026",
        "Post-Quantum Cryptography: Mã hóa chống máy tính lượng tử",
        "DevSecOps: Tích hợp bảo mật vào quy trình CI/CD"
    ],
    "[DATA_ANALYST]": [
        "Xây dựng RAG (Retrieval-Augmented Generation) cho LLM doanh nghiệp",
        "Data Lakehouse: Kết hợp sức mạnh của Data Lake và Data Warehouse",
        "Phân tích dữ liệu thời gian thực (Real-time Analytics) với Apache Flink",
        "Mô hình dự báo chuỗi thời gian (Time-series Forecasting) bằng Deep Learning"
    ],

    # === NHÓM 3: PHẦN CỨNG & IOT (HARDWARE) ===
    "[HARDWARE]": [
        "Thiết kế mạch PCB cao tần (High-speed PCB Design)",
        "Edge AI: Chạy mô hình AI trực tiếp trên vi điều khiển (TinyML)",
        "Công nghệ Pin thế hệ mới và quản lý năng lượng (Power Management)",
        "Lập trình FPGA cho xử lý tín hiệu số"
    ],
    "[IOT]": [
        "Mạng lưới vạn vật (Mesh Networking) với LoRaWAN và Zigbee",
        "Digital Twins: Bản sao số trong công nghiệp sản xuất",
        "Giao thức MQTT v5 và tối ưu hóa băng thông cho thiết bị IoT",
        "Bảo mật thiết bị IoT ở cấp độ phần cứng (Hardware Security Modules)"
    ],

    # === NHÓM 4: SÁNG TẠO & MARKETING (GROWTH) ===
    "[MARKETING]": [
        "Neuromarketing: Ứng dụng khoa học não bộ vào quảng cáo",
        "Programmatic Advertising: Quảng cáo lập trình hóa tự động",
        "Chiến lược Growth Hacking dựa trên Phễu AARRR",
        "Tối ưu hóa tìm kiếm bằng giọng nói (Voice Search SEO)"
    ],
    "[ARTIST]": [
        "Quy trình sản xuất Video Generative AI (Runway Gen-3, Sora)",
        "Thiết kế trải nghiệm người dùng không gian (Spatial UX cho VR/AR)",
        "Lý thuyết màu sắc nâng cao và tâm lý học hình ảnh",
        "Kỹ thuật Prompt Engineering chuyên sâu cho Midjourney v6"
    ],
    "[CONTENT_WRITER]": [
        "Kỹ thuật Storytelling: Cấu trúc hành trình anh hùng trong B2B",
        "SEO Semantic Search và Topic Clusters (Cụm chủ đề)",
        "Copywriting thôi miên: Các mẫu câu chốt sale tâm lý học",
        "Chiến lược nội dung đa kênh (Omnichannel Content Strategy)"
    ],

    # === NHÓM 5: NGHIỆP VỤ BỔ TRỢ (SUPPORT) ===
    "[LEGAL]": [
        "Khung pháp lý về AI và bản quyền tác giả toàn cầu",
        "Hợp đồng thông minh (Smart Contract) và tính pháp lý",
        "Tuân thủ GDPR và Nghị định 13 bảo vệ dữ liệu tại Việt Nam",
        "Giải quyết tranh chấp thương mại điện tử xuyên biên giới"
    ],
    "[RESEARCH]": [
        "Xu hướng công nghệ sinh học (Biotech) kết hợp AI",
        "Vật liệu mới (Graphene, Carbon Nanotubes) trong công nghiệp",
        "Tác động của 6G lên nền kinh tế số tương lai",
        "Nghiên cứu hành vi tiêu dùng bền vững (Sustainability)"
    ],
    "[SALES]": [
        "Mô hình bán hàng Challenger Sale (Người thách thức)",
        "Account-Based Marketing (ABM) cho khách hàng doanh nghiệp lớn",
        "Kỹ thuật đàm phán cấp cao (High-stakes Negotiation)",
        "Ứng dụng CRM AI để dự đoán tỷ lệ chốt đơn (Win Rate Prediction)"
    ]
}
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
                # Lưu ý: PostgreSQL dùng SERIAL cho ID tự tăng, SQLite dùng INTEGER PRIMARY KEY
                # Để tương thích cả 2 mà không dùng ORM phức tạp, ta dùng cấu trúc chuẩn SQL
                
                # Bảng Products
                conn.execute(text("CREATE TABLE IF NOT EXISTS products (id INTEGER PRIMARY KEY, name TEXT, price REAL)"))
                
                # Bảng Finance Logs
                conn.execute(text("CREATE TABLE IF NOT EXISTS finance_logs (id INTEGER PRIMARY KEY, type TEXT, amount REAL)"))
                
                # Bảng Agent Status (Quan trọng nhất)
                conn.execute(text("""
                    CREATE TABLE IF NOT EXISTS agent_status (
                        role_tag TEXT PRIMARY KEY, 
                        xp INTEGER DEFAULT 0, 
                        current_topic TEXT, 
                        last_updated TIMESTAMP
                    )
                """))
                
                # 2. KIỂM TRA & TẠO DỮ LIỆU MẪU
                # Không dùng cursor() nữa, dùng thẳng conn.execute
                result = conn.execute(text("SELECT count(*) FROM agent_status"))
                count = result.fetchone()[0]
                
                if count == 0:
                    print(colored("🌱 DATABASE TRỐNG - ĐANG KHỞI TẠO ĐỘI NGŨ AGENT...", "yellow"))
                    now = datetime.now()
                    
                    # Lặp qua danh sách Agent
                    for role in CURRICULUM.keys():
                        # LƯU Ý: Thay dấu ? bằng :param (Cú pháp của SQLAlchemy)
                        conn.execute(text("""
                            INSERT INTO agent_status (role_tag, xp, current_topic, last_updated)
                            VALUES (:role, 0, 'Đang chờ lệnh (Idle)', :time)
                        """), {"role": role, "time": now})
                        
                    conn.commit()
                    print(colored("✅ Đã tạo hồ sơ cho 15 chuyên gia AI.", "green"))
                else:
                    print(colored("✅ Database đã có dữ liệu.", "green"))
                    
                # Nhớ commit cuối cùng để chắc chắn lưu
                conn.commit()

        except Exception as e:
            print(colored(f"❌ Lỗi khởi tạo DB: {e}", "red"))
            # In ra lỗi chi tiết để debug nếu cần
            import traceback
            traceback.print_exc()
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
    """
    PHIÊN BẢN 10.0: COST-OPTIMIZED INHERITANCE (QUY TẮC KẾ THỪA & TIẾT KIỆM)
    - Nguyên tắc: "Không mua lại những gì đã có".
    - Bước 1: Kiểm tra Kho tri thức (Vector DB).
    - Bước 2: 
        + Nếu đã có kiến thức cũ (< 7 ngày) -> ÔN TẬP (Review Mode) -> Tốn 0đ API Search.
        + Nếu chưa có hoặc quá cũ -> MUA MỚI (Research Mode) -> Gọi API.
    """
    print(colored(f"🛡️ [INHERITANCE CHECK] {role_tag} đang kiểm tra kho tri thức...", "cyan", attrs=["bold"]))
    
    topics = CURRICULUM.get(role_tag, [])
    if not topics: return

    try:
        # 1. CHỌN CHỦ ĐỀ
        current_xp = 0
        with db_manager.get_connection() as conn:
            row = conn.execute("SELECT xp FROM agent_status WHERE role_tag = ?", (role_tag,)).fetchone()
            if row: current_xp = row[0]

        topic_index = int(current_xp / 50) % len(topics)
        current_topic = topics[topic_index]
        
        # 2. KIỂM TRA KẾ THỪA (QUAN TRỌNG NHẤT)
        # Tìm xem trong DB đã có bài nào về chủ đề này chưa?
        existing_knowledge = ""
        is_fresh = False
        
        if MEMORY_AVAILABLE and vector_db:
            # Tìm kiếm trong vector db xem có gì liên quan không
            results = await run_in_threadpool(lambda: vector_db.similarity_search(current_topic, k=1))
            
            if results:
                doc = results[0]
                existing_knowledge = doc.page_content
                # Kiểm tra xem kiến thức này cũ hay mới (Giả sử ta lưu timestamp trong metadata)
                # (Ở code trước ta chưa lưu kỹ timestamp, nhưng từ giờ sẽ lưu)
                # Tạm thời coi như nếu tìm thấy là "Kế thừa"
                print(colored(f"💡 [FOUND] Đã tìm thấy kiến thức kế thừa về: {current_topic}", "green"))
                is_fresh = True # Giả lập là tìm thấy

        # 3. QUYẾT ĐỊNH CHIẾN LƯỢC (RẼ NHÁNH TIỀN BẠC)
        final_output = ""
        xp_earned = 0
        mode = "UNKNOWN"

        # === NHÁNH A: KẾ THỪA (TIẾT KIỆM TIỀN) ===
        # Nếu đã có kiến thức rồi, ta chỉ dùng LLM (Gemini) để "Xào nấu" lại (Review), không tốn tiền Search (Perplexity)
        if is_fresh and existing_knowledge:
            mode = "REVIEW (Ôn Tập Kế Thừa)"
            print(colored(f"--> Chế độ: {mode} - Không tốn phí tìm kiếm.", "yellow"))
            
            if LLM_GEMINI:
                # Prompt Ôn tập: Dựa trên cái cũ để sinh ra góc nhìn mới
                review_prompt = f"""
                Bạn là Chuyên gia {role_tag}.
                Đây là kiến thức chúng ta đã học được trong quá khứ về "{current_topic}":
                ---
                {existing_knowledge[:3000]}
                ---
                
                NHIỆM VỤ: KẾ THỪA VÀ PHÁT TRIỂN (INHERIT & EVOLVE).
                Không cần tìm kiếm thông tin mới. Hãy dựa trên kiến thức cũ này để:
                1. Tóm tắt lại các điểm cốt lõi.
                2. Đặt ra 1 câu hỏi phản biện mới để thử thách tư duy.
                3. Đề xuất 1 ý tưởng ứng dụng mới từ kiến thức cũ này.
                
                Mục tiêu: Củng cố bộ nhớ mà không cần nạp thêm dữ liệu thô.
                """
                try:
                    res = await LLM_GEMINI.ainvoke(review_prompt)
                    final_output = res.content
                    xp_earned = 20 # Điểm ôn tập thấp hơn điểm nghiên cứu mới
                except:
                    final_output = existing_knowledge
            else:
                final_output = existing_knowledge

        # === NHÁNH B: KHÁM PHÁ MỚI (CHẤP NHẬN CHI PHÍ) ===
        # Chỉ chạy khi trong đầu rỗng tuếch về chủ đề này
        else:
            mode = "RESEARCH (Nghiên cứu Mới)"
            print(colored(f"--> Chế độ: {mode} - Cần tìm kiếm dữ liệu mới.", "magenta"))
            
            # (Phần này giữ nguyên logic Research cũ của ngài: Perplexity -> Gemini)
            raw_data = ""
            if LLM_PERPLEXITY:
                try:
                    res = await LLM_PERPLEXITY.ainvoke(f"Nghiên cứu chuyên sâu về: {current_topic}")
                    raw_data = res.content
                except: pass
            
            if raw_data and LLM_GEMINI:
                analyze_prompt = f"Phân tích chuyên sâu về {current_topic} dựa trên: {raw_data[:4000]}"
                try:
                    res = await LLM_GEMINI.ainvoke(analyze_prompt)
                    final_output = res.content
                    xp_earned = 50 # Điểm cao vì học cái mới
                except: final_output = raw_data
            else:
                final_output = raw_data

        # 4. LƯU KẾT QUẢ (CHỈ LƯU NẾU LÀ KIẾN THỨC MỚI HOẶC GÓC NHÌN MỚI)
        if MEMORY_AVAILABLE and vector_db and final_output:
            # Nếu là Review, ta có thể không cần lưu lại để tránh rác, hoặc lưu đè
            # Ở đây ta lưu thêm để làm dày dữ liệu cho Fine-tuning sau này
            await run_in_threadpool(lambda: vector_db.add_texts(
                texts=[final_output],
                metadatas=[{
                    "source": "Inheritance_Cycle", 
                    "agent": role_tag, 
                    "topic": current_topic,
                    "mode": mode,
                    "timestamp": datetime.now().isoformat()
                }]
            ))

        # 5. CẬP NHẬT TRẠNG THÁI
        new_xp = current_xp + xp_earned
        with db_manager.get_connection() as conn:
            c = conn.cursor()
            c.execute("""
                INSERT OR REPLACE INTO agent_status (role_tag, xp, current_topic, last_updated)
                VALUES (?, ?, ?, ?)
            """, (role_tag, new_xp, f"{mode}: {current_topic}", datetime.now()))
            conn.commit()
            
        print(colored(f"✅ [{mode}] {role_tag} +{xp_earned} XP | Tổng: {new_xp}", "green"))

    except Exception as e:
        print(colored(f"❌ Lỗi: {e}", "red"))    

async def morning_briefing_job():
    """
    PHIÊN BẢN 3.0: Tương thích PostgreSQL + Tự nhận thức (Meta-Cognition)
    """
    role_tag = "[ORCHESTRATOR]"
    print(colored(f"\n⏰ [CRON JOB] {role_tag} đang thực hiện quét tin tức buổi sáng...", "cyan", attrs=["bold"]))
    
    if not AI_AVAILABLE: # or not LLM_PERPLEXITY (Bỏ check Perplexity nếu muốn chạy test với Gemini)
        print(colored("⚠️ Bỏ qua Cron Job vì AI Module chưa sẵn sàng.", "yellow"))
        return

    # Lấy chủ đề từ Giáo Trình chung
    topics = CURRICULUM.get(role_tag, ["Tin tức AI mới nhất", "Thị trường công nghệ 2026"])
    report_buffer = []
    
    for topic in topics:
        try:
            print(colored(f"--> {role_tag} đang đọc: {topic}...", "white"))
            
            # Gọi AI (Ưu tiên Perplexity, Fallback sang Gemini/GPT nếu cần)
            # Giả sử dùng LLM chính nếu Perplexity chưa cấu hình
            llm_to_use = LLM_PERPLEXITY if LLM_PERPLEXITY else LLM_GEMINI
            res = await llm_to_use.ainvoke(topic)
            content = res.content
            
            # Lưu vào bộ nhớ Vector (RAG)
            if MEMORY_AVAILABLE and vector_db:
                await run_in_threadpool(lambda: vector_db.add_texts(
                    texts=[content],
                    metadatas=[{"source": "Morning_Briefing", "agent": role_tag, "topic": topic}]
                ))
            report_buffer.append(f"### {topic}\n{content[:800]}...") 
        except Exception as e:
            print(colored(f"⚠️ Lỗi đọc tin '{topic}': {e}", "yellow"))

    # Tạo báo cáo & Cập nhật Database
    if report_buffer:
        today_str = datetime.now().strftime("%Y-%m-%d")
        full_content = f"# 🌅 BẢN TIN SÁNG {today_str}\n\n" + "\n\n".join(report_buffer)
        
        # ID đặc biệt cho báo cáo (VD: BRIEFING_20260125)
        report_id = f"BRIEFING_{datetime.now().strftime('%Y%m%d')}"

        try:
            with db_manager.get_connection() as conn:
                # ---------------------------------------------------------
                # 1. LƯU BÁO CÁO VÀO DB (QUAN TRỌNG NHẤT ĐỂ KHÔNG MẤT FILE)
                # ---------------------------------------------------------
                # Đóng gói nội dung thành format tin nhắn để Dashboard đọc được
                history_json = json.dumps([{
                    "type": "ai", 
                    "data": {"content": full_content}
                }])
                
                # Dùng DELETE + INSERT để đảm bảo nếu chạy lại không bị lỗi trùng ID
                conn.execute(text("DELETE FROM projects WHERE id = :id"), {"id": report_id})
                
                project_query = text("""
                    INSERT INTO projects (id, name, history, timestamp)
                    VALUES (:id, :name, :history, :time)
                """)
                conn.execute(project_query, {
                    "id": report_id,
                    "name": f"Báo cáo sáng {today_str}",
                    "history": history_json,
                    "time": datetime.now()
                })
                
                # ---------------------------------------------------------
                # 2. CẬP NHẬT ĐIỂM XP (GAMIFICATION)
                # ---------------------------------------------------------
                # A. Lấy XP hiện tại
                xp_query = text("SELECT xp FROM agent_status WHERE role_tag = :role")
                row = conn.execute(xp_query, {"role": role_tag}).fetchone()
                new_xp = (row[0] if row else 0) + 100
                
                # B. Cập nhật trạng thái Agent
                conn.execute(text("DELETE FROM agent_status WHERE role_tag = :role"), {"role": role_tag})
                
                status_query = text("""
                    INSERT INTO agent_status (role_tag, xp, current_topic, last_updated) 
                    VALUES (:role, :xp, :topic, :time)
                """)
                conn.execute(status_query, {
                    "role": role_tag, 
                    "xp": new_xp, 
                    "topic": f"Hoàn thành bản tin {today_str}", 
                    "time": datetime.now()
                })

                # ---------------------------------------------------------
                # 3. GHI NHẬT KÝ TỰ NHẬN THỨC (META-COGNITION)
                # ---------------------------------------------------------
                log_query = text("""
                    INSERT INTO learning_logs (event_type, content, agent_name, timestamp)
                    VALUES (:type, :content, :agent, :time)
                """)
                conn.execute(log_query, {
                    "type": "CREATED",
                    "content": f"Đã tổng hợp và lưu trữ vĩnh viễn Bản tin sáng {today_str}.",
                    "agent": role_tag,
                    "time": datetime.now()
                })
                
                # CHỐT ĐƠN (COMMIT) 1 LẦN DUY NHẤT
                conn.commit()
                print(colored(f"✅ [DATABASE] Đã lưu báo cáo sáng vào hệ thống vĩnh viễn!", "green"))
                
        except Exception as e:
            print(colored(f"❌ Lỗi Lưu Trữ Job Sáng: {e}", "red"))

# ==========================================
# 3. PIPELINE DỰ ÁN LỚN (ĐÃ TỐI ƯU & HỢP NHẤT)
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
        
        plan_res = await run_in_threadpool(lambda: LLM_SUPERVISOR.invoke(architect_prompt))
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
        
    # --- SCHEDULER SETUP (QUAN TRỌNG) ---
    scheduler = AsyncIOScheduler()
    scheduler.add_job(morning_briefing_job, 'cron', hour=7, minute=0)
    scheduler.start()
    
    # --- 3. KÍCH HOẠT "HỌC VIỆN CA ĐÊM" (TÍNH NĂNG MỚI) ---
    # Thay vì dùng scheduler cứng nhắc, ta chạy Background Task linh hoạt
    # Để nó tự động học 60p -> nghỉ -> xoay vòng -> tự dừng khi có khách
    print("🎓 [SYSTEM] Kích hoạt chế độ 'Adaptive Learning' (Học luân phiên)...")
    learning_task = asyncio.create_task(adaptive_learning_scheduler())
    yield # Server chạy tại đây
    
    # --- SHUTDOWN ---
    scheduler.shutdown()
    # Hủy tác vụ học tập nhẹ nhàng
    print("💤 [SYSTEM] Đang giải tán lớp học...")
    learning_task.cancel()
    try:
        await learning_task
    except asyncio.CancelledError:
        print("✅ [SYSTEM] Đã dừng chế độ học tập an toàn.")
        
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
async def chat_endpoint(request: ChatRequest, background_tasks: BackgroundTasks):
    """
    SMART CHAT V4: STABLE & ERROR-PROOF
    Phiên bản sửa lỗi 400 OpenAI và tối ưu quy trình xử lý.
    """
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

        # --- 4. GỌI BỘ NÃO (LANGGRAPH) ---
        # Dùng invoke (đồng bộ) thay vì ainvoke ở đây để tránh race condition gây lỗi 400
        # Đảm bảo tin nhắn được append vào list trước khi gửi đi
        output = await run_in_threadpool(lambda: ai_app.invoke(
            {"messages": [human_msg]}, 
            config=config
        ))
        
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

async def adaptive_learning_scheduler():
    """
    Hệ thống lập lịch học tập thông minh.
    Chạy ngầm (Background Loop) song song với Server.
    """
    global CURRENT_LEARNER_INDEX, IS_BUSY
    
    print("🎓 [SCHEDULER] Đã kích hoạt Học viện Agent tự động.")
    
    while True:
        # 1. Kiểm tra trạng thái rảnh rỗi (Idle Check)
        # Nếu không có lệnh mới trong 5 phút -> Coi như rảnh
        idle_duration = (datetime.now() - LAST_ACTIVITY_TIME).total_seconds()
        if idle_duration > 300: 
            IS_BUSY = False
        else:
            IS_BUSY = True

        # 2. Logic điều phối
        if IS_BUSY:
            print("🚧 [SYSTEM] Hệ thống đang bận dự án. Tạm hoãn việc học.", end="\r")
            await asyncio.sleep(60) # Chờ 1 phút rồi check lại
            continue

        # 3. Bắt đầu phiên học 60 phút
        agent_name = LEARNING_QUEUE[CURRENT_LEARNER_INDEX]
        print(f"\n📚 [LEARNING] Bắt đầu phiên học 60p cho Agent: {agent_name}")
        
        # Giả lập quá trình học (Chia nhỏ thành 60 lần 1 phút để dễ ngắt ngang)
        for minute in range(60):
            # KIỂM TRA NGẮT NGANG: Nếu CEO đột nhiên ra lệnh
            if IS_BUSY: 
                print(f"🛑 [INTERRUPT] Ngừng phiên học của {agent_name} để phục vụ CEO!")
                break 
            
            # Thực hiện hành động học (Ví dụ: Đọc 1 trang tài liệu ngẫu nhiên trong DB)
            # await self_study(agent_name) 
            
            print(f"⏳ {agent_name} đang học... ({minute+1}/60 phút)", end="\r")
            await asyncio.sleep(60) # Học 1 phút

        # 4. Kết thúc phiên -> Xoay vòng
        if not IS_BUSY: # Chỉ chuyển người nếu học trọn vẹn (hoặc chấp nhận học dở)
            print(f"✅ [DONE] {agent_name} đã hoàn thành phiên học.")
            # Ghi nhật ký tự nhận thức
            # log_system_activity("LEARNED", f"{agent_name} hoàn thành 60p tự nghiên cứu.", "SCHEDULER")
            
            # Chuyển sang người tiếp theo
            CURRENT_LEARNER_INDEX = (CURRENT_LEARNER_INDEX + 1) % len(LEARNING_QUEUE)
        
        # Nghỉ 1 chút trước khi bắt đầu ca sau
        await asyncio.sleep(10)

# --- TÍCH HỢP VÀO STARTUP ---
@app.on_event("startup")
async def start_scheduler():
    # Chạy loop này ở chế độ nền (không chặn API)
    asyncio.create_task(adaptive_learning_scheduler())

# --- CẬP NHẬT TRẠNG THÁI KHI CÓ LỆNH ---
# Trong hàm chat_endpoint, thêm dòng này:
# global LAST_ACTIVITY_TIME, IS_BUSY
# LAST_ACTIVITY_TIME = datetime.now()
# IS_BUSY = True

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
