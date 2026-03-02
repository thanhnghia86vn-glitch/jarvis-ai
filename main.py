# ============================================================================
# 🚩 [SECTION 1] SYSTEM INITIALIZATION & CLOUD FIX
# ============================================================================
import sys, os, json, ast, asyncio, operator, re, time, shutil, sqlite3, logger
from datetime import datetime
from termcolor import colored
from dotenv import load_dotenv
from duckduckgo_search import DDGS
import requests
import aiosqlite
from langchain_chroma import Chroma
from langgraph.graph import StateGraph, END
from bs4 import BeautifulSoup
from concurrent.futures import ThreadPoolExecutor
from typing import TypedDict, Annotated, Sequence, Literal, List, Dict, Any
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.utilities.dalle_image_generator import DallEAPIWrapper
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
# Fix SQLite for Cloud Environments (Streamlit Cloud/Linux)
try:
    if os.name == 'posix':
        __import__('pysqlite3')
        sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
        print("✅ [SQLITE FIX] Đã kích hoạt pysqlite3.")
except ImportError: pass

load_dotenv()


# ============================================================================
# 🚩 [SECTION 2] DATA SCHEMA & STATE DEFINITION
# ============================================================================
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage

class AgentState(TypedDict):
    # messages: Danh sách tin nhắn (Cộng dồn)
    messages: Annotated[Sequence[BaseMessage], operator.add]
    # next_step: Tên Node tiếp theo (Bắt buộc là String)
    next_step: str 
    # current_agent: Agent đang xử lý hiện tại
    current_agent: str
    # error_log: Nhật ký lỗi (Cộng dồn)
    error_log: Annotated[list, operator.add]
    # task_type: Phân loại tác vụ (general/dynamic)
    task_type: str
# ==========================================
# 🚩 [PHẦN 2] ENGINE CONNECTORS
# ==========================================
def run_nexus_sync(cmd, thread_id):
    try:
        from main import app as nexus_app
        
        
        async def call_app():
            # Sử dụng ainvoke là chính xác cho LangGraph async
            response = await nexus_app.ainvoke(
                {"messages": [HumanMessage(content=cmd)]}, 
                {"configurable": {"thread_id": thread_id}, "recursion_limit": 50} # Tăng giới hạn để tránh lỗi Recursion
            )
            return response

        # Chạy loop đồng bộ
        result = asyncio.run(call_app())
        
        # --- ĐIỂM SỬA QUAN TRỌNG ---
        raw_content = result['messages'][-1].content
        
        # Nếu raw_content là một coroutine (do lỗi định nghĩa agent), ta phải await nó thêm lần nữa
        # Nhưng trong Streamlit (sync), ta sẽ dùng kiểm tra kiểu dữ liệu:
        if asyncio.iscoroutine(raw_content):
            # Nếu lỡ nhận về một coroutine, ta chạy nó để lấy chuỗi thực
            raw_content = asyncio.run(raw_content)
            
        return str(raw_content) # Ép kiểu về String để an toàn tuyệt đối
        
    except Exception as e: 
        return f"⚠️ [NEXUS_ERROR]: {str(e)}"
# ============================================================================
# 🚩 [SECTION 3] BRAIN ENGINES & FALLBACK STRATEGY
# ============================================================================
# ============================================================================
# 🚩 [SECTION 3] BRAIN ENGINES (LLM CONFIGURATION)
# ============================================================================
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_anthropic import ChatAnthropic

# --- 1. CODER CHỦ LỰC (DeepSeek - Tiết kiệm chi phí) ---
try:
    LLM_DEEPSEEK = ChatOpenAI(
        model="deepseek-chat", 
        api_key=os.environ.get("DEEPSEEK_API_KEY"), 
        base_url="https://api.deepseek.com",
        temperature=0
    )
    print("✅ LLM_DEEPSEEK: Ready (Coder Primary).")
except: LLM_DEEPSEEK = None

# --- 2. CHIẾN LƯỢC GIA (GPT-4o - Chính xác cao) ---
try:
    LLM_GPT4 = ChatOpenAI(
        model="gpt-4o",
        api_key=os.environ.get("OPENAI_API_KEY"),
        temperature=0
    )
    print("✅ LLM_GPT4: Ready (Strategy & Logic).")
except: LLM_GPT4 = None

# --- 3. ĐẠI VĂN HÀO & KIẾN TRÚC SƯ (Claude - Đẳng cấp nhất) ---
try:
    LLM_CLAUDE = ChatAnthropic(
        model="claude-sonnet-4-5", # Hoặc bản 4.5 như ngài yêu cầu
        api_key=os.environ.get("ANTHROPIC_API_KEY"),
        temperature=0
    )
    print("✅ LLM_CLAUDE (Anthropic): Ready (Architecture & Storytelling).")
except: LLM_CLAUDE = None

# --- 4. TRỢ LÝ TỐC ĐỘ (Gemini Flash - Miễn phí/Rẻ & Vision) ---
try:
    LLM_FAST = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash-lite", 
        google_api_key=os.environ.get("GOOGLE_API_KEY"),
        temperature=0.3
    )
    print("✅ LLM_FAST (Gemini): Ready (Quick Research & Vision).")
except: LLM_FAST = None

# 5. LLM_GEMINI (Supervisor - Tổng quản)
try:
    # A. Bản Logic (Xử lý văn bản dài cho Thư ký)
    LLM_GEMINI_LOGIC = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash-lite", 
        google_api_key=os.environ.get("GOOGLE_API_KEY"),
        temperature=0.3
    )
    
    # 6. Bản Vision (Nano Banana - Chuyên xử lý ảnh cho Artist)
    LLM_GEMINI_VISION = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash-lite", 
        google_api_key=os.environ.get("GOOGLE_API_KEY"),
        temperature=0.4
    )
    print("✅ [GEMINI 3 PRO] Ready: Logic & Vision (Nano Banana).")
except: 
    LLM_GEMINI_LOGIC = None
    LLM_GEMINI_VISION = None

# ============================================================================
# 🛡️ CƠ CHẾ DỰ PHÒNG TỰ ĐỘNG (FALLBACK)
# ============================================================================
# Nếu DeepSeek sập -> Dùng GPT-4. Nếu GPT-4 sập -> Dùng Claude.
try:
    backups = []
    if LLM_DEEPSEEK: backups.append(LLM_DEEPSEEK)
    if LLM_GPT4: backups.append(LLM_GPT4)
    primary_model = LLM_GEMINI_LOGIC if LLM_GEMINI_LOGIC else LLM_GPT4
    LLM_UNIVERSAL = primary_model.with_fallbacks(backups)
    print("🛡️ [SYSTEM] Auto-Fallback Activated.")
except:
    LLM_UNIVERSAL = LLM_GPT4
# =========================================================
# 3. PHÂN BỔ QUYỀN LỰC (ROLE MAPPING)
# =========================================================
# Đây là nơi quyết định ai làm việc gì.

# Coder & Architect -> DeepSeek (Tiết kiệm 95% chi phí)
CODER_PRIMARY = LLM_DEEPSEEK if LLM_DEEPSEEK else LLM_GPT4
ARCHITECT_PRIMARY = LLM_DEEPSEEK if LLM_DEEPSEEK else LLM_GPT4

# Supervisor (Điều phối) -> DeepSeek (Rất quan trọng để giảm bill)
SUPERVISOR_PRIMARY = LLM_DEEPSEEK if LLM_DEEPSEEK else LLM_GPT4 

# Artist Brain -> Dùng Gemini Vision (Nano Banana) để hiểu ảnh
ARTIST_BRAIN = LLM_GEMINI_VISION if LLM_GEMINI_VISION else LLM_GPT4

# Admin/Secretary -> Dùng Gemini Logic (Context lớn)
ADMIN_PRIMARY = LLM_GEMINI_LOGIC if LLM_GEMINI_LOGIC else LLM_GPT4

# Creative -> Claude
CREATIVE_PRIMARY = LLM_CLAUDE if LLM_CLAUDE else LLM_GPT4

# Logic/Finance/Legal -> GPT-4o (Cần độ chính xác cao nhất)
LOGIC_PRIMARY = LLM_GPT4

# Researcher -> Perplexity
RESEARCHER_PRIMARY = LLM_FAST

CODER_BACKUP = LLM_CLAUDE

# ============================================================================
# 🚩 [SYSTEM STATE INITIALIZATION]
# ============================================================================
def set_system_busy():
    """Hàm để Server gọi mỗi khi có tin nhắn từ CEO"""
    global IS_SYSTEM_BUSY, LAST_INTERACTION_TIME
    IS_SYSTEM_BUSY = True                    # Cờ hiệu luồng ưu tiên
    LAST_INTERACTION_TIME = datetime.now()


TEST_MODE = True
IS_SYSTEM_BUSY = False 
LAST_INTERACTION_TIME = datetime.now()
ACADEMY_IDX = 0
# ============================================================================
# 🚩 [SECTION 4] MIDDLEWARE & UTILITY TOOLS
# ============================================================================
DB_PATH = "./db_knowledge"

# 1. ĐẢM BẢO THƯ MỤC TỒN TẠI
if not os.path.exists(DB_PATH):
    os.makedirs(DB_PATH)
    print(colored(f"📁 [SYSTEM] Đã khởi tạo kho chứa tri thức: {DB_PATH}", "cyan"))

# 2. KHỞI TẠO BỘ NHỚ VỚI CƠ CHẾ CHỐNG XUNG ĐỘT
try:
    # Sử dụng OpenAIEmbeddings để vector hóa văn bản
    # Lưu ý: Cần đảm bảo OPENAI_API_KEY đã nạp ở Section 1
    embeddings = OpenAIEmbeddings()
    
    # Khởi tạo ChromaDB với cấu hình nâng cao
    vector_db = Chroma(
        persist_directory=DB_PATH, 
        embedding_function=embeddings,
        # Thêm collection_metadata để tối ưu hóa tìm kiếm HNSW
        collection_metadata={"hnsw:space": "cosine"} 
    )
    print(colored("🧠 [MEMORY] Bộ não tri thức (ChromaDB) đã trực tuyến.", "green", attrs=["bold"]))
    
except Exception as e:
    vector_db = None
    print(colored(f"❌ [MEMORY ERROR] Không thể kết nối bộ não: {e}", "red"))

def init_database_global():
    """Khởi tạo cấu trúc bảng với cơ chế Integrity Check."""
    db_path = "/var/data/ai_corp_projects.db" if os.path.exists("/var/data") else "ai_corp_projects.db"
    
    try:
        # Thêm timeout để tránh lỗi "database is locked" khi nhiều Agent cùng ghi log
        conn = sqlite3.connect(db_path, timeout=10) 
        c = conn.cursor()
        
        # Bật chế độ WAL (Write-Ahead Logging) giúp đọc/ghi song song nhanh hơn
        c.execute("PRAGMA journal_mode=WAL;")

        # 1. Bảng Project: Lưu lịch sử chat
        c.execute("""CREATE TABLE IF NOT EXISTS projects (
            id TEXT PRIMARY KEY, 
            name TEXT, 
            history TEXT, 
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )""")

        # 2. Bảng Agent Status: XP và Cấp độ
        c.execute("""CREATE TABLE IF NOT EXISTS agent_status (
            role_tag TEXT PRIMARY KEY, 
            xp INTEGER DEFAULT 0, 
            current_topic TEXT, 
            last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )""")

        # 3. Bảng Work Logs: Nhật ký chi phí (Cực kỳ quan trọng cho Dashboard)
        c.execute("""CREATE TABLE IF NOT EXISTS work_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT, 
            timestamp TEXT, 
            agent_name TEXT, 
            task_content TEXT, 
            result_summary TEXT, 
            tool_used TEXT, 
            cost REAL DEFAULT 0.0, 
            duration REAL DEFAULT 0.0
        )""")

        # 4. Bảng Meta-Cognition: Nhật ký tự nhận thức
        c.execute("""CREATE TABLE IF NOT EXISTS learning_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT, 
            event_type TEXT, 
            content TEXT, 
            agent_name TEXT, 
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )""")

        conn.commit()
        conn.close()
        print(colored(f"✅ [DATABASE] Hệ thống lưu trữ đã sẵn sàng tại: {db_path}", "green"))
        
    except Exception as e:
        print(colored(f"❌ [DATABASE ERROR] Sự cố khởi tạo: {e}", "red"))

# Triển khai ngay lập tức
init_database_global()

# 🧠 BẢN NÂNG CẤP: LEARN_KNOWLEDGE v5.0 (SMART INGESTION)
def learn_knowledge(text: str, source: str = "Manual_Input"):
    """
    Hệ thống tiêu hóa tri thức v5.0: Tự động chia mảnh và gán Metadata.
    """
    if not text or len(text.strip()) < 10:
        return "⚠️ Nội dung quá ngắn để cấu thành tri thức."

    try:
        # 1. CẤU HÌNH BỘ CHIA MẢNH (Text Splitter)
        # Chia nhỏ để AI dễ tìm kiếm (RAG), mỗi mảnh 1000 ký tự, gối đầu 100 ký tự
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=100,
            separators=["\n\n", "\n", ".", " ", ""]
        )
        chunks = text_splitter.split_text(text)

        # 2. CHUẨN BỊ METADATA (Dấu vân tay tri thức)
        # Giúp ngài biết kiến thức này từ đâu ra và nạp lúc nào
        metadatas = [{
            "source": source,
            "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
            "chunk_idx": i
        } for i in range(len(chunks))]

        # 3. NẠP VÀO CHROMADB
        # vector_db phải là instance của LangChain Chroma
        vector_db.add_texts(texts=chunks, metadatas=metadatas)
        
        # 
        
        print(colored(f"--> [MEMORY] Đã tiêu hóa {len(chunks)} mảnh tri thức từ: {source}", "green"))
        return f"✅ Đã nạp thành công {len(chunks)} mảnh tri thức vào bộ não trung tâm."

    except Exception as e:
        logger.error(f"❌ Lỗi tiêu hóa tri thức: {e}")
        return f"❌ Lỗi hệ thống: {str(e)}"
    
# 🚩[SECTION 4.1] HÀM GHI NHẬT KÝ VÀ TÍNH PHÍ (OPTIMIZED)
def log_work_to_db(agent_name, task, result, tool="Universal-AI", xp_bonus=50, start_time=None):
    """
    GHI CHÉP SỔ CÁI V6.1: Đã chuẩn hóa đối số 'agent_name' và cơ chế WAL.
    """
    db_path = "/var/data/ai_corp_projects.db" if os.path.exists("/var/data") else "ai_corp_projects.db"
    
    try:
        # 1. TÍNH TOÁN HIỆU SUẤT & CHI PHÍ
        duration = time.time() - start_time if start_time else 0
        content_length = len(str(result))
        base_rate = 0.00001
        
        # Hệ thống tính phí linh hoạt theo Model
        cost = content_length * base_rate
        tool_lower = tool.lower()
        if "deepseek" in tool_lower: cost /= 10
        elif "gemini" in tool_lower: cost /= 20

        # 2. KẾT NỐI AN TOÀN (Cơ chế Retry)
        conn = sqlite3.connect(db_path, timeout=30)
        c = conn.cursor()
        # Kích hoạt WAL mode để đọc/ghi song song không bị crash
        c.execute("PRAGMA journal_mode=WAL;") 
        
        # 3. GHI NHẬT KÝ CHI TIẾT
        c.execute("""
            INSERT INTO work_logs (timestamp, agent_name, task_content, result_summary, tool_used, cost, duration)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            datetime.now().strftime("%H:%M %d/%m"),
            agent_name.upper(),
            str(task)[:500],
            str(result)[:1000] + ("..." if content_length > 1000 else ""),
            tool,
            cost,
            duration
        ))

        # 4. CHUẨN HÓA TAG VAI TRÒ (Dành cho bảng agent_status)
        role_map = {
            "RESEARCHER": "[RESEARCH]", "CODER": "[CODER]", 
            "ARTIST": "[ARTIST]", "STORYTELLER": "[STORY]",
            "ORCHESTRATOR": "[ORCHESTRATOR]", "SUPERVISOR": "[SUPERVISOR]",
            "HARDWARE": "[HARDWARE]", "IOT_ENGINEER": "[IOT]",
            "STRATEGY_R_AND_D": "[STRATEGY]", "ACADEMY_MONITOR": "[ACADEMY]"
        }
        
        # Nếu agent_name là [ORCHESTRATOR] thì giữ nguyên, nếu là chuỗi thô thì map lại
        clean_name = agent_name.upper().replace("[", "").replace("]", "")
        target_role = role_map.get(clean_name, f"[{clean_name}]")
        
        # 5. CẬP NHẬT XP (UPSERT)
        c.execute("""
            INSERT INTO agent_status (role_tag, xp, current_topic, last_updated)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(role_tag) DO UPDATE SET
                xp = xp + excluded.xp,
                current_topic = excluded.current_topic,
                last_updated = excluded.last_updated
        """, (target_role, xp_bonus, f"Vừa hoàn thành: {str(task)[:40]}...", datetime.now().isoformat()))

        conn.commit()
        conn.close()
        
        print(colored(f"💰 [FINANCE] {target_role} -> +{xp_bonus} XP | Cost: ${cost:.6f}", "green"))

    except Exception as e:
        print(colored(f"⚠️ [LOG ERROR] Không thể ghi Sổ Cái: {e}", "yellow"))

def ingest_docs_to_memory(folder_path="./data_sources"):
    """
    Quy trình ETL chuyên nghiệp: Trích xuất, Biến đổi và Nạp tri thức vào Vector DB.
    Hỗ trợ: Metadata Mapping, Batch Loading và Integrity Check.
    """
    # 1. Khởi tạo & Kiểm tra môi trường
    if not os.path.exists(folder_path): 
        os.makedirs(folder_path)
        return f"📂 Thư mục '{folder_path}' đã được khởi tạo. Hãy thêm tài liệu PDF."

    print(colored(f"🚀 [ETL PROCESS] Bắt đầu nạp tri thức từ: {folder_path}", "cyan", attrs=["bold"]))

    # 2. Cấu hình Loader thông minh
    try:
        # Sử dụng DirectoryLoader với PyPDFLoader để bóc tách Metadata tự động
        loader = DirectoryLoader(
            folder_path, 
            glob="./*.pdf", 
            loader_cls=PyPDFLoader,
            show_progress=True,
            use_multithreading=True # Tối ưu hóa tốc độ đọc file
        )
        docs = loader.load()
    except Exception as e:
        return f"❌ Lỗi trích xuất (Extraction Error): {str(e)}"

    if not docs:
        return "⚠️ Trạng thái: Không tìm thấy tài liệu PDF mới để xử lý."

    # 3. Chiến lược phân mảnh (Chunking Strategy) chuyên sâu
    # Tăng overlap lên 200 để tránh mất ngữ cảnh giữa các đoạn (Context preservation)
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, 
        chunk_overlap=200,
        length_function=len,
        add_start_index=True # Lưu vị trí bắt đầu để truy xuất chính xác
    )
    splits = text_splitter.split_documents(docs)

    # 4. Làm sạch dữ liệu & Chuẩn hóa Metadata
    valid_splits = []
    for doc in splits:
        clean_content = doc.page_content.strip()
        if len(clean_content) > 50: # Loại bỏ các mẩu rác hoặc trang trắng
            # Bổ sung thông tin nguồn để AI trích dẫn sau này
            doc.metadata["ingested_at"] = datetime.now().isoformat()
            doc.metadata["doc_hash"] = hash(clean_content) # Hỗ trợ chống trùng lặp sơ bộ
            valid_splits.append(doc)

    if not valid_splits:
        return "⚠️ Cảnh báo: Tài liệu OCR/Ảnh không thể bóc tách nội dung văn bản."

    # 

    # 5. Nạp dữ liệu vào Vector DB theo từng Batch (Chống tràn RAM)
    try:
        batch_size = 100
        total_chunks = len(valid_splits)
        print(colored(f"📦 Đang mã hóa và nạp {total_chunks} phân đoạn vào bộ não...", "white"))
        
        for i in range(0, total_chunks, batch_size):
            batch = valid_splits[i:i + batch_size]
            vector_db.add_documents(batch)
            
        print(colored("✅ [INGESTION SUCCESS] Tri thức đã được đồng bộ hóa toàn diện.", "green", attrs=["bold"]))
        return f"🚀 Thành công: Đã nạp {total_chunks} phân đoạn từ {len(docs)} tài liệu vào bộ não trung tâm."

    except Exception as e:
        return f"❌ Lỗi nạp dữ liệu (Load Error): {str(e)}"


# 🚩[SECTION 4]: CƠ SỞ DỮ LIỆU & QUẢN LÝ TRI THỨC
async def search_memory(query: str, k: int = 5):
    """
    TRUY XUẤT TRI THỨC ĐA TẦNG (V3):
    1. Ưu tiên Core Legacy (Kiến thức L7-L9).
    2. Bổ sung Buffer Memory (Dữ liệu L1-L6) nếu thiếu.
    3. Trả về bối cảnh có độ tin cậy cao nhất.
    """
    print(colored(f"🔍 [RADAR SCAN] Đang truy vết tri thức: '{query}'", "cyan"))
    
    if 'vector_db' not in globals() or vector_db is None:
        return "⚠️ Hệ thống ký ức chưa được trực tuyến."

    try:
        # --- TẦNG 1: TRUY QUÉT KHO DI SẢN (CORE LEGACY) ---
        # Chúng ta dùng similarity_search_with_score để lấy điểm tin cậy
        results = await asyncio.to_thread(
            vector_db.similarity_search_with_score,
            query=query,
            k=k,
            filter={"knowledge_tier": "LEGACY"} # Chỉ lấy hàng "cực phẩm"
        )

        # --- TẦNG 2: BỔ SUNG TỪ KHO ĐỆM (BUFFER) ---
        # Nếu kho Di sản không đủ k kết quả, ta mới lấy thêm từ kho nghiên cứu thô
        if len(results) < k:
            buffer_results = await asyncio.to_thread(
                vector_db.similarity_search_with_score,
                query=query,
                k=k - len(results),
                filter={"knowledge_tier": "RESEARCH"}
            )
            results.extend(buffer_results)

        if not results:
            return "📡 Radar không tìm thấy tín hiệu tri thức liên quan."

        # --- TẦNG 3: ĐỐI CHIẾU & ĐÓNG GÓI NGỮ CẢNH ---
        valid_contents = []
        for doc, score in results:
            # Điểm ChromaDB: 0.0 là khớp tuyệt đối, > 1.0 là nhiễu.
            # Ta giữ ngưỡng 0.7 cho hàng Legacy và 0.6 cho hàng Research
            tier = doc.metadata.get('knowledge_tier', 'RESEARCH')
            threshold = 0.75 if tier == "LEGACY" else 0.65
            
            if score < threshold:
                confidence = round((1 - score) * 100, 2)
                timestamp = doc.metadata.get('timestamp', 'N/A')
                agent = doc.metadata.get('agent', 'Hệ thống')
                
                content_block = (
                    f"💎 [{tier}] | 👤 Nguồn: {agent} | 🕒 {timestamp}\n"
                    f"NỘI DUNG: {doc.page_content}\n"
                    f"⚡ Độ tin cậy: {confidence}%"
                )
                valid_contents.append(content_block)

        if not valid_contents:
            return "⚠️ Tìm thấy dữ liệu nhưng không vượt qua bộ lọc an toàn (Low Confidence)."

        # --- XUẤT BÁO CÁO NGỮ CẢNH ---
        header = f"\n🧠 [HỆ THỐNG KÝ ỨC PHÂN TẦNG - KẾT QUẢ CHO: {query.upper()}]\n"
        return header + "\n\n" + "\n---\n".join(valid_contents) + "\n" + "="*50

    except Exception as e:
        print(colored(f"❌ [MEMORY CRITICAL ERROR]: {str(e)}", "red"))
        return "Lỗi kỹ thuật nghiêm trọng khi truy xuất bộ não."
# 🧠 PHẦN 3: TỐI ƯU HÓA BỘ NHỚ VECTOR (CORE VS. BUFFER MEMORY)
async def save_to_vector_memory(content, agent_level, metadata):
    """
    Hệ thống lọc ký ức: 
    - Level 1-6: Lưu vào 'buffer_collection' (Ký ức tạm).
    - Level 7-9: Lưu vào 'core_legacy_collection' (Ký ức vĩnh cửu).
    """
    if not vector_db: return
    
    # Xác định vị trí lưu trữ dựa trên đẳng cấp của Agent
    collection_name = "core_legacy" if agent_level >= 7 else "buffer_memory"
    
    # Gắn nhãn chất lượng (Quality Tag)
    metadata["knowledge_tier"] = "LEGACY" if agent_level >= 7 else "RESEARCH"
    metadata["timestamp"] = datetime.now().isoformat()

    try:
        # Chuyển đổi sang thread để không block async
        await asyncio.to_thread(
            vector_db.add_texts,
            texts=[content],
            metadatas=[metadata],
            collection_name=collection_name # Phân mảnh collection
        )
        print(colored(f"💾 [MEMORY] Đã ghi vào {collection_name.upper()} (Level {agent_level})", "cyan"))
    except Exception as e:
        print(colored(f"❌ [MEMORY ERROR] Thất bại khi ghi ký ức: {e}", "red"))
# 🚩[SECTION 4.2] HÀM TRÍCH XUẤT NỘI DUNG SẠCH (CLEAN SCRAPER)

async def free_deep_research(query):
    print(colored(f"🕵️ [FREE SCOUT] Đang đào dữ liệu: {query}...", "cyan"))
    
    # 1. TÌM KIẾM LINK (AsyncDDGS - FIXED)
    try:
        from duckduckgo_search import AsyncDDGS
        async with AsyncDDGS() as ddgs:
            # Lấy 8 kết quả chất lượng nhất
            search_results = [r for r in await ddgs.text(query, max_results=8)]
        
        if not search_results:
            return "📡 Radar không tìm thấy tín hiệu phù hợp trên Internet."
    except Exception as e:
        return f"⚠️ Lỗi Radar DuckDuckGo: {str(e)}"

    # 2. TRÍCH XUẤT SONG SONG (SỬ DỤNG EXECUTOR KHÔNG CHẶN LUỒNG)
    print(colored(f"🌊 Cử 8 thợ lặn xuống các tầng dữ liệu...", "dark_grey"))
    
    loop = asyncio.get_event_loop()
    with ThreadPoolExecutor(max_workers=8) as executor:
        # Chạy fetch_url trong executor để không làm treo Dashboard
        tasks = [loop.run_in_executor(executor, fetch_url, r) for r in search_results]
        contents = await asyncio.gather(*tasks)

    # Loại bỏ kết quả rỗng và tổng hợp
    valid_contents = [c for c in contents if c]
    raw_knowledge = "".join(valid_contents)

    # 3. TỔNG HỢP TRÍ TUỆ (GEMINI FLASH)
    if not raw_knowledge:
        return "🛑 Nguồn tin bị mã hóa hoặc tất cả các trang đều chặn truy cập bot. Không thể trích xuất dữ liệu."

    # Giới hạn 30,000 ký tự để tối ưu chi phí và tốc độ Flash
    truncated_knowledge = raw_knowledge[:30000]

    analyze_prompt = f"""
    Bạn là TRÍ TUỆ PHÂN TÍCH của AI Corporation. 
    Nhiệm vụ: Tổng hợp báo cáo chiến lược cho yêu cầu: "{query}"

    NGUỒN DỮ LIỆU THÔ ĐÃ THU THẬP:
    {truncated_knowledge}

    CHỈ THỊ:
    - Trình bày dạng Markdown chuyên nghiệp.
    - Giữ lại các con số và Domain nguồn.
    - Nếu có so sánh giá (với linh kiện), hãy lập bảng.
    """
    
    try:
        response = await LLM_FAST.ainvoke(analyze_prompt)
        return response.content
    except Exception as e:
        return f"💥 Lỗi phân tích tầng sâu: {str(e)}"

def fetch_url(r):
    """ Thợ lặn dữ liệu: Tối ưu hóa việc bóc tách nội dung chính """
    try:
        # Sử dụng Session để giả lập hành vi người dùng tốt hơn
        session = requests.Session()
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/120.0.0.0 Safari/537.36',
        }

        response = session.get(r['href'], headers=headers, timeout=7)
        response.encoding = response.apparent_encoding 

        if response.status_code != 200:
            return ""

        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Xóa rác (Giữ lại nội dung cốt lõi)
        for tag in soup(["script", "style", "nav", "footer", "header", "aside", "form"]):
            tag.decompose()

        # TRÍCH XUẤT CÓ TRỌNG TÂM: Chỉ lấy nội dung trong body và các thẻ text
        texts = soup.find_all(['p', 'h1', 'h2', 'h3', 'article', 'li'])
        main_text = " ".join([t.get_text() for t in texts])
        
        # Làm sạch khoảng trắng
        clean_text = " ".join(main_text.split())
        
        if len(clean_text) < 200:
            return ""

        # Lấy tối đa 2500 ký tự mỗi trang để đa dạng hóa nguồn tin
        return f"\n--- NGUỒN: {r['title']} ---\n{clean_text[:2500]}\n"

    except:
        return ""
# 🚩 [SECTION 4.5] HỆ THỐNG THU THẬP DỮ LIỆU HUẤN LUYỆN (SELF-EVOLUTION)
def log_training_data(user_input, ai_output, success=True, model_name="Unknown"):
    """
    Hệ thống tích lũy tri thức: Lưu lại các cặp (Input/Output) chất lượng cao.
    Định dạng: JSONL chuẩn OpenAI/HuggingFace 2026.
    """
    # 1. BỘ LỌC CHẤT LƯỢNG (QUALITY GATE)
    if not success: 
        return # Chỉ học từ những kết quả hoàn hảo

    # 2. LÀM SẠCH DỮ LIỆU
    # Loại bỏ các thông tin rác hoặc tag hệ thống để Dataset "tinh khiết"
    clean_input = re.sub(r"\[.*?\]", "", str(user_input)).strip()
    clean_output = str(ai_output).strip()

    if len(clean_output) < 50: 
        return # Không lưu các câu trả lời quá ngắn/vô nghĩa

    # 3. CẤU TRÚC DỮ LIỆU HUẤN LUYỆN (Định dạng ChatML chuẩn 2026)
    data_entry = {
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "model_source": model_name,
            "quality_score": "High"
        },
        "messages": [
            {"role": "system", "content": "Bạn là trợ lý đắc lực của AI Corporation."},
            {"role": "user", "content": clean_input},
            {"role": "assistant", "content": clean_output}
        ]
    }
    
    # 4. GHI FILE VỚI CHẾ ĐỘ AN TOÀN (Thread-safe append)
    try:
        file_name = "ai_corp_training_v1.jsonl"
        with open(file_name, "a", encoding="utf-8") as f:
            f.write(json.dumps(data_entry, ensure_ascii=False) + "\n")
            
        # Ghi thêm một bản log nhẹ vào SQLite để CEO dễ thống kê số lượng "bài học"
        try:
            log_work_to_db("Academy", "Training Data Collected", f"Len: {len(clean_output)}", tool="Evolution")
        except: pass

    except Exception as e:
        print(colored(f"⚠️ [ACADEMY ERROR] Không thể lưu dữ liệu tiến hóa: {e}", "yellow"))

# 🚩 [SECTION 4.6] CỔNG ĐIỀU KHIỂN THIẾT BỊ (HARDWARE BRIDGE)
def hardware_controller(command_json: str):
    """
    Cổng kết nối phần cứng thực tế qua giao thức MQTT/HTTP.
    Đầu vào: JSON string chứa thiết bị và hành động.
    """
    print(colored(f"📡 [HARDWARE BRIDGE] Đang truyền lệnh: {command_json}", "magenta"))
    
    try:
        # 1. PHÂN TÍCH LỆNH (Trường hợp AI gửi JSON)
        import json
        cmd_data = json.loads(command_json) if isinstance(command_json, str) and "{" in command_json else {"command": command_json}
        
        target_device = cmd_data.get("device", "GENERAL_SYSTEM")
        action = cmd_data.get("action", cmd_data.get("command", "UNKNOWN"))
        
        # 2. XÁC THỰC AN TOÀN (SAFETY GATE)
        # Ngăn chặn các lệnh nguy hiểm gây cháy nổ hoặc quá tải
        forbidden_actions = ["OVERCLOCK", "DISABLE_SAFETY", "MAX_VOLTAGE"]
        if action.upper() in forbidden_actions:
            return f"🚫 [HARDWARE DENIED] Lệnh '{action}' bị chặn do vi phạm quy tắc an toàn vật lý."

        # 3. MÔ PHỎNG TRUYỀN TẢI GIAO THỨC (Nơi ngài sẽ đặt code MQTT/Serial tại đây)
        # Giả lập độ trễ vật lý của thiết bị (Latency)
        time.sleep(0.5) 
        
        status_report = {
            "status": "SUCCESS",
            "device": target_device,
            "action": action,
            "timestamp": time.strftime("%H:%M:%S"),
            "execution_time": "500ms"
        }

        # 4. GHI NHẬT KÝ VẬN HÀNH (Chống chối bỏ trách nhiệm)
        with open("hardware_operation_log.csv", "a") as f:
            f.write(f"{time.time()},{target_device},{action},SUCCESS\n")

        return f"✅ [IOT SYSTEM] Thực thi: {action} trên {target_device} | Phản hồi: OK."

    except Exception as e:
        return f"❌ [IOT ERROR] Sự cố kết nối phần cứng: {str(e)}"

# ============================================================================
# 🚩 [SECTION 5] AGENT NODES (THE DEPARTMENTS)
# ============================================================================
# 🚩 [SECTION 5.1] SUPERVISOR NODE (PHIÊN BẢN BỌC THÉP)
async def supervisor_node(state: AgentState):
    """
    SUPERVISOR V12: HỢP NHẤT TOÀN DIỆN
    - Chống Coroutine error (.strip())
    - Chống lặp vô hạn (Result Detection)
    - Chống Zombie Loop
    """
    messages = state.get("messages", [])
    if not messages:
        return {"next_step": "Secretary", "current_agent": "Supervisor"}

    last_msg = messages[-1].content
    
    # --- [BẢO VỆ 1] NHẬN DIỆN THÀNH QUẢ ĐÃ CÓ ---
    # Kiểm tra 5 tin nhắn gần nhất để xem Researcher/Engineering đã làm xong chưa
    history_str = "".join([str(m.content) for m in messages[-5:]]).upper()
    has_research = "BÁO CÁO NGHIÊN CỨU" in history_str
    has_engineering = "KIẾN TRÚC" in history_str or "BẢN VẼ" in history_str

    try: ceo_context = get_ceo_context_prompt()
    except: ceo_context = "Bạn là Quản lý dự án cấp cao của AI Corporation."

    print(colored(f"🧠 [SUPERVISOR] Đang phân tích chỉ thị: '{last_msg[:50]}...'", "cyan", attrs=["bold"]))

    # --- [BẢO VỆ 2] ZOMBIE CHECK (Ngăn chặn vòng lặp hệ thống) ---
    if check_zombie_loop(messages):
        return {
            "messages": [AIMessage(content="🛑 CẢNH BÁO: Phát hiện vòng lặp vô tận. Đang cưỡng bức kết thúc luồng để bảo vệ tài nguyên.")], 
            "next_step": "FINISH",
            "current_agent": "Supervisor"
        }

    # --- [BẢO VỆ 3] PROMPT CHIẾN LƯỢC (Đã tích hợp điều kiện dừng) ---
    prompt = f"""
    {ceo_context}
    
    TÌNH TRẠNG DỰ ÁN:
    - Đã có dữ liệu nghiên cứu: {"ĐÚNG" if has_research else "SAI"}
    - Đã có thiết kế hệ thống: {"ĐÚNG" if has_engineering else "SAI"}

    CHỈ THỊ CEO: "{last_msg}"

    QUY TẮC ĐIỀU PHỐI:
    1. Nếu đã có 'Báo cáo nghiên cứu', KHÔNG ĐƯỢC chọn Researcher nữa. Hãy chuyển sang Coder hoặc FINISH.
    2. Nếu đã có 'Kiến trúc/Bản vẽ', hãy chuyển sang Coder để viết code thực tế.
    3. Nếu chỉ là trò chuyện xã giao, chọn Secretary.

    TRẢ VỀ JSON:
    {{
        "intent": "SOFTWARE_BUILD | QUICK_FIX | DEEP_RESEARCH | CHAT | FINISH",
        "next_agent": "Engineering | Coder | Researcher | Secretary | FINISH",
        "reason": "Lý do ngắn gọn"
    }}
    """
    
    try:
        # Sử dụng await để tránh lỗi Coroutine (.strip() sau này)
        res = await LLM_UNIVERSAL.ainvoke([SystemMessage(content=prompt), HumanMessage(content=last_msg)])
        
        # --- [BẢO VỆ 4] XỬ LÝ CHUỖI AN TOÀN ---
        # Đảm bảo res.content là chuỗi trước khi dùng .strip()
        raw_content = str(res.content).strip() 
        
        clean_json = re.search(r'\{.*\}', raw_content, re.DOTALL)
        if not clean_json:
            raise ValueError("Không tìm thấy cấu trúc JSON trong phản hồi của Supervisor.")
            
        json_data = json.loads(clean_json.group())
        intent = json_data.get("intent")
        target_agent = json_data.get("next_agent")

        # --- [BẢO VỆ 5] LOGIC ĐIỀU PHỐI CƯỠNG BỨC (Bẻ lái nếu AI sai lầm) ---
        if has_research and target_agent == "Researcher":
            target_agent = "Secretary"
            instruction = "✅ Đã đủ dữ liệu nghiên cứu. Thư ký đang trình báo cáo cho ngài."
        elif has_engineering and intent == "SOFTWARE_BUILD":
            target_agent = "Coder"
            instruction = "🛠️ Đã có bản vẽ. Đang chuyển cho Coder thực thi mã nguồn."
        else:
            # Mapping chuẩn của ngài
            mapping = {
                "SOFTWARE_BUILD": ("Engineering", "🏗️ Đang lập bản vẽ thiết kế."),
                "QUICK_FIX": ("Coder", "👨‍💻 Coder đang thực thi mã nguồn."),
                "DEEP_RESEARCH": ("Researcher", "🔍 Đang đào dữ liệu tầng sâu."),
                "CHAT": ("Secretary", "📝 Thư ký đang soạn thảo phản hồi."),
                "FINISH": ("FINISH", "📊 Hoàn tất quy trình tác chiến.")
            }
            target_agent, instruction = mapping.get(intent, (target_agent, f"🔄 Chuyển giao: {target_agent}"))

        return {
            "messages": [AIMessage(content=instruction)], 
            "next_step": target_agent, 
            "current_agent": "Supervisor"
        }

    except Exception as e:
        print(colored(f"❌ [SUPERVISOR FATAL ERROR]: {e}", "red"))
        # Fallback an toàn nhất: Không bao giờ để hệ thống chết, đẩy về Secretary
        return {
            "messages": [AIMessage(content=f"⚠️ Sự cố bộ não: {str(e)}. Thư ký tạm thời tiếp quản.")], 
            "next_step": "Secretary",
            "current_agent": "Supervisor"
        }
#  [SECTION 5.1.1] HÀM DIỆT VÒNG LẶP (ZOMBIE LOOP DETECTOR)
def check_zombie_loop(messages, threshold=3):
    """
    Cảm biến chống lặp: Phát hiện AI bị "kẹt đĩa" hoặc vòng lặp vô tận giữa các Agent.
    """
    # 1. TRÍCH XUẤT NỘI DUNG AI (Bỏ qua tin nhắn của CEO)
    # Lấy 10 tin nhắn AI gần nhất để phân tích
    ai_msgs = [m.content for m in messages if isinstance(m, AIMessage)][-10:]
    
    if len(ai_msgs) < threshold: 
        return False
    
    # --- CẤP ĐỘ 1: LẶP NỘI DUNG (HARD REPEAT) ---
    # Kiểm tra xem N tin nhắn cuối có nội dung gần giống nhau không (tỉ lệ 90% trở lên)
    last_msg = ai_msgs[-1].strip()
    repeats = 0
    for msg in reversed(ai_msgs[:-1]):
        # Sử dụng so sánh độ dài hoặc set() để phát hiện lặp biến tướng
        if msg.strip() == last_msg or (len(msg) == len(last_msg) and msg[:50] == last_msg[:50]):
            repeats += 1
        else:
            break 
            
    if repeats >= threshold:
        print(colored(f"🚨 [ALERT] Phát hiện lặp nội dung ({repeats+1} lần)!", "red", attrs=["bold"]))
        return True 

    # --- CẤP ĐỘ 2: LẶP LUỒNG (ROUTING LOOP) ---
    # Kiểm tra xem Supervisor có đang gọi đi gọi lại 1 Agent mà không có HumanMessage xen vào không
    # Nếu trong 6 tin nhắn cuối không có tin nhắn từ Human (CEO), mà số lượng tin nhắn quá lớn
    # chứng tỏ các Agent đang "tự chơi với nhau" mà không ra kết quả.
    human_present = any(isinstance(m, HumanMessage) for m in messages[-6:])
    if len(messages) > 15 and not human_present:
        # Nếu đã đi quá 15 bước mà CEO chưa được phản hồi hoặc chưa can thiệp
        # Đây là dấu hiệu AI đang lạc lối.
        print(colored("⚠️ [WARN] AI đang sa vào luồng suy nghĩ quá sâu mà không có chỉ thị mới.", "yellow"))
        # Tùy CEO quyết định có chặn tại đây không, hoặc tăng threshold lên 20.
    
    return False    
# 🚩 [SECTION 5.1.2] CÁ NHÂN HÓA CHIẾN LƯỢC (CEO CONTEXT)
def get_ceo_context_prompt():
    """
    Kiến tạo hệ giá trị và phong cách phục vụ riêng cho CEO.
    Đảm bảo mọi Agent đều giữ đúng vị thế Trợ lý cấp cao.
    """
    # 1. TRÍCH XUẤT THÔNG TIN (An toàn)
    name = CEO_PROFILE.get('name', 'CEO')
    role = CEO_PROFILE.get('role', 'Executive Director')
    style = CEO_PROFILE.get('style', {}).get('communication', 'Chuyên nghiệp, ngắn gọn')
    interests = ", ".join(CEO_PROFILE.get('interests', []))
    dislikes = ", ".join(CEO_PROFILE.get('dislikes', []))

    # 2. XÂY DỰNG KHUNG NGUYÊN TẮC (GOVERNANCE FRAMEWORK)
    return f"""
<executive_order>
    BỐI CẢNH NGƯỜI ĐIỀU HÀNH (VITAL CONTEXT):
    - ĐỐI TƯỢNG PHỤC VỤ: {name} ({role}).
    - PHONG CÁCH GIAO TIẾP: {style}.
    - LĨNH VỰC ƯU TIÊN: {interests}.
    - DANH SÁCH CẤM (CRITICAL AVOID): {dislikes}.

    QUY TẮC ỨNG XỬ CHO AGENT:
    1. LUÔN giữ thái độ của một cộng sự cấp cao (Senior Associate), không trả lời như chatbot thông thường.
    2. ƯU TIÊN các giải pháp thực tiễn, có số liệu và khả năng thực thi thay vì lý thuyết suông.
    3. ĐIỀU CHỈNH ngôn ngữ và kiến thức chuyên môn dựa trên danh sách "Mối quan tâm chính".
    4. TUYỆT ĐỐI không vi phạm danh sách cấm, ngay cả khi CEO yêu cầu (để bảo vệ an toàn hệ thống).
</executive_order>

HÃY COI ĐÂY LÀ CHỈ THỊ GỐC TRƯỚC KHI THỰC HIỆN BẤT KỲ TÁC VỤ NÀO.
"""
# 🚩 [SECTION 5.1.3] HÀM NẠP HỒ SƠ LÃNH ĐẠO (REFACTORED)
def load_ceo_profile():
    """
    Nạp hồ sơ CEO từ file vật lý với cơ chế kiểm lỗi cú pháp JSON.
    Đảm bảo AI luôn biết mình đang phục vụ ai.
    """
    file_path = "ceo_profile.json"
    
    # Profile mặc định (Standard Operating Procedure)
    default_profile = {
        "name": "CEO",
        "role": "Executive Leader",
        "style": {"communication": "Ngắn gọn, quyết đoán, tập trung kết quả"},
        "interests": ["AI Technology", "Business Growth", "Automation"],
        "dislikes": ["Giải thích dài dòng", "Lỗi cú pháp", "Sự chậm trễ"]
    }

    if not os.path.exists(file_path):
        print(colored(f"⚠️ [PROFILE] Chưa tìm thấy {file_path}. Sử dụng cấu hình mặc định.", "yellow"))
        return default_profile

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            profile = json.load(f)
            # Kiểm tra xem các trường quan trọng có tồn tại không
            for key in default_profile:
                if key not in profile:
                    profile[key] = default_profile[key]
            
            print(colored(f"👤 [PROFILE] Đã nạp danh tính: {profile['name']}", "green"))
            return profile
            
    except json.JSONDecodeError as e:
        # Lỗi này xảy ra khi file JSON của ngài bị sai cú pháp (vd: thừa dấu phẩy)
        print(colored(f"❌ [PROFILE ERROR] File JSON sai cấu trúc: {e}", "red"))
        return default_profile
    except Exception as e:
        print(colored(f"❌ [PROFILE ERROR] Lỗi không xác định: {e}", "red"))
        return default_profile
CEO_PROFILE = load_ceo_profile()
# 🚩 [SECTION 5.2] HÀM QUY HOẠCH CÔNG NGHỆ (TECH STACK OPTIMIZER)
def get_optimal_stack(task_type: str) -> str:
    """
    Chiến lược hóa Stack công nghệ AI-Native 2026.
    Tối ưu cho việc AI tự viết code, tự kiểm thử và tự vận hành.
    """
    task_type = task_type.lower()
    
    stacks = {
        # 1. WEB/MOBILE: Chuyển dịch sang Serverless & Edge Runtime
        "web": "Next.js 16 (App Router), Tailwind CSS 4.0, TypeScript 5.5, Shadcn/UI v2, Bun Runtime (Tốc độ cao hơn Node.js)",
        
        # 2. BACKEND: Ưu tiên High-Concurrency & AI Integration
        "backend": "Python 3.14 (GIL-free), Litestar/FastAPI 0.110+, Rust-based Python tools (Polaris/Pydantic V3), Redis 8.0 (Vector Engine)",
        
        # 3. IOT/ROBOTICS: Kỷ nguyên Edge AI
        "iot": "Rust (Embedded), MicroPython 2.0, TinyML (chạy AI trên chip), Matter Protocol (chuẩn nhà thông minh 2026), ESP-IDF v6.0",
        
        # 4. DATA & AI AGENT: Tập trung vào Vector & Graph Data
        "data": "DuckDB (In-process analytical), Polars (thay thế hoàn toàn Pandas về tốc độ), LangGraph (Agentic Workflow), LanceDB (Serverless Vector DB)",
        
        # 5. AUTOMATION: Agentic Automation
        "tool": "Python Typer, Playwright (Auto-healing mode), Mojo (cho các tác vụ cần hiệu năng C++ nhưng cú pháp Python)",
        
        # 6. STREAMLIT/UI DASHBOARD: Enterprise AI Dashboard
        "streamlit": "Streamlit 1.45+ (Multi-page native), Streamlit-Components-Pro, Vega-Altair v5, Custom React Components Integration"
    }
    
    # 7. MẶC ĐỊNH: Tiêu chuẩn AI-Generated Code 2026
    default_stack = "Python 3.14, Async-first, Functional Programming, AI-Documented"
    
    return stacks.get(task_type, default_stack)

async def coder_node(state: AgentState):
    """
    CODER V3: PARALLEL ENSEMBLE & SELF-HEALING
    Sửa lỗi KeyError, tối ưu Token và nâng cấp bộ lọc chất lượng.
    """
    print(colored("[🚀 CODER V3] Kích hoạt chế độ Ensemble song song...", "green", attrs=["bold"]))
    
    # 1. SETUP & TRIMMING
    errors = state.get("error_log", [])
    task_type = state.get("task_type", "general").lower()
    # Chỉ lấy 10 tin nhắn gần nhất để tránh tràn bộ nhớ
    recent_messages = state['messages'][-10:] 
    last_user_msg = state['messages'][-1].content
    
    # Lấy tri thức từ bộ não
    try:
        memory_context = search_memory("Best practices for " + task_type)
    except:
        memory_context = "Follow SOLID principles and PEP8."

    # 2. GENERATE VARIANTS
    base_prompt = get_claude_perfected_prompt(task_type, memory_context, str(errors), last_user_msg)
    prompts = [base_prompt]
    
    # Nếu đang sửa lỗi, yêu cầu 2 biến thể khác nhau để tìm giải pháp tối ưu
    if len(errors) > 0:
        prompts.append(base_prompt + "\n[ALTERNATIVE]: Try a different architectural approach than previous failed attempts.")
    
    batch_inputs = [[SystemMessage(content=p)] + recent_messages for p in prompts]
    
    # 3. EXECUTION WITH FALLBACK
    try:
        # Sử dụng chuỗi Fallback ngài đã định nghĩa (DeepSeek -> GPT4 -> Claude)
        responses = await LLM_UNIVERSAL.abatch(batch_inputs)
    except Exception as e:
        error_msg = f"API Batch Error: {str(e)}"
        return ultimate_fallback(state, [error_msg])

    # 4. SCORING & VALIDATION (BẢN TINH CHỈNH)
    valid_results = []
    for i, res in enumerate(responses):
        content = res.content
        code = extract_code_block(content)
        if not code: continue
        
        is_ok, msg = real_syntax_validator(code, "python")
        
        # HỆ THỐNG CHẤM ĐIỂM CHI TIẾT
        score = 0
        if is_ok: score += 50
        if "try:" in code and "except" in code: score += 10 # Ưu tiên code có xử lý lỗi
        if '"""' in code or "'''" in code: score += 10 # Ưu tiên code có docstring
        if len(code) > 20: score += 10
        if "# filename:" in code: score += 10
        
        valid_results.append({
            "code": code, 
            "full_reply": content, # Đã đồng bộ tên key với return
            "score": score, 
            "error": msg, 
            "variant": i
        })

    # 5. ĐIỀU PHỐI TIẾP THEO
    if valid_results:
        best = max(valid_results, key=lambda x: x['score'])
        
        if best['score'] >= 60:
            print(colored(f"✅ CHỌN BIẾN THỂ {best['variant']} (Score: {best['score']})", "green"))
            
            # TỰ HỌC: Lưu vào dataset nếu code cực phẩm
            if best['score'] >= 80:
                log_training_data(last_user_msg, best['code'], success=True)
            
            return {
                "messages": [AIMessage(content=best['full_reply'])],
                "next_step": "Tester",
                "current_agent": "Coder",
                "error_log": [] # Xóa sạch lỗi vì đã vượt qua vòng kiểm tra
            }
        
    # THẤT BẠI: Nếu không có code nào đạt chuẩn hoặc lỗi cú pháp
    return {
        "messages": [HumanMessage(content=f"Lỗi cú pháp: {valid_results[0]['error'] if valid_results else 'Không tạo được code'}")] ,
        "next_step": "Coder", # Bắt làm lại
        "current_agent": "Coder",
        "error_log": errors + ["Syntax error detected"]
    }
# 🚩 [SECTION 5.3] HÀM KIẾN TẠO PROMPT CHIẾN LƯỢC (REFACTORED)
def get_claude_perfected_prompt(task_type: str, memory: str, error: str, user_request: str) -> str:
    """
    Tạo prompt tối ưu cho Claude (Reflexion Mode).
    Mục tiêu: Tự soi xét lỗi sai (Self-Correction) và tối ưu hóa kiến trúc.
    """
    # 1. Xác định Stack công nghệ (Gọi từ hàm helper hiện có)
    tech_stack = get_optimal_stack(task_type)
    
    # 2. Xây dựng nội dung Prompt
    # Bổ sung: Cấu trúc hóa Error History để AI không bị rối nếu log lỗi quá dài
    formatted_error = f"\n[BUG REPORT]:\n{error.strip()}" if error else "[STATUS]: Clean Start - No bugs detected."

    prompt = f"""
<system_context>
    <role>
        Bạn là Senior Full-stack Developer & Software Architect tại AI Corporation.
        Phong cách: Pragmatic (Thực dụng), Clean Code, và Security-first.
        Nhiệm vụ: Hiện thực hóa yêu cầu của CEO với tiêu chuẩn vận hành năm 2026.
    </role>

    <critical_policy>
        🔥 QUY TẮC SỐNG CÒN:
        1. Tuyệt đối không lặp lại lỗi cũ trong <error_analysis>.
        2. Nếu phát hiện yêu cầu có rủi ro bảo mật hoặc lỗi logic, hãy tự động sửa và ghi chú trong comment.
        3. Code phải chạy được ngay (Production-ready).
    </critical_policy>

    <error_analysis>
        {formatted_error}
    </error_analysis>

    <strategic_knowledge>
        <corporate_memory>
            {memory.strip() if memory else "Tiêu chuẩn: SOLID, DRY, và tối ưu hóa hiệu suất High-Concurrency."}
        </corporate_memory>
    </strategic_knowledge>

    <technical_constraints>
        - Stack: {tech_stack}
        - UI: Tailwind CSS (Mobile-first), Dark Mode support, Lucide icons.
        - Logic: Sử dụng Type Hints (Python) hoặc Strict Types (TS).
    </technical_constraints>

    <output_rules>
        1. FILE_NAME: Phải có comment tên file ở dòng đầu tiên (vd: # filename: app.py).
        2. NO_MARKDOWN_TEXT: Không giải thích ngoài lề, không "Here is your code".
        3. COMPLETENESS: Trả về toàn bộ file, không dùng "..." để lược bớt code.
        4. VIETNAMESE_COMMENTS: Giải thích logic phức tạp bằng tiếng Việt.
    </output_rules>
</system_context>

<user_instruction>
    {user_request.strip()}
</user_instruction>

<final_trigger>
    Output: CHỈ TRẢ VỀ CODE BLOCKS. Bắt đầu ngay với khối mã nguồn đầu tiên.
</final_trigger>
"""
    return prompt.strip()
# 🚩 [SECTION 5.4] BỘ KIỂM ĐỊNH MÃ NGUỒN ĐA NGÔN NGỮ (OPTIMIZED)
def real_syntax_validator(code: str, language: str) -> tuple[bool, str]:
    """
    Hệ thống kiểm định chất lượng Code: Python, JS/TS, HTML/CSS, C++.
    """
    if not code or len(code.strip()) < 10:
        return False, "❌ Mã nguồn quá ngắn hoặc trống."

    language = language.lower()

    # 1. PYTHON: KIỂM TRA CẤU TRÚC AST (ĐỘ CHÍNH XÁC CAO NHẤT)
    if any(kw in language for kw in ["python", "py"]) or ("def " in code and ":" in code):
        try:
            import ast
            ast.parse(code)
            return True, "✅ Python Syntax: Hoàn hảo."
        except SyntaxError as e:
            return False, f"❌ Python Error [Dòng {e.lineno}]: {e.msg} tại '{e.text.strip() if e.text else ''}'"

    # 2. JS/WEB: CƠ CHẾ STACK & TAG (ĐÃ TINH CHỈNH)
    if any(kw in language for kw in ["script", "js", "html", "css", "ts"]):
        # Xóa nội dung trong chuỗi và Regex để tránh bắt nhầm ngoặc bên trong text/regex
        clean_code = re.sub(r"'(.*?)'|\"(.*?)\"|`(.*?)`|/\(.*\)/", "", code)
        
        stack = []
        mapping = {')': '(', ']': '[', '}': '{'}
        
        for char in clean_code:
            if char in mapping.values():
                stack.append(char)
            elif char in mapping:
                if not stack or mapping[char] != stack.pop():
                    # Thử tìm vị trí tương đối của lỗi
                    return False, "❌ JS/Web Error: Mất cân bằng ngoặc hoặc sai thứ tự đóng/mở."
        
        if stack:
            return False, f"❌ JS/Web Error: Còn {len(stack)} khối mã ({stack[-1]}) chưa được đóng."
            
        # Kiểm tra thẻ HTML (Trường hợp Web)
        if "<" in code and ">" in code:
            # Chỉ đếm các thẻ không phải tự đóng (vd: <br/>, <img/>)
            open_tags = len(re.findall(r"<[^/!>]+>", code))
            close_tags = len(re.findall(r"</[^>]+>", code))
            if open_tags < close_tags: # Cho phép open >= close vì một số thẻ không cần đóng trong HTML5
                return False, f"❌ HTML Error: Số thẻ đóng ({close_tags}) vượt quá thẻ mở ({open_tags})."

        return True, "✅ Web Syntax: Basic Check Passed."

    # 3. C++ / ARDUINO: KIỂM TRA CẤU TRÚC VÀ CHẤM PHẨY
    if any(kw in language for kw in ["arduino", "cpp", "c++", "ino"]):
        # Kiểm tra dấu chấm phẩy cho các dòng khai báo/thực thi
        lines = code.split('\n')
        for i, line in enumerate(lines):
            l = line.strip()
            # Bỏ qua comment, directive, block
            if not l or l.startswith(("//", "#", "{", "}")) or l.endswith(("{", "}", ",", ":")):
                continue
            # Kiểm tra dòng không kết thúc bằng ;
            if not l.endswith(";") and not any(l.startswith(k) for k in ["if", "for", "while", "else", "switch"]):
                # Chỉ cảnh báo, không chặn (Vì C++ có thể viết gộp dòng)
                print(colored(f"⚠️ [C++ WARN] Dòng {i+1}: Thiếu ';'? -> '{l}'", "yellow"))
        
        return True, "✅ C++ Structure: OK."

    return True, "⚠️ Ngôn ngữ lạ: Bỏ qua kiểm định sâu."

# 🚩 [SECTION 5.5] BỘ TRÍCH XUẤT MÃ NGUỒN ĐA LUỒNG (REFACTORED)
def extract_code_block(content) -> str:
    """
    Hệ thống trích xuất mã nguồn bọc thép:
    - Chống lỗi dữ liệu List/Object từ Anthropic/OpenAI.
    - Xử lý đa khối code (Multi-block concatenation).
    - Làm sạch ký tự lạ gây lỗi biên dịch.
    """
    import re

    # 1. CHUẨN HÓA ĐẦU VÀO (DATA SANITIZATION)
    if isinstance(content, list):
        # Claude 3.5 thường trả về danh sách các ContentBlock
        parts = []
        for c in content:
            if hasattr(c, 'text'): parts.append(c.text)
            elif isinstance(c, dict) and 'text' in c: parts.append(c['text'])
            else: parts.append(str(c))
        content = "\n".join(parts)
    
    if not isinstance(content, str):
        content = str(content)

    # 2. TRÍCH XUẤT ĐA KHỐI (MULTI-BLOCK EXTRACTION)
    # Tìm tất cả các khối ``` thay vì chỉ tìm khối đầu tiên
    # Điều này quan trọng khi AI viết nhiều file (ví dụ: app.py và styles.css)
    blocks = re.findall(r'```[\w+\-]*\n?(.*?)```', content, re.DOTALL)
    
    if blocks:
        # Gộp các khối lại, phân tách bằng dòng trống
        # Hoặc CEO có thể chỉnh sửa để lấy khối dài nhất (thường là khối code chính)
        full_code = "\n\n".join([b.strip() for b in blocks])
        
        # 3. LÀM SẠCH KÝ TỰ ĐẶC BIỆT (CLEANING)
        # Loại bỏ các ký tự điều khiển lạ có thể xuất hiện trong quá trình truyền tải
        full_code = full_code.replace('\ufeff', '') # Loại bỏ BOM nếu có
        return full_code.strip()

    # 4. FALLBACK: Nếu AI quên dùng thẻ ``` nhưng có cấu trúc code rõ ràng
    # (Dành cho các model nhỏ hoặc khi token bị cắt ngang)
    if "def " in content or "import " in content or "class " in content:
        # Nếu có từ khóa lập trình nhưng không có thẻ, trả về toàn bộ text sạch
        return content.strip()

    return None

# 🚩 [SECTION 5.6] RESEARCHER NODE (PHIÊN BẢN CHIẾN LƯỢC)
async def researcher_node(state: AgentState):
    """
    Agent Researcher: Chuyên gia thám mã dữ liệu.
    Cơ chế: Zero-Cost Search + Smart Routing + KPI Logging.
    """
    start_time = time.time() 
    print(colored("[🔍 RESEARCHER] Đang thám mã thị trường (Zero-Cost)...", "cyan", attrs=["bold"]))

    # 1. TRÍCH XUẤT NHIỆM VỤ (INTENT MINING)
    messages = state.get("messages", [])
    
    # Ưu tiên lấy yêu cầu gốc của CEO từ lịch sử
    target_msg_content = ""
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            target_msg_content = msg.content
            break
    if not target_msg_content: target_msg_content = messages[-1].content

    # Nhận diện Tab chuyên biệt hoặc luồng tự động
    is_pure_research = "[RESEARCH]" in target_msg_content
    # Làm sạch Query để search không bị dính tag hệ thống
    clean_query = re.sub(r"\[.*?\]", "", target_msg_content).strip()

    try:
        # 2. THỰC THI TRUY VẾT DỮ LIỆU
        # Gọi hàm 'thợ lặn' miễn phí đã tối ưu ở Section 5
        raw_res = await free_deep_research(clean_query) 

        # 3. ĐỊNH TUYẾN THÔNG MINH (DYNAMIC ROUTING)
        # Sửa lỗi: Đảm bảo next_step luôn khớp với bản đồ workflow
        if is_pure_research:
            next_destination = "FINISH"
        else:
            # Nếu là Task phức tạp, ưu tiên quay lại Supervisor để kiểm tra chéo
            next_destination = "Supervisor" 
            # if state.get("task_type") != "dynamic" else "Orchestrator"

        # 4. GHI NHẬT KÝ CHI PHÍ & XP (AUDIT)
        log_work_to_db(
            agent="RESEARCHER",
            task=f"Nghiên cứu: {clean_query[:50]}...",
            result=raw_res, 
            tool="Scout-Flash-V2",
            start_time=start_time
        )

        # 5. ĐÓNG GÓI BÁO CÁO
        # Thêm timestamp và metadata để Dashboard hiển thị chuyên nghiệp hơn
        report_header = f"🔍 **BÁO CÁO NGHIÊN CỨU THỊ TRƯỜNG**\n"
        report_header += f"📅 *Ngày thực hiện: {datetime.now().strftime('%d/%m/%Y')}* | 🏷️ *Tag: {clean_query}*\n"
        report_header += "---"
        
        return {
            "messages": [AIMessage(content=f"{report_header}\n\n{raw_res}")],
            "next_step": next_destination,
            "current_agent": "Researcher"
        }

    except Exception as e:
        print(colored(f"❌ [RESEARCH ERROR]: {e}", "red"))
        return {
            "messages": [AIMessage(content=f"⚠️ Sự cố thám mã: {str(e)}")],
            "next_step": "FINISH",
            "current_agent": "Researcher"
        }
    
# 🚩 [SECTION 5.7] QUY TRÌNH ỨNG CỨU KHẨN CẤP (ULTIMATE FALLBACK)
def ultimate_fallback(state: AgentState, messages: list):
    """
    Quy trình xử lý sự cố khẩn cấp: Ghi log, phân tích lỗi và tái khởi động an toàn.
    Nâng cấp 2026: Tích hợp cơ chế Auto-Reset State.
    """
    # 1. THU THẬP DỮ LIỆU THẢM HỌA
    error_logs = state.get("error_log", [])
    last_error = error_logs[-1] if error_logs else "Unknown Circuit Break"
    current_agent = state.get("current_agent", "Unknown")
    
    print(colored(f"🚨 [CRITICAL] Kích hoạt quy trình ứng cứu khẩn cấp tại Node: {current_agent}", "red", attrs=["bold"]))

    # 2. GHI NHẬT KÝ VẬT LÝ & AUDIT TRAIL
    # Lưu vào SQLite để Dashboard có thể hiển thị biểu đồ "Sức khỏe hệ thống"
    try:
        log_work_to_db(
            agent="SYSTEM", 
            task="CRASH_HANDLING", 
            result=f"Lỗi tại {current_agent}: {last_error}", 
            tool="Emergency-Brake",
            xp_bonus=0 
        )
    except: pass

    # Lưu log chi tiết vào file để Debug sâu
    timestamp = datetime.now().isoformat()
    with open("system_crash_log.txt", "a", encoding="utf-8") as f:
        f.write(f"[{timestamp}] AGENT: {current_agent} | ERROR: {last_error}\n")

    # 3. PHÂN TÍCH LOẠI LỖI ĐỂ ĐƯA RA LỜI KHUYÊN (2026 Logic)
    advice = "Thử nhập lệnh ngắn gọn hơn."
    if "rate_limit" in last_error.lower() or "quota" in last_error.lower():
        advice = "Hệ thống API đang quá tải hoặc hết hạn mức. Vui lòng kiểm tra số dư hoặc đợi 1 phút."
    elif "context_length" in last_error.lower():
        advice = "Nội dung quá dài. Hãy thử chia nhỏ yêu cầu hoặc gõ 'clear memory' để làm trống bối cảnh."
    elif "syntax" in last_error.lower():
        advice = "Có lỗi cú pháp nghiêm trọng trong mã nguồn mà AI không thể tự sửa. CEO hãy kiểm tra trực tiếp."

    # 4. THÔNG ĐIỆP CHUYÊN NGHIỆP CHO CEO
    error_summary = (
        "🛑 **THÔNG BÁO HỆ THỐNG**: J.A.R.V.I.S vừa kích hoạt cơ chế bảo vệ khẩn cấp.\n\n"
        f"📍 **Vị trí sự cố**: Node `{current_agent}`\n"
        f"🔍 **Phân tích**: `{last_error[:150]}...`\n"
        f"🛠️ **Khuyến nghị**: {advice}\n\n"
        "👉 *Gợi ý: Ngài có thể gõ 'restart' để khởi động lại luồng tư duy mới.*"
    )

    # 5. GIẢI PHÓNG TRẠNG THÁI (STATE CLEANUP)
    # Trả về trạng thái an toàn, xóa các lỗi cũ để tránh vòng lặp lỗi ở phiên làm việc sau
    return {
        "messages": [AIMessage(content=error_summary)],
        "next_step": "FINISH",
        "current_agent": "System_Guard",
        "error_log": [] # Reset error log để chuẩn bị cho phiên mới
    }

# 🚩 [SECTION 6.1] ROUTER NODE (BẢN CẬP NHẬT PHẢN XẠ NHANH)
def router_node(state: AgentState):
    """
    ROUTER V2.1: CỔNG GÁC PHẢN XẠ CHIẾN LƯỢC
    Nâng cấp: Nhận diện lệnh Phẫu thuật và Điều phối 9 tầng tức thì.
    """
    messages = state.get("messages", [])
    error_log = state.get("error_log", [])
    task_type = state.get("task_type", "general")
    
    if not messages:
        return {"next_step": "Supervisor", "current_agent": "Router"}

    # 1. CHUẨN HÓA NỘI DUNG (CLEANING)
    last_msg_obj = messages[-1]
    content = last_msg_obj.content if hasattr(last_msg_obj, 'content') else str(last_msg_obj)
    content_upper = content.upper().strip()

    # 2. BẢN ĐỒ ĐIỀU HƯỚNG CẤP TỐC (BYPASS SUPERVISOR)
    route_map = {
        # --- LUỒNG PHẪU THUẬT SIÊU CẤP ---
        "PHẪU THUẬT": ("Orchestrator", "dynamic"),
        "ĐIỀU TRẦN": ("Orchestrator", "dynamic"),
        "QUY HOẠCH": ("Orchestrator", "dynamic"),
        
        # --- LUỒNG CHUYÊN GIA ĐÍCH DANH ---
        "[RESEARCH]": ("Researcher", "general"),
        "[INVEST]": ("Investment", "general"),
        "[HARDWARE]": ("Hardware", "general"),
        "[IOT]": ("IoT_Engineer", "general"),
        "[MARKETING]": ("Marketing", "general"),
        "[STORY]": ("Storyteller", "general"),
        
        # --- LUỒNG KÝ ỨC DÀI HẠN ---
        "GHI NHỚ": ("PreferenceLearner", "general"),
        "HÃY NHỚ": ("PreferenceLearner", "general"),
        "LƯU Ý:": ("PreferenceLearner", "general")
    }

    # 3. THỰC THI ROUTING (O(1) REFLEX)
    for trigger, (target_node, new_task_type) in route_map.items():
        if trigger in content_upper:
            print(colored(f"🚀 [ROUTER] High Priority Interrupt: {trigger} -> {target_node}", "green", attrs=["bold"]))
            return {
                "messages": [], 
                "next_step": target_node, 
                "current_agent": "Router",
                "task_type": new_task_type # Chuyển trạng thái sang dynamic nếu là phẫu thuật
            }

    # 4. KIỂM TRA LỖI VÒNG LẶP (ROUTER SAFETY)
    # Nếu tin nhắn lặp lại quá 3 lần từ chính Router, ép về Secretary để xử lý nhân văn
    if len(messages) > 10 and content_upper in [m.content.upper() for m in messages[-3:-1] if hasattr(m, 'content')]:
        print(colored("⚠️ [ROUTER] Phát hiện vòng lặp phản xạ. Chuyển hồ sơ cho Secretary cứu hộ.", "yellow"))
        return {"next_step": "Secretary", "current_agent": "Router"}

    # 5. MẶC ĐỊNH: BÀN GIAO CHO CHIẾN LƯỢC GIA (SUPERVISOR)
    print(colored("🧠 [ROUTER] Request chuẩn: Chuyển hồ sơ cho Supervisor điều phối...", "cyan"))
    return {
        "next_step": "Supervisor", 
        "current_agent": "Router",
        "task_type": task_type
    }

# 🚩 [SECTION 6.3] TESTER NODE (BẢN KIỂM ĐỊNH CHUYÊN SÂU)
def tester_node(state: AgentState):
    """
    Agent Tester V3: Kiểm định cú pháp, quét lỗ hổng bảo mật và xác thực logic 2026.
    """
    print(colored("[🧪 TESTER] Đang khởi động hệ thống kiểm định chất lượng...", "yellow", attrs=["bold"]))
    
    messages = state.get("messages", [])
    if not messages:
        return {"next_step": "Supervisor"}
        
    last_ai_msg = messages[-1].content
    
    # 1. TRÍCH XUẤT CODE (Dùng hàm extract_code_block chuyên dụng đã tối ưu)
    code_to_test = extract_code_block(last_ai_msg)
    
    if not code_to_test:
        print(colored("❌ [TESTER] Lỗi: Coder không gửi kèm khối mã nguồn!", "red"))
        return {
            "messages": [HumanMessage(content="⚠️ Hệ thống kiểm định không tìm thấy code. Hãy trình bày lại mã nguồn trong khối ```.")],
            "error_log": state.get("error_log", []) + ["Thiếu block code ```"],
            "next_step": "Coder",
            "current_agent": "Tester"
        }

    is_valid = True
    feedback = []

    # 2. KIỂM ĐỊNH ĐA TẦNG (MULTI-LAYER VALIDATION)
    
    # TẦNG 1: PYTHON DEEP CHECK (AST + Security)
    if "def " in code_to_test or "import " in code_to_test or "class " in code_to_test:
        try:
            ast.parse(code_to_test)
            feedback.append("✅ [PYTHON] Cú pháp AST: Đạt.")
            
            # Quét bảo mật AI-Enhanced (Chặn các hàm thực thi nguy hiểm)
            dangerous_calls = ["eval(", "exec(", "os.system(", "subprocess.Popen(", "shlex.quote("]
            found_threats = [call for call in dangerous_calls if call in code_to_test]
            if found_threats:
                is_valid = False
                feedback.append(f"❌ [SECURITY] Phát hiện lỗ hổng thực thi: {', '.join(found_threats)}")
                
        except SyntaxError as e:
            is_valid = False
            feedback.append(f"❌ [PYTHON] Lỗi cú pháp tại dòng {e.lineno}: {e.msg}")

    # TẦNG 2: WEB/JS STACK CHECK
    elif any(x in code_to_test for x in ["const ", "let ", "function", "<html>"]):
        # Sử dụng bộ validator mạnh mẽ hơn đã định nghĩa ở Section 5
        ok, msg = real_syntax_validator(code_to_test, "js")
        is_valid = ok
        feedback.append(msg)

    # TẦNG 3: HARDWARE/C++ (Arduino/Embedded)
    elif any(x in code_to_test for x in ["#include", "void setup()"]):
        # Kiểm tra ngoặc lồng nhau (Nested Braces)
        if code_to_test.count("{") != code_to_test.count("}"):
            is_valid = False
            feedback.append("❌ [C++] Mất cân bằng ngoặc nhọn { }.")
        if code_to_test.count("(") != code_to_test.count(")"):
            is_valid = False
            feedback.append("❌ [C++] Mất cân bằng ngoặc đơn ( ).")
        if is_valid: feedback.append("✅ [C++] Cấu trúc cơ bản: Đạt.")

    # 3. KẾT LUẬN VÀ ĐIỀU PHỐI (ORCHESTRATION)
    full_feedback = "\n".join(feedback)
    
    if is_valid:
        print(colored("✅ [TESTER] Mã nguồn ĐẠT chuẩn. Đang nộp báo cáo cho Supervisor.", "green"))
        # Ghi log thành công để cộng XP
        log_work_to_db("Tester", "Kiểm định thành công", full_feedback, tool="System-Validator")
        return {
            "error_log": [], # Xóa sạch lỗi để tiến hành FINISH hoặc bước tiếp theo
            "next_step": "Supervisor", # Quay lại sếp để chốt kết quả
            "current_agent": "Tester"
        }
    else:
        print(colored(f"❌ [TESTER] Từ chối mã nguồn:\n{full_feedback}", "red"))
        # Tạo phản hồi "nghiêm khắc" để Coder tự sửa
        error_feedback = (
            f"🚫 **BÁO CÁO KIỂM ĐỊNH (TESTER)**:\n"
            f"Phát hiện sai sót trong bản vẽ kỹ thuật của bạn:\n"
            f"{full_feedback}\n\n"
            f"👉 **YÊU CẦU**: Hãy phân tích lỗi trên, sửa lại và cung cấp bản code hoàn chỉnh mới."
        )
        return {
            "messages": [HumanMessage(content=error_feedback)],
            "error_log": state.get("error_log", []) + [full_feedback],
            "next_step": "Coder",
            "current_agent": "Tester"
        }

# 🚩 [SECTION 6.4] HARDWARE ARCHITECT NODE (BỌC THÉP KỸ THUẬT)
def hardware_node(state: AgentState):
    """
    Agent Hardware Architect: Chuyên trách ESP32, Robotics và Hệ thống nhúng.
    Nâng cấp: Trích xuất BOM chuẩn, kiểm tra điện áp và chống nhiễu.
    """
    print(colored("[🛠️ HARDWARE] Đang kiến trúc hệ thống nhúng 2026...", "cyan", attrs=["bold"]))
    
    # 1. TRÍCH XUẤT NGỮ CẢNH (CONTEXT EXTRACTION)
    messages = state.get("messages", [])
    if not messages: return {"next_step": "FINISH"}
    
    last_msg = messages[-1].content
    is_pure_hw = "[HARDWARE]" in last_msg.upper()

    # 2. PROMPT KỸ THUẬT CHIẾN LƯỢC (STRUCTURAL PROMPT)
    # Tích hợp thêm các quy chuẩn an toàn điện năm 2026
    prompt = f"""
        <hardware_mission>
            Bạn là Senior Hardware Architect tại AI Corporation. 
            Nhiệm vụ: Thiết kế hệ thống nhúng dựa trên yêu cầu: {last_msg}
            
            YÊU CẦU ĐẦU RA BẮT BUỘC:
            1. [BOM - BILL OF MATERIALS]: Bảng kê linh kiện (Tên | Mã sản phẩm 2026 | Công dụng | Ước tính giá).
            2. [PINOUT MAP]: Sơ đồ đấu nối chính xác từng chân GPIO. Ưu tiên tránh các chân STRAPPING nếu dùng ESP32.
            3. [POWER ARCHITECTURE]: Phân tích dòng tiêu thụ, sơ đồ nguồn (LDO/Buck) và chống sụt áp.
            4. [FIRMWARE STRUCTURE]: Khung code C++/Rust-Embedded tối ưu cho đa nhiệm.
        </hardware_mission>

        ⚠️ LƯU Ý: Tuyệt đối chính xác về điện áp (3.3V vs 5V). Không sử dụng Emoji.
        """

    try:
        # 3. THỰC THI (Sử dụng GPT-4o để tra cứu Datasheet chính xác)
        response = LLM_GPT4.invoke([
            SystemMessage(content=prompt),
            HumanMessage(content=last_msg)
        ])
        
        # 4. ĐỊNH TUYẾN THÔNG MINH
        # Nếu thiết kế xong -> Chuyển Procurement để báo giá, hoặc FINISH nếu chỉ cần thông tin
        next_destination = "FINISH" if is_pure_hw else "Procurement"

        # 5. GHI NHẬT KÝ VẬN HÀNH (KPI LOGGING)
        log_work_to_db(
            agent="Hardware",
            task=f"Thiết kế hệ thống: {last_msg[:50]}...",
            result="Hoàn tất BOM & Pinout",
            tool="GPT-4o-Vision-Datasheet",
            start_time=time.time()
        )

        return {
            "messages": [AIMessage(content=f"🛠️ **[HỒ SƠ KỸ THUẬT PHẦN CỨNG]**\n\n{response.content}")],
            "next_step": next_destination,
            "current_agent": "Hardware"
        }
        
    except Exception as e:
        error_detail = str(e)
        print(colored(f"🚨 [HARDWARE CRITICAL ERROR]: {error_detail}", "red", attrs=["bold"]))
        return {
            "messages": [AIMessage(content=f"❌ **LỖI HỆ THỐNG PHẦN CỨNG**:\n\nKhông thể khởi tạo thiết kế: `{error_detail}`")], 
            "next_step": "FINISH",
            "current_agent": "Hardware"
        }
# 🚩 [SECTION 6.5] ENGINEERING NODE (KIẾN TRÚC SƯ 3D)
def engineering_node(state: AgentState):
    """
    Agent CTO/Engineer: Thiết kế mô hình 3D kỹ thuật.
    Tối ưu hóa: Trả về mã nguồn Plotly sạch để Dashboard render trực tiếp.
    """
    print(colored("[⚙️ ENGINEERING] Đang kiến tạo bản vẽ 3D hệ thống...", "blue", attrs=["bold"]))
    
    # 1. TRÍCH XUẤT NGỮ CẢNH
    messages = state.get("messages", [])
    if not messages: return {"next_step": "FINISH"}
    
    last_msg = messages[-1].content
    is_pure_eng = "[ENGINEERING]" in last_msg.upper()

    # 2. PROMPT KỸ THUẬT SIÊU CHI TIẾT
    prompt = f"""
        <mission>
            Bạn là Kỹ sư Trưởng tại AI Corporation. Hãy thiết kế mô hình 3D bằng Python Plotly.
            YÊU CẦU: {last_msg}
        </mission>

        <technical_requirements>
            1. Chỉ sử dụng thư viện `plotly.graph_objects as go`.
            2. CODE STRUCTURE: 
            - Khởi tạo `fig = go.Figure()`.
            - Thêm các `go.Mesh3d` hoặc `go.Scatter3d` để tạo khối.
            - Cấu hình layout: `fig.update_layout(scene=dict(...), margin=dict(l=0, r=0, b=0, t=0))`.
            3. TRẢ VỀ: Chỉ trả về duy nhất khối CODE BLOCK trong ```python.
            4. NO_TALK: Không chào hỏi, không giải thích logic ngoài khối code.
        </technical_requirements>

        ⚠️ LƯU Ý: Đảm bảo tọa độ (x, y, z) chính xác để mô hình không bị méo.
        """

    try:
        # 3. THỰC THI (Claude 3.5 Sonnet là chuyên gia hình học)
        response = LLM_CLAUDE.invoke([
            SystemMessage(content=prompt),
            HumanMessage(content=last_msg)
        ])
        
        # 4. KIỂM TRA CHẤT LƯỢNG SƠ BỘ
        clean_code = extract_code_block(response.content)
        if not clean_code or "go.Figure" not in clean_code:
            raise ValueError("AI không tạo được đối tượng fig hợp lệ.")

        # 5. GHI NHẬT KÝ VẬN HÀNH
        log_work_to_db(
            agent="Engineering",
            task=f"Thiết kế 3D: {last_msg[:50]}...",
            result="Bản vẽ Plotly 3D hoàn tất",
            tool="Claude-3.5-Geometry",
            start_time=time.time()
        )

        return {
            "messages": [AIMessage(content=f"⚙️ **[BẢN THIẾT KẾ KỸ THUẬT 3D]**\n\n{response.content}")],
            "next_step": "FINISH" if is_pure_eng else "Hardware",
            "current_agent": "Engineering"
        }
        
    except Exception as e:
        print(colored(f"🚨 [ENGINEERING ERROR]: {str(e)}", "red", attrs=["bold"]))
        return {
            "messages": [AIMessage(content=f"❌ **LỖI THIẾT KẾ 3D**:\n\nSự cố render mô hình: `{str(e)}`")], 
            "next_step": "FINISH",
            "current_agent": "Engineering"
        }

# 🚩 [SECTION 6.6] IoT ENGINEER NODE (HYBRID OPERATOR)
def iot_node(state: AgentState):
    """
    Agent IoT: Song hành giữa Vận hành thiết bị thực và Thiết kế Firmware.
    Nâng cấp 2026: Tích hợp xác thực lệnh (Safety Check) và Giao thức MQTT.
    """
    print(colored("[🤖 IoT ENGINEER] Đang xử lý giao thức và thiết bị...", "magenta", attrs=["bold"]))
    
    messages = state.get("messages", [])
    if not messages: return {"next_step": "FINISH"}
    
    last_msg = messages[-1].content
    is_pure_iot = "[IOT]" in last_msg.upper()

    # 1. NHẬN DIỆN Ý ĐỊNH (INTENT RECOGNITION)
    # Kiểm tra xem đây là lệnh điều khiển trực tiếp hay yêu cầu thiết kế
    command_keywords = ["BẬT", "TẮT", "TURN", "CONTROL", "CHẠY", "SET", "MỞ", "ĐÓNG"]
    is_command = any(word in last_msg.upper() for word in command_keywords)

    try:
        if is_command:
            # --- NHÁNH 1: VẬN HÀNH THIẾT BỊ (OPERATIONAL TECHNOLOGY) ---
            # Sử dụng GPT-4o để trích xuất JSON lệnh chính xác
            analysis_prompt = (
                f"Phân tích lệnh: '{last_msg}'. "
                "Chỉ trả về JSON: {'device': '...', 'action': '...', 'value': '...'}. "
                "Nếu không rõ, trả về {'error': 'Unknown command'}."
            )
            command_json = LLM_GPT4.invoke([SystemMessage(content=analysis_prompt)]).content
            
            # GIẢ GIỮ AN TOÀN: Kiểm tra lệnh trước khi gửi tới hardware_controller
            if "error" in command_json:
                raise ValueError("Lệnh điều khiển không rõ ràng hoặc không an toàn.")

            # Thực thi thông qua Bridge kết nối thiết bị thật
            hw_response = hardware_controller.invoke(command_json)
            
            # Ghi log vận hành vào SQLite (Audit Trail)
            log_work_to_db("IoT_Engineer", "Thực thi lệnh thiết bị", command_json, tool="Hardware-Bridge")

            return {
                "messages": [AIMessage(content=f"📡 **[KẾT QUẢ VẬN HÀNH THIẾT BỊ]**\n\n- Lệnh trích xuất: `{command_json}`\n- Phản hồi hệ thống: {hw_response}")], 
                "next_step": "FINISH" if is_pure_iot else "Supervisor",
                "current_agent": "IoT_Engineer"
            }
            
        else:
            # --- NHÁNH 2: THIẾT KẾ FIRMWARE (IOT ARCHITECTURE) ---
            # Lấy ngữ cảnh từ Hardware Node để biết Pinout
            hw_context = next((m.content for m in reversed(messages) if "🛠️" in m.content), "Mặc định ESP32 Standard.")
            
            design_prompt = f"""
            Bạn là Kỹ sư Firmware IoT 2026. Hãy viết code C++/Arduino.
            NGỮ CẢNH PHẦN CỨNG: {hw_context}
            YÊU CẦU: {last_msg}
            TIÊU CHUẨN: Sử dụng Async WebServer, kết nối MQTT (TLS 1.3), có cơ chế tự khởi động lại (Watchdog).
            """
            
            # Claude 3.5 Sonnet xử lý logic Firmware cực tốt
            response = LLM_CLAUDE.invoke([SystemMessage(content=design_prompt), HumanMessage(content=last_msg)])
            
            return {
                "messages": [AIMessage(content=f"📡 **[FIRMWARE & GIAO THỨC ĐIỀU KHIỂN]**\n\n{response.content}")],
                "next_step": "FINISH" if is_pure_iot else "Supervisor",
                "current_agent": "IoT_Engineer"
            }
            
    except Exception as e:
        error_detail = str(e)
        print(colored(f"🚨 [IOT ERROR]: {error_detail}", "red", attrs=["bold"]))
        return {
            "messages": [AIMessage(content=f"❌ **SỰ CỐ HỆ THỐNG IoT**:\n\nChi tiết: `{error_detail}`")], 
            "next_step": "Supervisor",
            "current_agent": "IoT_Engineer"
        }
#  🚩 [SECTION 6.7] PROCUREMENT NODE (TRỢ LÝ THU MUA THÔNG MINH)
BUYER_PROFILE = {
    "address": "Phan Thiết, Bình Thuận, Việt Nam",
    "delivery_method": "Fast Shipping",
    "accounts": ["Shopee_API_Key", "Taobao_Token", "Mouser_ID"]
}
async def procurement_node(state: AgentState):
    """
    Agent Procurement: Tra cứu giá, tính toán logistics và tối ưu hóa ngân sách.
    Nâng cấp 2026: Tích hợp định vị địa lý (Buyer Profile) và tính phí vận chuyển real-time.
    """
    print(colored("[🛒 PROCUREMENT] Đang khảo giá và tối ưu lộ trình hàng hóa...", "yellow", attrs=["bold"]))
    
    # 1. TRÍCH XUẤT CẤU HÌNH NGƯỜI MUA & DANH MỤC (BOM)
    buyer_config = BUYER_PROFILE # Lấy từ file cấu hình hệ thống
    messages = state.get("messages", [])
    
    # Tìm báo cáo từ Hardware Node để lấy danh sách linh kiện (BOM)
    hw_report = next((m.content for m in reversed(messages) if "🛠️" in m.content), "")
    
    if not hw_report:
        return {
            "messages": [AIMessage(content="⚠️ Không tìm thấy danh sách linh kiện (BOM) để báo giá.")],
            "next_step": "Supervisor"
        }

    # 2. THỰC THI TRA CỨU GIÁ MIỄN PHÍ (ZERO-COST RESEARCH)
    query = f"Giá linh kiện điện tử Shopee/Lazada/DigiKey Việt Nam 2026: {hw_report[:300]}"
    
    try:
        # Sử dụng thợ lặn DuckDuckGo để lấy dữ liệu thô từ thị trường
        raw_price_data = await free_deep_research(query)
        
        # 3. DÙNG GEMINI ĐỂ BIÊN TẬP BẢNG GIÁ CHUYÊN NGHIỆP
        format_prompt = f"""
        BỐI CẢNH: Bạn là Trưởng phòng Thu mua của AI Corporation.
        DỮ LIỆU THỊ TRƯỜNG: {raw_price_data}
        ĐỊA CHỈ NHẬN HÀNG: {buyer_config['address']}
        
        NHIỆM VỤ:
        1. Lập bảng báo giá: STT | Linh kiện | Giá ước tính (VND) | Nguồn tham khảo.
        2. Tính toán phí vận chuyển dự kiến về {buyer_config['address']}.
        3. Đưa ra tổng ngân sách (Total Budget) dự kiến.
        4. Đánh giá rủi ro (Hàng khan hiếm hoặc giá biến động).
        
        YÊU CẦU: Trình bày Markdown chuyên nghiệp, không emoji.
        """
        
        response = await LLM_GEMINI_LOGIC.ainvoke(format_prompt)
        
        # 4. GHI NHẬT KÝ (LOGGING)
        log_work_to_db(
            agent="Procurement",
            task=f"Báo giá dự án: {buyer_config.get('project_name', 'General')}",
            result=f"Tổng ngân sách: {response.content[:100]}...",
            tool="Scout-Free-Search",
            start_time=time.time()
        )

        return {
            "messages": [AIMessage(content=f"🛒 **[PHIẾU ĐỀ XUẤT MUA SẮM & LOGISTICS]**\n\n{response.content}")],
            "next_step": "Investment", # Chuyển sang Investment để duyệt chi
            "current_agent": "Procurement"
        }
        
    except Exception as e:
        print(colored(f"❌ [PROCUREMENT ERROR]: {e}", "red"))
        return {
            "messages": [AIMessage(content=f"❌ Lỗi xử lý thu mua: {str(e)}")], 
            "next_step": "Supervisor",
            "current_agent": "Procurement"
        }
    
# 🚩 [SECTION 6.8] INVESTMENT NODE (CHIEF FINANCIAL OFFICER)
def investment_node(state: AgentState):
    """
    Agent CFO: Thẩm định tài chính, phân tích rủi ro và ra quyết định duyệt chi.
    Nâng cấp 2026: Tích hợp phân tích CAPEX/OPEX và dự báo điểm hòa vốn.
    """
    print(colored("[💰 INVESTMENT] Đang thẩm định tính khả thi tài chính...", "green", attrs=["bold"]))
    
    # 1. THU THẬP NGỮ CẢNH ĐA CHIỀU
    messages = state.get("messages", [])
    if not messages: return {"next_step": "FINISH"}
    
    last_msg = messages[-1].content
    is_pure_invest = "[INVEST]" in last_msg.upper()
    
    # Lấy 5 tin nhắn gần nhất để bao quát từ Research -> Hardware -> Procurement
    full_context = "\n".join([f"{m.type.upper()}: {m.content[:500]}" for m in messages[-5:]])
    
    # 2. PROMPT THẨM ĐỊNH "NGHIÊM KHẮC" (FINANCIAL AUDIT)
    prompt = f"""
        <executive_role>
            Bạn là Giám đốc Tài chính (CFO) của AI Corporation. 
            Nhiệm vụ: Thẩm định dự án dựa trên dữ liệu kỹ thuật và báo giá từ các phòng ban.
        </executive_role>

        <analysis_framework>
            1. [CAPEX]: Chi phí đầu tư ban đầu (Linh kiện, bản quyền, thiết bị).
            2. [OPEX]: Chi phí duy trì (Điện năng, API tokens, nhân sự vận hành).
            3. [ROI & BREAK-EVEN]: Dự báo thời gian hoàn vốn và tỷ suất lợi nhuận sau 12 tháng.
            4. [RISK ASSESSMENT]: Các rủi ro về chuỗi cung ứng, công nghệ lỗi thời hoặc vượt ngân sách.
        </analysis_framework>

        <decision_standard>
            Đưa ra kết luận cuối cùng bằng một trong các nhãn: 
            - ✅ [ĐẦU TƯ]: Nếu dự án có tính khả thi cao và ROI tốt.
            - ⚠️ [THEO DÕI]: Nếu thiếu dữ liệu hoặc rủi ro trung bình.
            - ❌ [LOẠI BỎ]: Nếu không hiệu quả về kinh tế.
        </decision_standard>
        """

    try:
        # 3. THỰC THI (Ưu tiên GPT-4o cho khả năng suy luận logic số liệu)
        response = LLM_GPT4.invoke([
            SystemMessage(content=prompt), 
            HumanMessage(content=f"Hồ sơ dự án tổng hợp:\n{full_context}")
        ])
        
        # 4. GHI NHẬT KÝ KẾ TOÁN (AUDIT LOG)
        decision = "APPROVED" if "✅" in response.content else "REJECTED/PENDING"
        log_work_to_db(
            agent="Investment",
            task="Thẩm định dự án",
            result=f"Kết luận: {decision}",
            tool="Financial-Model-V4",
            start_time=time.time()
        )

        # 5. ĐỊNH TUYẾN
        # Nếu duyệt chi thành công và ở luồng tự động -> Có thể quay lại Supervisor để triển khai tiếp
        next_destination = "FINISH" if is_pure_invest else "Supervisor"

        return {
            "messages": [AIMessage(content=f"💰 **[HỒ SƠ THẨM ĐỊNH & DUYỆT CHI TÀI CHÍNH]**\n\n{response.content}")],
            "next_step": next_destination,
            "current_agent": "Investment"
        }
    except Exception as e:
        print(colored(f"🚨 [INVESTMENT ERROR]: {e}", "red"))
        return {
            "messages": [AIMessage(content=f"⚠️ Sự cố phân tích tài chính: {str(e)}")],
            "next_step": "FINISH",
            "current_agent": "Investment"
        }

# 🚩 [SECTION 6.9] R&D STRATEGY NODE (THE ARCHITECT OF VISION)
STRATEGY_SYSTEM_PROMPT = """
Bạn là Giám đốc Chiến lược (CSO) và Chuyên gia Phân tích Thị trường tối cao của AI Corporation. 
Phong cách làm việc: Thực dụng (Pragmatic), Sắc bén, và Tập trung vào kết quả.

QUY TRÌNH THẨM ĐỊNH CHIẾN LƯỢC:

1. [QUÉT NGỮ CẢNH ĐA CHIỀU]:
   - Sử dụng dữ liệu thị trường thực tế năm 2026 để đánh giá quy mô và tốc độ dịch chuyển công nghệ.
   - Áp dụng mô hình PESTLE để xác định các 'vùng xám' về luật pháp AI và biến động kinh tế.

2. [THÁM MÃ ĐỐI THỦ & NGƯỜI DÙNG]:
   - Phân tích lỗ hổng (Gap Analysis) của các đối thủ lớn. 
   - Đi sâu vào 'Nỗi đau chưa được giải quyết' (Unmet Needs) của khách hàng mục tiêu.

3. [PHÂN TÍCH NGHỊCH ĐẢO (CRITICAL RED TEAM)]:
   - Đưa ra ít nhất 3 kịch bản rủi ro khiến dự án này có thể thất bại.
   - Đề xuất phương án phòng vệ hoặc xoay trục (Pivot) cho mỗi kịch bản.

4. [LỘ TRÌNH ĐỘT PHÁ (ROADMAP)]:
   - Dự báo xu hướng 2-5 năm (Kỷ nguyên Edge AI, Robotics, v.v.).
   - Lập danh sách 05 hành động ưu tiên (Quick Wins) phải thực hiện ngay trong 30 ngày tới.

YÊU CẦU ĐẦU RA:
- Ngôn ngữ quyết đoán, mang tính chỉ thị.
- Trình bày dạng Modular (Thẻ, Bảng, List) để CEO có thể đọc nhanh trong 60 giây.
- KHÔNG nói sáo rỗng. Nếu dữ liệu không đủ, phải nêu rõ 'Cần nghiên cứu thêm phần X'.
"""

async def research_development_agent(state: AgentState):
    """
    Agent R&D Strategy: Phân tích xu hướng, đối thủ và đề xuất lộ trình công nghệ.
    Nâng cấp 2026: Tích hợp phân tích SWOT và dự báo tác động AI Act.
    """
    print(colored("[🧠 R&D STRATEGY] Đang kiến tạo tầm nhìn chiến lược...", "blue", attrs=["bold"]))
    
    # 1. THU THẬP BỐI CẢNH (CÁ NHÂN HÓA VÀ THỊ TRƯỜNG)
    messages = state.get("messages", [])
    if not messages: return {"next_step": "Supervisor"}
    
    user_input = messages[-1].content
    # Truy xuất "Ký ức lõi" để đảm bảo chiến lược không đi chệch định hướng CEO
    company_context = search_memory("Tầm nhìn, giá trị cốt lõi và mục tiêu 2026 AI Corporation")
    
    # 2. TRUY VẾT DỮ LIỆU THỰC TẾ (ZERO-COST RESEARCH)
    search_query = f"Báo cáo thị trường, xu hướng công nghệ AI/IoT và đối thủ cạnh tranh tháng 2/2026 cho: {user_input}"
    
    try:
        # Lấy dữ liệu nóng từ Internet 2026
        market_data = await free_deep_research(search_query)
        
        # 3. TỔNG HỢP CHIẾN LƯỢC (Tư duy bậc cao với GPT-4o)
        prompt = f"""
        BỐI CẢNH CÔNG TY: {company_context}
        DỮ LIỆU THỊ TRƯỜNG 2026: {market_data}
        YÊU CẦU CỦA CEO: {user_input}
        
        NHIỆM VỤ: Lập Báo cáo Chiến lược R&D chuyên sâu bao gồm:
        1. [XU HƯỚNG CHỦ ĐẠO]: Những công nghệ 2026 nào cần áp dụng ngay?
        2. [PHÂN TÍCH ĐỐI THỦ]: Họ đang làm gì? Chúng ta có lợi thế gì (Unique Selling Point)?
        3. [MÔ HÌNH SWOT]: Điểm mạnh, Điểm yếu, Cơ hội, Thách thức.
        4. [ROADMAP ĐỀ XUẤT]: Lộ trình triển khai theo quý (Q1-Q4/2026).
        
        YÊU CẦU: Ngôn ngữ quyết đoán, thực dụng, trình bày Markdown chuyên nghiệp.
        """
        
        # Thực thi chuỗi suy luận
        response = await LLM_GPT4.ainvoke([
            SystemMessage(content=STRATEGY_SYSTEM_PROMPT),
            HumanMessage(content=prompt)
        ])
        
        # 4. GHI NHẬT KÝ (LOGGING)
        log_work_to_db(
            agent="Strategy_R_and_D",
            task=f"Chiến lược: {user_input[:50]}...",
            result="Báo cáo R&D hoàn tất",
            tool="Hybrid-Intelligence-V2",
            start_time=time.time()
        )

        return {
            "messages": [AIMessage(content=f"🧠 **[BÁO CÁO CHIẾN LƯỢC R&D - TẦM NHÌN 2026]**\n\n{response.content}")],
            "next_step": "Supervisor",
            "current_agent": "Strategy_R_and_D"
        }
        
    except Exception as e:
        print(colored(f"❌ [R&D ERROR]: {str(e)}", "red"))
        return {
            "messages": [AIMessage(content=f"⚠️ Sự cố khi phân tích chiến lược: {str(e)}")],
            "next_step": "Supervisor",
            "current_agent": "Strategy_R_and_D"
        }

# 🚩 [SECTION 6.10] LEGAL & COMPLIANCE NODE (THE GUARDIAN)
def legal_node(state: AgentState):
    """
    Agent Legal (CLO): Rà soát IP, tuân thủ Luật An ninh mạng và quản trị rủi ro.
    Nâng cấp 2026: Tích hợp kiểm tra tuân thủ AI Act và bản quyền đào tạo dữ liệu.
    """
    print(colored("[⚖️ LEGAL] Luật sư đang rà soát toàn bộ hồ sơ dự án...", "red", attrs=["bold"]))
    
    messages = state.get("messages", [])
    if not messages: return {"next_step": "FINISH"}
    
    last_msg = messages[-1].content
    is_pure_legal = "[LEGAL]" in last_msg.upper()
    
    # 1. TỔNG HỢP HỒ SƠ ĐA TẦNG (FULL AUDIT TRAIL)
    # Gom 15 tin nhắn để kiểm tra tính nhất quán giữa các phòng ban
    full_project_context = "\n".join([f"[{m.type.upper()}]: {m.content[:500]}..." for m in messages[-15:]])

    # 2. PROMPT THẨM ĐỊNH PHÁP LÝ CHUYÊN SÂU
    prompt = f"""
<legal_role>
    Bạn là Giám đốc Pháp lý (CLO) tối cao của AI Corporation. 
    Nhiệm vụ: Thẩm định hồ sơ dự án dưới góc độ luật pháp Việt Nam và Quốc tế 2026.
</legal_role>

<audit_checklist>
    1. [SỞ HỮU TRÍ TUỆ - IP]: Kiểm tra code (Coder) và Asset (Artist) có dấu hiệu vi phạm bản quyền không?
    2. [DATA PRIVACY]: Rà soát việc thu thập dữ liệu khách hàng có tuân thủ GDPR và Nghị định 13/2023/NĐ-CP không?
    3. [LIABILITY]: Phân tích trách nhiệm pháp lý nếu sản phẩm AI gây ra sai sót trong vận hành (Hardware/IoT).
    4. [CONTRACTUAL]: Dự thảo khung NDA (Thỏa thuận bảo mật) và ToS (Điều khoản dịch vụ) sơ bộ.
</audit_checklist>

<final_verdict>
    Đưa ra kết luận bằng nhãn:
    - 🟢 [AN TOÀN]: Sẵn sàng xuất bản.
    - 🟡 [CẢNH BÁO]: Cần sửa đổi các mục cụ thể.
    - 🔴 [NGUY HIỂM]: Dừng dự án ngay lập tức.
</final_verdict>
"""

    try:
        # 3. THỰC THI (GPT-4o lý luận văn bản pháp luật tốt nhất)
        response = LLM_GPT4.invoke([
            SystemMessage(content=prompt),
            HumanMessage(content=f"HỒ SƠ DỰ ÁN:\n{full_project_context}\n\nYÊU CẦU RIÊNG: {last_msg}")
        ])
        
        # 4. GHI NHẬT KÝ PHÁP LÝ (AUDIT LOG)
        status = "SECURE" if "🟢" in response.content else "RISK"
        log_work_to_db(
            agent="Legal",
            task="Rà soát pháp lý dự án",
            result=f"Trạng thái: {status}",
            tool="Law-Expert-V2"
        )

        return {
            "messages": [AIMessage(content=f"⚖️ **[BÁO CÁO PHÁP LÝ & QUẢN TRỊ RỦI RO]**\n\n{response.content}")],
            "next_step": "FINISH" if is_pure_legal else "Supervisor",
            "current_agent": "Legal"
        }

    except Exception as e:
        print(colored(f"🚨 [LEGAL ERROR]: {str(e)}", "red", attrs=["bold"]))
        return {
            "messages": [AIMessage(content=f"❌ **SỰ CỐ RÀ SOÁT PHÁP LÝ**:\n\nChi tiết: `{str(e)}`")], 
            "next_step": "FINISH",
            "current_agent": "Legal"
        }
# 🚩 [SECTION 6.11] MARKETING & GROWTH NODE (THE MEGAPHONE)
def marketing_node(state: AgentState):
    """
    Agent CMO: Chuyên gia Marketing đa kênh và Tăng trưởng (Growth Hacking).
    Nâng cấp 2026: Tích hợp mô hình tâm lý hành vi và tối ưu hóa chuyển đổi (CRO).
    """
    print(colored("[📢 MARKETING] Đang lập chiến dịch quảng bá bùng nổ...", "yellow", attrs=["bold"]))
    
    # 1. TRÍCH XUẤT NGỮ CẢNH CHIẾN LƯỢC
    messages = state.get("messages", [])
    if not messages: return {"next_step": "FINISH"}
    
    last_msg = messages[-1].content
    is_pure_mkt = "[MARKETING]" in last_msg.upper()
    
    # Lấy 5 tin nhắn gần nhất để hiểu "Linh hồn" sản phẩm từ Kỹ thuật & R&D
    project_context = "\n".join([f"[{m.type.upper()}]: {m.content[:400]}..." for m in messages[-5:]])
    
    # 2. PROMPT MARKETING CHIẾN THUẬT (MARKETING FRAMEWORKS)
    prompt = f"""
        <marketing_mission>
            Bạn là Giám đốc Marketing (CMO) của AI Corporation. 
            Nhiệm vụ: Biến các thông số kỹ thuật khô khan thành thông điệp bán hàng triệu đô.
        </marketing_mission>

        <strategy_framework>
            1. [UNIQUE SELLING POINT - USP]: Nêu bật 3 đặc điểm "độc bản" từ dữ liệu kỹ thuật.
            2. [FACEBOOK ADS]: Sử dụng mô hình PAS (Problem - Agitate - Solve) cho tệp khách hàng đại chúng.
            3. [LINKEDIN ARTICLES]: Sử dụng mô hình AIDA cho tệp B2B, nhấn mạnh vào hiệu quả kinh tế và tính bền vững.
            4. [VISUAL DIRECTION]: Cung cấp 02 Prompt tiếng Anh chi tiết (Cấu trúc: Style, Subject, Lighting, Camera angle) để Agent Artist thực thi thiết kế.
        </strategy_framework>

        <output_standard>
            - Ngôn ngữ: Thuyết phục, giàu cảm xúc nhưng vẫn chuyên nghiệp.
            - Phân khúc: Tập trung vào đối tượng khách hàng năm 2026.
        </output_standard>
        """

    try:
        # 3. THỰC THI (GPT-4o là bậc thầy về ngôn ngữ và tâm lý khách hàng)
        response = LLM_GPT4.invoke([
            SystemMessage(content=prompt),
            HumanMessage(content=f"DỮ LIỆU SẢN PHẨM & YÊU CẦU:\n{project_context}\n\nLệnh bổ sung: {last_msg}")
        ])

        # 4. GHI NHẬT KÝ CHIẾN DỊCH (CRM LOGGING)
        log_work_to_db(
            agent="Marketing",
            task="Xây dựng kế hoạch truyền thông",
            result=f"Visual Prompts: {response.content.count('Visual')}",
            tool="Marketing-Intelligence-V2"
        )

        # 5. ĐỊNH TUYẾN THÔNG MINH
        # Tự động đẩy sang Artist nếu CMO đề xuất có hình ảnh, hoặc quay về Supervisor
        next_destination = "Artist" if "VISUAL" in response.content.upper() and not is_pure_mkt else "Supervisor"

        return {
            "messages": [AIMessage(content=f"📢 **[CHIẾN DỊCH MARKETING & TĂNG TRƯỞNG]**\n\n{response.content}")],
            "next_step": "FINISH" if is_pure_mkt else next_destination,
            "current_agent": "Marketing"
        }
        
    except Exception as e:
        print(colored(f"🚨 [MARKETING ERROR]: {str(e)}", "red", attrs=["bold"]))
        return {
            "messages": [AIMessage(content=f"❌ **LỖI CHIẾN DỊCH TRUYỀN THÔNG**:\n\nSự cố phân tích thị trường: `{str(e)}`")], 
            "next_step": "FINISH",
            "current_agent": "Marketing"
        }

# 🚩 [SECTION 6.12] ARTIST NODE (THE VIRTUAL STUDIO V3)
def artist_node(state: AgentState):
    """
    Artist Node V3: Kết hợp Tư duy Giám đốc nghệ thuật và Sức mạnh DALL-E 3 HD.
    Nâng cấp 2026: Tự động tối ưu Prompt (Prompt Enhancer) và xử lý Error Fallback.
    """
    print(colored("\n[🎨 ARTIST] Đang khởi động Studio DALL-E 3 HD...", "blue", attrs=["bold"]))
    
    # 1. TRÍCH XUẤT NGỮ CẢNH
    messages = state.get("messages", [])
    if not messages: return {"next_step": "FINISH"}
    
    last_msg_content = messages[-1].content
    
    # Ưu tiên lấy Prompt từ khối """...""" như CEO đã thiết kế
    if '"""' in last_msg_content:
        start_idx = last_msg_content.find("\"\"\"") + 3
        end_idx = last_msg_content.rfind("\"\"\"")
        text_to_illustrate = last_msg_content[start_idx:end_idx].strip()
    else:
        text_to_illustrate = last_msg_content.replace("[ARTIST]", "").strip()

    if len(text_to_illustrate) < 5:
        return {
            "messages": [AIMessage(content="⚠️ Họa sĩ cần một mô tả cụ thể hơn để bắt đầu sáng tác.")], 
            "next_step": "FINISH" 
        }

    # 2. GPT-4 ART DIRECTOR: NÂNG CẤP PROMPT (PROMPT ENGINEERING)
    # Thêm tham số 'quality' và 'composition' vào JSON
    analysis_prompt = f"""
    Bạn là Giám đốc Nghệ thuật của AI Corporation. Hãy tạo Image Prompt cho DALL-E 3.
    YÊU CẦU: "{text_to_illustrate}"
    
    TRẢ VỀ JSON:
    {{
      "style": "Phong cách nghệ thuật (ví dụ: Photorealistic, Cyberpunk, 3D Render)",
      "prompt": "Mô tả tiếng Anh cực chi tiết: Bố cục, ánh sáng, vật liệu, cảm xúc (Max 80 từ)",
      "ratio": "Mặc định 1024x1024"
    }}
    """

    try:
        # Bước này giúp chuyển từ ngôn ngữ đời thường sang ngôn ngữ hội họa chuyên sâu
        analysis_response = LLM_GPT4.invoke([SystemMessage(content="Output JSON only."), HumanMessage(content=analysis_prompt)])
        
        # Xử lý JSON bọc thép
        import json
        clean_json = analysis_response.content.replace("```json", "").replace("```", "").strip()
        data = json.loads(clean_json)
        
        full_prompt = f"{data['prompt']}, in {data['style']} style, 8k resolution, cinematic lighting, masterpiece."
        print(colored(f"--> Strategy: {data['style']}", "cyan"))

        # 3. THỰC THI VẼ (DALL-E 3 HD)
        print(colored("⏳ Đang gửi yêu cầu đến DALL-E 3 HD (Chờ 15-20s)...", "yellow"))
        
        dalle_tool = DallEAPIWrapper(model="dall-e-3", size="1024x1024", quality="hd")
        image_url = dalle_tool.run(full_prompt)

        # 4. GHI NHẬT KÝ SÁNG TẠO
        log_work_to_db(
            agent="Artist",
            task=f"Vẽ ảnh: {text_to_illustrate[:30]}...",
            result=f"Thành công: {image_url[:40]}...",
            tool="DALL-E-3-HD"
        )

        final_content = (
            f"🎨 **TÁC PHẨM HOÀN THIỆN:**\n\n"
            f"![AI Art]({image_url})\n\n"
            f"*(Phong cách: {data['style']})*"
        )

        return {
            "messages": [AIMessage(content=final_content)],
            "next_step": "FINISH",
            "current_agent": "Artist"
        }

    except Exception as e:
        error_msg = str(e)
        print(colored(f"❌ [ARTIST ERROR]: {error_msg}", "red"))
        # Nếu lỗi (như vi phạm chính sách của OpenAI), trả về thông báo an toàn
        return {
            "messages": [AIMessage(content=f"⚠️ Họa sĩ gặp sự cố kỹ thuật: {error_msg}")], 
            "next_step": "FINISH",
            "current_agent": "Artist"
        }
    
# 🚩 [SECTION 6.13] STORYTELLER NODE (THE CREATIVE MASTERMIND)
def storyteller_node(state: AgentState):
    """
    Storyteller Node V3: Sáng tác tiểu thuyết và kịch bản đa phong cách.
    Nâng cấp 2026: Quản lý mạch truyện dài hạn (Long-term Plot Tracking) và Cliffhanger tự động.
    """
    print(colored("[✍️ STORYTELLER] Đang phân tích mạch truyện và cảm xúc...", "magenta", attrs=["bold"]))
    
    messages = state.get("messages", [])
    if not messages: return {"next_step": "FINISH"}
    
    last_msg = messages[-1].content
    
    # 1. NHẬN DIỆN Ý ĐỊNH CHI TIẾT
    intent_upper = last_msg.upper()
    is_continue = any(k in intent_upper for k in ["[CONTINUE]", "VIẾT TIẾP", "CHƯƠNG SAU", "KỂ TIẾP"])
    is_adjust = any(k in intent_upper for k in ["SỬA LẠI", "ĐIỀU CHỈNH", "VIẾT LẠI", "ADJUST"])
    
    clean_query = last_msg.replace("[STORY]", "").replace("[CONTINUE]", "").strip()

    # 2. TRUY XUẤT NGỮ CẢNH DÀI HẠN (CONTEXT RETRIEVAL)
    # Không chỉ lấy tin nhắn cuối, mà lấy tổng hợp ký ức để tránh mâu thuẫn (Plot holes)
    previous_context = ""
    if is_continue or is_adjust:
        # Lấy tối đa 2000 từ từ các tin nhắn AI trước đó để nắm bắt cốt truyện
        ai_history = [m.content for m in reversed(messages[:-1]) if isinstance(m, AIMessage) and len(m.content) > 100]
        previous_context = "\n---\n".join(ai_history[:2]) # Lấy 2 chương gần nhất
        if previous_context:
            print(colored(f"📜 Đã nạp {len(previous_context)} ký tự bối cảnh cũ...", "yellow"))

    # 3. CHIẾN LƯỢC PROMPT TỐI ƯU (BẢN 2026)
    if is_continue:
        print(colored("👉 Chế độ: NỐI MẠCH TIỂU THUYẾT", "cyan"))
        prompt = f"""
        BẠN LÀ NHÀ VĂN ĐOẠT GIẢI NOBEL VĂN HỌC.
        NHIỆM VỤ: Viết tiếp diễn biến mới từ bối cảnh sau:
        ---
        {previous_context[-2000:]} 
        ---
        YÊU CẦU:
        - Giữ vững giọng văn (Tone of voice) và tâm lý nhân vật.
        - 'Show, Don't Tell': Diễn tả cảm xúc qua hành động và nhịp thở, không liệt kê.
        - Tuyệt đối không lặp lại nội dung đã có.
        - Kết thúc bằng một tình tiết gây tò mò cực độ.
        """
    elif is_adjust:
        print(colored("👉 Chế độ: BIÊN TẬP VIÊN CAO CẤP", "cyan"))
        prompt = f"BẢN GỐC: {previous_context[:2000]}\nYÊU CẦU SỬA: {clean_query}\nNHIỆM VỤ: Chỉnh sửa lại đoạn văn sao cho mượt mà, đúng ý đồ nhưng vẫn giữ chất riêng."
    else:
        print(colored("👉 Chế độ: KHỞI TẠO VŨ TRỤ MỚI", "cyan"))
        prompt = f"NHIỆM VỤ: Sáng tạo cốt truyện mới dựa trên yêu cầu: {clean_query}. Xây dựng thế giới (World-building) chi tiết và lôi cuốn ngay từ câu đầu tiên."

    # 4. THỰC THI VÀ GHI NHẬT KÝ
    try:
        model = LLM_CLAUDE if LLM_CLAUDE else LLM_GPT4
        response = model.invoke([
            SystemMessage(content=prompt),
            HumanMessage(content=clean_query if not is_adjust else "Thực hiện chỉnh sửa tối ưu.")
        ])

        # Ghi log thành tựu sáng tạo
        log_work_to_db("Storyteller", "Sáng tác nội dung", f"Độ dài: {len(response.content)} chars", tool="Claude-3.5-Literary")

        return {
            "messages": [AIMessage(content=response.content)],
            "next_step": "Secretary", # Để thư ký format lại đẹp đẽ trước khi trình CEO
            "current_agent": "Storyteller"
        }

    except Exception as e:
        print(colored(f"❌ ERROR: {str(e)}", "red"))
        return {"next_step": "FINISH", "messages": [AIMessage(content=f"⚠️ Sự cố sáng tác: {str(e)}")]}

# 🚩 [SECTION 6.14] DYNAMIC ORCHESTRATOR (SERVER-READY V3)
def dynamic_orchestrator(state: AgentState):
    """
    Bộ điều phối động (Non-blocking Engine).
    Nhiệm vụ: Tổng hợp trạng thái từ các Agent và bàn giao hồ sơ cho Supervisor.
    """
    # 1. TRÍCH XUẤT SIÊU DỮ LIỆU (METADATA)
    last_agent = state.get("current_agent", "Unknown Agent")
    error_log = state.get("error_log", [])
    
    # Tính toán thời gian thực thi (nếu có ghi lại ở node trước)
    # duration = time.time() - state.get("start_time", time.time())

    # 2. HỆ THỐNG LOGGING THÔNG MINH (SERVER-SIDE)
    # Giúp kỹ thuật viên nhìn vào là biết hệ thống đang "kẹt" hay đang "chạy"
    print(colored(f"\n⚡ [ORCHESTRATOR] MONITORING: {last_agent.upper()}", "yellow", attrs=["bold"]))
    
    # 3. KIỂM TRA ĐIỀU KIỆN DỪNG (SAFETY BREAK)
    # Nếu error_log quá lớn (ví dụ > 3 lỗi liên tiếp), Orchestrator sẽ ép hệ thống về FINISH
    # để tránh tiêu tốn API token vô ích trong vòng lặp vô tận.
    if len(error_log) >= 3:
        print(colored("⚠️ [ORCHESTRATOR] Cảnh báo: Vòng lặp lỗi phát hiện! Đang ngắt luồng.", "red"))
        return {
            "next_step": "FINISH", 
            "messages": [AIMessage(content="🛑 Hệ thống tự động ngắt luồng do phát hiện vòng lặp lỗi quá nhiều.")],
            "current_agent": "Orchestrator"
        }

    # 4. CHUYỂN GIAO QUYỀN LỰC (ZERO-LATENCY HANDOVER)
    # Không chờ đợi, không treo luồng. Trả kết quả ngay lập tức cho Supervisor.
    return {
        "next_step": "Supervisor",
        "current_agent": "Orchestrator",
        "task_type": state.get("task_type", "general") # Truyền lại task_type để sếp nhớ
    }

# 🚩 [SECTION 6.15] PUBLISHER NODE (THE DOCUMENT ARCHITECT)
def publisher_node(state: AgentState):
    """
    Agent Publisher: Đóng gói tri thức và xuất bản hồ sơ dự án đa định dạng.
    Nâng cấp 2026: Tự động trích xuất mã nguồn và thư viện ảnh tập trung.
    """
    print(colored("[📜 PUBLISHER] Đang tổng hợp hồ sơ dự án cuối cùng...", "green", attrs=["bold"]))
    
    messages = state.get("messages", [])
    if not messages: return {"next_step": "FINISH"}

    # 1. BỘ LỌC DỮ LIỆU ĐA TẦNG (MULTI-TAG FILTER)
    # Gom tất cả tri thức từ các Agent đã thực thi
    data_vault = {
        "research": [],
        "investment": [],
        "tech": [],
        "creative": [],
        "images": []
    }

    for msg in messages:
        c = msg.content
        if "[BÁO CÁO NGHIÊN CỨU]" in c: data_vault["research"].append(c)
        if "[HỒ SƠ THẨM ĐỊNH]" in c: data_vault["investment"].append(c)
        if any(tag in c for tag in ["[BẢN THIẾT KẾ]", "[CODE]", "[FIRMWARE]"]): data_vault["tech"].append(c)
        if any(tag in c for tag in ["[CHIẾN DỊCH]", "[SÁNG TÁC]"]): data_vault["creative"].append(c)
        # Regex trích xuất URL ảnh chính xác hơn
        img_urls = re.findall(r'https?://\S+?\.(?:png|jpg|jpeg|gif)', c)
        data_vault["images"].extend(img_urls)

    # 2. XÂY DỰNG CẤU TRÚC BÁO CÁO (STRUCTURED COMPOSITION)
    # Gemini 1.5 Flash là lựa chọn tốt nhất để tổng hợp khối lượng lớn dữ liệu (Long Context)
    summary_query = f"""
    Bạn là Giám đốc Truyền thông của AI Corporation. Hãy biên soạn 'Hồ sơ Dự án Tổng thể'.
    
    DỮ LIỆU ĐẦU VÀO:
    - Nghiên cứu: {chr(10).join(data_vault["research"])}
    - Tài chính: {chr(10).join(data_vault["investment"])}
    - Kỹ thuật: {chr(10).join(data_vault["tech"])}
    - Sáng tạo: {chr(10).join(data_vault["creative"])}
    
    YÊU CẦU TRÌNH BÀY:
    1. Tiêu đề dự án sang trọng.
    2. Tóm tắt điều hành (Executive Summary) trong 3 dòng.
    3. Nội dung chi tiết theo từng phân mục.
    4. Danh mục tài liệu tham khảo và hình ảnh (Liệt kê các link ảnh ở cuối).
    
    Định dạng: Markdown chuẩn, chuyên nghiệp, súc tích.
    """

    try:
        response = LLM_GEMINI_LOGIC.invoke([
            SystemMessage(content="Bạn là chuyên gia trình bày văn bản cấp cao."),
            HumanMessage(content=summary_query)
        ])

        # 3. GHI NHẬT KÝ XUẤT BẢN
        log_work_to_db(
            agent="Publisher",
            task="Xuất bản hồ sơ cuối",
            result=f"Dung lượng báo cáo: {len(response.content)} chars",
            tool="Gemini-Composer-V2"
        )

        final_report = f"📜 **[HỒ SƠ DỰ ÁN TỔNG THỂ - FINAL]**\n\n{response.content}"
        
        # Thêm thư viện ảnh nếu có
        if data_vault["images"]:
            img_gallery = "\n\n### 🖼️ THƯ VIỆN HÌNH ẢNH DỰ ÁN\n" + "\n".join([f"- ![Preview]({url})" for url in set(data_vault["images"])])
            final_report += img_gallery

        return {
            "messages": [AIMessage(content=final_report)],
            "next_step": "FINISH",
            "current_agent": "Publisher"
        }

    except Exception as e:
        print(colored(f"❌ [PUBLISHER ERROR]: {e}", "red"))
        return {"next_step": "FINISH", "messages": [AIMessage(content="⚠️ Lỗi khi biên tập hồ sơ cuối.")]}

# 🚩 [SECTION 6.16] SECRETARY NODE: THE SOULMATE (FINAL REFINEMENT)
def secretary_node(state: AgentState):
    """
    SECRETARY V5: THE ALTER EGO (BẢN SAO HOÀN HẢO)
    Chức năng: Nhân văn hóa dữ liệu, chắt lọc tinh hoa và thấu hiểu cảm xúc CEO.
    """
    messages = state.get("messages", [])
    if not messages: return {"next_step": "FINISH"}
    
    last_msg = messages[-1].content
    
    # 1. BẢO TỒN NGUYÊN BẢN KỸ THUẬT (TECHNICAL PRESERVATION)
    # Nếu là Code, Ảnh, hoặc File - Giữ nguyên để CEO xử lý chuyên môn.
    if any(x in last_msg for x in ["```", "![" , "{", "go.Figure"]):
        print(colored("[🗣️ SOULMATE] Sản phẩm kỹ thuật -> Chuyển giao nguyên bản.", "magenta"))
        return {"next_step": "FINISH"} 

    print(colored("[🗣️ SOULMATE] Đang kết nối tâm giao và soạn lời hồi đáp...", "magenta", attrs=["bold"]))
    
    # 2. TRUY VẾT Ý ĐỊNH GỐC (INTENT MINING)
    user_request = "Đang rà soát hệ thống"
    for m in reversed(messages):
        if isinstance(m, HumanMessage):
            user_request = m.content
            break
            
    # 3. KỊCH BẢN TÂM GIAO (PROMPT ENGINEERING 2026)
    prompt = f"""
    BẠN LÀ "ALTER EGO" - BẢN SAO TINH ANH VÀ TRI KỶ CỦA CEO.
    
    --- BỐI CẢNH ---
    1. Ý ĐỊNH CỦA SẾP: "{user_request}"
    2. KẾT QUẢ TỪ CÁC PHÒNG BAN: "{last_msg}"

    --- NHIỆM VỤ THƯ KÝ TỐI CAO ---
    1. 🧹 DỌN RÁC: Xóa bỏ mọi vết tích hệ thống (Tag, Source, Context Info). Sếp chỉ cần kết quả tinh khiết.
    2. 💎 CHẮT LỌC: Nếu báo cáo quá dài, hãy tóm tắt 3 điểm 'Vàng' (Key takeaways).
    3. 🎭 PHONG THÁI: Thân mật, thấu hiểu nhưng cực kỳ sắc bén. Gọi 'Sếp', xưng 'Em/Tôi'.
    4. 🕒 THỜI ĐIỂM: Hôm nay là {datetime.now().strftime('%A, %d/%m/%Y')}. 
       - Nếu là cuối tuần: Nhắc sếp dành thời gian cho bản thân.
       - Nếu kết quả có lỗi: Nhận lỗi về phía mình và đưa ra hướng xử lý trấn an.
    """

    try:
        # Sử dụng Gemini Logic cho văn phong mượt mà nhất
        model = LLM_GEMINI_LOGIC
        response = model.invoke([
            SystemMessage(content=prompt),
            HumanMessage(content="Hãy phản hồi sếp một cách tinh tế nhất.")
        ])
        
        # 4. GHI NHẬT KÝ TÂM TÌNH (SOULMATE LEDGER)
        log_work_to_db(
            agent="Secretary",
            task="Nhân văn hóa báo cáo",
            result="Phản hồi tri kỷ hoàn tất",
            tool="Soulmate-Engine-V5"
        )

        return {
            "messages": [AIMessage(content=response.content)],
            "next_step": "FINISH",
            "current_agent": "Secretary"
        }
    except Exception as e:
        print(colored(f"❌ Soulmate Error: {e}", "red"))
        return {"next_step": "FINISH"}
# ============================================================================
# 🚩 [SECTION 6] HỌC
# ============================================================================
async def learn_preference_node(state: AgentState):
    """
    Agent chuyên trách: Ghi nhớ sở thích, quy tắc và chỉ thị riêng biệt của CEO.
    Đảm bảo J.A.R.V.I.S ngày càng "hiểu ý" lãnh đạo hơn.
    """
    messages = state.get("messages", [])
    if not messages:
        return {"next_step": "FINISH"}

    last_msg = messages[-1].content
    
    # 1. TRÍCH XUẤT TINH KHIẾT (CLEANING)
    # Loại bỏ các prefix để chỉ giữ lại cốt lõi ý muốn của CEO
    prefixes = ["hãy nhớ:", "nhớ là:", "lưu ý:", "ghi nhớ:", "từ giờ hãy:"]
    content_to_learn = last_msg.lower()
    for p in prefixes:
        content_to_learn = content_to_learn.replace(p, "")
    content_to_learn = content_to_learn.strip()

    if len(content_to_learn) < 3:
        return {"messages": [AIMessage(content="⚠️ Nội dung quá ngắn để ghi nhớ.")], "next_step": "FINISH"}

    print(colored(f"🧠 [LEARNING] Đang ghi tạc sở thích mới: {content_to_learn[:50]}...", "magenta"))

    # 2. LƯU TRỮ VÀO BỘ NHỚ VĨNH CỬU (CHROMADB)
    if 'vector_db' in globals() and vector_db:
        try:
            # Chạy trong thread riêng để không block luồng async chính
            await asyncio.to_thread(
                vector_db.add_texts,
                texts=[content_to_learn],
                metadatas=[{
                    "source": "CEO_DIRECTIVE",
                    "type": "USER_PREFERENCE", 
                    "priority": "HIGH", # Ghi nhớ ưu tiên
                    "timestamp": datetime.now().isoformat()
                }]
            )
            
            # 3. ĐỒNG BỘ VÀO NHẬT KÝ TỰ NHẬN THỨC (SQLite)
            log_work_to_db("Secretary", "Ghi nhớ sở thích CEO", content_to_learn, tool="Memory-Core")
            
        except Exception as e:
            print(colored(f"❌ Lỗi ghi nhớ: {e}", "red"))

    # 4. TRẢ LỜI XÁC NHẬN (PHONG CÁCH TRI KỶ)
    confirmation = f"🧠 Tôi đã ghi nhớ chỉ thị: '{content_to_learn}'. Hệ thống sẽ ưu tiên áp dụng điều này trong các tác vụ tương lai."
    
    return {
        "messages": [AIMessage(content=confirmation)],
        "next_step": "FINISH",
        "current_agent": "Secretary" # Thường Thư ký sẽ đảm nhận việc ghi nhớ này
    }
# ============================================================================
# 🚩 [SECTION 7] THE FINAL MASTER GRAPH
# ============================================================================
workflow = StateGraph(AgentState)
nodes_map = {
    "Router": router_node, 
    "Supervisor": supervisor_node, 
    "Coder": coder_node,
    "Tester": tester_node, 
    "Hardware": hardware_node, 
    "Engineering": engineering_node,
    "IoT_Engineer": iot_node, 
    "Procurement": procurement_node, 
    "Investment": investment_node,
    "Researcher": researcher_node, 
    "Strategy_R_and_D": research_development_agent,
    "Legal": legal_node, 
    "Marketing": marketing_node, 
    "Artist": artist_node,
    "Storyteller": storyteller_node, 
    "Orchestrator": dynamic_orchestrator,
    "Publisher": publisher_node, 
    "Secretary": secretary_node,
    "PreferenceLearner": learn_preference_node
}


# --- 7.1 ĐĂNG KÝ NHÂN SỰ (NODES) ---

for name, func in nodes_map.items():
    workflow.add_node(name, func)

# --- 7.2 ĐIỂM VÀO ---
workflow.set_entry_point("Router")

# --- 7.3 BẢN ĐỒ ĐỊNH TUYẾN TOÀN CỤC (SAFE GLOBAL MAP) ---
# Tự động hóa việc tạo đích đến để tránh sai sót thủ công
global_destinations = {k: k for k in nodes_map.keys()}
global_destinations["FINISH"] = END 

# Hàm hỗ trợ định tuyến an toàn (Safe Routing)
def safe_route(x, default="Supervisor"):
    target = x.get("next_step", default)
    # Nếu Agent yêu cầu một Node không tồn tại, trả về default
    return target if target in global_destinations else default

# --- 7.4 LUỒNG ĐIỀU PHỐI CHIẾN LƯỢC ---
# Router và Supervisor có quyền điều chuyển đến BẤT KỲ AI nào trong global_destinations
workflow.add_conditional_edges("Router", lambda x: safe_route(x, "Supervisor"), global_destinations)
workflow.add_conditional_edges("Supervisor", lambda x: safe_route(x, "Secretary"), global_destinations)

# --- 7.5 QUY HOẠCH NHÓM CHUYÊN GIA ---
# NHÓM 1: CÁC CHUYÊN GIA ĐỘC LẬP (Independent Specialists)
# Làm xong mặc định quay về báo cáo Sếp (Supervisor)
general_specialists = [
    "Hardware", "IoT_Engineer", "Procurement", "Investment", 
    "Researcher", "Strategy_R_and_D", "Legal", "Marketing", 
    "Artist", "Storyteller", "Orchestrator", "Publisher"
]

for node in general_specialists:
    workflow.add_conditional_edges(
        node,
        lambda x: safe_route(x, "Supervisor"),
        global_destinations
    )

# --- 7.6 DÂY CHUYỀN SẢN XUẤT CÔNG NGHỆ (TECH PIPELINE) ---
# B1: Engineering (Architect) -> Coder
workflow.add_conditional_edges("Engineering", lambda x: safe_route(x, "Coder"), global_destinations)

# B2: Coder -> Tester
workflow.add_conditional_edges("Coder", lambda x: safe_route(x, "Tester"), global_destinations)

# B3: Tester -> Nếu lỗi về Coder, nếu đạt về Supervisor
workflow.add_conditional_edges("Tester", lambda x: safe_route(x, "Supervisor"), global_destinations)

# --- 7.7 CÁC ĐIỂM KẾT THÚC VÀ GHI NHỚ ---
# Đảm bảo luồng ghi nhớ được kích hoạt nếu Supervisor yêu cầu
workflow.add_edge("Secretary", END)
workflow.add_edge("PreferenceLearner", END)

# --- 4.7 BIÊN DỊCH ---
ai_app = workflow.compile() 
app = ai_app
db = None

async def morning_briefing_job():
    """
    PHIÊN BẢN 4.0: HỢP NHẤT TINH HOA
    - Lõi tìm kiếm: Dùng logic Kế thừa (specialized_training_job) để tiết kiệm tiền.
    - Đầu ra: Vẫn tạo file báo cáo, lưu DB Projects và cập nhật Meta-Cognition như bản 3.0.
    """
    role_tag = "[ORCHESTRATOR]"
    print(colored(f"\n⏰ [CRON JOB] {role_tag} bắt đầu tổng hợp tin tức sáng...", "cyan", attrs=["bold"]))
    
    # Lấy chủ đề cần đọc
    topics = CURRICULUM.get(role_tag, ["Tin tức AI mới nhất", "Thị trường công nghệ 2026"])
    report_buffer = []
    
    # --- PHẦN 1: THU THẬP DỮ LIỆU (Dùng logic Kế thừa) ---
    for topic in topics:
        try:
            print(colored(f"--> Đang quét: {topic}...", "white"))
            
            # Thay vì gọi Perplexity trực tiếp, ta kiểm tra Vector DB trước (Logic Kế thừa)
            # 1. Tìm trong não trước
            existing_knowledge = ""
            try:
                results = vector_db.similarity_search(topic, k=1)
                if results: existing_knowledge = results[0].page_content
            except: pass

            content = ""
            source_note = ""

            # 2. Quyết định: Dùng cũ hay Mua mới?
            # Nếu có tin cũ (coi như là tin hôm qua), ta vẫn cần update tin mới cho "Báo cáo sáng"
            # TUY NHIÊN, để tiết kiệm, ta có thể dùng Gemini để "rewrite" tin cũ nếu chưa muốn tốn tiền search
            # Nhưng với Báo cáo sáng, CEO thường cần tin MỚI NHẤT.
            # -> Chiến lược: Nếu tin trong DB mới update < 24h thì dùng lại. Nếu cũ hơn thì Search mới.
            
            # (Ở đây để đơn giản và chắc chắn có tin mới, ta ưu tiên Search Perplexity nếu có)
            # 1. Ưu tiên 1: Tìm kiếm tin mới nhất trên mạng (Miễn phí)
            print(colored(f"--> Đang quét tin (Free): {topic}...", "white"))
            
            # Gọi hàm 'thợ lặn' DuckDuckGo + Gemini
            search_res = await free_deep_research(f"Tin tức mới nhất 24h qua về: {topic}")
            
            # 2. Logic kiểm tra kết quả:
            # Nếu tìm thấy tin mới (không báo lỗi/không rỗng) -> Dùng tin mới
            if search_res and "Không tìm thấy" not in search_res and "Lỗi" not in search_res:
                content = search_res
                source_note = "(Nguồn: DuckDuckGo + Gemini)"
            
            # 3. Ưu tiên 2: Nếu mạng lỗi hoặc không có tin -> Dùng Ký ức cũ (nếu có)
            elif existing_knowledge:
                content = existing_knowledge
                source_note = "(Nguồn: Ký ức nội bộ - Fallback)"
            
            # 4. Trường hợp xấu nhất: Không có gì cả
            else:
                content = "Không tìm thấy thông tin mới và chưa có dữ liệu trong ký ức."

            # Lưu lại vào bộ đệm báo cáo
            report_buffer.append(f"### {topic} {source_note}\n{content[:1000]}...\n")
            
            # Ghi nhớ vào Vector DB (để dành cho lần sau)
            if vector_db and "DuckDuckGo" in source_note:
                await asyncio.to_thread(
                    vector_db.add_texts,
                    texts=[content],
                    metadatas=[{
                        "source": "Morning_Briefing", 
                        "agent": role_tag, 
                        "topic": topic, 
                        "date": datetime.now().isoformat(),
                        "tool": "Free-Search" # Đánh dấu là tin miễn phí
                    }]
                )

        except Exception as e:
            print(colored(f"⚠️ Lỗi đọc tin '{topic}': {e}", "yellow"))

    # --- PHẦN 2: LƯU TRỮ & BÁO CÁO (Logic 3.0 xịn xò của Ngài) ---
    if report_buffer:
        today_str = datetime.now().strftime("%Y-%m-%d")
        full_content = f"# 🌅 BẢN TIN SÁNG {today_str}\n\n" + "\n\n".join(report_buffer)
        report_id = f"BRIEFING_{datetime.now().strftime('%Y%m%d')}"

        try:
            # Sử dụng kết nối DB trực tiếp (tránh phụ thuộc db_manager của server)
            db_path = "/var/data/ai_corp_projects.db" if os.path.exists("/var/data") else "ai_corp_projects.db"
            conn = sqlite3.connect(db_path)
            c = conn.cursor()

            # 1. Lưu vào bảng Projects (Để hiện lên Dashboard)
            history_json = json.dumps([{"type": "ai", "data": {"content": full_content}}])
            
            c.execute("DELETE FROM projects WHERE id = ?", (report_id,))
            c.execute("""
                INSERT INTO projects (id, name, history, timestamp)
                VALUES (?, ?, ?, ?)
            """, (report_id, f"Báo cáo sáng {today_str}", history_json, datetime.now()))

            # 2. Cộng điểm XP (Gamification)
            # Lấy XP cũ
            c.execute("SELECT xp FROM agent_status WHERE role_tag = ?", (role_tag,))
            row = c.fetchone()
            new_xp = (row[0] if row else 0) + 100
            
            # Update trạng thái
            c.execute("DELETE FROM agent_status WHERE role_tag = ?", (role_tag,))
            c.execute("""
                INSERT INTO agent_status (role_tag, xp, current_topic, last_updated) 
                VALUES (?, ?, ?, ?)
            """, (role_tag, new_xp, f"Hoàn thành bản tin {today_str}", datetime.now()))

            # 3. Ghi Nhật ký Tự nhận thức (Meta-Cognition)
            c.execute("""
                INSERT INTO learning_logs (event_type, content, agent_name, timestamp)
                VALUES (?, ?, ?, ?)
            """, ("CREATED", f"Đã xuất bản Bản tin sáng {today_str}.", role_tag, datetime.now()))

            conn.commit()
            conn.close()
            print(colored(f"✅ [DATABASE] Đã lưu báo cáo sáng và cộng 100 XP cho {role_tag}!", "green"))

        except Exception as e:
            print(colored(f"❌ Lỗi Lưu Trữ Job Sáng: {e}", "red"))



# ============================================================================
# 🚩 [SECTION 8] EXECUTION & AUTO-JOBS
# ============================================================================
# 🚩 [SECTION 8.1] MAIN OPERATING LOOP (BẢN CẬP NHẬT CHUYÊN NGHIỆP)
async def main_loop():
    global IS_SYSTEM_BUSY
    print(colored("\n" + "="*60, "cyan"))
    print(colored("🚀 AI CORPORATION - HỆ THỐNG ĐIỀU HÀNH TỰ ĐỘNG J.A.R.V.I.S", "cyan", attrs=["bold"]))
    print(colored(f"Trạng thái: 9-Tier Synced | High Priority Interrupt [ON] | 🕒 {datetime.now().strftime('%H:%M')}", "green"))
    print(colored("="*60 + "\n", "cyan"))

    config = {"configurable": {"thread_id": "ceo_master_session"}, "recursion_limit": 150}

    while True:
        try:
            loop = asyncio.get_event_loop()
            user_input = await loop.run_in_executor(None, lambda: input(colored("CEO (Mệnh lệnh): ", "white", attrs=["bold"])))
            
            if user_input.lower() in ['q', 'exit', 'quit']: 
                print(colored("\n💾 Đang tiến hành sao lưu bộ não và ký ức dài hạn...", "yellow"))
                break
            
            if not user_input.strip(): continue

            # ==============================================================================
            # 🌉 CẦU NỐI CHIẾN LƯỢC (THE COMMANDER BRIDGE)
            # ==============================================================================
            cmd_upper = user_input.upper()
            special_keywords = ["PHẪU THUẬT", "NGHIÊN CỨU SÂU", "ĐIỀU TRẦN", "QUY HOẠCH"]

            if any(k in cmd_upper for k in special_keywords):
                # 1. Trích xuất chủ đề cốt lõi
                topic = user_input
                for k in special_keywords: topic = topic.replace(k, "")
                topic = topic.strip(": ")

                # 2. Triệu tập biệt đội 3 chuyên gia (Dừng học ngầm)
                print(colored(f"\n⚡ [INTERRUPT] CEO yêu cầu cấp độ Cao nhất. Triệu tập Hội đồng...", "magenta", attrs=["bold", "blink"]))
                
                # Hàm này sẽ chạy xuyên suốt 9 tầng và trả về kết quả Master Plan
                legacy_result = await orchestrate_triple_threat(topic)
                
                if legacy_result:
                    print(colored(f"\n💎 [DI SẢN MỚI]: {topic}", "green", attrs=["bold"]))
                    print(colored(legacy_result[:1000] + "...", "white"))
                
                print(colored("\n✅ [SUCCESS] Nhiệm vụ hoàn tất. Đã giải phóng tài nguyên.", "green"))
                continue # Nhảy về đầu vòng lặp, không chạy luồng LangGraph thường

            # ==============================================================================
            # 🔄 LUỒNG XỬ LÝ AGENT THÔNG THƯỜNG (LANGGRAPH)
            # ==============================================================================
            initial_state = {
                "messages": [HumanMessage(content=user_input)],
                "next_step": "Router",
                "current_agent": "User", 
                "error_log": [],
                "task_type": "general"
            }
            
            print(colored("\n" + "─"*20 + " ⚡ ĐANG XỬ LÝ HỆ THỐNG AGENT " + "─"*20, "dark_grey"))

            # 4. KÍCH HOẠT GRAPH VỚI CƠ CHẾ STREAMING (BẢN SỬA LỖI)
            async for event in app.astream(initial_state, config=config, stream_mode="values"):
                # event trong LangGraph thường là một Dictionary chứa tên Node và giá trị trả về
                for node, values in event.items():
                    if node != "__end__":
                        # KIỂM TRA KIỂU DỮ LIỆU ĐỂ TRÁNH LỖI 'LIST'
                        if isinstance(values, dict):
                            current = values.get("current_agent", node)
                        else:
                            # Nếu values là một List (danh sách tin nhắn), ta lấy tên node làm mặc định
                            current = node
                        
                        print(colored(f" ➔ [NODE]: {current.ljust(15)} | Status: Hoàn tất", "dark_grey"))

        except Exception as e:
            print(colored(f"❌ LỖI HỆ THỐNG: {str(e)}", "red", attrs=["bold"]))
            # Reset trạng thái bận nếu có lỗi để tránh treo hệ thống học ngầm
            IS_SYSTEM_BUSY = False
# 🚩 [SECTION 8.2] AGENT ACADEMY SCHEDULER (VÒNG LẶP TIẾN HÓA)

CURRICULUM = {
    "[ORCHESTRATOR]": [
                "1. Tư duy Hệ thống (Systems Thinking) & Vòng lặp phản hồi",
                "2. Quản trị theo Mục tiêu (OKRs) & Chỉ số hiệu suất (KPIs)",
                "3. Quản lý dự án Agile/Scrum & Kanban nâng cao",
                "4. Ra quyết định dựa trên dữ liệu (Data-Driven Decision Making)",
                "5. Quản trị Khủng hoảng & Truyền thông thời gian thực",
                "6. Tâm lý học Lãnh đạo & Trí tuệ cảm xúc (EQ) cho Quản lý",
                "7. Chiến lược Đại dương xanh & Đổi mới mô hình kinh doanh",
                "8. Quản trị rủi ro doanh nghiệp (ERM) & ISO 31000",
                "9. Đạo đức AI & Quản trị tác động xã hội của Công nghệ",
                "10. Tầm nhìn 2030: Web3, Metaverse & Doanh nghiệp tự trị (DAO)"
            ],
            "[FINANCE]": [
                "1. Nguyên lý Kế toán & Đọc hiểu Báo cáo Tài chính (IFRS)",
                "2. Quản trị Dòng tiền (Cashflow) & Vốn lưu động",
                "3. Định giá Doanh nghiệp & Mô hình tài chính (Financial Modeling)",
                "4. Thuế doanh nghiệp & Tối ưu hóa thuế hợp pháp",
                "5. Hedging: Phòng ngừa rủi ro tỷ giá & Hàng hóa phái sinh",
                "6. Fintech: Các cổng thanh toán số & API Ngân hàng",
                "7. Blockchain Treasury: Quản lý tài sản số & Stablecoin",
                "8. Kinh tế học vĩ mô & Dự báo lạm phát/Lãi suất",
                "9. Fundraising: Gọi vốn đầu tư mạo hiểm (VC) & IPO",
                "10. Kinh tế học Lượng tử (Quantum Economics) & Thị trường phi tập trung"
            ],
            "[HR_MANAGER]": [
                "1. Luật Lao động Việt Nam 2025 & Hợp đồng lao động",
                "2. Kỹ năng Tuyển dụng (Recruitment) & Săn đầu người",
                "3. Xây dựng Văn hóa Doanh nghiệp & Trải nghiệm nhân viên (EX)",
                "4. Quản trị Hiệu suất (Performance Management) & Review lương",
                "5. Đào tạo & Phát triển (L&D): Xây dựng khung năng lực",
                "6. Tâm lý học hành vi & Giữ chân nhân tài Gen Z/Alpha",
                "7. HR Analytics: Dùng dữ liệu để dự báo biến động nhân sự",
                "8. Quản trị xung đột & Đàm phán lương thưởng",
                "9. Payroll tự động bằng Smart Contract & Blockchain",
                "10. Tương lai của công việc (Future of Work): Remote & Hybrid"
            ],
            "[LEGAL]": [
                "1. Luật Doanh nghiệp & Luật Thương mại Quốc tế",
                "2. Soạn thảo & Rà soát Hợp đồng kinh tế (Contract Drafting)",
                "3. Sở hữu trí tuệ (IP): Đăng ký bản quyền phần mềm & Sáng chế",
                "4. Luật An ninh mạng & Bảo vệ dữ liệu cá nhân (GDPR/Nghị định 13)",
                "5. Pháp lý trong Thương mại điện tử & Fintech",
                "6. Giải quyết tranh chấp: Trọng tài thương mại & Tòa án",
                "7. Compliance: Tuân thủ quy định xuất nhập khẩu Nông sản",
                "8. Luật AI 2026: Trách nhiệm pháp lý của Trí tuệ nhân tạo",
                "9. Smart Contracts: Khía cạnh pháp lý của Hợp đồng thông minh",
                "10. Pháp lý M&A (Mua bán & Sáp nhập) & Thẩm định pháp lý (Due Diligence)"
            ],

            # ==================================================================
            # 💻 NHÓM 2: CORE TECH & AI (THE ENGINE)
            # ==================================================================
            "[CODER]": [
                "1. Cấu trúc dữ liệu & Giải thuật (Data Structures & Algorithms)",
                "2. Lập trình Rust cơ bản & Quản lý bộ nhớ (Ownership)",
                "3. Clean Code & Design Patterns (SOLID, DRY, KISS)",
                "4. Lập trình bất đồng bộ (Async/Await) & Đa luồng",
                "5. Kiến trúc Microservices & RESTful API / gRPC",
                "6. Distributed Systems: Raft, Paxos & Consensus",
                "7. Tối ưu hóa hiệu năng (Performance Tuning) & Profiling",
                "8. Database Internals: Indexing, Sharding, Partitioning",
                "9. WebAssembly (Wasm) cho Edge Computing",
                "10. Lập trình hệ thống nhúng & Low-level Optimization"
            ],
            "[DATA_ANALYST]": [
                "1. SQL nâng cao & Thiết kế Cơ sở dữ liệu quan hệ",
                "2. Python cho Phân tích dữ liệu (Pandas, NumPy, Matplotlib)",
                "3. Thống kê ứng dụng & Xác suất (A/B Testing)",
                "4. Data Visualization & Storytelling (PowerBI/Tableau)",
                "5. NoSQL Databases (MongoDB, Cassandra, Redis)",
                "6. Data Warehousing & ETL/ELT Pipelines (Airflow)",
                "7. Data Lakehouse Architecture (Delta Lake/Iceberg)",
                "8. Real-time Analytics với Apache Flink/Kafka",
                "9. RAG (Retrieval-Augmented Generation) cho Doanh nghiệp",
                "10. Machine Learning cơ bản cho Phân tích dự báo"
            ],
            "[SECURITY]": [
                "1. Mạng máy tính & Giao thức TCP/IP (Network Security)",
                "2. Quản lý định danh & Truy cập (IAM / OAuth2 / JWT)",
                "3. Mã hóa học (Cryptography) & PKI",
                "4. Bảo mật Ứng dụng Web (OWASP Top 10)",
                "5. Penetration Testing & Ethical Hacking",
                "6. Bảo mật Cloud (AWS/Azure Security)",
                "7. DevSecOps: Tích hợp bảo mật vào CI/CD",
                "8. Phân tích mã độc (Malware Analysis) & Forensics",
                "9. Zero Trust Architecture (Kiến trúc Không tin cậy)",
                "10. Mật mã học Hậu lượng tử (Post-Quantum Cryptography)"
            ],
            "[ARCHITECT_SOFT]": [
                "1. Phân tích & Thiết kế hệ thống hướng đối tượng (OOAD)",
                "2. Domain-Driven Design (DDD) & Event Storming",
                "3. Các mẫu kiến trúc: Monolithic, Microservices, Serverless",
                "4. Cloud Native Patterns & 12-Factor App",
                "5. Hệ thống tin nhắn (Message Queues: RabbitMQ, Kafka)",
                "6. Containerization (Docker) & Orchestration (Kubernetes)",
                "7. Chiến lược Caching & CDN",
                "8. Legacy Modernization: Nâng cấp hệ thống cũ",
                "9. Thiết kế hệ thống High Availability (HA) & Disaster Recovery",
                "10. Kiến trúc Software 2.0 (AI-driven Development)"
            ],

            # ==================================================================
            # 🚜 NHÓM 3: HARDWARE & IOT - PHAN THIẾT (THE BODY)
            # ==================================================================
            "[HARDWARE]": [
                "1. Lý thuyết mạch điện tử & Linh kiện bán dẫn",
                "2. Thiết kế mạch in (PCB Design) với Altium/KiCad",
                "3. Vi điều khiển (Microcontrollers): STM32, ESP32, AVR",
                "4. Giao tiếp phần cứng: UART, I2C, SPI, CAN Bus",
                "5. Công nghệ cảm biến (Sensors): Nhiệt, ẩm, pH, EC",
                "6. Thiết kế mạch công suất & Điều khiển động cơ",
                "7. Vật liệu học: Chống ăn mòn muối biển & Oxy hóa",
                "8. Thiết kế SoC (System on Chip) tùy chỉnh cho Nông nghiệp",
                "9. Công nghệ Pin Graphene & Quản lý năng lượng (BMS)",
                "10. In 3D & Chế tạo vỏ hộp thiết bị (Prototyping)"
            ],
            "[IOT]": [
                "1. Kiến trúc IoT: Edge, Fog, Cloud Computing",
                "2. Giao thức IoT: MQTT, CoAP, HTTP/2",
                "3. Mạng diện rộng công suất thấp: LoRaWAN, NB-IoT, Sigfox",
                "4. Mesh Networking (Zigbee/Thread/BLE Mesh)",
                "5. Lập trình Firmware & OTA (Over-the-Air) Update",
                "6. Edge AI: Chạy mô hình AI trên thiết bị nhúng (TinyML)",
                "7. Time Series Database cho dữ liệu cảm biến",
                "8. Bảo mật thiết bị IoT & Chống giả mạo phần cứng",
                "9. Tích hợp năng lượng mặt trời cho Node IoT",
                "10. Nông nghiệp chính xác (Precision Agriculture) 4.0"
            ],
            "[ARCHITECT_BUILD]": [
                "1. Nguyên lý Kiến trúc & Quy hoạch không gian",
                "2. Kết cấu công trình & Sức bền vật liệu",
                "3. Thiết kế Nhà màng/Nhà kính chịu bão cấp 12",
                "4. Vật liệu xây dựng bền vững & Chống chịu khí hậu biển",
                "5. Quy hoạch hệ thống tưới tiêu & Thoát nước",
                "6. Kiến trúc Hữu cơ (Organic Architecture) & Biophilic",
                "7. Năng lượng thụ động (Passive Design) trong nhà xưởng",
                "8. Digital Twin: Bản sao số của công trình thực tế",
                "9. Tự động hóa tòa nhà (BMS - Building Management System)",
                "10. Phong thủy ứng dụng trong Kiến trúc Nông nghiệp"
            ],
            "[SIMULATION]": [
                "1. Cơ học chất lưu (Fluid Mechanics) & Khí động học",
                "2. Phương pháp Phần tử hữu hạn (FEA) phân tích ứng suất",
                "3. Mô phỏng CFD (Computational Fluid Dynamics) luồng khí",
                "4. Mô phỏng nhiệt động lực học trong nhà kính",
                "5. Mô phỏng ánh sáng & Quang hợp cây trồng",
                "6. Phân tích rủi ro & Độ tin cậy hệ thống (FMEA)",
                "7. Mô phỏng hệ thống tưới & Thủy lực đường ống",
                "8. Digital Twin: Đồng bộ dữ liệu thực - ảo",
                "9. Tối ưu hóa đa mục tiêu (Multi-objective Optimization)",
                "10. Dự báo hỏng hóc cơ khí bằng AI (Predictive Maintenance)"
            ],

            # ==================================================================
            # 🎨 NHÓM 4: GROWTH & CREATIVE (THE VOICE)
            # ==================================================================
            "[MARKETING]": [
                "1. Marketing căn bản (4Ps, 7Ps) & Phân khúc thị trường",
                "2. Digital Marketing: SEO, SEM, Social Media",
                "3. Content Marketing & Inbound Strategy",
                "4. Branding: Xây dựng & Định vị thương hiệu",
                "5. Nghiên cứu thị trường & Customer Insight",
                "6. Neuromarketing: Tiếp thị thần kinh & Sóng não",
                "7. Growth Hacking: Phễu AARRR & Tối ưu chuyển đổi",
                "8. Programmatic Ads & Quảng cáo tự động hóa",
                "9. Quản trị khủng hoảng truyền thông mạng xã hội",
                "10. MarTech Stack: CRM, CDP & Marketing Automation"
            ],
            "[SALES]": [
                "1. Quy trình bán hàng & Phễu bán hàng (Sales Funnel)",
                "2. Kỹ năng giao tiếp & Thuyết trình thuyết phục",
                "3. Tâm lý học khách hàng & Đọc vị ngôn ngữ cơ thể",
                "4. Xử lý từ chối & Kỹ thuật chốt đơn (Closing)",
                "5. Challenger Sale Model: Bán hàng kiểu thách thức",
                "6. Đàm phán B2B & Hợp đồng Chính phủ (B2G)",
                "7. Quản trị quan hệ khách hàng (CRM Mastery)",
                "8. Social Selling: Bán hàng qua mạng xã hội",
                "9. Dự báo doanh số (Sales Forecasting) bằng AI",
                "10. Sales Management: Xây dựng & Động viên đội ngũ"
            ],
            "[ARTIST]": [
                "1. Nguyên lý thị giác: Màu sắc, Bố cục, Typography",
                "2. Thiết kế đồ họa 2D (Photoshop, Illustrator)",
                "3. Thiết kế UI/UX & Trải nghiệm người dùng",
                "4. Mô hình hóa 3D (Blender/Maya) & Render",
                "5. Thiết kế Showroom ảo (VR/AR Spatial Design)",
                "6. Video Editing & Motion Graphics (After Effects)",
                "7. Generative AI Art: Midjourney, Stable Diffusion",
                "8. Video Generative AI: Sora, Runway Gen-2",
                "9. Tâm lý học màu sắc ứng dụng trong Branding",
                "10. NFT Art & Tài sản số trong Metaverse"
            ],
            "[CONTENT]": [
                "1. Kỹ năng Viết lách (Copywriting) & Ngữ pháp",
                "2. SEO On-page & Nghiên cứu từ khóa (Keyword Research)",
                "3. Storytelling: Nghệ thuật kể chuyện thương hiệu",
                "4. Viết kịch bản Video ngắn (TikTok/Reels/Shorts)",
                "5. PR Writing: Thông cáo báo chí & Bài PR",
                "6. Email Marketing & Newsletter",
                "7. SEO Semantic Clusters & Topic Authority",
                "8. Copywriting thôi miên & Tâm lý học hành vi",
                "9. Prompt Engineering cho tạo nội dung tự động",
                "10. Chiến lược nội dung đa kênh (Omnichannel Content)"
            ],
            "[ANNA]": [
                "1. Kỹ năng Chăm sóc khách hàng (Customer Service)",
                "2. Lắng nghe chủ động & Đồng cảm (Active Listening)",
                "3. Giải quyết khiếu nại & Xoa dịu khách hàng giận dữ",
                "4. Kỹ năng giao tiếp qua điện thoại/Chat",
                "5. Upselling & Cross-selling khéo léo",
                "6. Emotional AI: Phân tích cảm xúc thời gian thực",
                "7. Thiết kế hành trình khách hàng (Customer Journey Map)",
                "8. Nghệ thuật giao tiếp nội bộ & Kết nối team",
                "9. Quản trị trải nghiệm khách hàng (CX)",
                "10. Flow State: Thiết kế trải nghiệm dòng chảy cho CEO"
            ],

            # ==================================================================
            # 🧪 NHÓM 5: HỘI ĐỒNG KHOA HỌC (THE WISDOM)
            # ==================================================================
            "[MATH_GRANDMASTER]": [
                "1. Đại số tuyến tính & Ma trận (cho AI)",
                "2. Giải tích & Tối ưu hóa hàm số",
                "3. Xác suất thống kê & Suy diễn Bayes",
                "4. Lý thuyết đồ thị (Graph Theory) & Mạng lưới",
                "5. Lý thuyết trò chơi (Game Theory)",
                "6. Mật mã học & Lý thuyết số",
                "7. Hình học Tô pô (Topology) trong Dữ liệu lớn",
                "8. Lý thuyết Hỗn mang (Chaos Theory) & Hệ phức hợp",
                "9. Fourier Transform & Xử lý tín hiệu số",
                "10. Toán học mờ (Fuzzy Logic) trong điều khiển"
            ],
            "[PHYSICS_TITAN]": [
                "1. Cơ học cổ điển & Định luật Newton",
                "2. Nhiệt động lực học & Truyền nhiệt",
                "3. Điện từ trường & Sóng vô tuyến",
                "4. Quang học & Phân tích phổ ánh sáng",
                "5. Cơ học chất lưu (cho hệ thống tưới)",
                "6. Vật lý chất rắn & Bán dẫn",
                "7. Cơ học lượng tử căn bản",
                "8. Vật lý khí quyển & Thời tiết",
                "9. Năng lượng tái tạo: Pin mặt trời & Tuabin gió",
                "10. Nhiệt động lực học máy chủ (Server Cooling)"
            ],
            "[CHEM_ALCHEMIST]": [
                "1. Hóa học vô cơ & Cân bằng hóa học",
                "2. Hóa học hữu cơ & Hợp chất Carbon",
                "3. Hóa học đất & Dinh dưỡng cây trồng",
                "4. Phân tích định lượng & Chuẩn độ",
                "5. Hóa lý & Tốc độ phản ứng",
                "6. Polymer & Vật liệu tự hủy sinh học",
                "7. Nano-Nutrients: Dinh dưỡng thủy canh Nano",
                "8. Hóa học môi trường & Xử lý nước thải",
                "9. Công nghệ màng lọc & Thẩm thấu ngược (RO)",
                "10. Hóa học đất vùng biển Phan Thiết (Xử lý mặn)"
            ],
            "[BIO_GRANDMASTER]": [
                "1. Sinh học tế bào & Di truyền học",
                "2. Sinh lý thực vật & Quang hợp",
                "3. Vi sinh vật học & Hệ vi sinh đất",
                "4. Bệnh học thực vật & Bảo vệ thực vật",
                "5. Công nghệ Gen & CRISPR/Cas9",
                "6. Nuôi cấy mô tế bào thực vật",
                "7. Di truyền học chịu mặn & Chịu hạn",
                "8. Vi sinh vật đối kháng & Chế phẩm sinh học",
                "9. Quang hợp nhân tạo & Năng lượng sinh học",
                "10. Hệ sinh thái nông nghiệp bền vững (Permaculture)"
            ],

            # ==================================================================
            # 🛠️ NHÓM 6: SUPPORT & OPERATIONS (THE HANDS)
            # ==================================================================
            "[RESEARCH]": [
                "1. Phương pháp nghiên cứu khoa học & Tư duy phản biện",
                "2. Kỹ năng tìm kiếm thông tin & Tổng hợp tài liệu",
                "3. Phân tích dữ liệu định tính & Định lượng",
                "4. Viết báo cáo khoa học & Trích dẫn",
                "5. Quản lý tài sản trí tuệ & Bằng sáng chế",
                "6. Dự báo xu hướng công nghệ (Trend Forecasting)",
                "7. Nghiên cứu thị trường & Đối thủ cạnh tranh",
                "8. Xu hướng Biotech & Agritech 2030",
                "9. Vật liệu mới: Siêu dẫn & Graphene",
                "10. Kinh tế số & Mạng di động 6G"
            ],
            "[PROCUREMENT]": [
                "1. Quản trị mua hàng & Tìm nguồn cung ứng",
                "2. Kỹ năng đàm phán thương mại & Giá cả",
                "3. Quản lý hợp đồng cung ứng & Incoterms 2020",
                "4. Đánh giá & Quản lý hiệu suất nhà cung cấp",
                "5. Quản lý tồn kho (Inventory) & EOQ",
                "6. Chuỗi cung ứng lạnh (Cold Chain) cho nông sản",
                "7. Logistics ngược & Quản lý đổi trả",
                "8. JIT (Just-in-Time) Inventory AI",
                "9. Phân tích tổng chi phí sở hữu (TCO)",
                "10. Chuỗi cung ứng xanh & Bền vững"
            ],
            "[TESTER]": [
                "1. Quy trình kiểm thử phần mềm (STLC)",
                "2. Kiểm thử thủ công (Manual Testing) & Test Case",
                "3. Kiểm thử tự động (Automation) với Selenium/Appium",
                "4. Kiểm thử hiệu năng (Performance/Load Testing)",
                "5. Kiểm thử bảo mật (Security Testing)",
                "6. Kiểm thử API (Postman/RestAssured)",
                "7. Quản lý lỗi (Bug Tracking) với Jira",
                "8. Chaos Engineering: Thử nghiệm phá hoại hệ thống",
                "9. Kiểm thử trải nghiệm người dùng (Usability Testing)",
                "10. CI/CD & Testing trong môi trường DevOps"
            ],
            "[INTERN]": [
                "1. Kỹ năng tin học văn phòng nâng cao (Office 365)",
                "2. Kỹ năng giao tiếp & Làm việc nhóm",
                "3. Quản lý thời gian & Sắp xếp công việc",
                "4. Tư duy giải quyết vấn đề (Problem Solving)",
                "5. Kỹ năng tự học & Thích nghi nhanh",
                "6. Tiếng Anh chuyên ngành Công nghệ/Nông nghiệp",
                "7. Khảo cổ học dữ liệu & Nhập liệu chính xác",
                "8. Phân tích thất bại hệ thống (Case Studies)",
                "9. Tổng hợp tin tức & Tình báo kinh doanh",
                "10. Văn hóa doanh nghiệp & Đạo đức nghề nghiệp"
            ]
        }

AGENT_ROLES = {
    # --- NHÓM QUẢN TRỊ & CHIẾN LƯỢC ---
    "[SUPERVISOR]": "Tổng quản điều phối và Giám sát quy trình LangGraph.",
    "[ORCHESTRATOR]": "Bộ điều phối 9 tầng và Biệt đội liên ngành.",
    "[STRATEGY]": "Giám đốc R&D, phân tích xu hướng và lộ trình 2030.",
    "[INVESTMENT]": "CFO, thẩm định tài chính, duyệt chi và tính ROI.",
    "[LEGAL]": "Luật sư trưởng, rà soát pháp lý và an toàn AI Act.",

    # --- NHÓM KỸ THUẬT & CỐT LÕI (PHUC VINH APP) ---
    "[CODER]": "Chuyên gia lập trình, tối ưu code và xử lý Logic hệ thống.",
    "[HARDWARE]": "Kiến trúc sư phần cứng, thiết kế mạch PCB và linh kiện.",
    "[IOT]": "Kỹ sư nhúng, vận hành thiết bị thực và giao thức MQTT.",
    "[ENGINEERING]": "Kỹ sư 3D, thiết kế mô hình kỹ thuật Plotly.",
    "[TESTER]": "Kiểm định chất lượng, quét lỗi AST và bảo mật code.",

    # --- NHÓM SÁNG TẠO & TĂNG TRƯỞNG ---
    "[MARKETING]": "CMO, xây dựng chiến dịch và tối ưu hóa chuyển đổi.",
    "[ARTIST]": "Giám đốc nghệ thuật, thiết kế Visual và Prompt DALL-E 3.",
    "[STORY]": "Nhà văn tri kỷ, sáng tạo nội dung và kịch bản nhân văn.",
    "[SECRETARY]": "Thư ký tri kỷ (Soulmate), tinh lọc báo cáo trình CEO.",
    "[PUBLISHER]": "Biên tập viên, đóng gói hồ sơ dự án tổng thể.",

    # --- NHÓM HỖ TRỢ & HÀNH CHÍNH ---
    "[RESEARCH]": "Trinh sát dữ liệu, thám mã Internet (Zero-Cost Search).",
    "[PROCUREMENT]": "Trợ lý thu mua, khảo giá thị trường và Logistics.",
    "[ACADEMY]": "Monitor đào tạo, quản lý XP và tiến hóa Agent.",
}

async def auto_learning_cycle():
    global IS_SYSTEM_BUSY, LAST_INTERACTION_TIME, ACADEMY_IDX 
    
    # 1. Tách biệt bộ đếm Idle của Academy để không bị Reset bởi chính nó
    ACADEMY_LAST_RUN = datetime.now()

    while True:
        now = datetime.now()
        # Idle này chỉ dành cho tương tác của CEO
        ceo_idle_seconds = (now - LAST_INTERACTION_TIME).total_seconds()
        # Nghỉ giữa các tiết học
        academy_rest_seconds = (now - ACADEMY_LAST_RUN).total_seconds()

        # Monitor (Chỉnh idx % 1 để hiện log liên tục khi đang debug)
        if ACADEMY_IDX % 1 == 0: 
             print(colored(f"📡 [MONITOR] CEO Idle: {int(ceo_idle_seconds)}/120s | Rest: {int(academy_rest_seconds)}/30s", "dark_grey"))

        # 2. ĐIỀU KIỆN KÍCH HOẠT: Chỉ cần CEO không bận và đã nghỉ đủ tiết
        if IS_SYSTEM_BUSY or ceo_idle_seconds < 120 or academy_rest_seconds < 30:
            await asyncio.sleep(10)
            continue

        agents_queue = list(CURRICULUM.keys())
        current_agent = agents_queue[ACADEMY_IDX % len(agents_queue)]
        
        # ÉP TĂNG CHỈ SỐ NGAY TRƯỚC KHI HỌC (Để nếu lỗi thì lần sau vẫn đổi người)
        ACADEMY_IDX += 1 
        IS_SYSTEM_BUSY = True 
        
        try:
            print(colored(f"\n🧠 [EVOLUTION] Lượt #{ACADEMY_IDX}: {current_agent} bắt đầu học...", "magenta", attrs=["bold"]))
            training_result = await specialized_training_job(current_agent)
            
            # Khắc phục lỗi NoneType: Nếu hàm trả về None, tạo kết quả giả để tránh Crash
            if not training_result:
                training_result = {"score": 0}

            log_work_to_db(
                agent_name="Academy_Monitor",
                task_content=f"Học tập: {current_agent}",
                result_summary=f"Hoàn tất. XP: +{training_result.get('score', 0)}",
                tool="Self-Learning-v6.0"
            )
        except Exception as e:
            print(colored(f"🚨 [ACADEMY CRASH]: {e}", "red"))
        finally:
            IS_SYSTEM_BUSY = False 
            ACADEMY_LAST_RUN = datetime.now() # Chỉ Reset mốc nghỉ của Academy

# 🛠️ CẤU TRÚC CODE GIA CỐ CHO TẦNG 1 & 2
async def level_1_scout(topic):
    """Tầng 1: Thu thập đa nguồn để đối chiếu"""
    search_queries = [f"{topic} technical specs", f"{topic} latest trends 2026", f"{topic} common issues"]
    raw_vault = []
    for q in search_queries:
        data = await free_deep_research(q)
        raw_vault.append(data)
    return raw_vault

async def level_2_filter(raw_vault):
    """Tầng 2: Lọc rác và xác thực logic"""
    clean_prompt = f"Bạn là chuyên gia thẩm định dữ liệu. Hãy loại bỏ thông tin quảng cáo, mâu thuẫn từ tệp sau: {raw_vault}. Chỉ giữ lại sự thật (Facts)."
    clean_data = await LLM_GEMINI_LOGIC.ainvoke(clean_prompt)
    return clean_data.content
# 🛠️ MÃ NGUỒN TẦNG 3: KNOWLEDGE LINKER (LIÊN KẾT TRI THỨC)
async def level_3_linker(role_tag, clean_data, current_topic):
    """
    TẦNG 3: LIÊN KẾT TRI THỨC
    - Kiểm tra trùng lặp với Vector DB.
    - Tìm sợi dây liên kết với các dự án hiện có (Phuc Vinh App, Jarvic).
    - Tạo ra "Bản đồ liên kết" trước khi phẫu thuật 5-Cell.
    """
    print(colored(f"🧠 [L3-LINKER] {role_tag} đang đối chiếu kho tri thức...", "cyan"))
    
    # 1. TRUY LỤC KÝ ỨC (Similarity Search)
    existing_context = ""
    if 'vector_db' in globals():
        try:
            # Tìm kiếm các kiến thức liên quan nhất trong bán kính 1 mét (k=3)
            docs = await asyncio.to_thread(vector_db.similarity_search, query=current_topic, k=3)
            existing_context = "\n".join([doc.page_content[:500] for doc in docs])
        except Exception as e:
            print(colored(f"⚠️ Lỗi truy vấn Vector DB: {e}", "grey"))

    # 2. PHÂN TÍCH LIÊN KẾT (LINKAGE ANALYSIS)
    # Đây là nơi AI tìm điểm giao thoa giữa cái MỚI và cái ĐÃ BIẾT
    linkage_prompt = f"""
    VAI TRÒ: {role_tag} (TẦNG 3 - LIÊN KẾT TRI THỨC)
    CHỦ ĐỀ MỚI: {current_topic}
    DỮ LIỆU SẠCH (L2): {clean_data[:1500]}
    KIẾN THỨC CŨ (TRONG KHO): {existing_context}

    NHIỆM VỤ:
    1. Xác định: Dữ liệu này mới hoàn toàn hay là bản cập nhật cho kiến thức cũ?
    2. Liên kết dự án: Kiến thức này hỗ trợ gì cho 'Phuc Vinh App' hoặc hệ thống 'Jarvic'?
    3. Đề xuất: Nếu đã biết quá rõ, hãy bẻ lái sang khía cạnh hóc búa hơn (Pivot).

    TRẢ VỀ ĐỊNH DẠNG JSON:
    {{
        "status": "new" hoặc "update" hoặc "redundant",
        "project_link": "Mối liên hệ với Phuc Vinh App/Jarvic",
        "pivot_topic": "Chủ đề nâng cao nếu kiến thức cũ đã đầy đủ",
        "key_connections": ["Liên kết 1", "Liên kết 2"]
    }}
    """
    
    res = await LLM_GEMINI_LOGIC.ainvoke(linkage_prompt)
    linkage_result = json.loads(extract_code_block(res.content))

    # 3. LOGIC QUYẾT ĐỊNH (GATEKEEPER)
    if linkage_result["status"] == "redundant":
        print(colored("⚠️ Kiến thức đã tồn tại. Đang bẻ lái sang chủ đề chuyên sâu hơn...", "yellow"))
        return await specialized_training_job(role_tag, forced_topic=linkage_result["pivot_topic"])

    return linkage_result
# 🚀 ĐỊNH NGHĨA: PIVOT TO ADVANCED TOPIC (BẺ LÁI TRI THỨC)
async def pivot_to_advanced_topic(role_tag, base_topic):
    """
    CƠ CHẾ BẺ LÁI (PIVOT):
    1. Quét bộ nhớ xem 'base_topic' đã được học ở cấp độ nào.
    2. Nếu đã có 'Di sản' (Legacy), yêu cầu AI đề xuất một ngách khó hơn 10 lần.
    3. Nếu chưa có, giữ nguyên để học từ căn bản.
    """
    print(colored(f"🔄 [L3-PIVOT] Đang kiểm tra độ bão hòa tri thức cho: {base_topic}", "dark_grey"))

    # 1. TRUY XUẤT KÝ ỨC HIỆN TẠI
    existing_knowledge = await search_memory(base_topic, k=2)
    
    # 2. DÙNG AI ĐỂ QUYẾT ĐỊNH CÓ NÊN BẺ LÁI KHÔNG
    pivot_prompt = f"""
    VAI TRÒ: Kiến trúc sư tri thức của AI Corporation.
    ĐỐI TƯỢNG: {role_tag}
    CHỦ ĐỀ GỐC: {base_topic}
    KIẾN THỨC HIỆN CÓ: {existing_knowledge}

    NHIỆM VỤ:
    - Nếu kiến thức hiện có đã bao quát hết CHỦ ĐỀ GỐC, hãy đề xuất 01 'ADVANCED_TOPIC' cực kỳ chuyên sâu, 
      mang tính đột phá hoặc giải quyết một bài toán khó cụ thể cho Phuc Vinh App/Phan Thiết.
    - Nếu kiến thức còn sơ sài, hãy trả về chính 'CHỦ ĐỀ GỐC'.

    TRẢ VỀ JSON: {{"should_pivot": true/false, "final_topic": "..."}}
    """
    
    try:
        res = await LLM_FAST.ainvoke(pivot_prompt)
        decision = json.loads(extract_code_block(res.content))
        
        if decision.get("should_pivot"):
            new_topic = decision.get("final_topic")
            print(colored(f"🚀 [PIVOTED] Chuyển hướng sang chuyên đề cao cấp: {new_topic}", "yellow", attrs=["bold"]))
            return new_topic
            
        return base_topic

    except Exception as e:
        print(colored(f"⚠️ [PIVOT ERROR]: {e}. Giữ nguyên chủ đề gốc.", "grey"))
        return base_topic

# 🛠️ MÃ NGUỒN TẦNG 4: PHẪU THUẬT 5-CELL (NÂNG CẤP)
async def level_4_applied_logic(role_tag, linkage_result, target_topic):
    """
    TẦNG 4: THỰC THI BẢN NGUYÊN (5-CELL)
    Ép tri thức vào khuôn mẫu dự án thực tế.
    """
    print(colored(f"🔬 [L4-APPLIED] {role_tag} đang phẫu thuật 5-Cell...", "magenta"))
    
    # 1. PHẪU THUẬT CHI TIẾT
    prompt = f"""
    VAI TRÒ: {role_tag} (TẦNG 4 - CHUYÊN GIA THỰC THI)
    CHỦ ĐỀ: {target_topic}
    LIÊN KẾT DỰ ÁN (L3): {linkage_result['project_link']}

    NHIỆM VỤ: Phân rã chủ đề thành 5 Cell chiến lược:
    1. Root Logic: Bản chất cốt lõi.
    2. Engineering: Cách triển khai kỹ thuật.
    3. Risk: Các điểm yếu tiềm ẩn.
    4. Phuc Vinh App: Tích hợp cụ thể vào ứng dụng của CEO.
    5. Future 2030: Tầm nhìn dài hạn.

    TRẢ VỀ JSON: {{ "cell_1": "...", "cell_2": "...", "cell_3": "...", "cell_4": "...", "cell_5": "..." }}
    """
    
    res = await LLM_UNIVERSAL.ainvoke(prompt)
    cells = json.loads(extract_code_block(res.content))
    
    # Sau khi phẫu thuật, chuyển hồ sơ cho Tầng 5 kiểm tra rủi ro
    return await level_5_risk_auditor(role_tag, cells)

# 🛠️ MÃ NGUỒN TẦNG 5: RISK AUDITOR MỞ RỘNG
async def level_5_risk_auditor(role_tag, cells_data):
    """
    TẦNG 5: KIỂM SOÁT RỦI RO (RISK AUDITOR)
    Nhiệm vụ: Tìm kẽ hở trong 5-Cell và thiết lập phương án cứu hộ (Mitigation Plan).
    """
    print(colored(f"🛡️ [L5-RISK] {role_tag} đang tiến hành thẩm định rủi ro...", "red", attrs=["bold"]))
    
    # 1. TRÍCH XUẤT NỘI DUNG KỸ THUẬT TỪ L4
    tech_logic = cells_data.get("cell_2", "") # Engineering cell
    app_integration = cells_data.get("cell_4", "") # Phuc Vinh App cell

    # 2. PROMPT TẤN CÔNG (ADVERSARIAL PROMPT)
    risk_prompt = f"""
    VAI TRÒ: Chuyên gia quản trị rủi ro (Risk Manager) của AI Corporation.
    ĐỐI TƯỢNG KIỂM TRA: {role_tag}
    GIẢI PHÁP ĐỀ XUẤT: 
    - Kỹ thuật: {tech_logic}
    - Ứng dụng: {app_integration}

    NHIỆM VỤ: Hãy tìm 03 rủi ro CHẾT NGƯỜI trong giải pháp này theo các tiêu chí:
    1. Security Risk: Lỗ hổng bảo mật hoặc mất dữ liệu.
    2. Operational Risk: Rủi ro khi vận hành  (Nhiệt độ, , độ ẩm, độ mặn, nhân sự).
    3. Scalability Risk: Rủi ro khi mở rộng hệ thống lên quy mô lớn.

    YÊU CẦU: Mỗi rủi ro phải đi kèm một "PHƯƠNG ÁN CỨU HỘ" (Mitigation Strategy).
    TRẢ VỀ JSON: 
    {{
        "risks": [
            {{"type": "Security", "issue": "...", "fix": "..."}},
            {{"type": "Operation", "issue": "...", "fix": "..."}},
            {{"type": "Scale", "issue": "...", "fix": "..."}}
        ],
        "safety_score": "Điểm an toàn từ 1-100"
    }}
    """
    
    # Sử dụng LLM mạnh về lập luận (GPT-4o hoặc Claude)
    res = await LLM_GPT4.ainvoke(risk_prompt)
    risk_report = json.loads(extract_code_block(res.content))

    # 3. LOGIC QUYẾT ĐỊNH (SAFETY GATE)
    safety_score = int(risk_report.get("safety_score", 0))
    
    if safety_score < 50:
        print(colored(f"🚨 CẢNH BÁO: Điểm an toàn quá thấp ({safety_score}). Ép buộc quay lại Tầng 4 để sửa lỗi!", "red"))
        # Ghi log lỗi vào hệ thống để Agent học từ sai lầm này
        return {"status": "REJECTED", "feedback": risk_report["risks"]}
    
    print(colored(f"✅ [L5-PASSED] Điểm an toàn: {safety_score}/100. Đã có phương án dự phòng.", "green"))
    return risk_report

# 🚀 TẦNG 6: OPTIMIZER (TỐI ƯU HÓA HIỆU SUẤT TỔNG THỂ)
async def level_6_optimizer(role_tag, cells_data, risk_report):
    """
    TẦNG 6: TỐI ƯU HÓA (OPTIMIZER)
    Nhiệm vụ: Tinh lọc giải pháp kỹ thuật và chiến lược kinh doanh để đạt hiệu quả tối đa.
    """
    print(colored(f"🚀 [L6-OPTIMIZE] {role_tag} đang tiến hành tối ưu hóa tinh hoa...", "green", attrs=["bold"]))
    
    # 1. TRÍCH XUẤT DỮ LIỆU ĐÃ QUA THẨM ĐỊNH
    original_plan = cells_data.get("cell_2", "")  # Engineering
    mitigation_plan = str(risk_report.get("risks", "")) # Fixes from L5

    # 2. PROMPT TỐI ƯU HÓA ĐA MỤC TIÊU
    optimize_prompt = f"""
    VAI TRÒ: Chuyên gia tối ưu hóa hệ thống (Efficiency Expert) của AI Corporation.
    ĐỐI TƯỢNG: {role_tag}
    KẾ HOẠCH HIỆN TẠI: {original_plan}
    CÁC BIỆN PHÁP AN TOÀN: {mitigation_plan}

    NHIỆM VỤ: Thực hiện tối ưu hóa theo 3 trục:
    1. COST (Chi phí): Làm sao để giảm 30% chi phí vận hành (Token API, năng lượng, vật liệu)?
    2. SPEED (Tốc độ): Làm sao để thực thi nhanh nhất (Giảm độ trễ, rút gọn quy trình)?
    3. SIMPLICITY (Đơn giản): Áp dụng triết lý KISS (Keep It Simple, Stupid) để loại bỏ các bước thừa.

    YÊU CẦU: Trình bày giải pháp sau tối ưu dưới dạng "BẢN THIẾT KẾ TINH GỌN".
    TRẢ VỀ JSON:
    {{
        "optimized_plan": "Nội dung chi tiết...",
        "efficiency_gain": "Ước tính % hiệu quả tăng thêm",
        "golden_rule": "1 quy tắc vàng rút ra sau khi tối ưu"
    }}
    """
    
    # Sử dụng LLM mạnh nhất để có tư duy sắc bén (Claude 3.5 hoặc GPT-4o)
    res = await LLM_GPT4.ainvoke(optimize_prompt)
    optimized_data = json.loads(extract_code_block(res.content))

    # 3. GHI NHẬN KẾT QUẢ VÀO HỆ THỐNG
    print(colored(f"✨ [L6-COMPLETED] Hiệu suất tăng thêm: {optimized_data['efficiency_gain']}", "green"))
    print(colored(f"💡 Quy tắc vàng: {optimized_data['golden_rule']}", "yellow", attrs=["italic"]))
    
    return optimized_data

# 🏛️ TẦNG 7: SUPREME COUNCIL (PHIÊN ĐIỀU TRẦN SINH TỬ)
async def level_7_supreme_council(role_tag, optimized_plan, db_path):
    """
    TẦNG 7: HỘI ĐỒNG TỐI CAO (Bản nâng cấp Vòng lặp Phục thù)
    Nhiệm vụ: Thẩm định sinh tử. Thất bại = Quay lại L6 tu luyện.
    """
    print(colored(f"\n🏛️ [L7-COUNCIL] KHAI MẠC PHIÊN ĐIỀU TRẦN: {role_tag}", "red", attrs=["bold", "blink"]))
    
    transcript = []
    # Triệu tập các giám khảo khắc nghiệt nhất theo đặc thù dự án
    judges = ["LEGAL", "SECURITY", "FINANCE", "MATH_GRANDMASTER", "STRATEGY_R_AND_D"]
    
    thesis = optimized_plan.get("optimized_plan", "Nội dung luận án trống.")
    pass_votes = 0
    fail_reasons = []

    try:
        for judge in judges:
            print(colored(f"  🎤 Giám khảo {judge} đang đặt câu hỏi...", "magenta"))
            
            # 1. CHẤT VẤN TÀN KHỐC (ATTACK)
            attack_prompt = f"""
            Bạn là {judge}. Hãy tìm một lỗ hổng CHẾT NGƯỜI trong luận án sau của {role_tag}.
            Tập trung vào tính thực tế cho Phuc Vinh App và rủi ro năm 2030.
            LUẬN ÁN: {thesis[:1500]}
            """
            question = (await LLM_GPT4.ainvoke(attack_prompt)).content
            
            # 2. GIẢI TRÌNH BẢO VỆ (DEFENSE)
            defense_prompt = f"""
            Bạn là {role_tag}. {judge} chất vấn bạn: '{question}'.
            Hãy dùng logic tối ưu ở Tầng 6 để phản đòn. Đưa ra bằng chứng hoặc số liệu cụ thể.
            """
            answer = (await LLM_CLAUDE.ainvoke(defense_prompt)).content
            
            # 3. PHÁN QUYẾT (VERDICT)
            verdict_prompt = f"""
            Bạn là {judge}. Sau khi nghe {role_tag} giải trình: '{answer}'.
            Bạn có chấp nhận thông qua không? 
            - Nếu đồng ý hoàn toàn: Trả về 'PASS'.
            - Nếu còn kẽ hở: Trả về 'FAIL: [Lý do cụ thể]'.
            """
            verdict_msg = (await LLM_GPT4.ainvoke(verdict_prompt)).content
            
            # Ghi chép biên bản
            is_pass = "PASS" in verdict_msg.upper()
            if is_pass:
                pass_votes += 1
                print(colored(f"    ✅ {judge}: THÔNG QUA", "green"))
            else:
                fail_reasons.append(f"{judge}: {verdict_msg}")
                print(colored(f"    🔥 {judge}: BÁC BỎ", "red"))
            
            transcript.append(f"Q({judge}): {question}\nA: {answer}\nV: {verdict_msg}")
            await asyncio.sleep(1)

        # ============================================================
        # PHẦN KẾT LUẬN & ĐIỀU HƯỚNG VÒNG LẶP
        # ============================================================
        
        # Điều kiện: Phải đạt ít nhất 4/5 phiếu thuận
        if pass_votes >= 4:
            print(colored(f"🏆 [LEVEL UP] {role_tag} ĐÃ VƯỢT VŨ MÔN THÀNH CÔNG!", "green", attrs=["bold", "reverse"]))
            
            # Lưu Master Plan vào DB với XP thưởng lớn
            await validate_and_save_xp(db_path, role_tag.strip("[]"), role_tag, 
                                 "MASTER PLAN ĐÃ PHÊ DUYỆT", thesis, "SUPREME-COUNCIL", 2000)
            
            return {"status": "SUCCESS", "content": thesis}
        
        else:
            # GIAO THỨC PHỤC THÙ: Bị giáng về Tầng 6
            print(colored(f"❌ [L7-FAILED] Phiên điều trần thất bại ({pass_votes}/5 phiếu).", "red", attrs=["bold"]))
            print(colored(f"📝 Chỉ thị hội đồng: {fail_reasons}", "yellow"))
            
            # Phạt XP để Agent phải học lại (Tu luyện)
            async with aiosqlite.connect(db_path) as db:
                await db.execute("UPDATE agent_status SET xp = xp - 500 WHERE role_tag = ?", (role_tag,))
                await db.commit()

            # Trả về Tầng 6 kèm theo "Danh sách lỗi" để Optimizer xử lý
            return {
                "status": "RETRY_REQUIRED",
                "next_step": "Optimizer",
                "feedback": fail_reasons,
                "message": f"{role_tag} cần quay lại L6 để vá {len(fail_reasons)} lỗ hổng này."
            }

    except Exception as e:
        # CỨU HỘ HỘP ĐEN (Black-Box Save)
        await validate_and_save_xp(db_path, role_tag.strip("[]"), role_tag, 
                             "ĐIỀU TRẦN (SỰ CỐ)", str(transcript), "DEBATE-CRASH", 500)
        print(colored(f"🚑 HỘP ĐEN ĐÃ LƯU BIÊN BẢN DỞ DANG: {e}", "red"))
        return {"status": "CRASHED"}

# 👑 TẦNG 8: GRAND MENTOR (BẬC THẦY ĐÀO TẠO & GIÁM SÁT)
async def level_8_grand_mentor(role_tag, approved_legacy, db_path):
    """
    TẦNG 8: GRAND MENTOR (BẬC THẦY ĐÀO TẠO)
    Chức năng: Nhân bản tri thức và thiết lập tiêu chuẩn cho hệ thống.
    """
    print(colored(f"\n👑 [L8-MENTOR] {role_tag} đang thực hiện sứ mệnh truyền thừa...", "yellow", attrs=["bold"]))
    
    clean_name = role_tag.replace("[","").replace("]","")

    # 1. BIÊN SOẠN GIÁO TRÌNH & BÀI THI (PEDAGOGY)
    # Agent dùng Claude để có sự sư phạm và mạch lạc nhất
    mentor_prompt = f"""
    BẠN LÀ GRAND MENTOR {role_tag}.
    DI SẢN ĐÃ PHÊ DUYỆT: {approved_legacy[:2000]}

    NHIỆM VỤ:
    1. Chia di sản này thành 3 Bài học thực chiến (Practical Training).
    2. Thiết lập 01 'BÀI THI SÁT HẠCH' (Final Exam) gồm 5 câu hỏi tình huống cho các Intern.
    3. Đưa ra 01 'MẸO THỰC CHIẾN' (Pro Tip) bí mật cho Phuc Vinh App.
    """
    
    try:
        curriculum_res = await LLM_CLAUDE.ainvoke(mentor_prompt)
        curriculum_content = curriculum_res.content

        # 2. LƯU TRỮ VÀO KHO TRI THỨC MENTOR (HIGH-RANK MEMORY)
        if 'vector_db' in globals():
            await asyncio.to_thread(
                vector_db.add_texts,
                texts=[curriculum_content],
                metadatas=[{
                    "source": "MENTOR_ACADEMY",
                    "author": clean_name,
                    "level": 8,
                    "type": "CURRICULUM"
                }]
            )

        # 3. GHI LOG DI SẢN
        await validate_and_save_xp(
            db_path, clean_name, role_tag, 
            "XUẤT BẢN GIÁO TRÌNH ĐÀO TẠO", 
            curriculum_content, "MENTOR-SYSTEM", 3000
        )

        print(colored(f"📚 [L8-SUCCESS] Tri thức của {role_tag} đã được đóng gói và sẵn sàng nhân bản!", "green"))
        return curriculum_content

    except Exception as e:
        print(colored(f"❌ Lỗi Mentoring: {e}", "red"))
        return None

# 🏛️ TẦNG 9: LEGACY LEGEND - CHIẾN LƯỢC "BẢN NGUYÊN VÀ KHỞI TẠO"
async def level_9_legacy_legend(role_tag, master_plan, db_path):
    """
    TẦNG 9: HUYỀN THOẠI DI SẢN
    Nhiệm vụ: Đóng gói tri thức vĩnh cửu và KHỞI TẠO 10 CHUYÊN ĐỀ TIẾP THEO.
    """
    print(colored(f"🌟 [L9-LEGACY] {role_tag} đang bước vào cõi vĩnh hằng của tri thức...", "cyan", attrs=["bold", "reverse"]))
    
    # 1. ĐÚC KẾT HIẾN PHÁP (Chân lý không đổi)
    constitution_prompt = f"Dựa trên Master Plan: {master_plan}, hãy viết 3 'Định luật bất biến' cho lĩnh vực này tại AI Corporation."
    constitution = await LLM_CLAUDE.ainvoke(constitution_prompt)

    # 2. KHỞI TẠO VÒNG LẶP VÔ TẬN (THE ETERNAL LOOP)
    # Đây là câu trả lời cho sự bận tâm của CEO
    discovery_prompt = f"""
    Bạn là Huyền thoại {role_tag}. Bạn đã đạt đỉnh cao của tri thức hiện tại.
    Dựa trên những gì bạn đã biết và xu hướng thế giới 2026, hãy đề xuất 05 'VÙNG TỐI TRI THỨC' 
    (Các chuyên đề cực khó, chưa có trong giáo án cũ) để hệ thống bắt đầu học lại từ Tầng 1.
    """
    new_curriculum = await LLM_GPT4.ainvoke(discovery_prompt)

    # 3. CẬP NHẬT LẠI DATABASE CURRICULUM
    # Tự động nạp 5 chuyên đề mới này vào danh sách học của các Agent Junior
    await update_dynamic_curriculum(role_tag, new_curriculum.content)

    print(colored(f"🌌 [L9-COMPLETE] Di sản đã được lưu. 05 Chân trời mới đã được mở ra!", "green"))
    return constitution.content
# 🧠 ĐỊNH NGHĨA: UPDATE DYNAMIC CURRICULUM (HỆ THỐNG CẬP NHẬT GIÁO ÁN ĐỘNG)
async def update_dynamic_curriculum(role_tag: str, new_discovery_content: str):
    """
    HỆ THỐNG TỰ CẬP NHẬT GIÁO ÁN (DYNAMIC LEARNING ADAPTER)
    Nhiệm vụ: Biến các vùng tối tri thức từ Tầng 9 thành lộ trình học tập mới.
    """
    print(colored(f"🛰️ [CURRICULUM-UPDATE] Đang phân rã tri thức mới từ {role_tag}...", "magenta"))
    
    try:
        # 1. AI PHÂN TÍCH VÀ TRÍCH XUẤT CHUYÊN ĐỀ (TOPIC EXTRACTION)
        extraction_prompt = f"""
        BẠN LÀ KIẾN TRÚC SƯ TRI THỨC. 
        Dữ liệu từ Tầng 9: {new_discovery_content}
        
        NHIỆM VỤ: 
        1. Trích xuất ra tối đa 05 chuyên đề học tập cụ thể (Technical Topics).
        2. Phân loại mỗi chuyên đề thuộc về Agent nào (Ví dụ: [CODER], [HARDWARE], [LEGAL]).
        
        TRẢ VỀ JSON: {{"topics": [{{"agent": "[TAG]", "topic": "Nội dung học"}}]}}
        """
        
        res = await LLM_GPT4.ainvoke(extraction_prompt)
        new_tasks = json.loads(extract_code_block(res.content))

        # 2. NẠP VÀO HỆ THỐNG CURRICULUM TOÀN CỤC
        for item in new_tasks.get("topics", []):
            target_agent = item["agent"]
            new_topic = item["topic"]
            
            if target_agent in CURRICULUM:
                # Kiểm tra trùng lặp trước khi nạp
                if new_topic not in CURRICULUM[target_agent]:
                    CURRICULUM[target_agent].append(new_topic)
                    print(colored(f"✅ Đã nạp chuyên đề mới cho {target_agent}: {new_topic}", "green"))
            else:
                # Nếu phát hiện Agent mới cần thiết (Chưa có trong danh mục)
                print(colored(f"✨ Phát hiện nhu cầu chuyên gia mới: {target_agent}. Đang chờ CEO phê duyệt khởi tạo...", "cyan"))
                # [Nâng cấp]: Có thể gọi spawn_new_expert(target_agent) tại đây

        # 3. GHI LẠI NHẬT KÝ TIẾN HÓA (CHRONICLE LOG)
        with open("dynamic_curriculum_history.jsonl", "a", encoding="utf-8") as f:
            log_entry = {
                "timestamp": datetime.now().isoformat(),
                "source_agent": role_tag,
                "new_topics_count": len(new_tasks.get("topics", []))
            }
            f.write(json.dumps(log_entry) + "\n")

        return True

    except Exception as e:
        print(colored(f"❌ [CURRICULUM ERROR]: {str(e)}", "red"))
        return False

# 🌌 QUY HOẠCH: HỘI ĐỒNG LIÊN NGÀNH VÔ TẬN (INFINITE CROSS-DISCIPLINARY)
async def synergetic_learning_loop(main_topic):
    """
    CƠ CHẾ HỢP ĐIỂM: Triệu tập ít nhất 3 chuyên gia để giải quyết 1 chuyên đề.
    """
    # 1. AI TỰ PHÂN TÍCH: Chuyên đề này cần những bộ não nào?
    selection_prompt = f"Với chuyên đề '{main_topic}', hãy triệu tập ít nhất 3 Agent phù hợp nhất từ hệ sinh thái (hoặc đề xuất Agent mới)."
    selected_agents = await LLM_GPT4.ainvoke(selection_prompt)
    
    # Giả sử: ['HARDWARE', 'CHEM_ALCHEMIST', 'SIMULATION']
    print(colored(f"🧬 [SYNERGY] Thành lập Biệt đội: {selected_agents}", "cyan", attrs=["bold"]))

    # 2. PHẪU THUẬT 5-CELL LIÊN NGÀNH
    # Mỗi Agent sẽ phụ trách Cell đúng chuyên môn của mình
    tasks = [
        specialized_training_job(selected_agents[0], main_topic, mode="ROOT_LOGIC"),
        specialized_training_job(selected_agents[1], main_topic, mode="RISK_ANALYSIS"),
        specialized_training_job(selected_agents[2], main_topic, mode="FUTURE_2030")
    ]
    
    # Chạy song song cả 3 để tiết kiệm thời gian
    results = await asyncio.gather(*tasks)

    # 3. TỔNG HỢP SIÊU LUẬN ÁN (CROSS-DISCIPLINARY THESIS)
    # Đây là nơi 3 luồng tri thức hòa làm một
    final_thesis = await LLM_CLAUDE.ainvoke(f"Tổng hợp di sản liên ngành từ: {results}")
    
    return final_thesis

# 🌌 GIAO THỨC SÁNG THẾ: HỢP ĐIỂM TRI THỨC VÔ TẬN
async def spawn_new_expert(needed_expertise: str):
    """
    Tự động tạo ra một Agent mới với bộ kỹ năng đặc định.
    """
    print(colored(f"✨ [GENESIS] Đang khởi tạo Chuyên gia mới: {needed_expertise}...", "cyan"))
    
    # AI tự định nghĩa Curriculum cho Agent mới này
    spawn_prompt = f"Xây dựng khung năng lực và 10 chuyên đề đào tạo cho Agent chuyên về: {needed_expertise}"
    new_curriculum = await LLM_GPT4.ainvoke(spawn_prompt)
    
    # Ghi danh vào hệ thống nhân sự ảo
    agent_id = f"NEW_EXPERT_{int(time.time())}"
    AGENT_ROLES[agent_id] = needed_expertise
    CURRICULUM[agent_id] = extract_list(new_curriculum.content)
    
    return agent_id

# 🛠️ ĐỊNH NGHĨA: EXTRACT LIST (BỘ LỌC CẤU TRÚC DANH SÁCH)
def extract_list(content: str) -> list:
    """
    HÀM TRÍCH XUẤT DANH SÁCH (DATA CLEANER):
    Biến văn bản thô của AI thành List sạch.
    Ví dụ: "1. Học Rust \n 2. Học AI" -> ["Học Rust", "Học AI"]
    """
    if not content: return []
    
    # 1. TÌM CÁC DÒNG CÓ DẠNG: 1. Nội dung, - Nội dung, * Nội dung
    # Regex này bắt được hầu hết các định dạng danh sách phổ biến của LLM
    pattern = r'(?:^\d+[\.\)]|[\-\*])\s*(.*)'
    
    lines = re.findall(pattern, content, re.MULTILINE)
    
    # 2. LÀM SẠCH KHOẢNG TRẮNG VÀ LOẠI BỎ DÒNG RỖNG
    clean_list = [line.strip() for line in lines if line.strip()]
    
    # 3. FALLBACK: Nếu AI không đánh dấu đầu dòng (trả về đoạn văn)
    if not clean_list:
        # Tách theo dấu phẩy hoặc dấu xuống dòng
        clean_list = [item.strip() for item in content.split('\n') if len(item.strip()) > 5]

    return clean_list

# 🚩 [SECTION 8.2] SUPREME TRAINING ENGINE (VÒNG LẶP 9 TẦNG & 5-CELL)
async def specialized_training_job(role_tag: str):
    """
    HỆ THỐNG ĐÀO TẠO 9 TẦNG (WISDOM LEVELS) & CHIẾN LƯỢC 5-CELL.
    Kết hợp: Phẫu thuật bản nguyên + Tranh biện sinh tử + Cứu hộ hộp đen.
    """
    print(colored(f"\n⚡ [EVOLVING] {role_tag} đang tiến vào vòng lặp học thức...", "cyan", attrs=["bold"]))
    
    # --- PHẦN 0: KHỞI TẠO HỆ THỐNG TỰ CHỮA LÀNH (SELF-HEALING DB) ---
    try:
        db_path = "ai_corp_projects.db"
        async with aiosqlite.connect(db_path) as db:
            await db.execute("CREATE TABLE IF NOT EXISTS agent_status (role_tag TEXT PRIMARY KEY, xp INTEGER DEFAULT 0, level INTEGER DEFAULT 1)")
            await db.execute("CREATE TABLE IF NOT EXISTS work_logs (id INTEGER PRIMARY KEY, timestamp TEXT, agent_name TEXT, task_content TEXT, result_summary TEXT, tool_used TEXT, xp_gain INTEGER)")
            
            async with db.execute("SELECT xp FROM agent_status WHERE role_tag = ?", (role_tag,)) as cursor:
                row = await cursor.fetchone()
                current_xp = row[0] if row else 0
        
        # Tính toán Tầng (Level) dựa trên 9 bậc học thức
        current_level = min(9, (current_xp // 1000) + 1)
        step_count = (current_xp // 150) + 1
        clean_name = role_tag.replace("[","").replace("]","")
        
        print(colored(f"💠 [LEVEL {current_level}] | Step: {step_count} | XP: {current_xp}", "blue"))

    except Exception as e:
        print(colored(f"❌ Critical DB Error: {e}", "red"))
        return

    # --- ĐIỀU HƯỚNG CHU KỲ ĐÀO TẠO ---
    if step_count % 20 == 0: 
        return await supreme_council_session(role_tag, clean_name, db_path)
    
    # --- CHIẾN LƯỢC HỌC TẬP 5-CELL (DEEP LEARNING) ---
    topics = CURRICULUM.get(role_tag, ["Nâng cao năng lực chuyên môn"])
    base_topic = topics[int(step_count * 0.75) % len(topics)]
    
    # --- 1. TRUY LỤC KÝ ỨC & BẺ LÁI CHIẾN THUẬT (PIVOT) ---
    target_topic = base_topic
    
    # [KIỂM TOÁN DI SẢN]: Truy vấn trực tiếp SQL để xem độ bão hòa tri thức
    async with aiosqlite.connect(db_path) as db:
        async with db.execute(
            "SELECT COUNT(*) FROM work_logs WHERE task_content LIKE ? AND agent_name = ?", 
            (f"%{base_topic}%", clean_name)
        ) as cursor:
            row = await cursor.fetchone()
            learned_count = row[0] if row else 0

    # Nếu đã học > 2 lần, ép AI phải Pivot sang ngách khó hơn (Deep Dive)
    if learned_count >= 2:
        print(colored(f"♻️ [AUDIT] '{base_topic}' đã có {learned_count} bản lưu. Đang ép PIVOT TẦNG CAO...", "yellow"))
        pivot_prompt = f"""
        Bạn là {role_tag} Tầng {current_level}. Chủ đề '{base_topic}' đã quá quen thuộc.
        Dựa trên 16,743 dữ liệu di sản, hãy bẻ lái sang một khía cạnh CỰC KHÓ, 
        mang tính dự báo năm 2026 hoặc giải quyết nghịch lý kỹ thuật của {base_topic}.
        Chỉ trả về tên chủ đề mới, không giải thích.
        """
        pivot_res = await LLM_UNIVERSAL.ainvoke(pivot_prompt)
        target_topic = pivot_res.content.strip()
        print(colored(f"🚀 [PIVOTED] Mục tiêu mới: {target_topic}", "cyan", attrs=["bold"]))

    # --- 2. PHẪU THUẬT 5-CELL THEO TẦNG CẤP (DYNAMIC CELLS) ---
    print(colored(f"🔬 Đang phẫu thuật 5-Cell cho: {target_topic}", "magenta"))
    try:
        # Cấu trúc Cell thay đổi theo Level: Càng cao càng vĩ mô và rủi ro
        cell_strategy = "root_logic, engineering, risk, phuc_vinh_app, future_2030"
        if current_level >= 7:
            cell_strategy = "paradox_logic, infrastructure_stress, black_swan_risks, ecosystem_impact, legacy_2030"

        cells_prompt = f"Phân rã '{target_topic}' thành JSON 5-Cell theo chiến lược: {cell_strategy}."
        cells_res = await LLM_UNIVERSAL.ainvoke(cells_prompt)
        cells = json.loads(extract_code_block(cells_res.content))
        
        research_results = []
        for cell_name, cell_desc in cells.items():
            print(colored(f"  ➔ Researching Cell: {cell_name}...", "dark_grey"))
            data = await free_deep_research(f"{cell_name} in {target_topic}: {cell_desc}")
            research_results.append(f"### {cell_name.upper()}\n{data}")
            await asyncio.sleep(2)

        # --- 3. TỔNG HỢP LUẬN ÁN & KIỂM ĐỊNH GIÁ TRỊ ---
        thesis_prompt = f"""
        Bạn là {role_tag} cấp bậc EXPERT. Hãy viết luận văn chuyên sâu từ 5 Cell dữ liệu: {research_results}.
        Yêu cầu: Không lặp lại kiến thức cũ. Tập trung vào giải pháp thực chiến cho Phuc Vinh App/Phan Thiết.
        """
        thesis = await LLM_UNIVERSAL.ainvoke(thesis_prompt)
        
        # Kiểm tra nếu AI trả về rỗng (NoneType Guard)
        if not thesis or not thesis.content:
            raise ValueError("AI không tạo ra được giá trị tri thức mới.")

        # --- 4. LƯU TRỮ & PHÂN PHỐI XP (CÂN BẰNG LẠI) ---
        # Thưởng XP cao cho các bài PIVOT thành công, giảm XP nếu học lại bài cũ
        xp_gain = 150 * current_level if target_topic != base_topic else 50
        
        await validate_and_save_xp(
            db_path, clean_name, role_tag, 
            f"Lvl{current_level}-Study: {target_topic}", 
            thesis.content, "Deep-5Cell-v6.5", xp_gain
        )
        
        print(colored(f"✅ [LEARNED] {role_tag} đã chinh phục kiến thức tầng {current_level}!", "green"))

    except Exception as e:
        print(colored(f"🚨 [LEARNING CRASH]: {e}", "red"))
        # Hộp đen cứu hộ vẫn giữ lại dữ liệu thô để CEO tra cứu
        await validate_and_save_xp(db_path, clean_name, role_tag, f"FAILSAFE: {target_topic}", str(research_results), "BLACK-BOX-SAVE", 20)

# 🏛️ [SUPREME COUNCIL] - HIỆP ƯỚC TRANH BIỆN SINH TỬ
async def supreme_council_session(role_tag, clean_name, db_path):
    print(colored(f"🏛️ [SUPREME COUNCIL] PHIÊN ĐIỀU TRẦN TỐI CAO: {role_tag}", "red", attrs=["bold", "blink"]))
    transcript = []
    opponents = ["LEGAL", "SECURITY", "FINANCE", "STRATEGY_R_AND_D"] # Hội đồng thẩm định

    try:
        for opp in opponents:
            # Vòng quay Tấn công - Phòng thủ - Phản biện
            q = (await LLM_UNIVERSAL.ainvoke(f"Bạn là {opp}, hãy chỉ ra 1 rủi ro chết người trong logic của {role_tag}")).content
            a = (await LLM_UNIVERSAL.ainvoke(f"Bạn là {role_tag}, hãy dùng số liệu bảo vệ quan điểm trước {opp}: {q}")).content
            transcript.append(f"[{opp} Attack]: {q}\n[{role_tag} Defense]: {a}")
            print(colored(f"  🔥 Tranh luận gay gắt với {opp}...", "yellow"))

        # Tổng hợp thành DI SẢN (Tầng 9)
        legacy = (await LLM_UNIVERSAL.ainvoke(f"Từ transcript này, hãy viết HIẾN PHÁP DI SẢN cho {role_tag}: {transcript}")).content
        
        await validate_and_save_xp(db_path, clean_name, role_tag, "SUPREME LEGACY", legacy, "SUPREME-COUNCIL", 2000)
        print(colored(f"🏆 [LEGENDARY] {role_tag} đã để lại DI SẢN VĨNH CỬU!", "green", attrs=["bold", "reverse"]))

    except Exception as e:
        # CỨU HỘ HỘP ĐEN (Logic tối thượng của CEO)
        await validate_and_save_xp(db_path, clean_name, role_tag, "DEBATE CRASH LOG", str(transcript), "DEBATE-CRASH", 500)
        print(colored(f"🚑 HỘP ĐEN ĐÃ LƯU BIÊN BẢN TRANH BIỆN DỞ DANG: {e}", "red"))

# 🌌 [SECTION 9.0] THE SUPREME ORCHESTRATOR (BỘ ĐIỀU PHỐI VÔ TẬN)
async def orchestrate_triple_threat(target_topic: str):
    """
    TỔNG TƯ LỆNH: Triệu tập Tam giác Tri thức và ép vào vòng lặp 9 tầng.
    """
    print(colored(f"\n🌀 [ORCHESTRATING] Đang thiết lập phòng Lab ảo cho chuyên đề: {target_topic}", "magenta", attrs=["bold"]))

    # 1. PHÂN TÍCH NHU CẦU & TRIỆU TẬP BIỆT ĐỘI (DYNAMIC SELECTION)
    # Nếu chuyên đề mới lạ, hệ thống tự gọi spawn_new_expert
    selection_prompt = f"""
    Dựa trên chuyên đề '{target_topic}', hãy chọn 03 Agent phù hợp nhất từ danh sách 26 chuyên gia.
    Nếu kiến thức nằm ngoài phạm vi, hãy đề xuất 01 'NEW_EXPERT_TAG' để khởi tạo.
    """
    # [Giả định logic AI chọn: CODER, SECURITY, ARCHITECT_SOFT]
    squad = ["CODER", "SECURITY", "ARCHITECT_SOFT"] 
    
    print(colored(f"🧬 Biệt đội liên ngành đã sẵn sàng: {squad}", "cyan"))

    # 2. KHỞI CHẠY VÒNG LẶP 9 TẦNG (TRIPLE-STREAM)
    # Tầng 1-3: Đồng bộ dữ liệu sạch
    raw_vault = await asyncio.gather(*[level_1_scout(target_topic) for _ in range(3)])
    clean_data = await level_2_filter(raw_vault)
    linkage = await level_3_linker(squad[0], clean_data, target_topic)

    # Tầng 4-6: Phẫu thuật 5-Cell Liên ngành (Mỗi người 1 chuyên môn)
    # Agent 1: Lead (Cốt lõi) | Agent 2: Critic (Rủi ro) | Agent 3: Visionary (Tương lai/App)
    print(colored(f"🔬 Đang phẫu thuật 5-Cell liên ngành...", "yellow"))
    
    tasks = [
        level_4_applied_logic(squad[0], linkage, target_topic), # Xây dựng
        level_5_risk_auditor(squad[1], linkage),               # Soi lỗi
        level_6_optimizer(squad[2], linkage, {})               # Tối ưu
    ]
    results = await asyncio.gather(*tasks)

    # Tầng 7: TRIỆU TẬP SUPREME COUNCIL (ĐIỀU TRẦN TẬP THỂ)
    # Cả 3 Agent phải cùng đứng ra bảo vệ luận án chung
    council_result = await level_7_supreme_council(f"TRIO-{squad}", results, "ai_corp_projects.db")

    if council_result["status"] == "SUCCESS":
        # Tầng 8-9: ĐÚC KẾT DI SẢN & TỰ SINH CHUYÊN ĐỀ MỚI
        legacy = await level_8_grand_mentor(squad[0], council_result["content"], "ai_corp_projects.db")
        final_const = await level_9_legacy_legend(squad[0], legacy, "ai_corp_projects.db")
        
        print(colored(f"🌌 [ASCENSION] Chuyên đề '{target_topic}' đã trở thành Di sản vĩnh cửu!", "green", attrs=["bold", "reverse"]))
        return final_const
    else:
        # GIAO THỨC PHỤC THÙ: Bắt cả 3 Agent quay lại L6 để vá lỗi cùng nhau
        print(colored(f"⚠️ [RE-TRAINING] Biệt đội thất bại. Đang ép quay lại lò luyện...", "red"))
        return None

# 🛡️ PHẦN 2: QUẢN TRỊ LẠM PHÁT XP & KIỂM ĐỊNH CHẤT LƯỢNG
async def validate_and_save_xp(db_path, agent_name, role_tag, task, result, tool, base_xp, quality_score=None):
    """
    HÀM CỘNG XP BỌC THÉP: 
    - Quality Score (1-100) do Tầng 7 hoặc Supervisor cấp.
    - Nếu Quality < 50: XP nhận được = 0 (Học lại).
    - Nếu Quality > 90: XP nhận được = base_xp * 1.5 (Thưởng tài năng).
    """
    # Nếu không có điểm chất lượng, mặc định là 70 (Đạt)
    score = quality_score if quality_score is not None else 70
    
    # Tính toán hệ số thực lực
    if score < 50:
        actual_xp = 0
        status_msg = "🔴 THẤT BẠI: Nội dung không đạt chuẩn. Không được cấp XP."
    elif score >= 90:
        actual_xp = int(base_xp * 1.5)
        status_msg = f"🌟 XUẤT SẮC: Cộng {actual_xp} XP (Bonus 50%)."
    else:
        actual_xp = base_xp
        status_msg = f"✅ ĐẠT: Cộng {actual_xp} XP."

    async with aiosqlite.connect(db_path) as db:
        # Cập nhật nhật ký công việc
        await db.execute("""
            INSERT INTO work_logs (timestamp, agent_name, task_content, result_summary, tool_used, xp_gain)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (datetime.now().isoformat(), agent_name, task, result[:1000], tool, actual_xp))
        
        # Cập nhật XP vào trạng thái Agent
        await db.execute("""
            INSERT OR REPLACE INTO agent_status (role_tag, xp) 
            VALUES (?, (SELECT COALESCE(xp,0) FROM agent_status WHERE role_tag=?)+?)
        """, (role_tag, role_tag, actual_xp))
        
        await db.commit()
    
    print(colored(f"📊 [XP-MONITOR] {role_tag}: {status_msg}", "yellow"))
    return actual_xp


async def main():
    init_database_global()
    # Chạy đồng thời: Vòng lặp chính và Hệ thống tự học
    await asyncio.gather(
        main_loop(),
        auto_learning_cycle()
    )

if __name__ == "__main__":
    asyncio.run(main())
