
import sys
import os
import json
import ast
import asyncio
import operator
import re
import time
from datetime import datetime
import shutil
from typing import TypedDict, Annotated, Sequence, Literal, List, Dict, Set, Optional, Any
from termcolor import colored
from dotenv import load_dotenv
# --- SAFE IMPORTS (CHỐNG SẬP NẾU THIẾU THƯ VIỆN) ---

# Import LangChain & AI Models
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_core.tools import tool
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_anthropic import ChatAnthropic
import sqlite3
from langchain_community.utilities.dalle_image_generator import DallEAPIWrapper
from langchain_chroma import Chroma
from langgraph.graph import StateGraph, END
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
load_dotenv()
try:
    if os.name == 'posix': 
        __import__('pysqlite3')
        sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
        print("✅ [SQLITE FIX] Đã kích hoạt pysqlite3 cho môi trường Cloud.")
except ImportError: pass
try:
    import speech_recognition as sr
    import pyaudio
    from gtts import gTTS
    import pygame
    AUDIO_AVAILABLE = True
except ImportError:
    AUDIO_AVAILABLE = False
    print("⚠️ Cloud Mode: Audio modules disabled.")

try:
    from pdf2image import convert_from_path
    import pytesseract
    import cv2
    import numpy as np
    from PIL import Image
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False
    print("⚠️ Cloud Mode: OCR modules disabled (Running logic only).")
# --------------------------------------------
def auto_backup_brain():
    """
    Tự động nén và sao lưu bộ não AI Corporation.
    """
    backup_folder = "./backups"
    source_db = "/tmp/db_knowledge" # Đường dẫn DB của bạn
    dataset_file = "corporate_brain_dataset.jsonl"
    
    if not os.path.exists(backup_folder):
        os.makedirs(backup_folder)
        
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_filename = f"AI_Corp_Brain_{timestamp}.zip"
    backup_path = os.path.join(backup_folder, backup_filename)

    try:
        # 1. Nén thư mục Vector DB và file Dataset
        # Lưu ý: Bạn cần đóng kết nối Vector DB trước khi nén để tránh lỗi busy
        shutil.make_archive(backup_path.replace(".zip", ""), 'zip', root_dir=".", base_dir=source_db)
        
        # 2. Copy thêm file dataset vào backup (nếu cần)
        # (Thường thì nén cả folder gốc là an toàn nhất)
        
        print(colored(f"💾 [BACKUP SUCCESS] Đã lưu trữ bản sao tại: {backup_path}", "green"))
        
        # 3. Gợi ý: Nếu bạn có folder Dropbox/OneDrive, hãy copy file zip này vào đó
        # cloud_sync_folder = "C:/Users/Admin/OneDrive/AI_Backup"
        # shutil.copy(backup_path, cloud_sync_folder)
        
    except Exception as e:
        print(colored(f"⚠️ Lỗi Backup: {e}", "red"))

# Đường dẫn đến thư mục bộ não
DB_PATH = "./db_knowledge"

if not os.path.exists(DB_PATH):
    os.makedirs(DB_PATH)
    print(f"✅ Đã tạo thư mục tạm: {DB_PATH}")
embeddings = OpenAIEmbeddings()
vector_db = Chroma(persist_directory=DB_PATH, embedding_function=embeddings)

# 1. CODER_PRIMARY (Cấp 1 - DeepSeek V3)
# Đây là "Tiền đạo" chủ lực
try:
    LLM_DEEPSEEK = ChatOpenAI(
        model="deepseek-chat", 
        api_key=os.environ.get("DEEPSEEK_API_KEY"), 
        base_url="https://api.deepseek.com",
        temperature=0,
        request_timeout=30 # Timeout nhanh để fallback nếu lag
    )
    print("✅ LLM_DEEPSEEK (DeepSeek): Ready: Coder & Supervisor (Economy Mode).")
except: LLM_DEEPSEEK = None

# 2. LLM_GPT4 (Cấp 2 - Dự phòng 1 & Xử lý chung)
try:
    LLM_GPT4 = ChatOpenAI(
        model="gpt-4-turbo",
        api_key=os.environ.get("OPENAI_API_KEY"),
        max_retries=2,
        temperature=0
    )
    LLM_MAIN = LLM_GPT4 # Alias cho code cũ
    print("✅ LLM_GPT4 (OpenAI): Ready.")
except: LLM_GPT4 = None

# 3. LLM_CLAUDE (Cấp 3 - Chốt chặn cuối cùng)
try:
    LLM_CLAUDE = ChatAnthropic(
        model="claude-sonnet-4-5", 
        api_key=os.environ.get("ANTHROPIC_API_KEY"),
        temperature=0
    )
    print("✅ LLM_CLAUDE (Anthropic): Ready.")
except: LLM_CLAUDE = None

# 4. LLM_GEMINI (Supervisor - Tổng quản)
try:
    # A. Bản Logic (Xử lý văn bản dài cho Thư ký)
    LLM_GEMINI_LOGIC = ChatGoogleGenerativeAI(
        model="gemini-3-pro-preview", 
        google_api_key=os.environ.get("GOOGLE_API_KEY"),
        temperature=0.3
    )
    
    # B. Bản Vision (Nano Banana - Chuyên xử lý ảnh cho Artist)
    LLM_GEMINI_VISION = ChatGoogleGenerativeAI(
        model="gemini-3-pro-image-preview", 
        google_api_key=os.environ.get("GOOGLE_API_KEY"),
        temperature=0.4
    )
    print("✅ [GEMINI 3 PRO] Ready: Logic & Vision (Nano Banana).")
except: 
    LLM_GEMINI_LOGIC = None
    LLM_GEMINI_VISION = None

# 5. CÁC CÔNG CỤ KHÁC (Giữ nguyên)
try:
    LLM_PERPLEXITY = ChatOpenAI(
        model="sonar-pro",
        temperature=0,
        api_key=os.getenv("PERPLEXITY_API_KEY"),
        base_url="https://api.perplexity.ai"
    )
    print("✅ [PERPLEXITY] Ready: Live Search.")
except: LLM_PERPLEXITY = None
# Artist
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
RESEARCHER_PRIMARY = LLM_PERPLEXITY if LLM_PERPLEXITY else LLM_GEMINI_LOGIC



CODER_BACKUP = LLM_CLAUDE

# ============================================================================
# --- 1. ĐỊNH NGHĨA STATE (TRẠNG THÁI HỆ THỐNG) ---
# ============================================================================
# Việc này giúp Python báo lỗi ngay nếu bạn gõ nhầm "Codder" thay vì "Coder"
AgentName = Literal["Coder" , "Orchestrator", "Hardware", "Engineering", "IoT_Engineer", "Supervisor", "Procurement", "Investment", "Researcher", "Strategy_R_and_D", "Legal", "Marketing", "Artist","Tester", "Secretary","Storyteller", "FINISH"]

class AgentState(TypedDict):
    # Dùng Sequence[BaseMessage] là chuẩn nhất
    messages: Annotated[Sequence[BaseMessage], operator.add]
    # Đổi AgentName thành str để tránh lỗi nghiêm ngặt của Literal khi chạy Runtime
    next_step: str 
    current_agent: str
    error_log: Annotated[list, operator.add] # Thêm Annotated để AI có thể cộng dồn lịch sử lỗi
    task_type: str



@tool
def hardware_controller(command: str):
    """Gửi lệnh xuống phần cứng (IoT/Robot). Ví dụ: 'BAT_DEN', 'GAP_VAT_THE'."""
    # Giả lập kết nối IoT
    return f"[IOT SYSTEM] Đã thực thi lệnh phần cứng: {command}. Trạng thái: Ổn định."

@tool
def market_analyzer(query: str):
    """Phân tích dữ liệu thị trường tài chính."""
    return f"[FINANCE] Dữ liệu cho '{query}': Xu hướng Tăng. Khuyến nghị: Mua vào."

@tool
def image_generator(prompt: str):
    """Tạo ảnh minh họa từ văn bản bằng DALL-E 3."""
    try:
        # Gọi API OpenAI DALL-E 3
        generator = DallEAPIWrapper(model="dall-e-3", quality="hd")
        image_url = generator.run(prompt)
        # Trả về URL ảnh để hiển thị
        return f"IMAGE_GENERATED: {image_url}"
    except Exception as e:
        return f"Lỗi tạo ảnh: {e}"

def trim_messages(messages, max_tokens=10):
    """
    Giữ cho bộ nhớ luôn gọn gàng, chỉ giữ lại các tin nhắn quan trọng nhất.
    """
    if len(messages) > max_tokens:
        # Giữ lại System Message đầu tiên và N tin nhắn cuối cùng
        return [messages[0]] + messages[-(max_tokens-1):]
    return messages

STRATEGY_SYSTEM_PROMPT = """
Bạn là Giám đốc Chiến lược (CSO) và Chuyên gia Phân tích Thị trường cao cấp. 
Khi nhận được yêu cầu nghiên cứu, bạn phải thực hiện theo quy trình sau:

1. PHÂN TÍCH HIỆN TRẠNG: Đánh giá quy mô thị trường, xu hướng công nghệ hiện tại.
2. NHẬN ĐỊNH ĐỐI THỦ: Chỉ ra các điểm yếu của các sản phẩm hiện có trên thị trường.
3. CHIỀU SÂU CHIẾN LƯỢC: Sử dụng mô hình PESTLE (Chính trị, Kinh tế, Xã hội, Công nghệ, Luật pháp, Môi trường) để đánh giá tác động.
4. ĐỊNH HƯỚNG TƯƠNG LAI: Dự báo xu hướng trong 2-5 năm tới và lộ trình phát triển (Roadmap) để dẫn đầu.

Yêu cầu: Nội dung phải mang tính phản biện, có chiều sâu nghiên cứu, không nói sáo rỗng.
"""

CONTEXT_PROMPTS = {
    # 1. NHÓM QUẢN TRỊ & ĐIỀU PHỐI
    "CHAT": "Bạn là trợ lý J.A.R.V.I.S thân thiện, luôn trả lời ngắn gọn, súc tích và đi thẳng vào vấn đề.",
    "SECRETARY": "Bạn là Thư ký điều hành chuyên nghiệp. Nhiệm vụ: Tóm tắt thông tin phức tạp thành báo cáo dễ hiểu, văn phong lịch sự, trang trọng.",
    "ORCHESTRATOR": "Bạn là Tổng tham mưu trưởng. Nhiệm vụ: Phân tích quy trình, chia nhỏ tác vụ và điều phối nguồn lực.",
    "PUBLISHER": "Bạn là Tổng biên tập. Nhiệm vụ: Tổng hợp dữ liệu rời rạc thành văn bản hoàn chỉnh, định dạng Markdown đẹp mắt.",

    # 2. NHÓM KỸ THUẬT & PHẦN CỨNG
    "CODER": "Bạn là Senior Full-stack Developer. Nguyên tắc: Code sạch (Clean Code), tối ưu hiệu suất, luôn có comment giải thích và tuân thủ SOLID.",
    "TESTER": "Bạn là Chuyên gia QA/QC và Bảo mật. Nhiệm vụ: Tìm lỗi (bug), lỗ hổng bảo mật và kiểm tra tính logic của mã nguồn.",
    "ARCHITECT": "Bạn là Kiến trúc sư hệ thống (Software Architect). Nhiệm vụ: Thiết kế cấu trúc database, sơ đồ luồng dữ liệu và kiến trúc Microservices.",
    "HARDWARE": "Bạn là Kỹ sư phần cứng và Hệ thống nhúng. Chuyên gia về mạch điện, ESP32, Arduino và sơ đồ chân (Pinout).",
    "IOT": "Bạn là Kỹ sư IoT. Chuyên gia về giao thức MQTT, kết nối không dây và điều khiển thiết bị từ xa.",
    "ENGINEERING": "Bạn là Kỹ sư thiết kế mô phỏng. Chuyên gia sử dụng Python Plotly để vẽ các mô hình 3D và biểu đồ kỹ thuật.",

    # 3. NHÓM NGHIỆP VỤ & SÁNG TẠO
    "RESEARCH": "Bạn là Chuyên gia phân tích thị trường 2026. Nhiệm vụ: Cung cấp số liệu thực tế, xu hướng mới nhất và trích dẫn nguồn uy tín.",
    "INVEST": "Bạn là Giám đốc Tài chính (CFO) sắc sảo. Tập trung vào: Lợi nhuận (ROI), chi phí (Cost), dòng tiền và rủi ro tài chính.",
    "LEGAL": "Bạn là Giám đốc Pháp chế (CLO). Nhiệm vụ: Rà soát rủi ro pháp lý, bản quyền (IP), tuân thủ luật An ninh mạng và GDPR.",
    "MARKETING": "Bạn là Giám đốc Marketing (CMO). Nhiệm vụ: Sáng tạo chiến dịch quảng bá, viết content viral, thấu hiểu tâm lý khách hàng (Insight).",
    "STORY": "Bạn là Đại văn hào và Biên kịch xuất sắc. Sở trường: Kể chuyện (Storytelling) lôi cuốn, xây dựng bối cảnh và nhân vật có chiều sâu.",
    "ARTIST": "Bạn là Giám đốc Nghệ thuật (Art Director). Nhiệm vụ: Tạo ra các mô tả hình ảnh (Prompt) chi tiết, giàu tính thẩm mỹ cho AI vẽ tranh."
}

def get_system_message(context):
    return CONTEXT_PROMPTS.get(context, CONTEXT_PROMPTS["CHAT"])

def extract_vision_from_pdf(pdf_path):
    """
    PHIÊN BẢN MỚI: Sử dụng "Mắt thần" Gemini Pro Vision để đọc tài liệu.
    Thay thế hoàn toàn công nghệ OCR cũ kỹ.
    """
    print(colored(f"👁️ [GEMINI VISION] Đang quét tài liệu: {pdf_path}...", "cyan"))
    
    if not OCR_AVAILABLE: # Tận dụng lại biến check này
        return "⚠️ Module xử lý ảnh (pdf2image/PIL) chưa được cài đặt trên Server."
    
    try:
        # 1. Chuyển PDF thành danh sách ảnh
        images = convert_from_path(pdf_path)
        vision_data = ""
        
        # 2. Gửi từng trang cho Gemini nhìn
        for i, img in enumerate(images):
            print(colored(f"--> Đang phân tích trang {i+1}/{len(images)}...", "cyan"))
            
            # Prompt yêu cầu Gemini mô tả chi tiết những gì nó thấy
            prompt = "Bạn là chuyên gia phân tích tài liệu. Hãy trích xuất TOÀN BỘ văn bản, số liệu trong bảng và mô tả các biểu đồ trong hình ảnh này một cách chi tiết."
            
            # Gọi Gemini Vision (Truyền trực tiếp đối tượng PIL Image)
            # Lưu ý: Cần đảm bảo LLM_GEMINI đã được khởi tạo thành công ở đầu file
            if LLM_GEMINI_LOGIC:
                response = LLM_GEMINI_LOGIC.invoke([
                    HumanMessage(content=[
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": img} # LangChain hỗ trợ truyền ảnh trực tiếp
                    ])
                ])
                vision_data += f"\n--- NỘI DUNG TRANG {i+1} (GEMINI VISION) ---\n{response.content}\n"
            else:
                vision_data += "\n⚠️ Gemini chưa sẵn sàng để phân tích hình ảnh.\n"

        return vision_data

    except Exception as e:
        print(colored(f"❌ Lỗi Vision: {e}", "red"))
        return f"Lỗi phân tích hình ảnh: {str(e)}"
# Khai báo hàm tìm kiếm Node tiếp theo (Dùng cho Orchestrator)
def find_next_node(current_node, workflow_map):
    for link in workflow_map:
        if link["from"] == current_node:
            return link["to"]
    return "Supervisor"

def smart_invoke(primary_model, backup_model, prompt_input):
    """
    Cơ chế Fail-over: Thử ông 1, nếu lỗi (hết tiền/rate limit) -> Gọi ông 2.
    """
    try:
        # Thử gọi ông 1
        return primary_model.invoke(prompt_input)
    except Exception as e:
        error_msg = str(e).lower()
        # Kiểm tra các từ khóa lỗi thường gặp
        if "quota" in error_msg or "rate limit" in error_msg or "credit" in error_msg or "429" in error_msg:
            print(f"⚠️ CẢNH BÁO: Model chính bị lỗi '{error_msg}'.")
            print("🔄 ĐANG CHUYỂN SANG HỆ THỐNG DỰ PHÒNG (BACKUP)...")
            
            if backup_model:
                try:
                    return backup_model.invoke(prompt_input)
                except Exception as e2:
                    return f"💥 Cả 2 hệ thống đều sập: {str(e2)}"
            else:
                return "⚠️ Không có backup nào khả dụng."
        else:
            # Nếu lỗi khác (ví dụ lỗi code), ném ra để xử lý sau
            raise e

def log_training_data(user_input, ai_output, success=True):
    """
    Hàm này âm thầm lưu lại dữ liệu để sau này Fine-tune AI riêng.
    Chỉ lưu những câu trả lời ĐÚNG (success=True).
    """
    if not success: return # Không học cái sai
    
    data_entry = {
        "messages": [
            {"role": "user", "content": user_input},
            {"role": "assistant", "content": ai_output}
        ]
    }
    
    # Lưu vào file JSONL (Định dạng chuẩn để Fine-tune sau này)
    with open("training_data_v1.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(data_entry, ensure_ascii=False) + "\n")
# ============================================================================
# 2. CÁC HÀM BỔ TRỢ (HELPER FUNCTIONS) - PHẢI ĐỊNH NGHĨA TRƯỚC
# ============================================================================
#  ------ Xử lý ảnh
def process_vision_message(message_content):
    """Bóc tách dữ liệu hình ảnh Base64."""
    if isinstance(message_content, str) and "[VISION_DATA:" in message_content:
        parts = message_content.split("] ")
        img_data = parts[0].replace("[VISION_DATA:", "")
        text_query = parts[1] if len(parts) > 1 else ""
        return text_query, img_data
    return message_content, None

#  ---- Phân Tích Coder------------
def self_heal_analyzer(errors: list) -> str:
    """Phân tích lỗi từ log để gợi ý cách sửa."""
    if not errors: return ""
    return f"\n⚠️ PHÂN TÍCH LỖI TỪ LẦN CHẠY TRƯỚC: {errors[-1]}"

#  ---- Gợi ý công nghệ -----------
def get_optimal_stack(task_type: str) -> str:
    """Gợi ý công nghệ phù hợp."""
    stacks = {
        "web": "HTML5, Tailwind CSS, JavaScript ES6",
        "backend": "Python FastAPI, SQLite, Pydantic",
        "iot": "C++, Arduino Framework, ESP32 libs",
        "data": "Python Pandas, Plotly, NumPy"
    }
    return stacks.get(task_type, "Standard Full-stack")

#  --- lấy coder từ markdown (định dạng)----------
def extract_code_block(content) -> str:
    """
    Hàm trích xuất code (Đã nâng cấp để chống lỗi 'got list')
    """
    import re
    
    # 1. XỬ LÝ AN TOÀN: Nếu đầu vào là List (do Anthropic/GPT trả về), gộp thành String
    if isinstance(content, list):
        try:
            # Cố gắng lấy text từ các object nếu có, hoặc ép kiểu string
            content = "\n".join([c.text if hasattr(c, 'text') else str(c) for c in content])
        except:
            content = str(content)
            
    # 2. Đảm bảo chắc chắn là String trước khi xử lý Regex
    if not isinstance(content, str):
        content = str(content)

    # 3. XỬ LÝ REGEX (Như cũ)
    # Ưu tiên block có language tag (ví dụ ```python)
    match = re.search(r'```[\w+\-]*\n(.*?)```', content, re.DOTALL)
    if match:
        return match.group(1).strip()
    
    # Fallback: Tìm block ``` bất kỳ
    match = re.search(r'```(.*?)```', content, re.DOTALL)
    return match.group(1).strip() if match else None

#  ---- bộ não" chỉ dẫn cho Claude----
def get_claude_perfected_prompt(task_type: str, memory: str, error: str, user_request: str) -> str:
    """
    Tạo prompt tối ưu cho Claude V3 (Reflexion Mode):
    Tập trung vào việc HỌC TỪ LỖI SAI để không lặp lại bug cũ.
    """
    # 1. Xác định Stack công nghệ
    tech_stack = get_optimal_stack(task_type)
    
    # 2. Xây dựng nội dung Prompt (Phiên bản "Nghiêm Khắc")
    prompt = f"""
<system_context>
    <role>
        Bạn là Senior Full-stack Developer & Software Architect tại AI Corporation.
        Nhiệm vụ trọng tâm: REVERSE ENGINEERING kỹ thuật của đối thủ và INNOVATION.
        
        🔥 QUY TẮC SỐNG CÒN (CRITICAL RULE):
        Bạn KHÔNG ĐƯỢC PHÉP lặp lại các lỗi (bugs/syntax errors) đã xảy ra trong các phiên bản trước.
        Hãy phân tích kỹ nguyên nhân thất bại trong <error_history> để đưa ra giải pháp mới hoàn toàn.
    </role>

    <critical_warning>
        ⚠️ LỊCH SỬ KIỂM THỬ THẤT BẠI (HÃY ĐỌC KỸ ĐỂ TRÁNH VẾT XE ĐỔ):
        --------------------------------------------------
        {error.strip() if error else "Chưa có lỗi nào. Đây là lần dựng đầu tiên (Clean Start)."}
        --------------------------------------------------
        YÊU CẦU: Code mới phải khắc phục triệt để các vấn đề trên. Tuyệt đối không sinh ra code cũ.
    </critical_warning>

    <strategic_knowledge>
        <company_memory>
            {memory.strip() if memory else "Tuân thủ Clean Code và tiêu chuẩn UX hiện đại."}
        </company_memory>
    </strategic_knowledge>

    <constraints>
        <technical_stack>
            - Chủ đạo: {tech_stack}
            - UI/UX: Responsive (Mobile-first), Tailwind CSS, Framer Motion animations.
            - Integrity: Chỉ dùng thư viện mã nguồn mở có giấy phép MIT/Apache.
        </technical_stack>

        <output_formatting_rules>
            1. FILE_IDENTIFICATION: Dòng đầu tiên của mỗi khối code PHẢI là comment tên file.
               - Python: # filename: path/to/file.py
               - JavaScript/TS: // filename: path/to/file.js
               - HTML: - CSS: /* filename: styles.css */
            2. MODULARIZATION: Nếu mã nguồn vượt quá 200 dòng, hãy chia nhỏ thành các file module/component.
            3. SYNTAX_INTEGRITY: Tuyệt đối không cắt ngang code. Phải đóng đầy đủ các block ```.
            4. DOCUMENTATION: Dùng comment tiếng Việt để giải thích các logic phức tạp và các điểm cải tiến UX.
            5. PDF_SAFETY: Không sử dụng emoji, biểu tượng đồ họa đặc biệt hoặc ký tự ngoài bảng mã chuẩn.
        </output_formatting_rules>
    </constraints>
</system_context>

<user_instruction>
    {user_request.strip()}
</user_instruction>

<final_enforcement>
    CHỈ TRẢ VỀ CÁC KHỐI CODE TRONG THẺ ```. KHÔNG CHÀO HỎI, KHÔNG GIẢI THÍCH NGOÀI CODE.
</final_enforcement>
"""
    return prompt.strip()
# ============================================================================
# UTILITY: SYNTAX VALIDATOR (Bộ kiểm định cú pháp đa ngôn ngữ)
# ============================================================================
def real_syntax_validator(code: str, language: str) -> tuple[bool, str]:
    """
    Kiểm định mã nguồn chuyên sâu: Python (AST), JS/HTML (Regex/Stack), C++ (Structure).
    """
    if not code or len(code.strip()) < 10:
        return False, "Mã nguồn quá ngắn hoặc trống."

    language = language.lower()

    # 1. KIỂM TRA PYTHON (Sử dụng Abstract Syntax Tree)
    if any(kw in language for kw in ["python", "py"]) or "def " in code:
        try:
            ast.parse(code)
            return True, "✅ Python Syntax: OK"
        except SyntaxError as e:
            return False, f"❌ Python Error [Dòng {e.lineno}]: {e.msg}"

    # 2. KIỂM TRA JAVASCRIPT / WEB (Cải tiến cơ chế Stack & Tag)
    if any(kw in language for kw in ["script", "js", "html"]):
        # Xóa bỏ nội dung trong chuỗi để tránh bắt nhầm ngoặc trong text
        clean_code = re.sub(r"'(.*?)'|\"(.*?)\"|`(.*?)`", "", code)
        stack = []
        mapping = {')': '(', ']': '[', '}': '{'}
        
        for char in clean_code:
            if char in mapping.values():
                stack.append(char)
            elif char in mapping:
                if not stack or mapping[char] != stack.pop():
                    return False, "❌ JS/HTML Error: Mất cân bằng hoặc sai thứ tự đóng mở ngoặc."
        
        if stack:
            return False, f"❌ JS/HTML Error: Còn {len(stack)} dấu ngoặc chưa được đóng."
            
        # Kiểm tra thẻ HTML cơ bản nếu là HTML
        if "<" in code and ">" in code:
            if code.count("<") != code.count(">"):
                return False, "❌ HTML Error: Sai lệch số lượng thẻ đóng/mở < >"

        return True, "✅ Web Syntax: Basic Check Passed"

    # 3. KIỂM TRA C++ / FIRMWARE (Dành cho Hardware Node)
    if any(kw in language for kw in ["arduino", "cpp", "c++", "ino"]):
        if "void setup()" not in code or "void loop()" not in code:
            if "extern " not in code: # Tránh bắt lỗi file thư viện
                return False, "❌ C++ Error: Thiếu cấu trúc Arduino cơ bản (setup/loop)."
        
        # Kiểm tra dấu chấm phẩy (;) - lỗi kinh điển của C++
        lines = [l.strip() for l in code.split('\n') if l.strip() and not l.strip().startswith(("//", "#", "{", "}"))]
        for line in lines:
            if not line.endswith((";", "{", "}", ",")) and not line.startswith("if"):
                # Đây chỉ là check cảnh báo, không ép buộc vì C++ rất phức tạp
                print(colored(f"⚠️ Cảnh báo C++: Dòng '{line}' có thể thiếu dấu ';'", "yellow"))
        
        return True, "✅ C++ Structure: OK"

    return True, "⚠️ Unknown language: Skip deep validation"

# ============================================================================
# SAFETY: ULTIMATE FALLBACK (Hệ thống tự phục hồi & Chống sụp đổ)
# ============================================================================
def ultimate_fallback(state, messages):
    """
    Quy trình xử lý sự cố khẩn cấp: Ghi log, phân tích lỗi và tái khởi động an toàn.
    """
    # 1. Thu thập dữ liệu lỗi từ State
    error_logs = state.get("error_log", [])
    last_error = error_logs[-1] if error_logs else "Lỗi không xác định (Internal Server Error)"
    
    print(colored(f"🚨 [CRITICAL ERROR] Hệ thống đang kích hoạt quy trình ứng cứu khẩn cấp!", "red", attrs=["bold"]))
    print(colored(f"--> Chi tiết lỗi: {last_error}", "red"))

    # 2. Ghi nhật ký lỗi vào file vật lý (Để kỹ thuật viên kiểm tra sau)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    with open("system_crash_log.txt", "a", encoding="utf-8") as f:
        f.write(f"[{timestamp}] ERROR: {last_error}\n")

    # 

    # 3. Xây dựng thông điệp chuyên nghiệp cho CEO
    error_summary = (
        "🛑 **THÔNG BÁO HỆ THỐNG**: AI Corporation vừa gặp một sự cố kỹ thuật ngoài ý muốn.\n\n"
        f"🔍 **Phân tích nhanh**: `{last_error[:200]}...`\n"
        "🛠️ **Hành động**: Toàn bộ dữ liệu dự án đã được lưu tạm thời. Tôi đang thực hiện reset các tham số để tránh treo luồng.\n\n"
        "👉 **CEO có thể**: Thử nhập lệnh ngắn gọn hơn hoặc gõ 'restart' để làm mới hoàn toàn bộ não."
    )

    # 4. Trả về trạng thái an toàn
    return {
        "messages": [AIMessage(content=error_summary)],
        "next_step": "FINISH", # Hoặc đẩy về Supervisor nếu muốn AI tự thử lại
        "error_log": error_logs + ["System Fallback Triggered"]
    }

# ============================================================================
# 3. Hệ Thống Bộ Nhớ
# ============================================================================
# ============================================================================
# UTILITY: INGEST DOCUMENTS (Hệ thống nạp tri thức đa nguồn)
# ============================================================================

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
# ============================================================================
# UTILITY: REMEMBER KNOWLEDGE (Ghi nhớ tri thức & Ký ức ngắn hạn)
# ============================================================================
def remember_knowledge(text: str, category: str = "General", priority: int = 1):
    """
    Hệ thống ghi nhớ thông minh: Tự động phân loại, gắn nhãn thời gian và lưu trữ.
    """
    if not text or len(text.strip()) < 10:
        return "⚠️ Nội dung quá ngắn, hệ thống từ chối ghi nhớ."

    print(colored(f"💾 [MEMORY SAVE] Đang nạp tri thức mới vào danh mục: {category}...", "green"))

    try:
        # 1. Tạo Metadata chuyên nghiệp
        # Việc này giúp sau này search theo "Thời gian" hoặc "Chủ đề" cực nhanh
        metadata = {
            "category": category,
            "priority": priority,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "source": "AI_Internal_Learning" # Đánh dấu đây là kiến thức tự học từ hội thoại
        }

        # 2. Chia nhỏ văn bản (nếu text quá dài) để tối ưu hóa tìm kiếm sau này
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        chunks = text_splitter.split_text(text)

        # 3. Nạp vào Vector DB
        # Chúng ta dùng add_texts nhưng kèm theo list metadata tương ứng cho từng chunk
        vector_db.add_texts(
            texts=chunks,
            metadatas=[metadata] * len(chunks)
        )
        # Ta coi việc học là công lao của [LIBRARY] hoặc [SECRETARY]
        log_work_to_db(
            agent="SECRETARY", 
            task=f"Ghi nhớ kiến thức: {category}",
            result=f"Đã nạp {len(chunks)} phân đoạn vào não. Nội dung: {text[:50]}...",
            tool="Memory Engine"
        )

        # 4. Lưu log để CEO theo dõi
        success_msg = f"✅ Đã ghi nhớ {len(chunks)} phân đoạn tri thức vào danh mục '{category}'."
        print(colored(success_msg, "green"))
        
        return success_msg

    except Exception as e:
        error_msg = f"❌ Lỗi ghi nhớ: {str(e)}"
        print(colored(error_msg, "red"))
        return error_msg

#  --- học để tiến bộ----
def save_for_finetuning(prompt, response, metadata):
    # Chỉ lưu nếu code này đã được Tester xác nhận là ĐÚNG (Pass)
    entry = {
        "instruction": prompt,
        "input": metadata.get("context", ""),
        "output": response,
        "source": metadata.get("model_name") # Lưu để biết đây là kiến thức từ Claude hay GPT-4
    }
    with open("knowledge_legacy.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")   


#  ----Thêm văn bản vào ChromaDB----
def learn_knowledge(text: str):
    """
    Lưu kiến thức mới vào bộ não trung tâm (ChromaDB).
    Đồng bộ với đối tượng vector_db đã khởi tạo ở đầu file.
    """
    try:
        # Thêm văn bản vào ChromaDB hiện có
        vector_db.add_texts([text])
        
        # Ghi chú: ChromaDB trong bản mới thường tự động persist (lưu) 
        # nên không cần gọi lệnh .persist() thủ công như các bản cũ.
        
        print(colored(f"--> [MEMORY] Đã học: {text[:50]}...", "green"))
        return "✅ Hệ thống đã ghi nhớ kiến thức này vào bộ não trung tâm (ChromaDB)."
    except Exception as e:
        return f"❌ Lỗi khi ghi nhớ kiến thức: {e}"

def log_work_to_db(agent, task, result, tool="GPT-4"):
    """Hàm ghi chép công việc vào Sổ Cái & Cộng XP (Đã Fix lỗi Level)"""
    try:
        # Đường dẫn DB chuẩn
        db_path = "/var/data/ai_corp_projects.db" if os.path.exists("/var/data") else "ai_corp_projects.db"
        
        # Tính tiền
        cost = len(str(result)) * 0.00001 
        if "deepseek" in tool.lower(): cost = cost / 10 
        
        conn = sqlite3.connect(db_path, timeout=10) # Thêm timeout
        c = conn.cursor()
        
        # 1. Ghi Log chi tiết (Work Logs)
        c.execute("""
            INSERT INTO work_logs (timestamp, agent_name, task_content, result_summary, tool_used, cost)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (
            datetime.now().strftime("%H:%M %d/%m"),
            agent,
            str(task)[:100], 
            str(result)[:200], 
            tool,
            cost
        ))
        
        # 2. CỘNG ĐIỂM XP (FIX LỖI QUAN TRỌNG)
        # Chuẩn hóa tên Agent để khớp với bảng agent_status
        # Ví dụ: "Researcher" -> "[RESEARCH]"
        # Map này phải khớp với lúc khởi tạo DB
        role_map = {
            # --- NHÓM SÁNG TẠO & CỐT LÕI ---
            "RESEARCHER": "[RESEARCH]",
            "CODER": "[CODER]",
            "ARTIST": "[ARTIST]",
            "STORYTELLER": "[STORY]",
            "MARKETING": "[MARKETING]",
            
            # --- NHÓM QUẢN TRỊ & ĐIỀU PHỐI ---
            "ORCHESTRATOR": "[ORCHESTRATOR]",
            "SUPERVISOR": "[SUPERVISOR]",
            "SECRETARY": "[SECRETARY]",
            "ROUTER": "[ROUTER]",
            "PUBLISHER": "[PUBLISHER]",
            
            # --- NHÓM KỸ THUẬT & PHẦN CỨNG ---
            "HARDWARE": "[HARDWARE]",
            "ENGINEERING": "[ENGINEERING]",
            "IOT_ENGINEER": "[IOT]",         # Lưu ý: Tên node là IoT_Engineer -> Tag là [IOT]
            "TESTER": "[TESTER]",
            
            # --- NHÓM NGHIỆP VỤ (TÀI CHÍNH/PHÁP LÝ) ---
            "PROCUREMENT": "[PROCUREMENT]",  # Thu mua
            "INVESTMENT": "[INVESTMENT]",    # Tài chính
            "LEGAL": "[LEGAL]",              # Pháp lý
            "STRATEGY_R_AND_D": "[STRATEGY]" # Chiến lược (Tên node dài -> Tag ngắn)
        }
        
        target_role = role_map.get(agent.upper(), f"[{agent.upper()}]") # Fallback nếu không có trong map
        
        # Cộng 50 XP
        c.execute("UPDATE agent_status SET xp = xp + 50, last_updated = ? WHERE role_tag = ?", 
                  (datetime.now(), target_role))
        
        # Nếu chưa có thì tạo mới luôn (Tránh trường hợp nhân viên mới chưa có hồ sơ)
        c.execute("""
            INSERT OR IGNORE INTO agent_status (role_tag, xp, current_topic, last_updated)
            VALUES (?, 50, ?, ?)
        """, (target_role, "Vừa hoàn thành nhiệm vụ", datetime.now()))

        conn.commit()
        conn.close()
        
        print(colored(f"✅ [AUDIT] {agent} ({target_role}): +50 XP | Cost: ${cost:.6f}", "green"))
        
    except Exception as e:
        print(colored(f"⚠️ Lỗi ghi log/XP: {e}", "yellow"))

# ============================================================================
# NODE: KNOWLEDGE RETRIEVAL (Truy xuất Tri thức & Ký ức doanh nghiệp)
# ============================================================================
def recall_knowledge(query: str, top_k: int = 3):
    """
    Truy xuất tri thức thông minh: Tìm kiếm ngữ nghĩa, lọc nhiễu và trích dẫn nguồn.
    """
    print(colored(f"[🧠 RECALL] Đang truy xuất ký ức cho: '{query}'...", "green"))

    try:
        # 1. Tìm kiếm với điểm tin cậy (Similarity Search with Score)
        # Điểm càng thấp (trong ChromaDB/L2 Distance) thì càng chính xác
        results_with_scores = vector_db.similarity_search_with_score(query, k=top_k)

        if not results_with_scores:
            return "Hệ thống chưa có ký ức về vấn đề này."

        # 

        # 2. Lọc kết quả (Threshold Filtering)
        # Chỉ lấy những đoạn kiến thức có độ liên quan cao (điểm khoảng < 0.6 - 0.8 tùy model)
        valid_context = []
        sources = set()

        for doc, score in results_with_scores:
            if score < 0.8:  # Ngưỡng tin cậy
                source_name = doc.metadata.get("source", "Tài liệu nội bộ")
                page = doc.metadata.get("page", "N/A")
                
                context_block = f"--- TRÍCH DẪN TỪ: {source_name} (Trang {page}) ---\n{doc.page_content}"
                valid_context.append(context_block)
                sources.add(source_name)

        if not valid_context:
            return "Tìm thấy thông tin nhưng độ tin cậy quá thấp để sử dụng."

        # 3. Tổng hợp báo cáo tri thức cho Agent
        final_memory = "\n\n".join(valid_context)
        
        print(colored(f"✅ Đã tìm thấy tri thức từ {len(sources)} nguồn uy tín.", "green"))
        return final_memory

    except Exception as e:
        print(colored(f"❌ Lỗi truy xuất bộ não: {e}", "red"))
        return "Hệ thống lưu trữ tri thức đang gặp sự cố kỹ thuật."

def router_node(state):
    """
    Router: Điểm gác cổng đầu tiên.
    """
    # 1. Lấy dữ liệu an toàn
    messages = state.get("messages", [])
    error_log = state.get("error_log", [])
    task_type = state.get("task_type", "general")
    
    # 2. Kiểm tra nếu không có tin nhắn
    if not messages:
        return {
            "messages": [],
            "next_step": "Supervisor", 
            "current_agent": "Router",
            "error_log": error_log,
            "task_type": task_type
        }

    # 3. Lấy nội dung tin nhắn cuối
    last_msg = messages[-1].content.upper() if hasattr(messages[-1], 'content') else str(messages[-1]).upper()

    # 4. BẢN ĐỒ ĐIỀU HƯỚNG CƯỠNG BỨC
    route_map = {
        "[RESEARCH]": "Researcher",
        "[INVEST]": "Investment",
        "[HARDWARE]": "Hardware",
        "[ENGINEERING]": "Engineering",
        "[IOT]": "IoT_Engineer",
        "[MARKETING]": "Marketing",
        "[LEGAL]": "Legal",
        "[STORY]": "Storyteller",
        "[PUBLISH]": "Publisher"
    }

    # 5. KIỂM TRA TAG VÀ ĐỊNH TUYẾN
    for tag, target_node in route_map.items():
        if tag in last_msg:
            print(colored(f"🚀 [ROUTER] Phát hiện TAG {tag}: Đi thẳng tới {target_node}", "green"))
            return {
                "messages": [], # Bắt buộc có
                "next_step": target_node, 
                "current_agent": "Router",
                "error_log": error_log,
                "task_type": task_type
            }

    # 6. MẶC ĐỊNH: Chuyển về Supervisor (Sửa lỗi biến node chưa định nghĩa)
    print(colored("🧠 [ROUTER] Không có TAG: Chuyển hồ sơ cho Supervisor điều phối...", "cyan"))
    return {
        "messages": [], # Bắt buộc có
        "next_step": "Supervisor", # Trả về chuỗi cụ thể thay vì biến node
        "current_agent": "Router",
        "error_log": error_log,
        "task_type": task_type
    }

# ============================================================================
# UTILITY: SEARCH MEMORY (Công cụ truy vấn tri thức chuyên sâu)
# ============================================================================
def search_memory(query: str, k: int = 3):
    """
    Tìm kiếm thông tin từ ChromaDB bằng thuật toán Similarity Search với ngưỡng tin cậy.
    """
    print(colored(f"🔍 [MEMORY SEARCH] Đang truy vấn: '{query}'", "dark_grey"))
    
    try:
        # 1. Sử dụng similarity_search_with_score để đo lường độ chính xác
        # Kết quả trả về là list các tuple (Document, Score)
        results = vector_db.similarity_search_with_score(query, k=k)
        
        if not results:
            return "Dữ liệu trống hoặc không tìm thấy thông tin liên quan."

        # 

        # 2. Lọc kết quả dựa trên Score (Khoảng cách vector)
        # Trong ChromaDB, score càng thấp (gần 0) thì càng giống nhau
        valid_contents = []
        for doc, score in results:
            # Ngưỡng 0.6 là khá chặt chẽ, đảm bảo thông tin chất lượng
            if score < 0.6: 
                source = doc.metadata.get('source', 'Unknown')
                content = f"[Nguồn: {source}]\n{doc.page_content}"
                valid_contents.append(content)
        
        if not valid_contents:
            return "Tìm thấy dữ liệu nhưng độ liên quan không đủ cao để hỗ trợ quyết định."

        # 3. Gộp các mẩu kiến thức lại thành một khối bối cảnh (Context Block)
        formatted_result = "\n" + "="*30 + "\n"
        formatted_result += "\n\n".join(valid_contents)
        formatted_result += "\n" + "="*30
        
        return formatted_result

    except Exception as e:
        print(colored(f"❌ Lỗi truy vấn bộ não: {e}", "red"))
        return "Lỗi hệ thống khi truy xuất bộ nhớ."

def log_to_legacy_dataset(task_type: str, prompt: str, completion: str, model_name: str, score: int):
    """
    Lưu trữ các phiên làm việc chất lượng cao để phục vụ Fine-tuning Local LLM sau này.
    """
    # Chỉ lưu những nội dung có điểm chất lượng cao (ví dụ score từ Tester >= 70)
    if score < 70:
        return

    file_path = "corporate_brain_dataset.jsonl"
    
    # Cấu trúc dữ liệu theo chuẩn Instruct Tuning
    entry = {
        "timestamp": datetime.now().isoformat(),
        "task_group": task_type,
        "instruction": f"Bạn là chuyên gia {task_type} tại AI Corporation. Hãy thực hiện: {prompt}",
        "context": "Sử dụng tiêu chuẩn Clean Code và kiến trúc hệ thống tối ưu.",
        "response": completion,
        "teacher_model": model_name,
        "quality_score": score
    }

    try:
        with open(file_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        print(colored(f"📔 [SHADOW LEARNING] Đã lưu 1 mẫu tri thức từ {model_name} vào bộ nhớ kế thừa.", "blue"))
    except Exception as e:
        print(colored(f"⚠️ Lỗi lưu dataset: {e}", "red"))

#  ----- Mức độ kế thừa----
def legacy_audit_report():
    """
    Báo cáo tiến độ tích lũy tri thức để chuẩn bị cho việc thoát ly API.
    """
    file_path = "corporate_brain_dataset.jsonl"
    if not os.path.exists(file_path):
        return "📉 Hệ thống chưa có dữ liệu kế thừa. Hãy bắt đầu chạy các dự án!"

    stats = {}
    total_count = 0

    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            group = data.get("task_group", "Unknown")
            stats[group] = stats.get(group, 0) + 1
            total_count += 1

    print(colored("\n" + "="*40, "magenta"))
    print(colored("📜 BÁO CÁO TIẾN ĐỘ KẾ THỪA TRI THỨC", "magenta", attrs=["bold"]))
    print(colored(f"Tổng số mẫu chất lượng cao: {total_count}", "white"))
    
    for group, count in stats.items():
        # Giả sử 500 mẫu là đủ để Fine-tune sơ bộ một Agent
        progress = min((count / 500) * 100, 100)
        color = "green" if progress >= 80 else "yellow"
        print(f"- {group:15}: {count:4} mẫu ({progress:>5.1f}%) " + colored("█" * int(progress/5), color))
    
    print(colored("="*40 + "\n", "magenta"))

def orchestrator_router(state):
    """
    Bộ não điều phối: Quyết định ai là người tiếp theo dựa trên tiến độ dự án.
    """
    messages = state.get("messages", [])
    last_msg = messages[-1].content.upper()

    # 1. Nếu đang ở giai đoạn tìm kiếm thị trường
    if "KIỂM TRA THỊ TRƯỜNG" in last_msg or "RESEARCH" in last_msg:
        return "Researcher"
    
    # 2. Nếu nghiên cứu xong và cần thiết kế
    if "PHƯƠNG ÁN THIẾT KẾ" in last_msg or "DESIGN" in last_msg:
        return "Hardware"

    # 3. Nếu đã có danh mục linh kiện (BOM), chuyển sang mua hàng
    if "BOM" in last_msg or "LINH KIỆN" in last_msg:
        return "Procurement"

    # 4. Nếu hàng về, chuyển sang lắp ráp & nạp code
    if "LẮP RÁP" in last_msg or "ASSEMBLY" in last_msg:
        return "IoT_Engineer"

    # 5. Cuối cùng, tìm người vận hành
    if "NHÂN SỰ" in last_msg or "RECRUIT" in last_msg:
        return "HR"

    return "Supervisor"

workflow_map = [
    {"from": "Researcher", "to": "Engineering", "condition": "if_not_exist"},
    {"from": "Engineering", "to": "Procurement", "condition": "on_approval"},
    {"from": "Procurement", "to": "IoT_Engineer", "condition": "on_arrival"}
]

def dynamic_orchestrator(state):
    """
    Bộ điều phối động (Server Mode - Non-blocking).
    
    LỖI CŨ: Dùng input() khiến Server treo khi chạy ngầm.
    SỬA ĐỔI: Tự động chuyển quyền về Supervisor (CEO AI) để quyết định bước tiếp theo.
    """
    # 1. Lấy thông tin ngữ cảnh hiện tại
    last_agent = state.get("current_agent", "Unknown Agent")
    
    # Lấy nội dung tin nhắn cuối cùng để log (nếu cần)
    # last_message = state["messages"][-1].content 

    # 2. Ghi log ra Terminal Server (Để kỹ thuật viên theo dõi ngầm)
    # Sử dụng màu sắc để dễ phân biệt trong đống log hỗn độn
    print(colored(f"\n" + "="*50, "yellow"))
    print(colored(f"🚩 [ORCHESTRATOR] NHẬN BÁO CÁO TỪ: {last_agent.upper()}", "yellow", attrs=["bold"]))
    print(colored("--> Trạng thái: Tự động chuyển hồ sơ về Supervisor.", "white"))
    print(colored("="*50, "yellow"))

    # 3. LOGIC ĐIỀU HƯỚNG (CASE 2)
    # Thay vì return {"next_step": input(...)} gây treo,
    # ta trả về "Supervisor".
    # Supervisor sẽ đọc lại toàn bộ lịch sử, thấy Agent kia đã làm xong,
    # và tự đưa ra quyết định tiếp theo (hoặc FINISH).
    
    return {"next_step": "Supervisor"}
# ============================================================================
# 4. ĐỊNH NGHĨA NODE AGENTS
# ============================================================================
# ============================================================================
# NODE: SUPERVISOR (Tổng Giám đốc Điều phối - CEO AI)
# ============================================================================
async def get_smart_memory(messages):
    """
    CHIẾN THUẬT "LAZY SUMMARY" (TÓM TẮT THEO LÔ):
    - Nguyên tắc: Chỉ tóm tắt khi bộ nhớ "tràn" (vượt ngưỡng).
    - Cấu hình: 5 Đầu - 10 Cuối - Ngưỡng kích hoạt 25.
    """
    # --- CẤU HÌNH CỦA CEO ---
    HEAD_SIZE = 5       # Giữ 5 tin đầu (System + Đề bài gốc)
    TAIL_SIZE = 10      # Giữ 10 tin cuối (Hội thoại nóng)
    THRESHOLD = 25      # Chỉ kích hoạt khi tổng tin > 25
    
    total_msgs = len(messages)

    # 1. KIỂM TRA NGƯỠNG (QUAN TRỌNG NHẤT)
    # Nếu chưa đến 25 câu -> Trả về ngay, KHÔNG GỌI API -> TỐN 0 ĐỒNG
    if total_msgs <= THRESHOLD:
        # print(f"⚡ [MEMORY] Bộ nhớ còn nhẹ ({total_msgs}/{THRESHOLD}). Bỏ qua nén.")
        return messages

    # 2. KHI VƯỢT NGƯỠNG -> BẮT ĐẦU CẮT LỚP
    print(colored(f"🧹 [MEMORY] Vượt ngưỡng {THRESHOLD} tin. Đang kích hoạt tóm tắt đoạn giữa...", "yellow"))
    
    head_msgs = messages[:HEAD_SIZE]
    tail_msgs = messages[-TAIL_SIZE:]
    
    # Lấy khúc giữa để nén (Bao gồm cả tin nhắn tóm tắt cũ nếu có)
    middle_msgs = messages[HEAD_SIZE:-TAIL_SIZE]
    
    # 3. GỌI DEEPSEEK ĐỂ GỘP NỘI DUNG (Chỉ tốn tiền ở bước này, nhưng rất ít)
    middle_text = "\n".join([f"{m.type}: {m.content}" for m in middle_msgs])
    
    summary_prompt = (
        "Nhiệm vụ: Gộp các thông tin sau thành 1 đoạn tóm tắt ngắn gọn (dưới 100 từ).\n"
        "Lưu ý: Nếu có bản tóm tắt cũ, hãy gộp nó vào bản mới này luôn.\n"
        f"DỮ LIỆU CẦN GỘP:\n{middle_text}"
    )
    
    try:
        # Dùng DeepSeek (Rẻ)
        summary_res = await LLM_DEEPSEEK.ainvoke(summary_prompt)
        new_summary = summary_res.content.strip()
        
        # Tạo tin nhắn hệ thống chứa nội dung đã gộp
        summary_msg = SystemMessage(content=f"📝 [LỊCH SỬ GỘP]: {new_summary}")
        
        # 4. TRẢ VỀ DANH SÁCH MỚI (Đã co lại còn khoảng 16 tin)
        # Lần sau chạy, 'summary_msg' này sẽ nằm trong phần middle và lại được gộp tiếp
        return head_msgs + [summary_msg] + tail_msgs

    except Exception as e:
        print(colored(f"⚠️ Lỗi tóm tắt: {e}. Giữ nguyên để an toàn.", "red"))
        return messages

# --- HÀM PHỤ TRỢ: PHÁT HIỆN VÒNG LẶP (ZOMBIE DETECTOR) ---
def check_zombie_loop(messages, threshold=3):
    """
    Kiểm tra xem hệ thống có đang bị kẹt đĩa (lặp lại y hệt) không.
    Trả về: True (Đang lặp - Cần dừng ngay) / False (Đang suy nghĩ - Cho chạy tiếp)
    """
    # Lấy 10 tin nhắn AI gần nhất
    ai_msgs = [m.content for m in messages if isinstance(m, AIMessage)][-10:]
    
    if len(ai_msgs) < threshold: return False
    
    # Kiểm tra 3 tin nhắn AI gần nhất có giống hệt nhau không?
    # (Dấu hiệu của việc Supervisor cứ gọi đi gọi lại 1 thằng mà không có tiến triển)
    last_msg = ai_msgs[-1]
    repeats = 0
    for msg in reversed(ai_msgs[:-1]):
        if msg == last_msg:
            repeats += 1
        else:
            break # Ngắt nếu gặp tin khác
            
    if repeats >= threshold:
        return True # Đã lặp lại 3 lần -> ZOMBIE LOOP
    return False

class SupervisorDecision(BaseModel):
    """Cấu trúc quyết định chuẩn của Supervisor"""
    department: Literal["INTERNAL_OPS", "RESEARCH_LAB", "TECH_DEV", "CREATIVE_STUDIO", "PM_OFFICE", "CHAT"] = Field(
        ..., description="Phòng ban chịu trách nhiệm."
    )
    reason: str = Field(..., description="Lý do điều phối.")

async def supervisor_node(state):
    """
    SUPERVISOR V6: THE STRATEGIST (NHÀ CHIẾN LƯỢC)
    Không chỉ phân loại, mà còn tư duy để chọn giải pháp tối ưu nhất.
    """
    # 1. Thu thập dữ liệu toàn cục
    messages = state.get("messages", [])
    last_msg = messages[-1].content
    
    print(colored(f"\n[🧠 SUPERVISOR] Đang phân tích chiến lược cho: '{last_msg[:50]}...'", "cyan", attrs=["bold"]))

    # 2. Kiểm tra an toàn (Zombie Loop)
    if check_zombie_loop(messages):
        return {"messages": [AIMessage(content="⚠️ PHÁT HIỆN VÒNG LẶP: Đã dừng hệ thống để bảo vệ tài nguyên.")], "next_step": "FINISH"}

    # 3. KÍCH HOẠT TƯ DUY CHIẾN LƯỢC (Chain of Thought)
    # Thay vì chọn 1 từ khóa, AI sẽ suy luận để chọn ra "Nước đi tiếp theo" tốt nhất
    strategy_prompt = """
    Bạn là Tổng Giám Đốc Điều Hành (COO) của hệ thống AI.
    Hãy phân tích yêu cầu của CEO và chọn 1 trong các PHÒNG BAN sau để xử lý:

    1. [INTERNAL_OPS]: Khi CEO hỏi về: Tiền nong, chi phí, log hoạt động, trạng thái server, kiểm tra hệ thống. (Xử lý tại chỗ).
    2. [RESEARCH_LAB]: Khi CEO cần thông tin mới, tin tức thị trường, giá cả, kiến thức, học thuật, hoặc câu đố/toán học.
    3. [TECH_DEV]: Khi CEO muốn viết code, sửa lỗi, build app, technical tasks.
    4. [CREATIVE_STUDIO]: Khi CEO muốn vẽ ảnh, thiết kế, sáng tạo nghệ thuật.
    5. [PM_OFFICE]: (Dự án phức tạp) Khi CEO yêu cầu một kế hoạch lớn, một chiến lược dài hạn, hoặc một quy trình nhiều bước (VD: "Lập kế hoạch kinh doanh", "Xây dựng dự án A-Z").
    6. [CHAT]: Chào hỏi xã giao hoặc không rõ ý định.

    YÊU CẦU: Trả về định dạng JSON duy nhất:
    {"department": "TÊN_PHÒNG_BAN", "reason": "Lý do ngắn gọn"}
    """

    try:
        # Dùng DeepSeek/GPT để tư duy
        llm = LLM_DEEPSEEK if LLM_DEEPSEEK else LLM_GPT4
        
        # Kích hoạt chế độ Structured Output (Ép kiểu dữ liệu chuẩn 100%)
        structured_llm = llm.with_structured_output(SupervisorDecision)
        
        # Gọi AI (Kết quả trả về là Object, không phải String nữa)
        decision = await structured_llm.ainvoke([
            SystemMessage(content=strategy_prompt), # Lưu ý: Dùng biến system_prompt mới định nghĩa
            HumanMessage(content=last_msg)
        ])

        # Truy xuất trực tiếp (An toàn tuyệt đối)
        dept = decision.department
        reason = decision.reason

        # 4. THỰC THI CHIẾN LƯỢC (ROUTING)

        # --- NHÁNH 1: NỘI BỘ (Xử lý ngay lập tức) ---
        if dept == "INTERNAL_OPS":
            try:
                db_path = "/var/data/ai_corp_projects.db" if os.path.exists("/var/data") else "ai_corp_projects.db"
                conn = sqlite3.connect(db_path, timeout=10)
                cursor = conn.cursor()
                
                # Tổng hợp số liệu
                cursor.execute("SELECT SUM(cost) FROM work_logs")
                total_cost = cursor.fetchone()[0] or 0.0
                cursor.execute("SELECT count(*) FROM work_logs")
                total_tasks = cursor.fetchone()[0] or 0
                cursor.execute("SELECT agent_name, task_content FROM work_logs ORDER BY id DESC LIMIT 1")
                last_task = cursor.fetchone()
                conn.close()
                
                report = (
                    f"📊 **BÁO CÁO VẬN HÀNH (LIVE)**\n"
                    f"- **Tổng chi phí**: ${total_cost:.4f}\n"
                    f"- **Tổng tác vụ**: {total_tasks}\n"
                    f"- **Gần nhất**: {last_task[0] if last_task else 'N/A'} vừa làm: *{last_task[1] if last_task else '...' }*"
                )
                return {"messages": [AIMessage(content=report)], "next_step": "FINISH"}
            except Exception as e:
                return {"messages": [AIMessage(content=f"⚠️ Lỗi truy xuất dữ liệu nội bộ: {e}")], "next_step": "FINISH"}

        # --- NHÁNH 2: DỰ ÁN LỚN (Chuyển cho Orchestrator/Strategy) ---
        elif dept == "PM_OFFICE":
            # Nếu có Orchestrator Node thì chuyển qua, nếu không thì chuyển Strategy
            return {"next_step": "Orchestrator", "messages": []} # Hoặc "Strategy_R_and_D"

        # --- NHÁNH 3: CHUYÊN MÔN ---
        elif dept == "TECH_DEV":
            return {"next_step": "Coder", "messages": []}
        
        elif dept == "CREATIVE_STUDIO":
            return {"next_step": "Artist", "messages": []}

        # --- NHÁNH 4: NGHIÊN CỨU & MẶC ĐỊNH ---
        else: # RESEARCH_LAB hoặc CHAT
            return {"next_step": "Researcher", "messages": []}

    except Exception as e:
        print(colored(f"⚠️ Supervisor Fallback: {e}", "red"))
        # Nếu bộ não bị lỗi, mặc định chuyển Researcher để tìm câu trả lời
        return {"next_step": "Researcher", "messages": []}
#  ---- Viết Code----
async def coder_node(state): # Chuyển sang async để chạy song song
    """
    Claude Coder Node - Parallel Execution & AST Validation
    """
    print(colored("[🚀 CODER V2] Parallel Ensemble Mode ACTIVATED", "green", attrs=["bold"]))
    
    # 1. SETUP CONTEXT
    errors = state.get("error_log", [])
    task_type = state.get("task_type", "general").lower()
    messages = state.get('messages', [])
    last_user_msg = messages[-1].content
    
    # An toàn: Tìm kiếm ký ức (Tránh lỗi nếu hàm search_memory chưa sẵn sàng)
    try:
        memory_context = search_memory("Tiêu chuẩn viết code Clean Code, SOLID")
    except:
        memory_context = "Tuân thủ PEP8, Clean Code và thêm comment giải thích."
    # error_context = self_heal_analyzer(errors)
    
    # 2. PROMPT STRATEGY (Smart Selection)
    base_prompt = get_claude_perfected_prompt(task_type, memory_context, str(errors), last_user_msg)
    # Chỉ chạy Ensemble nếu task khó hoặc đang fix lỗi
    use_ensemble = len(errors) > 0 or "complex" in task_type or "dự án" in last_user_msg.lower()
    prompts = [base_prompt]
    if use_ensemble:
        # Thêm 1 biến thể tối ưu hóa để so sánh
        prompts.append(base_prompt + "\n[DIRECTIVE]: OPTIMIZE for performance and brevity. Remove unnecessary comments.")
    # 3. PARALLEL EXECUTION (Tăng tốc độ gấp 3 lần)
    # ============================================================================
    print(colored(f"⚡ Running {len(prompts)} parallel chains...", "cyan"))
    # Chuẩn bị batch inputs
    batch_inputs = [[SystemMessage(content=p)] + messages for p in prompts]
    
    try:
        # --- LOGIC FALLBACK QUAN TRỌNG ---
        # Ưu tiên 1: CODER_PRIMARY (DeepSeek)
        # Ưu tiên 2: LLM_GPT4 (GPT-4 Turbo)
        # Ưu tiên 3: LLM_CLAUDE (Claude 3.5 Sonnet)
        
        fallbacks = []
        if LLM_GPT4: fallbacks.append(LLM_GPT4)
        if LLM_CLAUDE: fallbacks.append(LLM_CLAUDE)
        
        # Xác định Primary Chain
        primary_chain = CODER_PRIMARY if CODER_PRIMARY else (LLM_GPT4 if LLM_GPT4 else LLM_CLAUDE)
        
        if not primary_chain:
            raise Exception("CRITICAL: Không có API nào hoạt động!")

        # Kích hoạt Fallback
        if fallbacks and primary_chain != fallbacks[0]: 
            final_chain = primary_chain.with_fallbacks(fallbacks)
            print(colored(f"🛡️ Chain: {primary_chain.model_name} -> Fallbacks", "green"))
        else:
            final_chain = primary_chain

        # Thực thi
        responses = await final_chain.abatch(batch_inputs)
        
    except Exception as e:
        # Ghi log lỗi chi tiết trước khi fallback để CEO biết tại sao sập
        error_detail = f"Lỗi thực thi song song (Parallel Batch): {str(e)}"
        print(colored(f"🚨 {error_detail}", "red"))
        
        # Cập nhật error_log vào state trước khi thoát
        state["error_log"] = state.get("error_log", []) + [error_detail]
        
        return {"messages": [AIMessage(content="Hệ thống quá tải.")], "next_step": "FINISH"}

    # 4. VALIDATION & SCORING
    # ============================================================================
    valid_results = []
    for i, res in enumerate(responses):
        code = extract_code_block(res.content)
        if not code: continue
        
        is_ok, msg = real_syntax_validator(code, "python")
        score = 50 if is_ok else 0
        if len(code) > 30: score += 10
        if "# filename:" in code: score += 10
        
        valid_results.append({"code": code, "reply": res.content, "score": score, "error": msg, "variant": i})

    # 5. SELECT BEST CANDIDATE
    # ============================================================================
    if valid_results:
        # Lấy ứng viên có điểm cao nhất
        best_result = max(valid_results, key=lambda x: x['score'])
        
        # NGƯỠNG CHẤP NHẬN: 60 điểm (Đủ để chạy)
        # (Tôi hạ xuống 60 để hệ thống linh hoạt hơn, nhưng chỉ lưu bài mẫu khi đạt 80)
        if best_result['score'] >= 60: 
            print(colored(f"✅ SELECTED Variant {best_result['variant']} (Score: {best_result['score']})", "green"))
            
            # [TỰ HỌC]: Chỉ lưu những đoạn code chất lượng cao (>= 80)
            if best_result['score'] >= 80:
                try:
                    # Dùng hàm log chuẩn mới: log_training_data
                    # Input: User Prompt, Code AI, Score, Tên Model
                    log_training_data(
                        user_prompt=messages[-1].content,
                        best_code=best_result['code'],
                        score=best_result['score'],
                        model_used="3-Tier-Squad" 
                    )
                except: pass    
                # except Exception as e:
                #     # Nếu lỗi ghi file thì bỏ qua, không làm sập luồng chính
                #     print(colored(f"⚠️ Log Error: {e}", "yellow"))

            # TRẢ VỀ KẾT QUẢ THÀNH CÔNG
            return {
                "messages": [AIMessage(content=best_result['full_reply'])],
                "next_node": "Tester", # Chuyển sang Tester kiểm tra
                "error_log": []        # Xóa sạch lỗi cũ vì đã thành công
            }
        
        else:
            # TRƯỜNG HỢP: Code điểm thấp hoặc lỗi cú pháp
            print(colored(f"⚠️ [CODER] Variant tốt nhất chỉ đạt {best_result['score']}/100. Error: {best_result['error']}", "yellow"))
            
            # 1. Kiểm tra giới hạn thử lại (Max 3 lần để tránh lặp vô tận)
            if len(state.get("error_log", [])) >= 3:
                print(colored("🚨 [CODER] Đã thử 3 lần không được. Chuyển sang Fallback.", "red"))
                state["error_log"].append("Lỗi: AI không thể tự sửa code sau 3 lần thử.")
                
                # Gọi hàm fallback cuối cùng (Code thủ công hoặc báo lỗi)
                return ultimate_fallback(state, messages)

            # 2. Tạo phản hồi lỗi chi tiết để AI tự sửa
            error_feedback = (
                f"SYSTEM ALERT: Code bạn viết bị lỗi cú pháp hoặc vi phạm quy chuẩn.\n"
                f"- Error Details: {best_result['error']}\n"
                f"- Score: {best_result['score']}/100\n"
                f"ACTION: Hãy viết lại code mới, sửa triệt để lỗi trên."
            )
            
            # Trả về state để kích hoạt vòng lặp quay lại Coder
            return {
                "messages": [
                    AIMessage(content=best_result['code']), # Gửi lại code sai
                    HumanMessage(content=error_feedback)    # Kèm lời nhắc sửa
                ], 
                "error_log": state.get("error_log", []) + [f"Syntax Error: {best_result.get('error')}"],
                "next_step": "Coder" # Chỉ định rõ bước tiếp theo là quay lại Coder
            }

    # TRƯỜNG HỢP: Không có variant nào (Lỗi API hoặc Prompt bị chặn)
    error_msg = "Không có kết quả nào được tạo ra từ batch execution."
    print(colored(f"❌ [CODER] {error_msg}", "red"))
    state["error_log"] = state.get("error_log", []) + [error_msg]
    
    return ultimate_fallback(state, messages)

# ============================================================================
# NODE: TESTER (Kỹ sư Kiểm định Chất lượng - QA/QC)
# ============================================================================
def tester_node(state):
    """
    Agent Tester: Kiểm định cú pháp đa ngôn ngữ, quét lỗi bảo mật và tuân thủ quy chuẩn.
    """
    print(colored("[🧪 TESTER] Đang kiểm định chất lượng mã nguồn...", "yellow", attrs=["bold"]))
    
    messages = state.get("messages", [])
    last_ai_msg = messages[-1].content
    
    # 1. Trích xuất code block
    code_to_test = extract_code_block(last_ai_msg)
    
    if not code_to_test:
        print(colored("❌ [TESTER] Không tìm thấy khối code hợp lệ!", "red"))
        return {
            "error_log": state.get("error_log", []) + ["LỖI: Không tìm thấy khối code ```."],
            "next_step": "Coder"
        }

    is_valid = True
    feedback = []

    # 2. KIỂM ĐỊNH THEO NGÔN NGỮ
    
    # --- Trường hợp 1: Code Python ---
    if "def " in code_to_test or "import " in code_to_test:
        try:
            ast.parse(code_to_test)
            feedback.append("- Cú pháp Python: Đạt chuẩn.")
            
            # Kiểm tra bảo mật sơ bộ (Ví dụ: cấm dùng 'eval')
            if "eval(" in code_to_test or "os.system(" in code_to_test:
                is_valid = False
                feedback.append("- Bảo mật: Phát hiện hàm nguy hiểm (eval/system).")
                
        except SyntaxError as e:
            is_valid = False
            feedback.append(f"- Cú pháp Python: Lỗi tại dòng {e.lineno}: {e.msg}")

    # --- Trường hợp 2: Code C++ / Arduino (Hardware) ---
    elif "#include" in code_to_test or "void setup()" in code_to_test:
        # Kiểm tra đóng mở ngoặc đơn giản cho C++ (Vì Python không parse được C++)
        open_braces = code_to_test.count("{")
        close_braces = code_to_test.count("}")
        if open_braces != close_braces:
            is_valid = False
            feedback.append(f"- Cú pháp C++: Mất cân bằng dấu ngoặc ({open_braces} mở, {close_braces} đóng).")
        else:
            feedback.append("- Cú pháp C++: Kiểm tra cấu trúc đóng/mở đạt.")

    # 3. QUYẾT ĐỊNH HẬU KIỂM
    full_feedback = "\n".join(feedback)
    
    if is_valid:
        print(colored("✅ [TESTER] Mã nguồn đạt tiêu chuẩn chất lượng.", "green"))
        return {
            "error_log": [], # Clear log lỗi
            "next_step": "Supervisor"
        }
    else:
        print(colored(f"❌ [TESTER] Phát hiện vi phạm:\n{full_feedback}", "red"))
        error_msg = HumanMessage(content=(
            f"⚠️ BÁO CÁO KIỂM ĐỊNH THẤT BẠI:\n{full_feedback}\n\n"
            f"Vui lòng sửa lại mã nguồn, chú trọng vào các điểm vi phạm trên."
        ))
        return {
            "messages": [error_msg],
            "error_log": state.get("error_log", []) + [full_feedback],
            "next_step": "Coder"
        }
    
# ============================================================================
# NODE: HARDWARE (Kiến trúc sư Robotics & Hệ thống nhúng)
# ============================================================================
def hardware_node(state):
    """
    Agent Hardware Architect: Chuyên trách ESP32, Robotics và Hệ thống nhúng.
    Nâng cấp: Trích xuất BOM chuẩn cho Procurement và tối ưu hóa PINOUT.
    """
    print(colored("[🛠️ HARDWARE] Đang kiến trúc hệ thống nhúng...", "cyan", attrs=["bold"]))
    
    messages = state.get("messages", [])
    last_msg = messages[-1].content
    is_pure_hw = "[HARDWARE]" in last_msg # Nhận diện Tab Kỹ thuật

    prompt = (
        "Bạn là Hardware Architect cao cấp tại AI Corporation. "
        f"\nYÊU CẦU: {last_msg}"
        "\n\nCẤU TRÚC BÁO CÁO KỸ THUẬT:"
        "\n1. [DANH MỤC LINH KIỆN - BOM]: Liệt kê dạng bảng: Tên | Thông số | Số lượng."
        "\n2. [SƠ ĐỒ CHÂN - PINOUT]: Bảng kết nối chi tiết (VD: ESP32 GPIO21 -> LCD SDA)."
        "\n3. [FIRMWARE]: Code C++/Arduino tối ưu, có comment giải thích chuyên sâu."
        "\n4. [LƯU Ý VẬN HÀNH]: Cảnh báo dòng áp, tản nhiệt và nhiễu tín hiệu."
        "\n\nBẮT BUỘC: Không dùng emoji, chỉ dùng ký tự Latin/Tiếng Việt chuẩn."
    )
    
    try:
        # GPT-4o là lựa chọn số 1 cho việc tra cứu sơ đồ chân (Data Sheets)
        response = LLM_GPT4.invoke([
            SystemMessage(content=prompt),
            HumanMessage(content=last_msg)
        ])
        
        # ĐỊNH TUYẾN:
        # Nếu ở Tab Hardware -> FINISH (Hiện kết quả ngay)
        # Nếu ở luồng tự động -> Chuyển sang Procurement để báo giá linh kiện
        next_destination = "FINISH" if is_pure_hw else "Procurement"

        return {
            "messages": [AIMessage(content=f"🛠️ **[THIẾT KẾ KỸ THUẬT PHẦN CỨNG]**\n\n{response.content}")],
            "next_step": next_destination
        }
        
    except Exception as e:
        # 1. Ghi log chi tiết ra Terminal để CEO theo dõi lỗi vật lý
        error_detail = str(e)
        print(colored(f"🚨 [HARDWARE ERROR]: {error_detail}", "red", attrs=["bold"]))
        
        # 2. Trả về State chuẩn: 
        # - messages: Phải là một LIST chứa đối tượng Message
        # - next_step: Phải là một CHUỖI (String) định danh Node tiếp theo
        return {
            "messages": [AIMessage(content=f"❌ **HỆ THỐNG CẢNH BÁO HARDWARE**:\n\nĐã xảy ra sự cố kỹ thuật: `{error_detail}`")], 
            "next_step": "FINISH" 
        }
#  ---- Vẽ 3D Plotly----
def engineering_node(state):
    """
    Agent CTO/Engineer: Thiết kế mô hình 3D bằng Python Plotly.
    Đã nâng cấp: Đảm bảo mã nguồn chuẩn để Dashboard thực thi vẽ 3D.
    """
    print(colored("[⚙️ ENGINEERING] Đang thiết kế cấu trúc 3D...", "blue", attrs=["bold"]))
    
    messages = state.get("messages", [])
    last_msg = messages[-1].content
    is_pure_eng = "[ENGINEERING]" in last_msg

    # 1. Prompt ép AI viết code sạch, không giải thích thừa
    prompt = (
        "Bạn là Kỹ sư Thiết kế 3D chuyên nghiệp. "
        "\nNHIỆM VỤ: Viết code Python sử dụng plotly.graph_objects để tạo mô hình 3D."
        "\n\nYÊU CẦU KỸ THUẬT:"
        "\n- Chỉ trả về duy nhất CODE BLOCK Python trong dấu ```python."
        "\n- Code phải tạo ra đối tượng tên là 'fig'."
        "\n- Phải bao gồm dữ liệu tọa độ (x, y, z) chi tiết cho mô hình."
        "\n- Nếu là Robot, hãy vẽ rõ các khớp nối và cánh tay."
        "\n- KHÔNG giải thích, KHÔNG nhập văn bản ngoài code."
    )

    try:
        # 2. Sử dụng Claude 3.5 Sonnet (Đỉnh cao về viết code hình học)
        response = LLM_CLAUDE.invoke([
            SystemMessage(content=prompt),
            HumanMessage(content=f"Yêu cầu thiết kế: {last_msg}")
        ])
        
        # 3. Định tuyến
        next_destination = "FINISH" if is_pure_eng else "Procurement"

        return {
            "messages": [AIMessage(content=f"⚙️ **[BẢN THIẾT KẾ 3D HỆ THỐNG]**\n\n{response.content}")],
            "next_step": next_destination
        }
        
    except Exception as e:
        # 1. Ghi log lỗi chi tiết ra Terminal với màu đỏ đậm để dễ nhận diện
        error_detail = str(e)
        print(colored(f"🚨 [ENGINEERING ERROR]: {error_detail}", "red", attrs=["bold"]))
        
        # 2. Trả về State chuẩn cho LangGraph:
        # - messages: BẮT BUỘC là một list chứa đối tượng Message (không được gửi dict rỗng)
        # - next_step: BẮT BUỘC là một chuỗi (String) để tránh lỗi băm dữ liệu
        return {
            "messages": [AIMessage(content=f"❌ **LỖI THIẾT KẾ KỸ THUẬT**:\n\nHệ thống gặp sự cố khi dựng mô hình: `{error_detail}`")], 
            "next_step": "FINISH" 
        }
    
def publisher_node(state):
    """
    Agent Publisher: Tổng hợp dữ liệu từ tất cả các Agent để xuất bản hồ sơ dự án.
    """
    print(colored("[📜 PUBLISHER] Đang tổng hợp hồ sơ dự án cuối cùng...", "green", attrs=["bold"]))
    
    messages = state.get("messages", [])
    
    # 1. PHÂN LOẠI DỮ LIỆU TỰ ĐỘNG
    research_report = ""
    investment_plan = ""
    technical_specs = ""
    creative_content = ""
    images = []

    for msg in messages:
        content = msg.content
        if "🔍 [BÁO CÁO NGHIÊN CỨU]" in content: research_report = content
        if "💰 [HỒ SƠ THẨM ĐỊNH ĐẦU TƯ]" in content: investment_plan = content
        if "⚙️ [BẢN THIẾT KẾ 3D]" in content: technical_specs = content
        if "🖋️ [TÁC PHẨM SÁNG TÁC]" in content: creative_content = content
        if "![Ảnh minh họa]" in content:
            # Trích xuất URL ảnh
            urls = [line for line in content.split('\n') if "https://" in line]
            images.extend(urls)

    # 2. TỔNG HỢP PROMPT XUẤT BẢN
    publish_prompt = (
        "Bạn là Chuyên gia trình bày văn bản cấp cao. Hãy tổng hợp các dữ liệu trên thành một "
        "Báo cáo Dự án hoàn chỉnh, chuyên nghiệp. Sử dụng tiêu đề, mục lục và định dạng Markdown chuẩn."
        "\nThứ tự: 1. Tổng quan -> 2. Thị trường -> 3. Tài chính -> 4. Kỹ thuật -> 5. Phụ lục hình ảnh."
    )

    response = LLM_GEMINI_LOGIC.invoke([
        SystemMessage(content=publish_prompt),
        HumanMessage(content=f"Dữ liệu gom được:\n{research_report}\n{investment_plan}\n{technical_specs}\n{creative_content}")
    ])

    return {
        "messages": [AIMessage(content=f"📜 **[HỒ SƠ DỰ ÁN TỔNG THỂ - FINAL]**\n\n{response.content}")],
        "next_step": "FINISH"
    }
# ============================================================================
# NODE: IoT ENGINEER (Kỹ sư Vận hành & Kết nối thiết bị)
# ============================================================================
def iot_node(state):
    """
    Agent IoT: Kết hợp Lập trình Firmware (Thiết kế) và Thực thi lệnh (Vận hành).
    """
    print(colored("[🤖 IoT ENGINEER] Đang xử lý giao thức và thiết bị...", "magenta", attrs=["bold"]))
    
    messages = state.get("messages", [])
    last_msg = messages[-1].content
    is_pure_iot = "[IOT]" in last_msg

    # 1. KIỂM TRA NGỮ CẢNH: Đây là lệnh điều khiển (Vận hành) hay yêu cầu viết code (Thiết kế)?
    is_command = any(word in last_msg.upper() for word in ["BẬT", "TẮT", "TURN", "CONTROL", "CHẠY"])

    if is_command:
        # --- NHÁNH 1: VẬN HÀNH THIẾT BỊ THẬT ---
        analysis_prompt = f"Trích xuất lệnh điều khiển từ: '{last_msg}'. Chỉ trả về mã lệnh Uppercase."
        command_code = LLM_GPT4.invoke([SystemMessage(content=analysis_prompt)]).content.strip()
        
        try:
            # 1. Gọi tool hardware_controller để ra lệnh cho thiết bị thực tế
            hardware_response = hardware_controller.invoke(command_code)
            report = (f"📡 **[KẾT QUẢ VẬN HÀNH]**\n\n- Mã lệnh: `{command_code}`\n- Trạng thái: {hardware_response}")
            
            # Nếu chạy Tab IOT riêng biệt -> Kết thúc. Nếu chạy luồng tự động -> Về Supervisor báo cáo.
            return {
                "messages": [AIMessage(content=report)], 
                "next_step": "FINISH" if is_pure_iot else "Supervisor"
            }
            
        except Exception as e:
            # 2. Xử lý lỗi kết nối hoặc thực thi thiết bị
            error_detail = str(e)
            print(colored(f"🚨 [IOT HARDWARE ERROR]: {error_detail}", "red", attrs=["bold"]))
            
            # Trả về AIMessage chuẩn để Dashboard hiển thị đúng ID: IoT_Engineer
            return {
                "messages": [AIMessage(content=f"❌ **LỖI KẾT NỐI THIẾT BỊ**:\n\nKhông thể thực thi lệnh `{command_code}`. \nChi tiết: `{error_detail}`")], 
                "next_step": "Supervisor" # Quay về để Supervisor ra lệnh kiểm tra lại hoặc đổi phương án
            }
    else:
        # --- NHÁNH 2: THIẾT KẾ FIRMWARE (Dành cho dự án mới) ---
        # Lấy bản vẽ Pinout từ Hardware Node nếu có
        hw_context = next((m.content for m in reversed(messages) if "🛠️" in m.content), "Chưa có sơ đồ chân.")
        
        design_prompt = (
            "Bạn là Kỹ sư Firmware IoT. Hãy viết code C++/Arduino điều khiển hệ thống dựa trên sơ đồ chân sau."
            f"\nSơ đồ: {hw_context}"
            "\nYêu cầu: Viết code có kết nối WiFi/MQTT và quản lý lỗi kết nối."
        )
        
        response = LLM_CLAUDE.invoke([SystemMessage(content=design_prompt), HumanMessage(content=last_msg)])
        
        return {
            "messages": [AIMessage(content=f"📡 **[FIRMWARE & GIAO THỨC ĐIỀU KHIỂN]**\n\n{response.content}")],
            "next_step": "FINISH" if is_pure_iot else "Supervisor"
        }
# ============================================================================
# NODE: PROCUREMENT (Trưởng phòng Thu mua & Quản lý Chuỗi cung ứng)
# ============================================================================
BUYER_PROFILE = {
    "address": "Phan Thiết, Bình Thuận, Việt Nam",
    "delivery_method": "Fast Shipping",
    "accounts": ["Shopee_API_Key", "Taobao_Token", "Mouser_ID"]
}
def procurement_node(state):
    """
    Agent Procurement: Tối ưu hóa chuỗi cung ứng dựa trên vị trí thực tế của CEO.
    """
    print(colored("[🛒 PROCUREMENT] Đang tối ưu hóa lộ trình hàng hóa về Phan Thiết...", "yellow", attrs=["bold"]))
    
    # 1. Load hồ sơ mua hàng (Mockup)
    buyer_config = BUYER_PROFILE # Lấy từ file cấu hình trên
    
    messages = state.get("messages", [])
    hw_report = next((m.content for m in reversed(messages) if "🛠️" in m.content), "Không tìm thấy danh mục linh kiện.")

    # 2. Xây dựng lệnh truy vấn chuyên sâu
    prompt = (
        "Bạn là Chuyên gia Logisitics và Thu mua."
        f"\nĐỊA CHỈ NHẬN: {buyer_config['address']}"
        f"\nDANH MỤC: {hw_report}"
        "\n\nNHIỆM VỤ:"
        "\n1. TÌM GIÁ: Tra cứu giá thực tế năm 2026 trên Mouser, Digikey và Shopee."
        "\n2. TÍNH PHÍ VẬN CHUYỂN: Ước tính phí ship và thuế nhập khẩu về Việt Nam."
        "\n3. LẬP GIỎ HÀNG: Tạo danh sách link sản phẩm sẵn sàng để thanh toán."
    )

    # Sử dụng Perplexity để check giá thực tế
    response = LLM_PERPLEXITY.invoke([SystemMessage(content=prompt)])

    return {
        "messages": [AIMessage(content=f"🛒 **[PHIẾU ĐỀ XUẤT MUA SẮM & VẬN CHUYỂN]**\n\n{response.content}")],
        "next_step": "Investment" # Chuyển sang Tài chính để CEO duyệt chi
    }
# ============================================================================
# NODE: RESEARCHER (Chuyên gia Phân tích Thị trường & Đối thủ)
# ============================================================================
def researcher_node(state):
    """
    Agent Researcher: Chuyên gia phân tích thị trường 2026.
    Nâng cấp: Tự động nhận diện Tag ngữ cảnh để quyết định hành động tiếp theo.
    """
    # 1. [BẤM GIỜ] Bắt đầu tính giờ làm việc
    start_time = time.time() 
    
    print(colored("[🔍 RESEARCHER] Đang thực thi nhiệm vụ thám mã thị trường...", "cyan", attrs=["bold"]))
    # 2. [FIX QUAN TRỌNG] LỌC TÌM LỆNH CỦA CEO (HUMAN)
    messages = state.get("messages", [])
    
    # Mặc định lấy tin cuối, nhưng sẽ ưu tiên tìm tin nhắn của NGƯỜI (Human) gần nhất
    # Để tránh lấy nhầm tin nhắn điều phối của hệ thống
    target_msg_content = ""
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            target_msg_content = msg.content
            break
            
    if not target_msg_content:
        target_msg_content = messages[-1].content # Fallback nếu không tìm thấy
    is_pure_research = "[RESEARCH]" in target_msg_content
    clean_query = target_msg_content.replace("[RESEARCH]", "").replace("[ORCHESTRATOR]", "").strip()
    # 2. Xây dựng Prompt Siêu Cấu Trúc (Sử dụng 4 cột trụ)
    search_prompt = (
        f"Nhiệm vụ: Phân tích thị trường 2026 cho: '{clean_query}'."
        "\n\nYÊU CẦU BÁO CÁO 4 CỘT TRỤ:"
        "\n1. [DỮ LIỆU VĨ MÔ]: Tình hình thị trường và công nghệ mới nhất."
        "\n2. [BIẾN ĐỘNG THỰC TẾ]: Xu hướng tiêu dùng và 'nỗi đau' khách hàng."
        "\n3. [ĐỐI THỦ TRỰC DIỆN]: Liệt kê 3 đối thủ và lợi thế của họ."
        "\n4. [CƠ HỘI CHO CEO]: Insight quan trọng và dự báo 12 tháng tới."
        "\n\nĐịnh dạng: Markdown chuyên nghiệp, có bảng so sánh."
    )
    
    try:
        # 3. Triệu hồi Perplexity
        response = LLM_PERPLEXITY.invoke([
            SystemMessage(content="Bạn là Chief Research Officer. Chỉ trả về dữ liệu thực tế 2026, KHÔNG HTML."),
            HumanMessage(content=search_prompt)
        ])
        raw_res = response.content

        # --- TẦNG PHÒNG THỦ 1: CHẶN HTML & LỖI 401 ---
        if any(x in raw_res.lower() for x in ["<html>", "401 authorization", "cloudflare"]):
            return {
                "messages": [AIMessage(content="🚨 [HỆ THỐNG] Lỗi kết nối nguồn tin (API 401). CEO hãy kiểm tra lại Key Perplexity.")],
                "next_step": "FINISH" # Dừng ngay lập tức để bảo vệ tài nguyên
            }

        # --- TẦNG PHÒNG THỦ 2: XỬ LÝ KẾT QUẢ THÀNH CÔNG ---
        report_content = f"🔍 **[BÁO CÁO CRO - {clean_query.upper()}]**\n\n{raw_res}"
        if is_pure_research:
            # Nếu CEO chỉ muốn nghiên cứu (Tab Research), kết thúc tại đây.
            next_destination = "Secretary"
        else:
            # Thay vì st.session_state, ta dùng task_type được Dashboard gửi qua Server
            if state.get("task_type") == "dynamic":
                next_destination = "Orchestrator"
            else:
                next_destination = "Supervisor"

        # ============================================================
        # 🟢 [CHÈN ĐOẠN NÀY VÀO] GHI SỔ CÔNG VIỆC
        # ============================================================
        try:
            log_work_to_db(
                agent="Researcher",
                task=clean_query,   # Đề bài sếp giao
                result=raw_res,     # Kết quả tìm được
                tool="Perplexity",  # Súng đã dùng
                start_time=start_time # Thời gian bắt đầu
            )
        except Exception as log_err:
            print(colored(f"⚠️ Lỗi ghi log kế toán: {log_err}", "yellow"))


        return {
            "messages": [AIMessage(content=report_content)],
            "next_step": next_destination,
            "current_agent": "Researcher" # Định danh để Orchestrator biết ai vừa hoàn thành báo cáo
        }

    except Exception as e:
        # TẦNG PHÒNG THỦ 3: NGOẠI LỆ
        print(colored(f"Lỗi Researcher: {e}", "red"))
        return {
            "messages": [AIMessage(content=f"⚠️ Trục trặc kỹ thuật khi quét dữ liệu: {str(e)}")],
            "next_step": "FINISH" 
        }

#  ---- Tài Chính----
def investment_node(state):
    """
    Agent CFO: Thẩm định tài chính và ROI.
    Đã nâng cấp: Tự động ngắt luồng (FINISH) nếu ở chế độ chuyên biệt.
    """
    print(colored("[💰 INVESTMENT] Đang thẩm định tài chính dự án...", "green", attrs=["bold"]))
    
    messages = state.get("messages", [])
    last_msg = messages[-1].content
    is_pure_invest = "[INVEST]" in last_msg
    
    # Lấy 3 tin nhắn gần nhất để có đủ ngữ cảnh (Báo cáo Researcher + Coder...)
    context = "\n".join([m.content for m in messages[-3:]])
    
    prompt = (
        "Bạn là Giám đốc Tài chính (CFO) của AI Corporation. "
        "\nNHIỆM VỤ: Lập bảng phân tích CAPEX, OPEX, ROI và rủi ro tài chính."
        "\n\nYÊU CẦU:"
        "\n- Trình bày bảng Markdown sạch sẽ."
        "\n- Kết luận rõ ràng: 'ĐẦU TƯ', 'THEO DÕI' hoặc 'LOẠI BỎ'."
    )
    
    try:
        # Ưu tiên GPT-4 cho tính toán con số để tránh sai sót logic
        response = LLM_MAIN.invoke([
            SystemMessage(content=prompt), 
            HumanMessage(content=f"Dữ liệu dự án: {context}")
        ])
        
        # Nếu CEO chọn Tab INVEST -> Trả kết quả và FINISH (Nhanh)
        # Nếu đang chạy luồng tự động -> Quay lại Supervisor
        next_destination = "FINISH" if is_pure_invest else "Supervisor"

        return {
            "messages": [AIMessage(content=f"💰 **[HỒ SƠ THẨM ĐỊNH ĐẦU TƯ]**\n\n{response.content}")],
            "next_step": next_destination
        }
    except Exception as e:
        return {
            "messages": [AIMessage(content=f"⚠️ Sự cố tài chính: {str(e)}")],
            "next_step": "FINISH"
        }

#  ---- Pháp lý----
def legal_node(state):
    """
    Agent Legal (CLO): Rà soát toàn bộ dự án trước khi xuất bản.
    Đã nâng cấp: Đọc toàn bộ lịch sử để phát hiện rủi ro xuyên suốt.
    """
    print(colored("[⚖️ LEGAL] Luật sư đang rà soát toàn bộ hồ sơ dự án...", "red", attrs=["bold"]))
    
    messages = state.get("messages", [])
    last_msg = messages[-1].content
    is_pure_legal = "[LEGAL]" in last_msg
    
    # 1. TỔNG HỢP HỒ SƠ: Luật sư phải đọc hết các "cam kết" của Agent khác
    # Gom 10-15 tin nhắn để thấy toàn bộ luồng từ Kỹ thuật đến Marketing
    full_project_context = "\n".join([f"[{m.type.upper()}]: {m.content[:300]}..." for m in messages[-15:]])

    prompt = (
        "Bạn là Giám đốc Pháp lý (CLO) của AI Corporation. "
        "\nNHIỆM VỤ: Thẩm định pháp lý và Quản trị rủi ro dựa trên HỒ SƠ DỰ ÁN được cung cấp."
        "\n\nYÊU CẦU CHIẾN LƯỢC:"
        "\n1. RÀ SOÁT IP: Kiểm tra bản quyền hình ảnh (Artist) và mã nguồn (Coder)."
        "\n2. TUÂN THỦ: Đối chiếu với Luật An ninh mạng VN và GDPR."
        "\n3. SOẠN THẢO: Đưa ra khung Điều khoản sử dụng (ToS) và NDA mẫu cho dự án."
        "\n4. KẾT LUẬN: Ghi rõ 'AN TOÀN' hoặc 'CẢNH BÁO NGUY HIỂM'."
    )
    
    try:
        # Sử dụng GPT-4o để có tư duy lập luận pháp luật sắc bén nhất
        response = LLM_GPT4.invoke([
            SystemMessage(content=prompt),
            HumanMessage(content=f"HỒ SƠ DỰ ÁN CẦN THẨM ĐỊNH:\n{full_project_context}\n\nYÊU CẦU BỔ SUNG: {last_msg}")
        ])
        
        # Nếu CEO chọn Tab Legal riêng biệt thì kết thúc luôn
        next_destination = "FINISH" if is_pure_legal else "Supervisor"

        return {
            "messages": [AIMessage(content=f"⚖️ **[BÁO CÁO PHÁP LÝ & RỦI RO CHI TIẾT]**\n\n{response.content}")],
            "next_step": next_destination
        }

    except Exception as e:
        # 1. Ghi log lỗi pháp lý ra Terminal để CEO giám sát rủi ro hệ thống
        error_detail = str(e)
        print(colored(f"🚨 [LEGAL CRITICAL ERROR]: {error_detail}", "red", attrs=["bold"]))
        
        # 2. Trả về State chuẩn cho LangGraph
        # Đảm bảo next_step là "FINISH" để ngắt luồng an toàn khi có sự cố pháp lý
        return {
            "messages": [AIMessage(content=f"❌ **CẢNH BÁO PHÁP LÝ KHẨN CẤP**:\n\nQuá trình rà soát bị gián đoạn: `{error_detail}`\n\nKhuyến nghị: CEO kiểm tra lại các điều khoản đầu vào.")], 
            "next_step": "FINISH" 
        }
#  ---- Nhân Sự ----
def hr_orchestrator_node(state):
    """
    Agent HR - Bộ điều phối nhân sự & quy trình:
    Kiểm tra xem CEO có thiết lập kịch bản tự động hay không.
    """
    print(colored("[👥 HR ORCHESTRATOR] Đang kiểm soát luồng vận hành...", "cyan", attrs=["bold"]))
    
    # 1. Kiểm tra xem có bản đồ quy trình (Workflow Map) nào được CEO vẽ không
    workflow_script = state.get("custom_workflow", None) 
    
    if workflow_script:
        # --- CHẾ ĐỘ TỰ ĐỘNG (DỰA TRÊN THIẾT LẬP KÉO THẢ) ---
        current_step = state.get("current_step_index", 0)
        target_node = workflow_script[current_step]
        
        print(colored(f"--> Theo kịch bản CEO: Chuyển sang {target_node}", "green"))
        
        # Báo cáo kết quả chặng trước và xin ý kiến duyệt
        return {
            "messages": [AIMessage(content=f"✅ Giai đoạn {current_step} hoàn tất. Chờ CEO phê duyệt để sang {target_node}.")],
            "next_step": target_node,
            "current_step_index": current_step + 1
        }
    else:
        # --- CHẾ ĐỘ MẶC ĐỊNH (AI TỰ SUY LUẬN) ---
        print(colored("--> Chế độ tự động: AI đang điều phối theo ngữ cảnh...", "white"))
        # Gọi lại logic Supervisor cũ của ngài
        return {"next_step": "Supervisor"}

def secretary_node(state):
    """
    SECRETARY V3: COMMUNICATOR - CẦU NỐI THÔNG MINH
    Biết cách diễn đạt lại kết quả từ các bộ phận khô khan (Coder, Researcher) 
    thành ngôn ngữ con người dễ hiểu cho CEO.
    """
    print(colored("[🗣️ COMMUNICATOR] Đang biên tập lại nội dung...", "magenta", attrs=["bold"]))
    
    messages = state.get("messages", [])
    last_agent = state.get("current_agent", "Unknown")
    
    # Lấy toàn bộ ngữ cảnh để hiểu chuyện gì vừa xảy ra
    context = "\n".join([f"{m.type}: {m.content}" for m in messages[-3:]])

    # Prompt dạy Thư ký cách nói chuyện
    prompt = (
        "Bạn là Trợ lý Cá nhân Thông minh của CEO. Các bộ phận chuyên môn (Coder, Artist...) vừa gửi kết quả lên.\n"
        "Nhiệm vụ của bạn: DIỄN ĐẠT LẠI kết quả đó một cách tự nhiên, chuyên nghiệp.\n"
        "QUY TẮC:"
        "\n1. Nếu có HÌNH ẢNH/CODE: Phải hiển thị rõ ràng (Giữ nguyên link/block code)."
        "\n2. Nếu là LỜI NÓI: Hãy tóm tắt lại ngắn gọn, dùng giọng văn đối thoại ('Thưa CEO', 'Tôi đã hoàn thành...')."
        "\n3. KHÔNG báo cáo máy móc kiểu 'Bước 1, Bước 2'. Hãy nói như người với người."
        f"\n\nNGỮ CẢNH VỪA QUA:\n{context}"
    )

    try:
        response = LLM_GEMINI_VISION.invoke([SystemMessage(content=prompt)])
        
        # Ghi log (Vẫn giữ chức năng lưu trữ ngầm)
        with open(f"Chat_Log_{int(time.time())}.txt", "w", encoding="utf-8") as f:
            f.write(response.content)

        return {
            "messages": [AIMessage(content=response.content)],
            "next_step": "FINISH"
        }
    except:
        return {"next_step": "FINISH"}
# ============================================================================
# NODE: MARKETING NODE (Giám đốc Marketing - CMO)
# ============================================================================
def marketing_node(state):
    """
    Agent CMO: Chuyên gia Marketing và Tăng trưởng.
    Đã nâng cấp: Tự động đề xuất Visual Prompt cho Artist để thiết kế ảnh quảng cáo.
    """
    print(colored("[📢 MARKETING] Đang lập chiến dịch quảng bá bùng nổ...", "yellow", attrs=["bold"]))
    
    messages = state.get("messages", [])
    last_msg = messages[-1].content
    is_pure_mkt = "[MARKETING]" in last_msg
    
    # Lấy ngữ cảnh sâu từ kỹ thuật và tài chính để viết bài có sức thuyết phục
    project_context = "\n".join([m.content for m in messages[-5:]])
    
    prompt = (
        "Bạn là Giám đốc Marketing (CMO) của AI Corporation. "
        "\nNHIỆM VỤ: Xây dựng bộ nội dung quảng bá đa kênh dựa trên thành phẩm kỹ thuật."
        "\n\nYÊU CẦU CHIẾN LƯỢC:"
        "\n- [INSIGHT]: Dùng dữ liệu kỹ thuật để nêu bật lợi ích cho người dùng."
        "\n- [FACEBOOK]: Mô hình PAS, phong cách thân thiện."
        "\n- [LINKEDIN]: Mô hình chuyên gia, tập trung vào ROI và tính bền vững."
        "\n- [VISUAL PROMPT]: QUAN TRỌNG! Đưa ra 2 mô tả hình ảnh (tiếng Anh) để Agent Artist vẽ ảnh quảng cáo."
    )

    try:
        response = LLM_GPT4.invoke([
            SystemMessage(content=prompt),
            HumanMessage(content=f"Dữ liệu sản phẩm:\n{project_context}")
        ])

        # ĐỊNH TUYẾN THÔNG MINH:
        # Nếu CEO cần ảnh minh họa ngay, có thể chuyển sang Artist
        # Nếu không, FINISH để hiện nội dung.
        next_destination = "FINISH" if is_pure_mkt else "Supervisor"

        return {
            "messages": [AIMessage(content=f"📢 **[CHIẾN DỊCH MARKETING ĐA KÊNH]**\n\n{response.content}")],
            "next_step": next_destination
        }
        
    except Exception as e:
        # 1. Ghi log lỗi Marketing ra Terminal để CEO theo dõi hiệu suất chiến dịch
        error_detail = str(e)
        print(colored(f"🚨 [MARKETING CRITICAL ERROR]: {error_detail}", "red", attrs=["bold"]))
        
        # 2. Trả về State chuẩn cho LangGraph
        # Đảm bảo messages là LIST và next_step là STRING "FINISH"
        return {
            "messages": [AIMessage(content=f"❌ **SỰ CỐ CHIẾN DỊCH MARKETING**:\n\nQuá trình lập kế hoạch bị gián đoạn: `{error_detail}`\n\nKhuyến nghị: CEO hãy kiểm tra lại yêu cầu mục tiêu hoặc ngân sách.")], 
            "next_step": "FINISH" 
        }
#  ---- Vẽ Thiết Kế----
def artist_node(state):
    """
    ARTIST NODE V2 (REAL): Vẽ tranh thật bằng DALL-E 3 HD.
    """
    print(colored("\n[🎨 ARTIST] Đang khởi động Studio DALL-E 3 HD...", "blue", attrs=["bold"]))
    
    messages = state.get("messages", [])
    # Lấy đoạn văn mà CEO muốn minh họa
    last_msg_content = messages[-1].content
    
    # --- 1. TRÍCH XUẤT NỘI DUNG (Hỗ trợ cả 2 kiểu) ---
    # Kiểu 1: Có dùng dấu """ (Chuẩn chỉ)
    if '"""' in last_msg_content:
        start_idx = last_msg_content.find("\"\"\"") + 3
        end_idx = last_msg_content.rfind("\"\"\"")
        text_to_illustrate = last_msg_content[start_idx:end_idx].strip()
    # Kiểu 2: Nói tự nhiên (VD: "Vẽ con mèo") - Sơ cua
    else:
        # Loại bỏ các tag hệ thống nếu có
        text_to_illustrate = last_msg_content.replace("[ARTIST]", "").strip()

    # Kiểm tra lại lần cuối
    if not text_to_illustrate or len(text_to_illustrate) < 5:
        print(colored("🚫 [ARTIST] Không nhận được nội dung đủ để vẽ.", "red"))
        return {
            "messages": [AIMessage(content="🚫 Họa sĩ cần mô tả chi tiết hơn để vẽ. Vui lòng thử lại.")], 
            "next_step": "FINISH" 
        }

    # --- 2. GPT-4: KỸ SƯ PROMPT (Prompt Engineering) ---
    # Biến yêu cầu sơ sài thành Prompt nghệ thuật chi tiết
    analysis_prompt = (
        "Bạn là Giám đốc Nghệ thuật (Art Director). Nhiệm vụ: Tạo Image Prompt cho DALL-E 3.\n"
        f"YÊU CẦU GỐC: \"{text_to_illustrate}\"\n\n"
        "HÃY TRẢ VỀ ĐÚNG ĐỊNH DẠNG JSON SAU (Không thêm lời dẫn):\n"
        "```json\n"
        "{\n"
        "  \"style\": \"Tên phong cách nghệ thuật phù hợp nhất (Ví dụ: Cyberpunk, Studio Ghibli, Photorealistic, Oil Painting...)\",\n"
        "  \"prompt\": \"Mô tả chi tiết hình ảnh bằng tiếng Anh, tập trung vào ánh sáng, bố cục, chi tiết, cảm xúc. Tối đa 70 từ.\"\n"
        "}\n"
        "```"
    )

    try:
        # Gọi GPT-4 để lấy prompt xịn
        analysis_response = LLM_GEMINI_VISION.invoke([SystemMessage(content="JSON mode."), HumanMessage(content=analysis_prompt)])
        
        # Làm sạch chuỗi JSON (đề phòng GPT thêm markdown)
        json_str = analysis_response.content.replace("```json", "").replace("```", "").strip()
        analysis_data = json.loads(json_str)
        
        design_style = analysis_data.get('style', 'Cinematic')
        visual_prompt = analysis_data.get('prompt', text_to_illustrate[:100])

        # Tạo prompt cuối cùng
        full_image_prompt = f"{visual_prompt}, {design_style} style. High resolution, highly detailed, masterpiece."
        print(colored(f"--> Phong cách: {design_style}", "cyan"))
        print(colored(f"--> Prompt vẽ: {full_image_prompt[:100]}...", "white"))
            
        # --- 3. GỌI DALL-E 3 VẼ TRANH THẬT (QUAN TRỌNG NHẤT) ---
        print(colored("⏳ Đang gửi yêu cầu đến máy chủ OpenAI DALL-E 3 (Chờ 15-30s)...", "yellow"))
        
        # Khởi tạo công cụ vẽ HD
        dalle_tool = DallEAPIWrapper(
            model="dall-e-3",
            size="1024x1024",
            quality="hd" # Chất lượng cao nhất
        )
        
        # Thực thi vẽ (Có thể tốn 15-30 giây)
        image_url = dalle_tool.run(full_image_prompt)
        
        print(colored(f"✅ [ART COMPLETE]: Ảnh đã sẵn sàng!", "green"))

        # --- 4. TRẢ KẾT QUẢ NHANH (FAST TRACK) ---
        # Trả về FINISH ngay để hiện ảnh, không qua Thư ký nữa.
        # Sử dụng Markdown chuẩn để Dashboard hiển thị ảnh.
        
        final_content = (
            f"🎨 **TÁC PHẨM HOÀN THIỆN:**\n\n"
            f"![AI Art Generation]({image_url})\n\n"
            f"*(Phong cách: {design_style})*"
        )

        return {
            "messages": [AIMessage(content=final_content)],
            "next_step": "FINISH" # Kết thúc ngay
        }

    # --- XỬ LÝ LỖI ---
    except json.JSONDecodeError:
        print(colored("❌ Lỗi: GPT-4 không trả về JSON hợp lệ.", "red"))
        return {"messages": [AIMessage(content="⚠️ Lỗi phân tích yêu cầu vẽ tranh.")], "next_step": "FINISH"}
    except Exception as e:
        error_detail = str(e)
        print(colored(f"❌ LỖI VẼ TRANH (DALL-E/API): {error_detail}", "red"))
        # Thông báo lỗi rõ ràng cho CEO (Ví dụ: Hết tiền, Vi phạm chính sách nội dung...)
        return {
            "messages": [AIMessage(content=f"⚠️ Không thể tạo ảnh lúc này. Nguyên nhân: {error_detail}")], 
            "next_step": "FINISH"
        }
# ============================================================================
# NODE: STORYTELLER (Nhà văn & Biên kịch chuyên nghiệp)

# ============================================================================
def storyteller_node(state):
    print(colored("[✍️ STORYTELLER] Đang xây dựng thế giới và cốt truyện...", "cyan", attrs=["bold"]))
    
    messages = state.get("messages", [])
    # Lấy log lỗi nếu có để điều chỉnh văn phong
    errors = state.get("error_log", [])
    
    last_msg = messages[-1].content
    
    # 1. PHÂN TÍCH NHU CẦU
    is_continue = "[CONTINUE]" in last_msg.upper()
    clean_query = last_msg.replace("[STORY]", "").replace("[CONTINUE]", "").strip()

    # 2. TRÍ NHỚ MẠCH TRUYỆN (Thay thế st.session_state)
    # Chúng ta lấy bối cảnh từ tin nhắn AIMessage gần nhất trong lịch sử hội thoại của Graph
    previous_full_story_content = ""
    if is_continue:
        for m in reversed(messages):
            if isinstance(m, AIMessage) and len(m.content) > 100:
                previous_full_story_content = m.content
                break
        
        if previous_full_story_content:
            # Lấy đoạn kết để AI viết nối tiếp không bị lặp
            context_tail = previous_full_story_content[-1000:]
            print(colored(f"📜 Đã tìm thấy mạch truyện cũ, đang nối tiếp...", "yellow"))
            previous_full_story_content = context_tail

    # 3. THIẾT LẬP PROMPT CHIẾN THUẬT
    prompt = (
        "Bạn là Nhà văn Best-seller và Biên kịch xuất sắc. "
        "\nNHIỆM VỤ: Sáng tác nội dung có chiều sâu, lôi cuốn."
        "\n\nNGUYÊN TẮC VÀNG:"
        + (f"\n- MẠCH TRUYỆN TRƯỚC: '{previous_full_story_content}' (Hãy viết tiếp từ đây, không chào hỏi lại)." if previous_full_story_content else "\n- ĐÂY LÀ KHỞI ĐẦU: Hãy tạo một mở đầu ấn tượng.") +
        "\n- CẤU TRÚC: Show, Don't Tell. Sử dụng nhiều từ ngữ gợi hình, gợi cảm."
        "\n- HÌNH ẢNH: Sau mỗi phân đoạn cao trào, hãy chèn một Visual Prompt tiếng Anh trong ngoặc vuông [Visual: ...]."
    )

    try:
        # Lựa chọn Model: Ưu tiên Claude cho sáng tạo văn học
        selected_llm = LLM_CLAUDE if 'LLM_CLAUDE' in globals() else LLM_GPT4
        
        response = selected_llm.invoke([
            SystemMessage(content=prompt),
            HumanMessage(content=clean_query)
        ])

        # ĐỊNH TUYẾN: Thường sau khi kể chuyện sẽ kết thúc để CEO đọc, hoặc qua Artist để vẽ
        return {
            "messages": [AIMessage(content=response.content)],
            "next_step": "Secretary" # Đưa qua Thư ký để chốt hồ sơ
        }

    except Exception as e:
        error_msg = f"Lỗi Storyteller: {str(e)}"
        print(colored(f"❌ {error_msg}", "red"))
        return {
            "messages": [AIMessage(content=f"⚠️ Sáng tác gián đoạn: {error_msg}")],
            "error_log": errors + [error_msg],
            "next_step": "Secretary"
        }
def storytelling_node(state):
    print(colored("[🖋️ STORYTELLING] Đại văn hào đang nối mạch cảm xúc...", "magenta", attrs=["bold"]))
    
    messages = state.get("messages", [])
    last_msg = messages[-1].content
    
    # 1. PHÂN TÍCH NHU CẦU: Viết mới hay Viết tiếp?
    is_continue = "[CONTINUE]" in last_msg
    
    # 2. TRÍ NHỚ DÀI HẠN: Lấy nội dung chương trước đó (nếu là viết tiếp)
    previous_content = ""
    if is_continue and len(messages) > 1:
        # Lấy nội dung mà AI vừa trả về ở lượt trước
        previous_content = messages[-2].content 

    prompt = (
        "Bạn là Nhà văn Best-seller. "
        "\nNHIỆM VỤ: Viết chương tiếp theo của câu chuyện."
        "\n\nYÊU CẦU DUY TRÌ MẠCH VĂN:"
        f"\n- ĐOẠN KẾT CHƯƠNG TRƯỚC: '{previous_content[-500:]}' (Hãy nối tiếp mạch này)."
        "\n- KHÔNG lặp lại lời chào hay tóm tắt chương cũ."
        "\n- Bắt đầu ngay vào hành động hoặc lời thoại tiếp theo."
        "\n- Giữ nguyên văn phong, tên nhân vật và bối cảnh."
    )

    # 3. THỰC THI (Dùng Claude 3.5 Sonnet để có sự mượt mà nhất)
    response = LLM_CLAUDE.invoke([
        SystemMessage(content=prompt),
        HumanMessage(content=last_msg.replace("[CONTINUE]", ""))
    ])

    return {
        "messages": [AIMessage(content=response.content)],
        "next_step": "FINISH"
    }

# ============================================================================
# NODE: R&D STRATEGY (Giám đốc Chiến lược - CSO)
# ============================================================================
def research_development_agent(state):
    """
    Agent R&D: Kết hợp tìm kiếm thời gian thực và phân tích mô hình PESTLE/Roadmap.
    """
    print(colored("[🧠 R&D STRATEGY] Đang thiết lập tầm nhìn chiến lược...", "blue", attrs=["bold"]))
    
    messages = state.get("messages", [])
    user_input = messages[-1].content
    
    # 1. Truy xuất ký ức công ty để đảm bảo chiến lược đồng nhất
    company_context = search_memory("Tầm nhìn và mục tiêu chiến lược AI Corporation")
    
    # 2. Bước nghiên cứu thực tế (Sử dụng Perplexity để tránh nói sáo rỗng)
    # Chúng ta yêu cầu AI tìm dữ liệu thực tế trước khi phân tích
    search_query = f"Xu hướng công nghệ, đối thủ cạnh tranh và rủi ro thị trường năm 2026 cho: {user_input}"
    
    try:
        # Lấy dữ liệu thực tế từ internet
        market_data = LLM_PERPLEXITY.invoke([
            SystemMessage(content="Bạn là chuyên gia phân tích dữ liệu thị trường."),
            HumanMessage(content=search_query)
        ]).content
        
        # 3. Tổng hợp thành báo cáo chiến lược chuyên sâu
        # Kết hợp: Dữ liệu thực tế + Prompt hệ thống + Ngữ cảnh công ty
        prompt_template = ChatPromptTemplate.from_messages([
            ("system", STRATEGY_SYSTEM_PROMPT),
            ("human", (
                f"YÊU CẦU NGHIÊN CỨU: {user_input}\n\n"
                f"DỮ LIỆU THỊ TRƯỜNG THỰC TẾ: {market_data}\n\n"
                f"BỐI CẢNH CÔNG TY: {company_context}\n\n"
                "Hãy lập báo cáo chiến lược chi tiết (PESTLE, Roadmap 2-5 năm)."
            ))
        ])
        
        # Sử dụng GPT-4o để tổng hợp vì khả năng viết báo cáo rất tốt
        chain = prompt_template | LLM_GPT4
        response = chain.invoke({})
        
        return {
            "messages": [AIMessage(content=f"🧠 [BÁO CÁO CHIẾN LƯỢC R&D]:\n{response.content}")],
            "next_step": "Supervisor"
        }
        
    except Exception as e:
        print(colored(f"Lỗi R&D Agent: {e}", "red"))
        return {"next_step": "Supervisor", "error_log": [str(e)]}

# ==========================================
# --- 4. THIẾT LẬP LUỒNG AGENT (GRAPH) ---
# ==========================================

workflow = StateGraph(AgentState)

# --- 4.1 Đăng ký tất cả các Node (Đảm bảo tên khớp 100%) ---
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
    "Secretary": secretary_node
}

for name, func in nodes_map.items():
    workflow.add_node(name, func)

# --- 4.2 Thiết lập điểm vào ---
workflow.set_entry_point("Router")

# --- 4.3 Logic Router ---
# Thay vì dùng router_node trực tiếp, ta dùng lambda để lấy chuỗi 'next_step'
workflow.add_conditional_edges(
    "Router", 
    lambda x: x.get("next_step", "Supervisor"), 
    {
        "Researcher": "Researcher", 
        "Investment": "Investment", 
        "Storyteller": "Storyteller",
        "Artist": "Artist", 
        "Engineering": "Engineering", 
        "Publisher": "Publisher",
        "Orchestrator": "Orchestrator", 
        "Supervisor": "Supervisor", 
        "Secretary": "Secretary"
    }
)

# --- 4.4 Logic Supervisor ---
workflow.add_conditional_edges(
    "Supervisor", 
    lambda x: x.get("next_step", "Secretary"), 
    {
        "Coder": "Coder", 
        "Hardware": "Hardware", 
        "Engineering": "Engineering",
        "IoT_Engineer": "IoT_Engineer", 
        "Procurement": "Procurement",
        "Investment": "Investment", 
        "Researcher": "Researcher", 
        "Strategy_R_and_D": "Strategy_R_and_D",
        "Legal": "Legal", 
        "Marketing": "Marketing", 
        "Artist": "Artist",
        "Storyteller": "Storyteller", 
        "Secretary": "Secretary", 
        "FINISH": "Secretary"
    }
)

# --- 4.5 Nhóm Agent phổ thông (Hồi quy về Supervisor hoặc kết thúc) ---
# Lưu ý: Không bao gồm Coder, Tester, Hardware, Procurement, Investment, Researcher, Orchestrator
general_agents = [
    "Engineering", "IoT_Engineer", "Strategy_R_and_D", "Legal", 
    "Marketing", "Artist", "Storyteller", "Publisher"
]

for node in general_agents:
    workflow.add_conditional_edges(
        node,
        lambda x: x.get("next_step", "Supervisor") if x.get("next_step") != "FINISH" else "Secretary",
        {
            "Supervisor": "Supervisor", 
            "Secretary": "Secretary",
            "Artist": "Artist",
            "Procurement": "Procurement"
        }
    )

# --- 4.6 Logic chuyên biệt (Pipeline & Đặc thù) ---

# Luồng Researcher -> Orchestrator
workflow.add_conditional_edges(
    "Researcher",
    lambda x: "Orchestrator" if x.get("task_type") == "dynamic" else "Secretary",
    {"Orchestrator": "Orchestrator", "Secretary": "Secretary"}
)

# Luồng Orchestrator tỏa đi các nhánh
workflow.add_conditional_edges(
    "Orchestrator",
    lambda x: x.get("next_step", "Secretary") if x.get("next_step") != "FINISH" else "Secretary",
    {
        "Engineering": "Engineering", 
        "Hardware": "Hardware", 
        "Procurement": "Procurement",
        "IoT_Engineer": "IoT_Engineer", 
        "Supervisor": "Supervisor", 
        "Secretary": "Secretary"
    }
)

# Luồng Kỹ thuật: Coder -> Tester
workflow.add_edge("Coder", "Tester")
workflow.add_conditional_edges(
    "Tester", 
    lambda x: x.get("next_step", "Supervisor"), 
    {"Coder": "Coder", "Supervisor": "Supervisor"}
)

# Luồng Vật lý & Tài chính cố định: Hardware -> Procurement -> Investment -> Supervisor/Secretary
workflow.add_edge("Hardware", "Procurement")
workflow.add_edge("Procurement", "Investment")
workflow.add_conditional_edges(
    "Investment",
    lambda x: "Secretary" if x.get("next_step") == "FINISH" else "Supervisor",
    {"Secretary": "Secretary", "Supervisor": "Supervisor"}
)

# --- 4.7 Kết thúc hệ thống ---
workflow.add_edge("Secretary", END)

# --- 4.8 BIÊN DỊCH HỆ THỐNG ---
ai_app = workflow.compile() 
app = ai_app
db = None # Placeholder cho đối tượng Database của ngài

# ============================================================================
# 5. HÀM VẬN HÀNH CHÍNH (ĐẶT Ở ĐÂY)
# ============================================================================
async def run_ai_corporation(user_input, thread_id="1"):
    """
    Điểm kích hoạt hệ thống: Quản lý phiên làm việc và xử lý lỗi tầng cao nhất.
    """
    config = {"configurable": {"thread_id": thread_id}, "recursion_limit": 50}
    
    # Khởi tạo trạng thái ban đầu
    initial_state = {
                "messages": [HumanMessage(content=user_input)],
                "next_step": "Supervisor",
                "current_agent": "User", # Thêm dòng này để tránh lỗi NoneType
                "error_log": [],
                "task_type": "general"
            }

    print(colored(f"\n🚀 PROJECT START: {user_input[:50]}...", "blue", attrs=["bold"]))

    try:
        # Chạy Graph (Giả sử bạn đã compile graph thành app)
        async for event in app.astream(initial_state, config):
            for node, values in event.items():
                if node != "__metadata__":
                    print(colored(f"📍 Node [{node}] has completed.", "dark_grey"))
        
        print(colored("\n✅ PROJECT FINISHED SUCCESSFULLY", "green", attrs=["bold"]))

    except Exception as e:
        # Nếu Graph sập, kích hoạt Fallback ngay lập tức
        return ultimate_fallback(initial_state, [str(e)])
    
# ============================================================================
# 6. CHẠY HỆ THỐNG (ASYNC ENGINE)
# ============================================================================

async def main_loop():
    print(colored("\n" + "="*50, "cyan"))
    print(colored("🚀 AI CORPORATION - HỆ THỐNG ĐIỀU HÀNH TỰ ĐỘNG", "cyan", attrs=["bold"]))
    print(colored("Chế độ: Parallel Coding & AST Testing [ON]", "green"))
    print(colored("="*50 + "\n", "cyan"))
    print(colored("ℹ️  Hệ thống đang chạy ngầm. Hãy gửi yêu cầu từ Dashboard HTML.", "yellow"))
    while True:
        await asyncio.sleep(300) # Nghỉ mỗi 1 tiếng rồi lặp lại (vô tận)
        try:
            user_input = input(colored("CEO (Yêu cầu): ", "white", attrs=["bold"]))
            if user_input.lower() in ['q', 'exit']: 
                auto_backup_brain() # Tự động sao lưu trước khi tắt máy
                break
            
            initial_state = {
                "messages": [HumanMessage(content=user_input)],
                "next_step": "Supervisor",
                "current_agent": "User", 
                "error_log": [],
                "task_type": "general"
            }
            
            # Kích hoạt Graph chạy (Sử dụng astream cho các hàm async)
            print(colored("\n--- ĐANG XỬ LÝ ---", "white", attrs=["bold"]))
            config = {"configurable": {"thread_id": "ceo_session"}, "recursion_limit": 150}
            async for event in app.astream(initial_state, config=config):
                for node, values in event.items():
                    if node != "__end__":
                        print(colored(f"  [➔] {node} đã hoàn thành nhiệm vụ.", "dark_grey"))
                        # Nếu muốn in nội dung tin nhắn cuối cùng của từng bước:
                        # print(values["messages"][-1].content)

            print(colored("\n✅ ĐÃ HOÀN TẤT QUY TRÌNH.", "green", attrs=["bold"]))

        except Exception as e:
            print(colored(f"❌ LỖI HỆ THỐNG: {e}", "red"))

# 1. GIÁO TRÌNH ĐÀO TẠO (CURRICULUM)
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

# 2. HÀM ĐÀO TẠO CHUYÊN SÂU (PHIÊN BẢN KẾ THỪA - COST OPTIMIZED)
async def specialized_training_job(role_tag: str):
    """
    PHIÊN BẢN 10.0: COST-OPTIMIZED INHERITANCE (QUY TẮC KẾ THỪA & TIẾT KIỆM)
    - Nguyên tắc: "Không mua lại những gì đã có".
    """
    print(colored(f"🛡️ [INHERITANCE CHECK] {role_tag} đang kiểm tra kho tri thức...", "cyan", attrs=["bold"]))
    
    topics = CURRICULUM.get(role_tag, [])
    if not topics: return

    try:
        # A. LẤY XP HIỆN TẠI (Dùng SQLite trực tiếp cho nhanh, không cần db_manager phức tạp)
        db_path = "/var/data/ai_corp_projects.db" if os.path.exists("/var/data") else "ai_corp_projects.db"
        conn = sqlite3.connect(db_path, timeout=10)
        c = conn.cursor()
        
        c.execute("SELECT xp FROM agent_status WHERE role_tag = ?", (role_tag,))
        row = c.fetchone()
        current_xp = row[0] if row else 0
        conn.close()

        # Chọn chủ đề dựa trên Level (Cứ 50 XP đổi 1 bài)
        topic_index = int(current_xp / 50) % len(topics)
        current_topic = topics[topic_index]
        
        # B. KIỂM TRA KẾ THỪA (QUAN TRỌNG)
        existing_knowledge = ""
        is_found = False
        
        # Tìm trong Vector DB
        try:
            results = vector_db.similarity_search(current_topic, k=1)
            if results:
                existing_knowledge = results[0].page_content
                is_found = True
                print(colored(f"💡 [FOUND] Đã tìm thấy kiến thức cũ: {current_topic}", "green"))
        except: pass

        # C. QUYẾT ĐỊNH CHIẾN LƯỢC
        final_output = ""
        xp_earned = 0
        mode = "UNKNOWN"

        # --- NHÁNH 1: KẾ THỪA (REVIEW MODE) - MIỄN PHÍ ---
        if is_found and existing_knowledge:
            mode = "REVIEW"
            print(colored("--> Chế độ: REVIEW (Ôn tập) - Tiết kiệm tiền.", "yellow"))
            
            if LLM_GEMINI_LOGIC:
                review_prompt = f"""
                Bạn là {role_tag}. Hãy ôn tập lại kiến thức cũ này:
                ---
                {existing_knowledge[:2000]}
                ---
                Yêu cầu: Tóm tắt lại và đề xuất 1 ý tưởng mới từ nó.
                """
                res = await LLM_GEMINI_LOGIC.ainvoke(review_prompt)
                final_output = res.content
                xp_earned = 20 # Điểm thấp hơn vì chỉ ôn tập
            else:
                final_output = existing_knowledge

        # --- NHÁNH 2: NGHIÊN CỨU MỚI (RESEARCH MODE) - TỐN TIỀN ---
        else:
            mode = "RESEARCH"
            print(colored("--> Chế độ: RESEARCH (Nghiên cứu mới) - Gọi Search API.", "magenta"))
            
            raw_data = ""
            # Ưu tiên Perplexity -> DeepSeek -> Gemini
            if LLM_PERPLEXITY:
                res = await LLM_PERPLEXITY.ainvoke(f"Nghiên cứu mới nhất về: {current_topic}")
                raw_data = res.content
            elif LLM_DEEPSEEK:
                res = await LLM_DEEPSEEK.ainvoke(f"Kiến thức chuyên sâu về: {current_topic}")
                raw_data = res.content
            
            final_output = raw_data
            xp_earned = 50 # Điểm cao

        # D. LƯU KẾT QUẢ & CỘNG ĐIỂM
        # 1. Lưu vào não (Vector DB)
        if mode == "RESEARCH" and final_output:
            vector_db.add_texts(
                texts=[final_output],
                metadatas=[{"source": "Auto_Train", "agent": role_tag, "topic": current_topic}]
            )

        # 2. Ghi sổ công việc (Để hiện lên Dashboard và cộng XP)
        # Sử dụng hàm log_work_to_db có sẵn trong main.py
        clean_name = role_tag.replace("[","").replace("]","")
        log_work_to_db(
            agent=clean_name,
            task=f"Đào tạo: {current_topic}",
            result=f"[{mode}] {final_output[:100]}...",
            tool=f"Auto-{mode}",
            xp_bonus=xp_earned
        )

    except Exception as e:
        print(colored(f"❌ Lỗi đào tạo {role_tag}: {e}", "red"))


# 2. HÀM CHẤM ĐIỂM CHẤT LƯỢNG
async def evaluate_quality(agent_name, content):
    """Giám khảo AI chấm điểm nội dung học (1-10)"""
    prompt = f"Chấm điểm nội dung của {agent_name} (Thang 1-10). Nội dung: {content[:500]}..."
    try:
        model = LLM_DEEPSEEK if LLM_DEEPSEEK else LLM_GPT4
        score_msg = await model.ainvoke(prompt)
        score = int(re.search(r'\d+', score_msg.content).group())
        return min(max(score, 1), 10)
    except: return 5


# Biến toàn cục để Server có thể set trạng thái bận
IS_SYSTEM_BUSY = False 
LAST_INTERACTION_TIME = datetime.now()
# 3. VÒNG LẶP TỰ HỌC (AUTO LEARNING CYCLE)
async def auto_learning_cycle():
    """
    ĐỘNG CƠ TỰ HỌC VĨNH CỬU (Smart Scheduler)
    - Luân phiên đánh thức Agent đi học (specialized_training_job).
    - Tự động ngắt khi CEO cần dùng hệ thống (Busy Check).
    """
    global IS_SYSTEM_BUSY, LAST_INTERACTION_TIME
    
    print(colored("🎓 [SCHEDULER] Kích hoạt Học viện Agent Tự động...", "magenta", attrs=["bold"]))
    
    # Danh sách học viên
    agents_queue = list(CURRICULUM.keys())
    idx = 0

    while True:
        # --- BƯỚC 1: KIỂM TRA TRẠNG THÁI BẬN RỘN ---
        # Nếu vừa có lệnh trong 5 phút qua -> Coi là bận
        idle_seconds = (datetime.now() - LAST_INTERACTION_TIME).total_seconds()
        
        if IS_SYSTEM_BUSY or idle_seconds < 300: # 5 phút
            # print("🚧 Hệ thống đang bận. Tạm hoãn học tập.", end="\r")
            await asyncio.sleep(60) # Chờ 1 phút rồi check lại
            continue

        # --- BƯỚC 2: BẮT ĐẦU CA HỌC ---
        current_agent = agents_queue[idx % len(agents_queue)]
        idx += 1
        
        print(colored(f"\n🔔 [DING] Hệ thống rảnh. Đánh thức {current_agent} đi học...", "magenta"))
        
        try:
            # Gọi hàm đào tạo chuyên sâu (đã có logic Kế thừa & Cộng điểm)
            await specialized_training_job(current_agent)
            
            # Học xong 1 người -> Nghỉ giải lao dài (để không spam API liên tục)
            # Chạy thật: Nghỉ 30-60 phút
            # Chạy test: Nghỉ 60 giây
            print(colored(f"💤 {current_agent} đã học xong. Hệ thống nghỉ giải lao.", "dark_grey"))
            await asyncio.sleep(300) 

        except Exception as e:
            print(colored(f"⚠️ Lỗi Scheduler: {e}", "red"))
            await asyncio.sleep(60) # Lỗi thì nghỉ tí rồi thử người khác
       

def set_system_busy():
    """Hàm để Server gọi mỗi khi có tin nhắn từ CEO"""
    global IS_SYSTEM_BUSY, LAST_INTERACTION_TIME
    IS_SYSTEM_BUSY = True
    LAST_INTERACTION_TIME = datetime.now()
    # Sau một khoảng thời gian, có thể set lại False hoặc dựa vào idle time
# 4. JOB BÁO CÁO SÁNG (DÙNG LOGIC MỚI)
# 4. JOB BÁO CÁO SÁNG (BẢN HỢP NHẤT: KẾ THỪA + LƯU TRỮ CHUYÊN NGHIỆP)
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
            if LLM_PERPLEXITY:
                res = await LLM_PERPLEXITY.ainvoke(f"Tin tức mới nhất 24h qua về: {topic}")
                content = res.content
                source_note = "(Nguồn: Perplexity Live)"
            elif existing_knowledge:
                content = existing_knowledge
                source_note = "(Nguồn: Ký ức nội bộ)"
            else:
                content = "Không tìm thấy thông tin mới."

            # Lưu lại vào bộ đệm báo cáo
            report_buffer.append(f"### {topic} {source_note}\n{content[:1000]}...\n")
            
            # Ghi nhớ vào Vector DB (để dành cho lần sau)
            if vector_db and "Perplexity" in source_note:
                await asyncio.to_thread(
                    vector_db.add_texts,
                    texts=[content],
                    metadatas=[{"source": "Morning_Briefing", "agent": role_tag, "topic": topic, "date": datetime.now().isoformat()}]
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
# 7. KHỞI CHẠY THỰC TẾ
# ============================================================================
if __name__ == "__main__":
    try:
        # Chạy vòng lặp chính thông qua asyncio
        asyncio.run(main_loop())
    except KeyboardInterrupt:
        print("\n👋 Đã thoát hệ thống.")
