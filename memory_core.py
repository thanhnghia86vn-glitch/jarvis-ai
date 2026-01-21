import logging
from datetime import datetime
from langchain_core.documents import Document

# Cấu hình Log để dễ debug trên Render
logger = logging.getLogger("MEMORY_CORE")

# --- KẾT NỐI VÀO BỘ NÃO CHÍNH (SAFE IMPORT) ---
# Thay vì tự tạo DB mới, ta "mượn" DB đã fix SQLite từ main.py
# Dùng try/except để tránh lỗi vòng lặp (Circular Import)
try:
    from main import vector_db, LLM_GPT4, AIMessage, SystemMessage, HumanMessage
    CORE_AVAILABLE = True
    logger.info("✅ MEMORY CORE: Đã kết nối với Bộ não trung tâm.")
except ImportError:
    CORE_AVAILABLE = False
    vector_db = None
    LLM_GPT4 = None
    logger.warning("⚠️ MEMORY CORE: Không thể kết nối Main Brain (Chạy chế độ Offline).")

def recall_relevant_memories(query: str, k=3):
    """
    Hồi tưởng: Tìm kiếm ký ức liên quan đến câu nói hiện tại.
    """
    if not CORE_AVAILABLE or not vector_db:
        return "" # Trả về rỗng nếu hệ thống chưa sẵn sàng

    try:
        print(f"🧠 [MEMORY] Đang lục lọi ký ức về: '{query}'...")
        # Tìm kiếm tương đồng
        results = vector_db.similarity_search_with_score(query, k=k)
        
        memories = []
        for doc, score in results:
            # Score của Chroma (L2): Càng thấp càng giống (0 là giống hệt)
            # Ngưỡng 1.2 là khá rộng, có thể hạ xuống 0.8 nếu muốn chính xác hơn
            if score < 1.2: 
                time_str = doc.metadata.get('timestamp', 'Unknown Time')
                memories.append(f"- {doc.page_content} (Ghi lúc: {time_str})")
        
        return "\n".join(memories) if memories else ""
        
    except Exception as e:
        logger.error(f"❌ Lỗi hồi tưởng: {e}")
        return ""

def extract_and_save_memory(user_input: str, ai_response: str):
    """
    Ghi nhớ chủ động: Dùng AI lọc thông tin quan trọng để lưu.
    """
    if not CORE_AVAILABLE or not vector_db or not LLM_GPT4:
        return False

    # Prompt tối ưu hóa để tiết kiệm Token và tăng độ chính xác
    prompt = f"""
    Bạn là Thư ký Ghi nhớ của hệ thống J.A.R.V.I.S.
    
    Hội thoại:
    User: {user_input}
    AI: {ai_response}

    NHIỆM VỤ:
    1. Chỉ trích xuất thông tin CỐT LÕI mang tính lâu dài (Sở thích, Tên tuổi, Dự án, Lịch hẹn, Quan điểm).
    2. Bỏ qua các câu chào hỏi, lệnh code, hoặc hội thoại tán gẫu vô thưởng vô phạt.
    3. Nếu không có gì đáng nhớ, trả về đúng 1 từ: NONE

    Định dạng đầu ra (nếu có): [Thông tin đã cô đọng thành 1 câu khẳng định]
    """
    
    try:
        # Gọi LLM (Dùng invoke để an toàn)
        analysis_msg = LLM_GPT4.invoke([
            SystemMessage(content="Nhiệm vụ: Trích xuất ký ức."), 
            HumanMessage(content=prompt)
        ])
        analysis = analysis_msg.content.strip()
        
        # Logic lọc rác
        if "NONE" not in analysis and len(analysis) > 5:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print(f"💾 [MEMORY SAVE] Đang ghi vào não bộ: {analysis}")
            
            # Lưu vào Vector DB (Dùng lại vector_db của main)
            doc = Document(
                page_content=analysis,
                metadata={"timestamp": timestamp, "source": "conversation"}
            )
            # Chạy hàm add_documents (ChromaDB tự xử lý embedding)
            vector_db.add_documents([doc])
            return True
            
    except Exception as e:
        logger.error(f"⚠️ Lỗi quá trình ghi nhớ: {e}")
    
    return False