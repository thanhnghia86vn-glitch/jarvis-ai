import json
import random
import logging
import asyncio
import re
from datetime import datetime
from termcolor import colored
from duckduckgo_search import DDGS

# Cấu hình log riêng cho Thợ lặn
logger = logging.getLogger("RESEARCH_AGENT")

class ResearchAgent:
    def __init__(self):
        """
        Khởi tạo Agent nghiên cứu. 
        Không nhận db_manager qua tham số để tránh vòng lặp Import ngay khi load file.
        """
        self._db = None
        self.is_autopilot_on = False

    @property
    def db(self):
        """Lấy db_manager chỉ khi cần thiết (Lazy Loading)"""
        if self._db is None:
            try:
                from server import db_manager
                self._db = db_manager
            except ImportError:
                logger.error("❌ [CRITICAL] Không thể kết nối db_manager từ server.py")
        return self._db

    def _get_chat_model(self):
        """Lấy CHAT_MODEL từ server chỉ khi thực thi lệnh"""
        try:
            from server import CHAT_MODEL
            return CHAT_MODEL
        except ImportError:
            return None

    # --- TẦNG 1: ĐI SĂN (KNOWLEDGE DIVER) ---
    async def auto_knowledge_diver(self):
        """Hệ thống tự hành quét Internet tìm dự án và tri thức mới"""
        chat_model = self._get_chat_model()
        if not self.is_autopilot_on or not chat_model:
            return

        print(colored("\n🌊 [RESEARCH] Thợ lặn đang quét internet...", "cyan"))

        hunting_grounds = [
            "AI Multi-Agent Systems 2026",
            "Blockchain Security Vulnerabilities",
            "Python Automation for Business",
            "Global Digital Currency Trends"
        ]
        target = random.choice(hunting_grounds)
        
        try:
            search_content = ""
            def _fetch():
                with DDGS() as ddgs:
                    return list(ddgs.text(f"{target} latest news 2026", max_results=5))
            
            # Sử dụng run_in_threadpool từ server để tránh block
            from fastapi.concurrency import run_in_threadpool
            results = await run_in_threadpool(_fetch)

            for r in results:
                search_content += f"- {r['title']}: {r['body']}\n"

            # AI Supervisor thẩm định mục tiêu
            analysis_prompt = f"""
            Phân tích dữ liệu internet về: {target}
            Dữ liệu: {search_content}
            Đề xuất 1 chủ đề tiềm năng nhất. 
            Trả về JSON duy nhất: {{"subject": "tên", "difficulty": 1-5}}
            """
            
            res = await chat_model.ainvoke(analysis_prompt)
            # Làm sạch JSON bọc thép
            clean_json = re.search(r'\{.*\}', res.content, re.DOTALL)
            if not clean_json: return
            
            data = json.loads(clean_json.group())

            # Phân phối thành các Task nhỏ cho Worker
            await self.distribute_to_workers(
                subject=f"🚀 [AUTO] {data['subject']}", 
                num_tasks=8, 
                reward=round(0.02 + (data.get('difficulty', 2) * 0.01), 3)
            )
            logger.info(f"✅ Đã phân phối mẩu tri thức mới về: {data['subject']}")

        except Exception as e:
            logger.error(f"❌ Research Error: {e}")

    # --- TẦNG 2: PHÂN RÃ (KNOWLEDGE DISTRIBUTOR) ---
    async def distribute_to_workers(self, subject: str, num_tasks: int, reward: float):
        """Bẻ nhỏ chủ đề lớn thành các Task nghiên cứu chuyên sâu"""
        chat_model = self._get_chat_model()
        if not chat_model or not self.db: return 0

        prompt = f"Chia nhỏ chủ đề '{subject}' thành {num_tasks} nhiệm vụ nghiên cứu. Trả về JSON list: [{{'topic', 'task_type', 'content'}}]"
        try:
            res = await chat_model.ainvoke(prompt)
            clean_json = re.search(r'\[.*\]', res.content, re.DOTALL)
            if not clean_json: return 0
            
            tasks = json.loads(clean_json.group())
            
            from sqlalchemy import text
            with self.db.get_connection() as conn:
                for t in tasks:
                    conn.execute(text("""
                        INSERT INTO learning_tasks (topic, task_type, reward, content, status) 
                        VALUES (:t, :type, :r, :c, 'PENDING')
                    """), {
                        "t": f"[{subject.upper()}] {t['topic']}", 
                        "type": t.get('task_type', 'RESEARCH'), 
                        "r": reward, 
                        "c": t.get('content', '')
                    })
                conn.commit()
            return len(tasks)
        except Exception as e:
            logger.error(f"❌ Distribution Error: {e}")
            return 0

    # --- TẦNG 3: TIÊU HÓA (KNOWLEDGE INGESTION) ---
    async def ingest_pdf(self, file_path):
        """Nạp tài liệu PDF vào bộ nhớ dài hạn"""
        try:
            # Import muộn để tránh Circular Import với main.py
            from main import ingest_docs_to_memory
            from fastapi.concurrency import run_in_threadpool
            
            result = await run_in_threadpool(lambda: ingest_docs_to_memory(file_path))
            return result
        except Exception as e:
            return f"Lỗi nạp PDF: {e}"

# Khởi tạo instance SẠCH (Không tham chiếu db_manager ngay lập tức)
research_agent = ResearchAgent()