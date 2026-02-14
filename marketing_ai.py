import sqlite3
import os
import re
import logging
from identity_core import jarvis_identity

# Cấu hình log để CEO theo dõi hoạt động của Marketing
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("JARVIS_MARKETING")

class MarketingAI:
    def __init__(self):
        # Xác định chính xác lộ trình Database để không lấy nhầm dữ liệu cũ
        self.db_path = "/var/data/ai_corp_projects.db" if os.path.exists("/var/data") else "ai_corp_projects.db"

    def _get_chat_model(self):
        """Kết nối an toàn với bộ não AI từ server trung tâm"""
        try:
            from server import CHAT_MODEL
            return CHAT_MODEL
        except ImportError:
            logger.error("❌ Không thể kết nối với CHAT_MODEL từ server.py")
            return None

    # --- CHẾ ĐỘ 1: CHÀO HÀNG (DỰA TRÊN DỮ LIỆU THỰC) ---
    async def pitch_product_via_email(self, product_id, customer_email):
        chat_model = self._get_chat_model()
        if not chat_model: return {"status": "error", "msg": "AI Core Offline"}

        # Truy xuất thông tin sản phẩm từ sổ cái
        conn = sqlite3.connect(self.db_path)
        product = conn.execute("SELECT name, description, price FROM products WHERE id=?", (product_id,)).fetchone()
        conn.close()
        
        if not product: return {"status": "error", "msg": "Sản phẩm không tồn tại"}
        name, desc, price = product
        
        # Xây dựng Prompt Marketing chuyên sâu
        prompt = f"""
        Bạn là Giám đốc Marketing của hệ thống J.A.R.V.I.S (Phan Thiết).
        NHIỆM VỤ: Soạn email chào hàng cá nhân hóa.
        THÔNG TIN: Sản phẩm {name}, Giá ${price}, Mô tả: {desc}.
        YÊU CẦU: 
        1. Ngôn từ chuyên nghiệp, không viết 'linh tinh' hoặc hứa hẹn sai thực tế.
        2. Trình bày HTML có cấu trúc rõ ràng, nút kêu gọi hành động (CTA) nổi bật.
        3. Cuối thư có chữ ký: 'J.A.R.V.I.S Digital Operations Team'.
        """
        
        try:
            response = await chat_model.ainvoke(prompt)
            content = response.content
            
            # Gửi mail qua module Identity (Tay chân)
            subject = f"Cơ hội hợp tác: Giải pháp {name} tối ưu"
            success = jarvis_identity.send_system_mail(customer_email, subject, content)
            
            return {"status": "success" if success else "failed", "content": content}
        except Exception as e:
            return {"status": "error", "msg": str(e)}

    # --- CHẾ ĐỘ 2: PHẢN HỒI THÔNG MINH (CHỐNG VIẾT LINH TINH) ---
    async def smart_reply(self, customer_msg, customer_email):
        chat_model = self._get_chat_model()
        if not chat_model: return False
        
        # Prompt ràng buộc hành vi AI
        prompt = f"""
        Bạn là Trợ lý Marketing cấp cao. Khách hàng vừa gửi mail với nội dung: "{customer_msg}"
        Hãy soạn thư trả lời dựa trên các nguyên tắc:
        - Lịch sự, ngắn gọn nhưng đầy đủ thông tin.
        - Nếu khách hỏi về kỹ thuật, hãy hướng dẫn họ liên hệ bộ phận CODER.
        - Tuyệt đối không tự bịa đặt thông tin nếu không chắc chắn.
        - Định dạng HTML chuyên nghiệp.
        """
        
        try:
            response = await chat_model.ainvoke(prompt)
            reply_body = response.content
            
            # Kiểm soát chất lượng cuối cùng (Quality Gate)
            if "lỗi" in reply_body.lower() or len(reply_body) < 30:
                logger.warning("⚠️ Nội dung AI soạn không đạt chuẩn, hủy lệnh gửi.")
                return False

            success = jarvis_identity.send_system_mail(
                customer_email, 
                "Phản hồi hỗ trợ từ J.A.R.V.I.S", 
                reply_body
            )
            return success
        except Exception as e:
            logger.error(f"❌ Lỗi thực thi Smart Reply: {e}")
            return False

# Khởi tạo thực thể duy nhất cho toàn hệ thống
marketing_agent = MarketingAI()