import os
import imaplib
import email
import re
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from dotenv import load_dotenv

# Nạp biến môi trường từ .env hoặc cấu hình Cloud
load_dotenv()

class IdentityCore:
    def __init__(self):
        # Thông tin được lấy từ biến môi trường ngài đã cài đặt
        self.email_user = os.getenv("MAIL_USERNAME")
        self.email_pass = os.getenv("MAIL_PASSWORD")
        self.imap_url = "imap.gmail.com"
        self.smtp_url = "smtp.gmail.com"

    # --- BƯỚC 2: MODULE ĐỌC EMAIL & TRÍCH XUẤT OTP ---
    def fetch_latest_otp(self, keyword="verification"):
        """
        Tự động kết nối IMAP để lấy mã xác nhận mới nhất từ hòm thư.
        """
        try:
            # 1. Kết nối và đăng nhập
            mail = imaplib.IMAP4_SSL(self.imap_url)
            mail.login(self.email_user, self.email_pass)
            mail.select("inbox")

            # 2. Tìm kiếm thư dựa trên từ khóa (vd: OTP, Code, Verify)
            # Tìm kiếm trong cả tiêu đề và nội dung
            status, messages = mail.search(None, f'(OR SUBJECT "{keyword}" BODY "{keyword}")')
            
            if status != "OK" or not messages[0]:
                print(f"🔍 [IDENTITY] Không tìm thấy thư phù hợp với từ khóa: {keyword}")
                return None

            # 3. Lấy thư mới nhất (ID cuối cùng trong danh sách)
            latest_id = messages[0].split()[-1]
            status, data = mail.fetch(latest_id, "(RFC822)")
            
            raw_email = data[0][1]
            msg = email.message_from_bytes(raw_email)

            # 4. Trích xuất nội dung văn bản (Plain Text)
            body = ""
            if msg.is_multipart():
                for part in msg.walk():
                    if part.get_content_type() == "text/plain":
                        body = part.get_payload(decode=True).decode(errors='ignore')
            else:
                body = msg.get_payload(decode=True).decode(errors='ignore')

            # 5. Dùng Regex lọc mã OTP (Dãy số từ 4 đến 6 chữ số)
            otp_match = re.search(r'\b\d{4,6}\b', body)
            
            mail.logout() # Đăng xuất để bảo mật

            if otp_match:
                return otp_match.group(0)
            return None

        except Exception as e:
            print(f"❌ [IDENTITY] Lỗi truy cập hòm thư: {str(e)}")
            return None

    # --- BƯỚC 3: MODULE GIAO TIẾP & BÁO CÁO ---
    def send_system_mail(self, to_email, subject, html_content):
        """
        Gửi email thông báo cho CEO hoặc khách hàng qua SMTP.
        """
        try:
            msg = MIMEMultipart()
            msg['From'] = f"J.A.R.V.I.S Core <{self.email_user}>"
            msg['To'] = to_email
            msg['Subject'] = subject
            msg.attach(MIMEText(html_content, 'html'))

            server = smtplib.SMTP(self.smtp_url, 587)
            server.starttls()
            server.login(self.email_user, self.email_pass)
            server.send_message(msg)
            server.quit()
            return True
        except Exception as e:
            print(f"❌ [IDENTITY] Lỗi gửi email: {str(e)}")
            return False

# Khởi tạo instance để dùng chung
jarvis_identity = IdentityCore()