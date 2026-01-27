# file: api_server.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import sqlite3
import uvicorn

app = FastAPI()

# Cho phép HTML gọi API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

DB_PATH = "ai_finance.db" # Đảm bảo file này đã được tạo từ main.py
AGENT_DB_PATH = "agents_data.db" # Database chứa Level/XP

@app.get("/api/costs")
def get_costs():
    """Lấy dữ liệu chi phí chi tiết để đối chiếu"""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    # Lấy 100 giao dịch mới nhất
    rows = conn.execute("SELECT * FROM audit_log ORDER BY timestamp DESC LIMIT 100").fetchall()
    conn.close()
    return [dict(row) for row in rows]

@app.get("/api/agents")
def get_agents_activity():
    """Lấy tình trạng học tập và Level của Agent"""
    # 1. Lấy thông tin Level
    conn_ag = sqlite3.connect(AGENT_DB_PATH)
    conn_ag.row_factory = sqlite3.Row
    agents = {row['name']: dict(row) for row in conn_ag.execute("SELECT * FROM agents").fetchall()}
    conn_ag.close()
    
    # 2. Lấy lịch sử hoạt động (Học gì, Làm gì)
    conn_log = sqlite3.connect(DB_PATH)
    conn_log.row_factory = sqlite3.Row
    logs = conn_log.execute("SELECT * FROM audit_log ORDER BY timestamp DESC LIMIT 50").fetchall()
    conn_log.close()
    
    # Gộp dữ liệu
    result = []
    for log in logs:
        agent_name = log['agent']
        agent_info = agents.get(agent_name, {'level': 1, 'xp': 0})
        result.append({
            "time": log['timestamp'],
            "agent": agent_name,
            "level": agent_info['level'],
            "action": log['action_type'],
            "learned": log['knowledge_gained'],
            "result": log['application'],
            "cost": log['cost_usd']
        })
    return result

if __name__ == "__main__":
    print("🚀 API Server đang chạy tại http://localhost:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000)