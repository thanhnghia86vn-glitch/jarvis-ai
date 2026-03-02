@app.post("/api/admin/sync_brain_structure")
async def sync_brain_structure(file: UploadFile = File(...), x_api_key: str = Header(None)):
    """Giao thức nạp đè bộ não Vector (ChromaDB) từ Local lên Cloud"""
    if x_api_key != ADMIN_SECRET: return {"error": "Unauthorized"}
    
    import zipfile, shutil
    # Đường dẫn thư mục não bộ trên Cloud
    target_dir = os.path.join(BASE_DATA_DIR, "db_knowledge")
    zip_path = os.path.join(BASE_DATA_DIR, "brain_transfer.zip")

    try:
        # 1. Lưu file zip gửi lên
        with open(zip_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # 2. Xóa bộ não cũ để tránh xung đột dữ liệu
        if os.path.exists(target_dir):
            shutil.rmtree(target_dir)

        # 3. Giải nén bộ não mới từ Phan Thiết
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(BASE_DATA_DIR)

        os.remove(zip_path) # Dọn dẹp
        return {"status": "BRAIN_SYNCED", "msg": "Cấu trúc tri thức đã được cập nhật!"}
    except Exception as e:
        return {"status": "ERROR", "msg": str(e)}
