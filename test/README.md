# SkillStruct Test Suite

Bộ test tự động cho tất cả các API và component của SkillStruct platform.

## 📁 Cấu trúc Test

```
test/
├── README.md                      # Tài liệu này
├── run_all_tests.py              # Script chạy tất cả test APIs
├── test_components.py            # Test các utility functions
├── test_ocr_clustering_api.py    # Test OCR + Clustering API (Port 8000)
├── test_json_generation_api.py   # Test JSON Generation API (Port 8001)
├── test_graph_api.py             # Test Graph Management API (Port 8002)
└── test_recommendation_api.py    # Test Recommendation API (Port 8003)
```

## 🚀 Cách sử dụng

### 1. Chạy tất cả test APIs

```bash
# Chạy tuần tự (dễ đọc output)
python test/run_all_tests.py

# Chạy song song (nhanh hơn)
python test/run_all_tests.py --parallel

# Test chỉ một API cụ thể
python test/run_all_tests.py --api "OCR + Clustering"
```

### 2. Test từng API riêng lẻ

```bash
# Test OCR + Clustering API
python test/test_ocr_clustering_api.py

# Test JSON Generation API  
python test/test_json_generation_api.py

# Test Graph Management API
python test/test_graph_api.py

# Test Recommendation API
python test/test_recommendation_api.py
```

### 3. Test các component utilities

```bash
# Test tất cả utility functions
python test/test_components.py
```

## 📋 Điều kiện tiên quyết

### 1. Cài đặt dependencies

```bash
pip install requests pandas
```

### 2. Khởi động các API

Trước khi chạy test, đảm bảo các API đang chạy:

```bash
# Khởi động tất cả APIs
python scripts/start_api.py

# Hoặc khởi động từng API riêng:
python services/ocr+clusteringAPI.py      # Port 8000
python services/genjsongraphAPI.py        # Port 8001  
python services/GraphAPI.py               # Port 8002
python services/recommendapi.py           # Port 8003
```

## 🧪 Các loại test

### API Tests

| API | Port | Test Cases |
|-----|------|------------|
| **OCR + Clustering** | 8000 | Health check, Resume processing, Batch processing |
| **JSON Generation** | 8001 | Health check, JSON generation, Data formatting, Validation |
| **Graph Management** | 8002 | Health check, Graph creation, Path analysis, Clustering, Visualization |
| **Recommendation** | 8003 | Health check, Skill recommendations, Career paths, Job matching, Learning resources |

### Component Tests

- **Core Utils**: Text processing, skill extraction, validation
- **File Utils**: File operations, PDF handling, directory management
- **Data Utils**: DataFrame operations, data cleaning, export functions
- **Database Utils**: Connection management, query execution, backup
- **Config Management**: Configuration loading and validation

## 📊 Output mẫu

```
🚀 SKILLSTRUCT API TEST SUITE
================================================================================
Running tests for all APIs sequentially...
Note: RAG API is excluded from testing
================================================================================

============================================================
🧪 TESTING OCR + CLUSTERING API (Port 8000)
============================================================
✅ Health Check - Status: 200
   Response: {'status': 'healthy', 'message': 'API is running'}

📄 Testing file: data/resume/AdalarasuVN[5_3].pdf
   Status: 200
   ✅ Success: Resume processed successfully
   📊 Extracted skills: 15 skills
   👤 Name: ADALARASU V.N
   📧 Email: tamialadal@gmail.com

================================================================================
📊 TEST SUMMARY
================================================================================
   OCR + Clustering    : ✅ PASSED
   JSON Generation     : ✅ PASSED
   Graph Management    : ✅ PASSED
   Recommendation      : ✅ PASSED
--------------------------------------------------------------------------------
   Total APIs tested: 4
   Passed: 4
   Failed: 0
   Success Rate: 100.0%
================================================================================
```

## 🛠️ Troubleshooting

### Lỗi thường gặp

1. **API không phản hồi**
   ```
   ❌ OCR + Clustering API is not running on port 8000
   ```
   - **Giải pháp**: Khởi động API bằng `python services/ocr+clusteringAPI.py`

2. **Connection refused**
   ```
   ❌ Health Check failed: Connection refused
   ```
   - **Giải pháp**: Đảm bảo API đang chạy và port không bị chiếm

3. **Module not found**
   ```
   ModuleNotFoundError: No module named 'requests'
   ```
   - **Giải pháp**: Cài đặt dependencies: `pip install requests`

4. **File not found**
   ```
   ❌ No resume files found to test
   ```
   - **Giải pháp**: Đảm bảo có file PDF trong thư mục `data/resume/`

### Debug mode

Để debug chi tiết hơn, chạy từng test riêng lẻ:

```bash
# Chạy test với debug output
python -u test/test_ocr_clustering_api.py

# Hoặc dùng verbose mode
python test/run_all_tests.py --api "OCR + Clustering"
```

## 📝 Thêm test mới

### Để thêm test cho API mới:

1. Tạo file `test_new_api.py` theo pattern có sẵn
2. Thêm vào `APIS` dictionary trong `run_all_tests.py`
3. Implement các test functions cần thiết

### Để thêm test cho component mới:

1. Thêm function test mới vào `test_components.py`
2. Thêm vào list `tests` trong function `main()`

## 🎯 Best Practices

1. **Luôn test health check trước**: Đảm bảo API đang chạy
2. **Sử dụng dữ liệu test thực tế**: Test với file resume có sẵn
3. **Kiểm tra cả success và error cases**: Test các tình huống lỗi
4. **Timeout appropriate**: Set timeout hợp lý cho các test
5. **Clean up**: Dọn dẹp dữ liệu test sau khi chạy

## 📞 Liên hệ

Nếu có vấn đề với test suite, hãy kiểm tra:
1. Tất cả dependencies đã được cài đặt
2. Các API đang chạy trên đúng port
3. File dữ liệu test có sẵn trong thư mục `data/`
4. Cấu hình hệ thống phù hợp
