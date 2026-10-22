# Agriculture RAG Backend

Backend RAG (Retrieval-Augmented Generation) hỏi đáp khuyến nông, xây dựng bằng FastAPI, ChromaDB và mô hình embedding tiếng Việt chạy local.

---

## Yêu cầu hệ thống

| Phần mềm | Phiên bản tối thiểu |
|---|---|
| Python | 3.10+ |
| Java | 11+ (bắt buộc cho OpenDataLoader PDF) |
| pip | 23+ |

> **Lưu ý GPU:** Mặc định embedding chạy trên CPU. Nếu có GPU, đổi `EMBEDDING_DEVICE=cuda` và cài `torch` phiên bản CUDA tương ứng.

---

## Cài đặt

### 1. Tạo virtual environment

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate
```

### 2. Cài đặt dependencies

```bash
pip install -r requirements.txt
```

### 3. Cấu hình environment variables

```bash
cp .env.example .env
```

Mở `.env` và điền `GEMINI_API_KEY` của bạn:

```env
GEMINI_API_KEY=your_real_gemini_api_key_here
```

Các biến khác đã có giá trị mặc định hợp lý, không cần thay đổi cho lần chạy đầu tiên.

### 4. Tạo thư mục dữ liệu

```bash
mkdir -p data/uploads data/vector_db
```

---

## Chạy server

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Truy cập Swagger UI tại: **http://localhost:8000/docs**

---

## API Endpoints

| Method | Endpoint | Mô tả |
|---|---|---|
| `POST` | `/api/v1/documents` | Upload một hoặc nhiều file PDF |
| `GET` | `/api/v1/documents` | Danh sách tài liệu |
| `GET` | `/api/v1/documents/{id}` | Chi tiết tài liệu |
| `DELETE` | `/api/v1/documents/{id}` | Xóa tài liệu và vector |
| `POST` | `/api/v1/query` | Đặt câu hỏi tiếng Việt |
| `GET` | `/health` | Kiểm tra trạng thái hệ thống |
| `GET` | `/docs` | Swagger UI |

---

## Luồng sử dụng cơ bản

### Upload tài liệu PDF

```bash
curl -X POST http://localhost:8000/api/v1/documents \
  -F "files=@ky-thuat-trong-lua.pdf"
```

Hệ thống trả về `document_id` và status `processing`. Quá trình index chạy nền — kiểm tra trạng thái bằng:

```bash
curl http://localhost:8000/api/v1/documents/{document_id}
```

Chờ đến khi status là `indexed` trước khi đặt câu hỏi.

### Đặt câu hỏi

```bash
curl -X POST http://localhost:8000/api/v1/query \
  -H "Content-Type: application/json" \
  -d '{"question": "Làm thế nào để phòng bệnh đạo ôn trên cây lúa?", "top_k": 5}'
```

Response mẫu:
```json
{
  "question": "Làm thế nào để phòng bệnh đạo ôn trên cây lúa?",
  "answer": "Theo tài liệu được cung cấp...",
  "sources": [
    {
      "document_id": "4ecdb281-...",
      "file_name": "ky-thuat-trong-lua.pdf",
      "page": 12,
      "chunk_id": "chunk-...",
      "score": 0.82,
      "excerpt": "Để phòng bệnh đạo ôn..."
    }
  ],
  "model": "gemini-flash-latest"
}
```

---

## Batch ingestion (script)

Để nhập nhiều file PDF mà không cần HTTP API:

```bash
python scripts/ingest_documents.py path/to/pdf_folder/
# hoặc chỉ định từng file
python scripts/ingest_documents.py doc1.pdf doc2.pdf
```

---

## Re-index khi đổi embedding model

Nếu bạn thay đổi `EMBEDDING_MODEL` trong `.env`, toàn bộ tài liệu phải được index lại:

1. Dừng server.
2. Xóa thư mục `data/vector_db/` và `data/documents.db`.
3. Khởi động lại server.
4. Upload lại toàn bộ tài liệu hoặc dùng script batch ingestion.

> **Lý do:** Mỗi embedding model tạo ra vector không tương thích với nhau. Trộn vector từ nhiều model trong cùng một collection sẽ cho kết quả retrieval sai.

---

## Chạy bằng Docker

```bash
# Build image
docker build -t agriculture-rag .

# Chạy container
docker run -p 8000:8000 \
  -e GEMINI_API_KEY=your_key_here \
  -v $(pwd)/data:/app/data \
  agriculture-rag
```

---

## Cấu trúc project

```
agriculture-rag/
├── app/
│   ├── main.py                   # FastAPI app entry point
│   ├── api/
│   │   ├── dependencies.py       # Middleware (request ID)
│   │   └── routes/
│   │       ├── documents.py      # Upload / list / detail / delete
│   │       ├── query.py          # Q&A endpoint
│   │       └── health.py         # Health check
│   ├── core/
│   │   ├── config.py             # Pydantic Settings (env vars)
│   │   ├── logging.py            # Structured logging + request_id
│   │   └── exceptions.py         # Custom exceptions + handlers
│   ├── models/
│   │   └── document.py           # SQLAlchemy model + SQLite setup
│   ├── schemas/
│   │   ├── document.py           # API request/response schemas
│   │   └── query.py
│   ├── services/
│   │   ├── document_service.py   # Document CRUD
│   │   ├── ingestion_service.py  # PDF → chunk → embed → index
│   │   ├── retrieval_service.py  # Query → vector search
│   │   └── rag_service.py        # Retrieval + LLM orchestration
│   ├── providers/
│   │   ├── pdf/                  # OpenDataLoader + pypdf fallback
│   │   ├── embedding/            # Vietnamese embedding (local)
│   │   ├── llm/                  # LLMProvider base + Gemini impl
│   │   └── vector_store/         # ChromaDB persistent store
│   └── prompts/
│       └── agriculture_qa.py     # System + user prompt templates
├── data/                         # Runtime data (gitignored)
├── scripts/
│   └── ingest_documents.py       # Batch PDF ingestion script
├── .env.example
├── .gitignore
├── requirements.txt
├── Dockerfile
└── README.md
```

---

## Hạn chế đã biết (MVP)

- Không có authentication / authorization.
- Chỉ hỗ trợ PDF có sẵn text; PDF scan (ảnh) cần OCR bổ sung.
- Không hỗ trợ hội thoại nhiều lượt (multi-turn).
- Không có streaming response.
- Background task ingestion sử dụng FastAPI `BackgroundTasks` (single-process); cần Celery/Redis cho production multi-worker.
- Embedding model lần đầu chạy sẽ tải về (~1.5 GB) từ HuggingFace Hub.
- Chưa có hybrid search hoặc reranking.
