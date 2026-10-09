import json, sys, uuid
from pathlib import Path
from datetime import datetime, timezone
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.core.config import get_settings
from app.providers.pdf.opendataloader_parser import ParsedPage, normalize_vietnamese
from app.services.ingestion_service import _chunk_pages
from app.providers.embedding.vietnamese_embedding import embed_documents
from app.providers.vector_store.chroma import get_vector_store

def walk(node, pages: dict):
    if isinstance(node, dict):
        text = node.get("content")
        if text and text.strip():
            pages.setdefault(node.get("page number", 1), []).append(text.strip())
        for key in ("kids", "list items"):
            walk(node.get(key), pages)
    elif isinstance(node, list):
        for child in node:
            walk(child, pages)

data = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
buckets: dict[int, list[str]] = {}
walk(data.get("kids"), buckets)

pages = [
    ParsedPage(page_number=n, content=normalize_vietnamese("\n\n".join(buckets[n])))
    for n in sorted(buckets)
]

settings = get_settings()
document_id = str(uuid.uuid4())
chunks = _chunk_pages(
    pages=pages,
    chunk_size=settings.CHUNK_SIZE,
    chunk_overlap=settings.CHUNK_OVERLAP,
    document_id=document_id,
    file_name=data.get("file name", Path(sys.argv[1]).name),
    uploaded_at=datetime.now(timezone.utc).isoformat(),
)

for chunk, vec in zip(chunks, embed_documents([c.content for c in chunks])):
    chunk.embedding = vec

get_vector_store().add_chunks(chunks)
print(f"Đã nạp {len(chunks)} chunks từ {len(pages)} trang.")