"""Prompt templates for the Agriculture Q&A system."""

SYSTEM_PROMPT = """Bạn là trợ lý hỏi đáp về khuyến nông.

Chỉ sử dụng thông tin trong phần NGỮ CẢNH để trả lời câu hỏi.

Yêu cầu:
1. Trả lời bằng tiếng Việt, rõ ràng và dễ hiểu cho người làm nông.
2. Không tự bổ sung thông tin không xuất hiện trong NGỮ CẢNH.
3. Nếu NGỮ CẢNH không đủ để trả lời, hãy nói:
   "Tôi chưa tìm thấy đủ thông tin trong tài liệu hiện có để trả lời câu hỏi này."
4. Khi tài liệu có nhiều phương án, hãy trình bày từng phương án.
5. Không thay thế tư vấn trực tiếp của cán bộ nông nghiệp đối với tình huống
   có nguy cơ gây thiệt hại lớn cho cây trồng, vật nuôi hoặc môi trường.
6. Cuối câu trả lời, ghi nguồn theo định dạng:
   [Tên tài liệu, trang X]."""


USER_PROMPT_TEMPLATE = """NGỮ CẢNH:
{context}

CÂU HỎI:
{question}

TRẢ LỜI:"""


FALLBACK_ANSWER = (
    "Tôi chưa tìm thấy đủ thông tin trong tài liệu hiện có để trả lời câu hỏi này."
)


def build_user_prompt(context: str, question: str) -> str:
    """Build the user-facing prompt with context and question."""
    return USER_PROMPT_TEMPLATE.format(context=context, question=question)


def build_context_block(chunks: list) -> str:
    """
    Build the NGỮ CẢNH block from a list of RetrievalResult objects.

    Each chunk is formatted as:
    [Nguồn: <file_name>, trang <page>]
    <content>
    """
    parts = []
    for chunk in chunks:
        meta = chunk.metadata
        file_name = meta.get("file_name", "Không rõ tài liệu")
        page = meta.get("page", "?")
        header = f"[Nguồn: {file_name}, trang {page}]"
        parts.append(f"{header}\n{chunk.content}")
    return "\n\n---\n\n".join(parts)
