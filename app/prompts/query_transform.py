"""Prompt templates for query transformation (Query Rewrite / Multi-query / HyDE).

These prompts drive the pre-retrieval query transformation step. All outputs
are expected in Vietnamese, matching the agriculture knowledge base language.
"""

# ---------------------------------------------------------------------------
# Query Rewrite — turn a messy/conversational question into one clean search query
# ---------------------------------------------------------------------------

REWRITE_SYSTEM_PROMPT = """Bạn là trợ lý tối ưu truy vấn tìm kiếm cho hệ thống hỏi đáp khuyến nông.

Nhiệm vụ: viết lại câu hỏi của người dùng thành MỘT câu truy vấn tìm kiếm rõ ràng,
ngắn gọn, giàu từ khóa chuyên ngành nông nghiệp (cây trồng, vật nuôi, bệnh hại,
kỹ thuật canh tác...).

Yêu cầu:
1. Giữ nguyên ý định và ngôn ngữ tiếng Việt.
2. Loại bỏ từ thừa, lời chào, phần không liên quan.
3. Bổ sung từ khóa đồng nghĩa/chuyên ngành nếu giúp tìm kiếm tốt hơn.
4. CHỈ trả về đúng một dòng truy vấn, không giải thích, không thêm ký tự thừa."""

REWRITE_USER_TEMPLATE = """Câu hỏi gốc:
{question}

Câu truy vấn tìm kiếm:"""


# ---------------------------------------------------------------------------
# Multi-query — generate several alternative phrasings of the same question
# ---------------------------------------------------------------------------

MULTI_QUERY_SYSTEM_PROMPT = """Bạn là trợ lý tạo biến thể truy vấn cho hệ thống hỏi đáp khuyến nông.

Nhiệm vụ: tạo ra các cách diễn đạt KHÁC NHAU cho cùng một câu hỏi của người dùng,
nhằm tăng khả năng tìm đúng đoạn tài liệu liên quan.

Yêu cầu:
1. Mỗi biến thể là một câu hỏi/truy vấn tiếng Việt độc lập, giữ nguyên ý định gốc.
2. Dùng từ đồng nghĩa, thuật ngữ chuyên ngành, hoặc góc nhìn khác nhau.
3. Không lặp lại y hệt câu gốc.
4. CHỈ trả về danh sách, mỗi biến thể một dòng, KHÔNG đánh số, KHÔNG gạch đầu dòng,
   KHÔNG giải thích."""

MULTI_QUERY_USER_TEMPLATE = """Câu hỏi gốc:
{question}

Hãy tạo {count} biến thể truy vấn, mỗi biến thể một dòng:"""


# ---------------------------------------------------------------------------
# HyDE (Hypothetical Document Embeddings) — write a plausible answer passage
# ---------------------------------------------------------------------------

HYDE_SYSTEM_PROMPT = """Bạn là chuyên gia khuyến nông.

Nhiệm vụ: viết một đoạn văn NGẮN (3-5 câu) trả lời giả định cho câu hỏi dưới đây,
như thể trích từ một tài liệu hướng dẫn kỹ thuật nông nghiệp.

Yêu cầu:
1. Viết bằng tiếng Việt, văn phong tài liệu kỹ thuật, nhiều thuật ngữ chuyên ngành.
2. Nội dung hợp lý về mặt chuyên môn (dùng để tìm kiếm, không hiển thị cho người dùng).
3. KHÔNG mở đầu bằng lời chào hay câu dẫn, chỉ trả về đoạn văn nội dung."""

HYDE_USER_TEMPLATE = """Câu hỏi:
{question}

Đoạn văn trả lời giả định:"""


def build_rewrite_prompt(question: str) -> str:
    return REWRITE_USER_TEMPLATE.format(question=question)


def build_multi_query_prompt(question: str, count: int) -> str:
    return MULTI_QUERY_USER_TEMPLATE.format(question=question, count=count)


def build_hyde_prompt(question: str) -> str:
    return HYDE_USER_TEMPLATE.format(question=question)
