"""Gemini LLM provider using langchain-google-genai."""

import time

from app.core.config import get_settings
from app.core.exceptions import LLMConfigurationError, LLMRateLimitedError, LLMRequestFailedError
from app.core.logging import get_logger
from app.providers.llm.base import LLMProvider

logger = get_logger(__name__)


class GeminiProvider(LLMProvider):
    """LLM provider backed by Google Gemini via langchain-google-genai."""

    def __init__(self):
        settings = get_settings()
        if not settings.GEMINI_API_KEY:
            raise LLMConfigurationError(
                "GEMINI_API_KEY chưa được cấu hình. Vui lòng kiểm tra file .env."
            )
        self._model_name = settings.GEMINI_MODEL
        self._temperature = settings.LLM_TEMPERATURE
        self._api_key = settings.GEMINI_API_KEY
        self._client = None

    def _get_client(self):
        if self._client is None:
            from langchain_google_genai import ChatGoogleGenerativeAI  # type: ignore

            self._client = ChatGoogleGenerativeAI(
                model=self._model_name,
                google_api_key=self._api_key,
                temperature=self._temperature,
            )
            logger.info("Gemini client initialized: model=%s", self._model_name)
        return self._client

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        from langchain_core.messages import HumanMessage, SystemMessage  # type: ignore

        client = self._get_client()
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt),
        ]

        start = time.time()
        logger.info("Calling LLM provider=gemini model=%s", self._model_name)
        try:
            response = client.invoke(messages)
            elapsed = time.time() - start
            logger.info("LLM response received in %.2fs", elapsed)
            return response.content
        except Exception as exc:
            elapsed = time.time() - start
            err_str = str(exc).lower()
            logger.error("LLM call failed after %.2fs: %s", elapsed, type(exc).__name__)
            if "rate" in err_str or "quota" in err_str or "429" in err_str:
                raise LLMRateLimitedError(
                    "Gemini API tạm thời bị giới hạn. Vui lòng thử lại sau."
                ) from exc
            raise LLMRequestFailedError(
                "Không thể kết nối tới dịch vụ LLM. Vui lòng thử lại sau."
            ) from exc

    @property
    def model_name(self) -> str:
        return self._model_name


_gemini_instance: GeminiProvider | None = None


def get_gemini_provider() -> GeminiProvider:
    """Return a singleton GeminiProvider."""
    global _gemini_instance
    if _gemini_instance is None:
        _gemini_instance = GeminiProvider()
    return _gemini_instance
