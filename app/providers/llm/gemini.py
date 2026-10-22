"""Gemini LLM provider using langchain-google-genai."""

import time

from app.core.config import get_settings
from app.core.exceptions import LLMConfigurationError, LLMRateLimitedError, LLMRequestFailedError
from app.core.logging import get_logger
from app.core.metrics import LLM_CALL_DURATION, LLM_CALLS, record_tokens
from app.providers.llm.base import LLMProvider

logger = get_logger(__name__)

_PROVIDER = "gemini"


def _record_usage(response) -> None:
    """Extract token usage from a LangChain response and feed the counters.

    ``usage_metadata`` is the normalised LangChain field; fall back to the
    raw ``response_metadata`` when the model/SDK does not populate it.
    """
    usage = getattr(response, "usage_metadata", None) or {}
    prompt = int(usage.get("input_tokens", 0) or 0)
    completion = int(usage.get("output_tokens", 0) or 0)
    total = int(usage.get("total_tokens", 0) or 0)
    if not (prompt or completion or total):
        return
    record_tokens(_PROVIDER, prompt=prompt, completion=completion, total=total)


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
            LLM_CALL_DURATION.labels(provider=_PROVIDER).observe(elapsed)
            LLM_CALLS.labels(provider=_PROVIDER, status="success").inc()
            _record_usage(response)
            logger.info("LLM response received in %.2fs", elapsed)
            return response.content
        except Exception as exc:
            elapsed = time.time() - start
            LLM_CALL_DURATION.labels(provider=_PROVIDER).observe(elapsed)
            err_str = str(exc).lower()
            logger.error("LLM call failed after %.2fs: %s", elapsed, type(exc).__name__)
            if "rate" in err_str or "quota" in err_str or "429" in err_str:
                LLM_CALLS.labels(provider=_PROVIDER, status="rate_limited").inc()
                raise LLMRateLimitedError(
                    "Gemini API tạm thời bị giới hạn. Vui lòng thử lại sau."
                ) from exc
            LLM_CALLS.labels(provider=_PROVIDER, status="error").inc()
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
