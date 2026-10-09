"""Base abstraction for LLM providers."""

from abc import ABC, abstractmethod


class LLMProvider(ABC):
    """Abstract base class for LLM providers.

    This abstraction allows swapping Gemini for Ollama, HuggingFace, or any
    other backend without changing the API or retrieval pipeline.
    """

    @abstractmethod
    def generate(self, system_prompt: str, user_prompt: str) -> str:
        """Generate a response from the LLM.

        Args:
            system_prompt: Instructions for the model.
            user_prompt: The user-facing prompt (context + question).

        Returns:
            The generated text response.
        """

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Return the model identifier used for this provider."""
