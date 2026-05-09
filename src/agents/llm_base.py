import os
from functools import lru_cache
from typing import Any, Iterator

from google import genai
from langchain_google_genai import ChatGoogleGenerativeAI
from pyhocon import ConfigFactory

DEFAULT_GEMINI_MODEL = "gemini-2.5-flash-lite"


class LLMBase:
    """
    Shared LLM bootstrap for config-driven Gemini model initialization.

    Centralizes model parameter loading, API key resolution, client creation,
    and lightweight streaming helpers for Gemini-backed components.
    """

    def __init__(
        self,
        llm_config_path: str,
        config_section: str,
        default_temperature: float,
        default_top_p: float,
        default_max_tokens: int,
    ) -> None:
        """
        Load model parameters from ``llm.<config_section>`` in the LLM config file.

        Args:
            llm_config_path: Filesystem path to the HOCON LLM config file.
            config_section: Section name under the top-level ``llm`` key
                (for example: ``"helper_agent"``).
            default_temperature: Fallback temperature when not present in config.
            default_top_p: Fallback top-p value when not present in config.
            default_max_tokens: Fallback max token limit when not present in config.
        """
        llm_cfg = self._load_llm_config_cached(llm_config_path)
        section_prefix = f"llm.{config_section}"

        self.model_name = str(
            llm_cfg.get(f"{section_prefix}.model", DEFAULT_GEMINI_MODEL)
        )
        self.temperature = float(
            llm_cfg.get(f"{section_prefix}.temperature", default_temperature)
        )
        self.top_p = float(llm_cfg.get(f"{section_prefix}.top_p", default_top_p))
        self.max_tokens = int(
            llm_cfg.get(f"{section_prefix}.max_tokens", default_max_tokens)
        )

    @staticmethod
    @lru_cache(maxsize=8)
    def _load_llm_config_cached(config_path: str) -> Any:
        """
        Parse and cache a HOCON LLM config file by path.

        Args:
            config_path: Filesystem path to the config file.

        Returns:
            Parsed config object with dot-notation key access.
        """
        return ConfigFactory.parse_file(config_path)

    @staticmethod
    def resolve_api_key(allow_test_key: bool = False) -> str:
        """
        Resolve a Gemini API key from environment variables.

        Args:
            allow_test_key: If True, returns ``"test-key"`` when no real key is found.

        Returns:
            API key string from ``GEMINI_API_KEY`` or ``GOOGLE_API_KEY``.

        Raises:
            ValueError: If no key is found and ``allow_test_key`` is False.
        """
        api_key = os.getenv("GEMINI_API_KEY", "").strip()
        if not api_key:
            # Local dev fallback: read the dedicated Gemini secret file directly.
            try:
                with open("secrets/google_gemini_api_key", "r", encoding="utf-8") as f:
                    api_key = f.read().strip()
            except OSError:
                api_key = ""

        if not api_key:
            api_key = os.getenv("GOOGLE_API_KEY", "").strip()

        if api_key:
            return api_key
        if allow_test_key:
            return "test-key"
        raise ValueError("Missing GEMINI_API_KEY or GOOGLE_API_KEY")

    def init_chat_model(
        self,
        model_name: str | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        max_tokens: int | None = None,
        allow_test_key: bool = False,
    ) -> ChatGoogleGenerativeAI:
        """
        Initialize a chat model from base defaults with optional overrides.

        Args:
            model_name: Optional model identifier override.
            temperature: Optional temperature override.
            top_p: Optional top-p override.
            max_tokens: Optional max output token override.
            allow_test_key: Whether to allow ``"test-key"`` when no API key is set.

        Returns:
            Configured ``ChatGoogleGenerativeAI`` instance.
        """
        return ChatGoogleGenerativeAI(
            model=model_name or self.model_name,
            temperature=self.temperature if temperature is None else temperature,
            top_p=self.top_p if top_p is None else top_p,
            max_output_tokens=self.max_tokens if max_tokens is None else max_tokens,
            google_api_key=self.resolve_api_key(allow_test_key=allow_test_key),
        )

    def init_genai_client(self) -> genai.Client | None:
        """
        Initialize a low-level Gemini client.

        Returns:
            ``genai.Client`` when a valid API key is available, otherwise ``None``.
        """
        try:
            return genai.Client(api_key=self.resolve_api_key(allow_test_key=False))
        except ValueError:
            return None

    @staticmethod
    def _content_to_text(content: Any) -> str:
        """
        Normalize model output content into plain text.

        Args:
            content: Content payload from model output.

        Returns:
            A plain-text representation of the content.
        """
        if isinstance(content, str):
            return content

        if isinstance(content, list):
            chunks: list[str] = []
            for item in content:
                if isinstance(item, str):
                    chunks.append(item)
                elif isinstance(item, dict):
                    text = item.get("text")
                    if isinstance(text, str):
                        chunks.append(text)
            return "".join(chunks)

        if isinstance(content, dict):
            text = content.get("text")
            return text if isinstance(text, str) else ""

        text_attr = getattr(content, "text", None)
        return text_attr if isinstance(text_attr, str) else ""

    @staticmethod
    def _normalize_chat_history(
        chat_history: list[dict[str, Any]] | None,
        limit: int,
    ) -> list[dict[str, str]]:
        """
        Normalize chat history into validated ``{"role", "text"}`` turns.

        Args:
            chat_history: Optional prior conversation turns.
            limit: Maximum number of latest turns to keep.

        Returns:
            Filtered and normalized turns for user/assistant roles only.
        """
        if not isinstance(chat_history, list) or not chat_history:
            return []

        normalized: list[dict[str, str]] = []
        for item in chat_history[-max(1, int(limit)) :]:
            if not isinstance(item, dict):
                continue
            role = str(item.get("role", "")).strip().lower()
            text = str(item.get("text", "") or "").strip()
            if role in {"user", "assistant"} and text:
                normalized.append({"role": role, "text": text})
        return normalized

    @classmethod
    def history_lines(
        cls,
        chat_history: list[dict[str, Any]] | None,
        limit: int = 10,
    ) -> list[str]:
        """
        Render compact ``"role: text"`` lines from chat history.

        Args:
            chat_history: Optional prior conversation turns.
            limit: Maximum number of latest turns to include.

        Returns:
            List of compact role-prefixed history lines.
        """
        turns = cls._normalize_chat_history(chat_history=chat_history, limit=limit)
        return [f"{turn['role']}: {turn['text']}" for turn in turns]

    @classmethod
    def messages_with_history(
        cls,
        question: str,
        chat_history: list[dict[str, Any]] | None,
        max_history_messages: int = 20,
    ) -> list[dict[str, str]]:
        """
        Build ``{"role", "content"}`` messages from history plus a question.

        Args:
            question: Current user question to append as the final message.
            chat_history: Optional prior conversation turns.
            max_history_messages: Max number of latest turns to prepend.

        Returns:
            Message list ready for LangChain/LangGraph invocation.
        """
        turns = cls._normalize_chat_history(
            chat_history=chat_history,
            limit=max_history_messages,
        )
        messages = [{"role": t["role"], "content": t["text"]} for t in turns]
        messages.append({"role": "user", "content": question})
        return messages

    @classmethod
    def build_synthesis_messages(
        cls,
        question: str,
        chat_history: list[dict[str, Any]] | None,
        tool_outputs: list[str],
        system_prompt: str,
        history_limit: int = 10,
    ) -> list[dict[str, str]]:
        """Build generic synthesis messages from question, context, and tool outputs."""
        history_lines = cls.history_lines(chat_history, limit=history_limit)
        return [
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": (
                    f"Question:\n{question}\n\n"
                    f"Recent chat context:\n{chr(10).join(history_lines) if history_lines else '(none)'}\n\n"
                    f"Tool outputs (JSON/text):\n{chr(10).join(tool_outputs)}\n\n"
                    "Now write the final answer to the user."
                ),
            },
        ]

    @classmethod
    def synthesize_from_tool_outputs(
        cls,
        model: Any,
        question: str,
        chat_history: list[dict[str, Any]] | None,
        tool_outputs: list[str],
        fallback_text: str,
        system_prompt: str,
    ) -> str:
        """Run optional synthesis using a provided model and return fallback on failure."""
        if not tool_outputs:
            return fallback_text

        synthesis_messages = cls.build_synthesis_messages(
            question=question,
            chat_history=chat_history,
            tool_outputs=tool_outputs,
            system_prompt=system_prompt,
        )

        try:
            try:
                synthesis_result = model.invoke({"messages": synthesis_messages})
            except Exception:
                synthesis_result = model.invoke(synthesis_messages)

            if isinstance(synthesis_result, dict):
                messages = synthesis_result.get("messages", [])
                if messages:
                    synthesized = cls._content_to_text(
                        getattr(messages[-1], "content", "")
                    ).strip()
                    if synthesized:
                        return synthesized

            synthesized = cls._content_to_text(
                getattr(synthesis_result, "content", "")
            ).strip()
            if synthesized:
                return synthesized
        except Exception:
            pass

        return fallback_text

    def stream(
        self,
        user_input: str,
        model: ChatGoogleGenerativeAI | None = None,
    ) -> Iterator[str]:
        """
        Stream model output tokens for a plain user input string.

        Args:
            user_input: Raw user prompt text.
            model: Optional pre-initialized chat model. If omitted, uses
                ``init_chat_model()`` with base defaults.

        Yields:
            Text tokens from the streamed model response.
        """
        active_model = model or self.init_chat_model()
        for chunk in active_model.stream(user_input):
            token = self._content_to_text(getattr(chunk, "content", chunk))
            if token:
                yield token
