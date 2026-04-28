import json
from dataclasses import dataclass
from time import time

import pandas as pd
from google.genai import types
from src.intelligence.llm_base import LLMBase


CATEGORIZE_CATEGORIES = [
    "Food",
    "Shopping",
    "Medical",
    "Transport",
    "Utilities",
    "Entertainment",
    "Travel",
    "Health & Fitness",
    "Other",
]


@dataclass
class CategorizeResult:
    """
    Container for the output of a categorization run.

    Attributes:
        frame: The input DataFrame with a ``category`` column appended.
        summary: Token/cost/latency metadata from the LLM call(s).
    """

    frame: pd.DataFrame
    summary: dict


class TransactionCategorizer(LLMBase):
    """
    Categorize transaction rows using a Gemini LLM.

    Processes the DataFrame in chunks to stay within token limits and records
    cumulative token usage and estimated cost across all LLM calls.
    """

    def __init__(self, config: object, llm_config_path: str = "config/llm_config.conf"):
        """
        Initialize the categorizer from app and LLM configuration objects.

        Args:
            config: Parsed application config (HOCON) exposing ``categorize.*`` keys.
            llm_config_path: Path to the HOCON file controlling model selection
                and sampling parameters.
        """
        super().__init__(
            llm_config_path=llm_config_path,
            config_section="categorization",
            default_temperature=0.0,
            default_top_p=1.0,
            default_max_tokens=1200,
        )

        self.primary_model = self.model_name

        # Allow categorization to be disabled entirely via config
        self.enabled = bool(
            config.get("categorize.enabled", config.get("enrichment.enabled", True))
        )
        self.chunk_size = int(
            config.get(
                "categorize.chunk_size", config.get("enrichment.chunk_size", 100)
            )
        )

        self.primary_client = self.init_genai_client()

        # Running totals for usage reporting
        self.usage = {
            "model": self.primary_model,
            "inputTokens": 0,
            "outputTokens": 0,
            "llmCalls": 0,
            "fallbackCalls": 0,
            "estimatedTotalCostUsd": 0.0,
            "latencySeconds": 0.0,
            "rowsProcessed": 0,
        }

    @staticmethod
    def _input_token_rate_per_million(model_name: str) -> float:
        """
        Return the input token cost in USD per one million tokens for a given model.

        Args:
            model_name: Gemini model identifier string.

        Returns:
            Cost per million input tokens in USD. Defaults to 0.10 for unknown models.
        """
        rates = {
            "gemini-2.5-flash-lite": 0.10,
        }
        return rates.get(model_name, 0.10)

    @staticmethod
    def _output_token_rate_per_million(model_name: str) -> float:
        """
        Return the output token cost in USD per one million tokens for a given model.

        Args:
            model_name: Gemini model identifier string.

        Returns:
            Cost per million output tokens in USD. Defaults to 0.40 for unknown models.
        """
        rates = {
            "gemini-2.5-flash-lite": 0.40,
        }
        return rates.get(model_name, 0.40)

    def _record_usage(
        self, usage: object | None, model_name: str, is_fallback: bool
    ) -> None:
        """
        Accumulate token counts and estimated cost from a single LLM response.

        Args:
            usage: The usage metadata object attached to the Gemini response, or
                ``None`` if the response did not include usage information.
            model_name: Model identifier used for per-model cost lookup.
            is_fallback: Whether this call was made on the fallback model path.
        """
        if usage is None:
            return

        # Support both OpenAI-style and Gemini-style attribute names
        input_tokens = int(
            getattr(usage, "prompt_tokens", None)
            or getattr(usage, "prompt_token_count", 0)
            or 0
        )
        output_tokens = int(
            getattr(usage, "completion_tokens", None)
            or getattr(usage, "candidates_token_count", 0)
            or 0
        )

        self.usage["inputTokens"] += input_tokens
        self.usage["outputTokens"] += output_tokens
        self.usage["llmCalls"] += 1
        if is_fallback:
            self.usage["fallbackCalls"] += 1

        # Accumulate cost using per-model rates
        input_cost = (input_tokens / 1e6) * self._input_token_rate_per_million(
            model_name
        )
        output_cost = (output_tokens / 1e6) * self._output_token_rate_per_million(
            model_name
        )
        self.usage["estimatedTotalCostUsd"] += input_cost + output_cost

    @staticmethod
    def _sanitize_category(value: object) -> str:
        """
        Coerce an LLM-returned category string to a valid ``CATEGORIZE_CATEGORIES`` value.

        Args:
            value: Raw category value from the parsed JSON response.

        Returns:
            A title-cased string from ``CATEGORIZE_CATEGORIES``, or ``"Other"`` if
            the value is missing, ``None``, or not in the allowed set.
        """
        if value is None:
            return "Other"
        text = str(value).strip().title()
        return text if text in CATEGORIZE_CATEGORIES else "Other"

    @staticmethod
    def _sanitize_confidence(value: object) -> float:
        """
        Clamp an LLM-returned confidence value to the range ``[0.0, 1.0]``.

        Args:
            value: Raw confidence value from the parsed JSON response.

        Returns:
            A float between 0.0 and 1.0, or 0.0 if the value cannot be parsed.
        """
        try:
            conf = float(value)
        except (TypeError, ValueError):
            return 0.0
        if conf < 0.0:
            return 0.0
        if conf > 1.0:
            return 1.0
        return conf

    @staticmethod
    def _strip_code_fence(text: str) -> str:
        """
        Remove a markdown code fence wrapper (``` ... ```) if present.

        Args:
            text: Raw string that may be wrapped in a triple-backtick fence.

        Returns:
            The inner content with fence markers and language hint stripped,
            or the original string if no fence was detected.
        """
        stripped = text.strip()
        if stripped.startswith("```"):
            lines = stripped.splitlines()
            # A valid fence has at least an opening line, content, and closing line
            if len(lines) >= 3:
                return "\n".join(lines[1:-1]).strip()
        return stripped

    def _categorize_chunk(self, chunk: pd.DataFrame) -> list[dict]:
        """
        Send a single chunk of rows to the LLM and return parsed category assignments.

        Args:
            chunk: A slice of the full DataFrame, indexed with original row indices.

        Returns:
            A list of dicts, each with ``idx`` (original DataFrame index) and
            ``category`` (validated category string) keys.
        """
        # Build a minimal row representation to keep prompt tokens low
        rows = []
        for idx, row in chunk.iterrows():
            rows.append(
                {
                    "idx": int(idx),
                    "business_name": str(row.get("business_name", "") or "").strip(),
                }
            )

        categories_json = json.dumps(CATEGORIZE_CATEGORIES)
        prompt = (
            "You are a transaction categorizer.\n"
            "Classify each row using business_name only.\n"
            "Choose exactly one category per row from this enum: "
            f"{categories_json}.\n"
            "Return only JSON with this shape: "
            '{"items":[{"idx":123,"category":"Food"}]}.\n'
            "Rules:"
            "\n- category must be one of the enum values exactly"
            "\n- if uncertain, use category 'Other' with lower confidence"
            "\nRows to classify:\n"
            f"{json.dumps(rows)}"
        )

        completion_kwargs = {
            "temperature": self.temperature,
            "top_p": self.top_p,
            "max_output_tokens": self.max_tokens,
            # Request structured JSON output directly to avoid parsing fence wrappers
            "response_mime_type": "application/json",
        }

        response = None
        used_model = self.primary_model

        if self.primary_client is not None:
            try:
                response = self.primary_client.models.generate_content(
                    model=self.primary_model,
                    contents=prompt,
                    config=types.GenerateContentConfig(**completion_kwargs),
                )
            except Exception:
                response = None

        if response is None:
            raise RuntimeError(
                "No Gemini client available for transaction categorization"
            )

        self._record_usage(
            getattr(response, "usage_metadata", None),
            model_name=used_model,
            is_fallback=False,
        )

        raw = str(getattr(response, "text", "") or "").strip()
        payload_text = self._strip_code_fence(raw)

        try:
            parsed = json.loads(payload_text)
        except json.JSONDecodeError:
            # Treat malformed JSON as an empty result rather than crashing
            parsed = {"items": []}

        items = parsed.get("items", []) if isinstance(parsed, dict) else []
        if not isinstance(items, list):
            items = []

        # Validate and normalise each item; silently drop anything malformed
        normalized = []
        for item in items:
            if not isinstance(item, dict):
                continue
            try:
                idx = int(item.get("idx"))
            except (TypeError, ValueError):
                continue

            normalized.append(
                {
                    "idx": idx,
                    "category": self._sanitize_category(item.get("category")),
                }
            )

        return normalized

    def categorize_dataframe(self, frame: pd.DataFrame) -> CategorizeResult:
        """
        Categorize every row in *frame* and return a ``CategorizeResult``.

        Rows are processed in chunks of ``self.chunk_size`` to stay within token
        limits. The ``category`` column is written back to the output DataFrame.
        When categorization is disabled the column is set to ``"Other"`` for all rows.

        Args:
            frame: Input DataFrame. Must contain at least a ``business_name`` column.

        Returns:
            A ``CategorizeResult`` with the annotated DataFrame and a usage summary dict.
        """
        if frame is None or frame.empty:
            out = frame.copy() if isinstance(frame, pd.DataFrame) else pd.DataFrame([])
            out["category"] = []
            return CategorizeResult(
                out,
                {
                    "model": self.primary_model,
                    "rowsProcessed": 0,
                    "llmCalls": 0,
                    "fallbackCalls": 0,
                    "inputTokens": 0,
                    "outputTokens": 0,
                    "estimatedTotalCostUsd": 0.0,
                    "latencySeconds": 0.0,
                },
            )

        if not self.enabled:
            out = frame.copy()
            out["category"] = "Other"
            return CategorizeResult(
                out,
                {
                    "model": "disabled",
                    "rowsProcessed": int(len(out)),
                    "llmCalls": 0,
                    "fallbackCalls": 0,
                    "inputTokens": 0,
                    "outputTokens": 0,
                    "estimatedTotalCostUsd": 0.0,
                    "latencySeconds": 0.0,
                },
            )

        start = time()
        # Reset index so chunk offsets align with DataFrame positions
        out = frame.copy().reset_index(drop=True)
        out["category"] = "Other"

        chunk_size = max(1, self.chunk_size)
        for offset in range(0, len(out), chunk_size):
            chunk = out.iloc[offset : offset + chunk_size]
            chunk_results = self._categorize_chunk(chunk)
            # Write results back by original index; skip any out-of-range indices
            for item in chunk_results:
                idx = int(item["idx"])
                if idx < 0 or idx >= len(out):
                    continue
                out.at[idx, "category"] = item["category"]

        self.usage["rowsProcessed"] = int(len(out))
        self.usage["latencySeconds"] = round(time() - start, 3)

        summary = {
            "model": str(self.usage["model"]),
            "rowsProcessed": int(self.usage["rowsProcessed"]),
            "llmCalls": int(self.usage["llmCalls"]),
            "fallbackCalls": int(self.usage["fallbackCalls"]),
            "inputTokens": int(self.usage["inputTokens"]),
            "outputTokens": int(self.usage["outputTokens"]),
            "estimatedTotalCostUsd": round(
                float(self.usage["estimatedTotalCostUsd"]), 6
            ),
            "latencySeconds": float(self.usage["latencySeconds"]),
        }
        return CategorizeResult(out, summary)
