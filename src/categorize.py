import json
import os
from dataclasses import dataclass
from time import time

import pandas as pd
from openai import OpenAI

from src.utils.utils import (
    GEMINI_FLASH_LITE_MODEL,
    OPENAI_MODEL,
    setup_gemini,
    setup_openai,
)


CATEGORIZE_CATEGORIES = [
    "Food",
    "Shopping",
    "Medical",
    "Transport",
    "Utilities",
    "Entertainment",
    "Travel",
    "Other",
]


@dataclass
class CategorizeResult:
    frame: pd.DataFrame
    summary: dict


class TransactionCategorizer:
    def __init__(self, config: object):
        self.primary_provider = (
            str(config.get("llm.default_provider", "gemini_flash_lite")).strip().lower()
        )
        self.primary_model = str(
            config.get("llm.primary_model", GEMINI_FLASH_LITE_MODEL)
        )
        self.fallback_model = str(config.get("llm.fallback_model", OPENAI_MODEL))
        self.gemini_base_url = str(
            config.get(
                "llm.gemini_base_url",
                "https://generativelanguage.googleapis.com/v1beta/openai/",
            )
        )

        self.enabled = bool(
            config.get("categorize.enabled", config.get("enrichment.enabled", True))
        )
        self.chunk_size = int(
            config.get(
                "categorize.chunk_size", config.get("enrichment.chunk_size", 100)
            )
        )

        setup_openai()
        self.openai_client = OpenAI()

        setup_gemini()
        gemini_key = os.getenv("GEMINI_API_KEY", "").strip()
        self.gemini_client = (
            OpenAI(api_key=gemini_key, base_url=self.gemini_base_url)
            if gemini_key
            else None
        )

        if self.primary_provider == "openai":
            self.primary_client = self.openai_client
            self.primary_model = self.fallback_model
            self.fallback_client = self.gemini_client
            self.fallback_model = GEMINI_FLASH_LITE_MODEL
        else:
            self.primary_client = self.gemini_client
            self.fallback_client = self.openai_client

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
        rates = {
            "gpt-4o-mini": 0.15,
            "gemini-2.5-flash-lite": 0.10,
        }
        return rates.get(model_name, 0.15)

    @staticmethod
    def _output_token_rate_per_million(model_name: str) -> float:
        rates = {
            "gpt-4o-mini": 0.60,
            "gemini-2.5-flash-lite": 0.40,
        }
        return rates.get(model_name, 0.60)

    def _record_usage(
        self, usage: object | None, model_name: str, is_fallback: bool
    ) -> None:
        if usage is None:
            return

        input_tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
        output_tokens = int(getattr(usage, "completion_tokens", 0) or 0)

        self.usage["inputTokens"] += input_tokens
        self.usage["outputTokens"] += output_tokens
        self.usage["llmCalls"] += 1
        if is_fallback:
            self.usage["fallbackCalls"] += 1

        input_cost = (input_tokens / 1_000_000) * self._input_token_rate_per_million(
            model_name
        )
        output_cost = (output_tokens / 1_000_000) * self._output_token_rate_per_million(
            model_name
        )
        self.usage["estimatedTotalCostUsd"] += input_cost + output_cost

    @staticmethod
    def _sanitize_category(value: object) -> str:
        if value is None:
            return "Other"
        text = str(value).strip().title()
        return text if text in CATEGORIZE_CATEGORIES else "Other"

    @staticmethod
    def _sanitize_confidence(value: object) -> float:
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
        stripped = text.strip()
        if stripped.startswith("```"):
            lines = stripped.splitlines()
            if len(lines) >= 3:
                return "\n".join(lines[1:-1]).strip()
        return stripped

    def _categorize_chunk(self, chunk: pd.DataFrame) -> list[dict]:
        rows = []
        for idx, row in chunk.iterrows():
            rows.append(
                {
                    "idx": int(idx),
                    "business_name": str(row.get("business_name", "") or "").strip(),
                    "total": float(row.get("total", 0.0) or 0.0),
                    "date": str(row.get("date", "") or ""),
                    "currency": str(row.get("currency", "") or "").strip().upper(),
                }
            )

        categories_json = json.dumps(CATEGORIZE_CATEGORIES)
        prompt = (
            "You are a transaction categorizer.\n"
            "Choose exactly one category per row from this enum: "
            f"{categories_json}.\n"
            "Return only JSON with this shape: "
            '{"items":[{"idx":123,"category":"Food","confidence":0.92}]}.\n'
            "Rules:"
            "\n- category must be one of the enum values exactly"
            "\n- confidence must be 0..1"
            "\n- if uncertain, use category 'Other' with lower confidence"
            "\nRows to classify:\n"
            f"{json.dumps(rows)}"
        )

        completion_kwargs = {"temperature": 0.0, "max_tokens": 1200}
        messages = [{"role": "user", "content": prompt}]

        response = None
        used_model = self.primary_model
        is_fallback = False

        if self.primary_client is not None:
            try:
                response = self.primary_client.chat.completions.create(
                    model=self.primary_model,
                    messages=messages,
                    **completion_kwargs,
                )
            except Exception:
                response = None

        if response is None:
            if self.fallback_client is None:
                raise RuntimeError(
                    "No LLM client available for transaction categorization"
                )
            response = self.fallback_client.chat.completions.create(
                model=self.fallback_model,
                messages=messages,
                **completion_kwargs,
            )
            used_model = self.fallback_model
            is_fallback = True

        self._record_usage(
            response.usage, model_name=used_model, is_fallback=is_fallback
        )

        raw = str(response.choices[0].message.content or "").strip()
        payload_text = self._strip_code_fence(raw)

        try:
            parsed = json.loads(payload_text)
        except json.JSONDecodeError:
            parsed = {"items": []}

        items = parsed.get("items", []) if isinstance(parsed, dict) else []
        if not isinstance(items, list):
            items = []

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
        out = frame.copy().reset_index(drop=True)
        out["category"] = "Other"

        chunk_size = max(1, self.chunk_size)
        for offset in range(0, len(out), chunk_size):
            chunk = out.iloc[offset : offset + chunk_size]
            chunk_results = self._categorize_chunk(chunk)
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
