import re
from datetime import date
from typing import Any

import pandas as pd


class HelperAgent:
    """Answer natural-language questions over validated transaction rows."""

    @staticmethod
    def _to_frame(validated_rows: list[dict[str, Any]]) -> pd.DataFrame:
        if not isinstance(validated_rows, list) or not validated_rows:
            return pd.DataFrame(
                [],
                columns=[
                    "Transaction Business Name",
                    "Transaction Total",
                    "Transaction Date",
                    "Transaction Category",
                ],
            )

        frame = pd.DataFrame(validated_rows).copy()

        for col in [
            "Transaction Business Name",
            "Transaction Total",
            "Transaction Date",
            "Transaction Category",
        ]:
            if col not in frame.columns:
                frame[col] = None

        frame["Transaction Total"] = pd.to_numeric(
            frame["Transaction Total"], errors="coerce"
        )
        frame["Transaction Date"] = pd.to_datetime(
            frame["Transaction Date"], errors="coerce"
        )
        frame["Transaction Category"] = (
            frame["Transaction Category"].fillna("Other").astype(str)
        )

        frame = frame.dropna(subset=["Transaction Total", "Transaction Date"])
        return frame

    @staticmethod
    def _category_from_question(question: str) -> str | None:
        q = question.lower()
        category_keywords = {
            "food": ["food", "restaurant", "dining", "meal", "coffee", "cafe"],
            "grocery": ["grocery", "grocer", "supermarket"],
            "travel": ["travel", "flight", "hotel", "airfare"],
            "transport": ["transport", "uber", "lyft", "taxi", "train", "bus"],
            "shopping": ["shopping", "retail", "store"],
            "entertainment": ["entertainment", "movie", "theater", "concert"],
            "utilities": ["utilities", "electric", "water", "gas", "internet"],
            "health": ["health", "medical", "pharmacy", "doctor"],
        }

        for category, words in category_keywords.items():
            if any(word in q for word in words):
                return category
        return None

    @staticmethod
    def _is_this_month(question: str) -> bool:
        q = question.lower()
        return "this month" in q or "current month" in q

    @staticmethod
    def _metric_from_question(question: str) -> str:
        q = question.lower()
        if "how many" in q or "count" in q or "number of" in q:
            return "count"
        if "average" in q or "avg" in q or "mean" in q:
            return "avg"
        return "sum"

    @staticmethod
    def _top_n_from_question(question: str) -> int | None:
        match = re.search(r"top\s+(\d+)", question.lower())
        if not match:
            return None
        try:
            return int(match.group(1))
        except ValueError:
            return None

    def answer(
        self,
        question: str,
        validated_rows: list[dict[str, Any]],
        today: date | None = None,
    ) -> dict[str, Any]:
        frame = self._to_frame(validated_rows)
        plan = {
            "metric": self._metric_from_question(question),
            "category": self._category_from_question(question),
            "thisMonth": self._is_this_month(question),
            "topN": self._top_n_from_question(question),
        }

        if frame.empty:
            return {
                "answer": "No validated transactions are available yet.",
                "rowsScanned": 0,
                "queryPlan": plan,
                "confidence": "low",
            }

        scoped = frame.copy()

        if plan["thisMonth"]:
            now = pd.Timestamp(today or date.today())
            scoped = scoped[
                (scoped["Transaction Date"].dt.year == now.year)
                & (scoped["Transaction Date"].dt.month == now.month)
            ]

        if plan["category"]:
            category_token = str(plan["category"]).lower()
            scoped = scoped[
                scoped["Transaction Category"]
                .str.lower()
                .str.contains(re.escape(category_token), na=False)
            ]

        if scoped.empty:
            return {
                "answer": "I could not find validated transactions matching that question.",
                "rowsScanned": int(len(frame)),
                "queryPlan": plan,
                "confidence": "medium",
            }

        if plan["topN"]:
            grouped = (
                scoped.groupby("Transaction Category", dropna=False)[
                    "Transaction Total"
                ]
                .sum()
                .sort_values(ascending=False)
            )
            top_n = max(1, int(plan["topN"]))
            top = grouped.head(top_n)
            lines = [f"{idx}: ${float(val):,.2f}" for idx, val in top.items()]
            answer = "Top categories by spend:\n" + "\n".join(lines)
            confidence = "high"
        elif plan["metric"] == "count":
            answer = f"You have {len(scoped)} validated transactions in that scope."
            confidence = "high"
        elif plan["metric"] == "avg":
            avg_val = float(scoped["Transaction Total"].mean())
            answer = f"Average spend is ${avg_val:,.2f}."
            confidence = "high"
        else:
            total = float(scoped["Transaction Total"].sum())
            scope_bits = []
            if plan["category"]:
                scope_bits.append(f"on {plan['category']}")
            if plan["thisMonth"]:
                scope_bits.append("this month")
            scope_text = " " + " ".join(scope_bits) if scope_bits else ""
            answer = f"You spent ${total:,.2f}{scope_text}."
            confidence = "high"

        return {
            "answer": answer,
            "rowsScanned": int(len(frame)),
            "queryPlan": plan,
            "confidence": confidence,
        }
