import numpy as np
import pandas as pd
from time import time
from dataclasses import dataclass
import re
from concurrent.futures import ThreadPoolExecutor

from pyhocon import ConfigFactory

from fuzzywuzzy import process, fuzz
from src.agents.categorize import TransactionCategorizer

pd.set_option("display.max_columns", None)


@dataclass
class Results:
    """
    Container for all outputs produced by a single validation run.

    Attributes:
        validated_transactions: Rows where transaction and proof amounts matched exactly.
        discrepancies: Matched pairs where the totals differed (non-zero delta).
        unmatched_transactions: Transactions that could not be paired with any proof.
        unmatched_proofs: Proofs that could not be paired with any transaction.
    """

    def __init__(
        self,
        validated_transactions: pd.DataFrame = None,
        discrepancies: pd.DataFrame = None,
        unmatched_transactions: pd.DataFrame = None,
        unmatched_proofs: pd.DataFrame = None,
    ):
        self.validated_transactions = validated_transactions
        self.discrepancies = discrepancies
        self.unmatched_transactions = unmatched_transactions
        self.unmatched_proofs = unmatched_proofs


class Validator:
    """
    Match transactions against receipts/proofs and surface discrepancies.

    Uses fuzzy business-name matching and date windowing to build candidate
    pairs, then resolves conflicts greedily by similarity score and total delta.
    """

    def __init__(
        self,
        transactions: pd.DataFrame,
        proofs: pd.DataFrame,
        config_path: str = "config/config.conf",
        parsed_config: object | None = None,
    ):
        """
        Initialize the Validator with transaction and proof DataFrames.

        Args:
            transactions: DataFrame of bank/card transactions with columns
                ``business_name``, ``total``, ``date``, and ``currency``.
            proofs: DataFrame of receipts/proofs with the same schema.
            config_path: Path to the HOCON app config file. Ignored when
                *parsed_config* is supplied.
            parsed_config: Pre-parsed config object. Takes precedence over
                *config_path* to avoid redundant disk I/O.
        """
        self.transactions = transactions
        self.proofs = proofs
        # Prefer an already-parsed config to avoid re-reading the file
        self.config = (
            parsed_config
            if parsed_config is not None
            else ConfigFactory.parse_file(config_path)
        )
        self.categorize_cost: dict = {}
        self.recommendation_matching = self._load_recommendation_matching_config()

    def _load_recommendation_matching_config(self) -> dict[str, float | int]:
        """Load recommendation matching thresholds from config with safe defaults."""
        defaults: dict[str, float | int] = {
            "name_similarity_threshold": 0.70,
            "date_window_days": 1,
            "amount_threshold": 0.05,
        }

        config_section: object = {}
        if self.config is not None and hasattr(self.config, "get"):
            config_section = self.config.get("recommendation_matching", {})

        if config_section is None:
            config_section = {}

        if not hasattr(config_section, "get"):
            config_section = {}

        name_similarity_threshold = defaults["name_similarity_threshold"]
        raw_name_threshold = config_section.get("name_similarity_threshold")
        if raw_name_threshold is not None:
            try:
                parsed_name_threshold = float(raw_name_threshold)
                if 0.0 <= parsed_name_threshold <= 1.0:
                    name_similarity_threshold = parsed_name_threshold
            except (TypeError, ValueError):
                pass

        date_window_days = defaults["date_window_days"]
        raw_date_window = config_section.get("date_window_days")
        if raw_date_window is not None:
            try:
                parsed_date_window = int(raw_date_window)
                if parsed_date_window >= 0:
                    date_window_days = parsed_date_window
            except (TypeError, ValueError):
                pass

        amount_threshold = defaults["amount_threshold"]
        raw_amount_threshold = config_section.get("amount_threshold")
        if raw_amount_threshold is not None:
            try:
                parsed_amount_threshold = float(raw_amount_threshold)
                if parsed_amount_threshold >= 0:
                    amount_threshold = parsed_amount_threshold
            except (TypeError, ValueError):
                pass

        return {
            "name_similarity_threshold": name_similarity_threshold,
            "date_window_days": date_window_days,
            "amount_threshold": amount_threshold,
        }

    def _categorize_inputs(self) -> None:
        """
        Categorize transaction and proof rows in parallel and track aggregate model usage.

        Runs two ``TransactionCategorizer`` instances concurrently via a thread pool
        and merges their usage summaries into ``self.categorize_cost``.
        """
        categorizer_tx = TransactionCategorizer(self.config)
        categorizer_pr = TransactionCategorizer(self.config)

        # Categorize transactions and proofs concurrently to reduce wall-clock time
        with ThreadPoolExecutor(max_workers=2) as executor:
            tx_future = executor.submit(
                categorizer_tx.categorize_dataframe,
                self.transactions,
            )
            pr_future = executor.submit(
                categorizer_pr.categorize_dataframe,
                self.proofs,
            )
            tx_enriched = tx_future.result()
            pr_enriched = pr_future.result()

        self.transactions = tx_enriched.frame
        self.proofs = pr_enriched.frame

        # Merge usage summaries from both categorizers into a single aggregate dict
        costs = [tx_enriched.summary, pr_enriched.summary]
        models = [str(cost.get("model", "unknown")) for cost in costs if cost]
        self.categorize_cost = {
            "model": "+".join(models) if models else "unknown",
            "rowsProcessed": int(
                sum(int(cost.get("rowsProcessed", 0) or 0) for cost in costs)
            ),
            "llmCalls": int(sum(int(cost.get("llmCalls", 0) or 0) for cost in costs)),
            "fallbackCalls": int(
                sum(int(cost.get("fallbackCalls", 0) or 0) for cost in costs)
            ),
            "inputTokens": int(
                sum(int(cost.get("inputTokens", 0) or 0) for cost in costs)
            ),
            "outputTokens": int(
                sum(int(cost.get("outputTokens", 0) or 0) for cost in costs)
            ),
            "estimatedTotalCostUsd": round(
                sum(
                    float(cost.get("estimatedTotalCostUsd", 0.0) or 0.0)
                    for cost in costs
                ),
                6,
            ),
            "latencySeconds": round(
                sum(float(cost.get("latencySeconds", 0.0) or 0.0) for cost in costs),
                3,
            ),
        }

    @staticmethod
    def match_business_names(transaction_name: str, proofs: list[str], threshold=80):
        """
        Find the closest matching proof business name for a given transaction name.

        Args:
            transaction_name: Business name from the transaction row.
            proofs: List of business name strings from available proof rows.
            threshold: Minimum fuzzy-match score (0–100) required for a match.
                Defaults to 80.

        Returns:
            The best-matching proof name as a string, or ``None`` if no candidate
            meets the *threshold* or if either input is empty.
        """
        if transaction_name is None or proofs is None or len(proofs) == 0:
            return None

        candidate = process.extractOne(
            str(transaction_name), proofs, scorer=fuzz.partial_ratio
        )
        if candidate is None:
            return None

        match, score = candidate
        return match if score >= threshold else None

    @staticmethod
    def _normalize_date_value(value: object) -> pd.Timestamp:
        """
        Parse raw or noisy date text into a normalised midnight ``Timestamp`` or ``NaT``.

        First attempts a direct ``pd.to_datetime`` parse. If that fails, extracts a
        date-shaped token via regex before retrying.

        Args:
            value: Any value that may represent a date (string, datetime, etc.).

        Returns:
            A ``pd.Timestamp`` normalised to midnight, or ``pd.NaT`` on failure.
        """
        raw = str(value).strip()
        parsed = pd.to_datetime(raw, errors="coerce")
        if pd.notna(parsed):
            return parsed.normalize()

        # Fall back to extracting a recognisable date substring
        token = re.search(
            r"(\d{4}[-/]\d{1,2}[-/]\d{1,2}|\d{1,2}[-/]\d{1,2}[-/]\d{2,4})",
            raw,
        )
        if not token:
            return pd.NaT

        parsed_token = pd.to_datetime(token.group(1), errors="coerce")
        if pd.isna(parsed_token):
            return pd.NaT

        return parsed_token.normalize()

    @staticmethod
    def _date_key(value: object) -> str:
        """
        Build a stable ``YYYY-MM-DD`` string key for date-based matching.

        Using a normalised string key lets us group rows by calendar date without
        being sensitive to time-of-day or format noise.

        Args:
            value: Any value representing a date.

        Returns:
            A ``"YYYY-MM-DD"`` string, or the lowercased raw string if parsing fails.
        """
        parsed = Validator._normalize_date_value(value)
        if pd.notna(parsed):
            return parsed.strftime("%Y-%m-%d")
        return str(value).strip().lower()

    @staticmethod
    def _name_similarity(left: object, right: object) -> int:
        """
        Compute a fuzzy partial-ratio similarity score for two business name strings.

        Both inputs are lowercased and stripped before comparison to reduce noise.

        Args:
            left: First business name (transaction side).
            right: Second business name (proof side).

        Returns:
            An integer score in the range ``[0, 100]``.
        """
        return int(
            fuzz.partial_ratio(str(left).strip().lower(), str(right).strip().lower())
        )

    @staticmethod
    def validate_totals(merged_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Split matched pairs into validated transactions and discrepancies.

        Rows with a zero delta (amounts match exactly) go into the validated set;
        rows with a non-zero delta go into discrepancies.

        Args:
            merged_df: DataFrame of matched transaction–proof pairs with
                ``total_transaction`` and ``total_proof`` columns.

        Returns:
            A tuple ``(validated, discrepancies)`` where both are DataFrames with
            human-readable column names applied.
        """
        merged_df["delta"] = merged_df["total_transaction"] - merged_df["total_proof"]

        discrepancies: pd.DataFrame = np.round(merged_df[merged_df["delta"] != 0.0], 2)
        validated = merged_df[merged_df["delta"] == 0.0]

        # Drop internal matching columns before presenting results to the caller
        validated = validated.drop(
            columns=["delta", "currency_proof", "currency_transaction"],
            errors="ignore",
        )
        discrepancies = discrepancies.drop(
            columns=["currency_proof", "currency_transaction"],
            errors="ignore",
        )

        validated = validated.rename(
            columns={
                "business_name_transaction": "Transaction Business Name",
                "total_transaction": "Transaction Total",
                "date_transaction": "Transaction Date",
                "business_name_proof": "Proof Business Name",
                "total_proof": "Proof Total",
                "date_proof": "Proof Date",
                "category_transaction": "Transaction Category",
                "category_proof": "Proof Category",
            }
        )
        validated["Result"] = ["Validated"] * len(validated)

        return validated, discrepancies

    def find_unmatched_transactions(self, used_tx_indices: set[int]) -> pd.DataFrame:
        """
        Identify transactions that were not used in any matched pair.

        Args:
            used_tx_indices: Set of transaction DataFrame indices that were already
                paired with a proof during the matching step.

        Returns:
            A DataFrame of unmatched transaction rows with internal helper columns removed.
        """
        unmatched = self.transactions.loc[
            ~self.transactions.index.isin(list(used_tx_indices))
        ]

        return unmatched.drop(
            columns=["name_key", "date_key", "currency"], errors="ignore"
        )

    def find_unmatched_proofs(self, used_proof_indices: set[int]) -> pd.DataFrame:
        """
        Identify proofs that were not used in any matched pair.

        Args:
            used_proof_indices: Set of proof DataFrame indices that were already
                paired with a transaction during the matching step.

        Returns:
            A DataFrame of unmatched proof rows with internal helper columns removed.
        """
        unmatched = self.proofs.loc[~self.proofs.index.isin(list(used_proof_indices))]

        return unmatched.drop(
            columns=["name_key", "date_key", "currency"], errors="ignore"
        )

    @staticmethod
    def update_unmatched_dataframes(
        accepted_recommendations: pd.DataFrame,
        unmatched_transactions: pd.DataFrame,
        unmatched_proofs: pd.DataFrame,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Remove accepted recommendation pairs from the unmatched DataFrames.

        When the user accepts a recommendation, those rows should no longer appear
        in the unmatched lists. This method performs that removal via an outer merge.

        Args:
            accepted_recommendations: DataFrame of user-accepted pairings. The first
                three columns map to transactions; the next columns map to proofs.
            unmatched_transactions: Current unmatched transactions DataFrame.
            unmatched_proofs: Current unmatched proofs DataFrame.

        Returns:
            A tuple ``(remaining_transactions, remaining_proofs)`` with accepted
            rows removed from each.
        """
        # Nothing to remove if there are no accepted recommendations
        if accepted_recommendations.empty:
            return unmatched_transactions, unmatched_proofs

        accepted_transactions: pd.DataFrame = accepted_recommendations.iloc[:, :3]
        accepted_transactions = accepted_transactions.map(
            lambda x: x.strip() if isinstance(x, str) else x
        )
        accepted_proofs: pd.DataFrame = accepted_recommendations.iloc[:, 3:-1]
        accepted_proofs = accepted_proofs.map(
            lambda x: x.strip() if isinstance(x, str) else x
        )

        correct_cols = unmatched_transactions.columns

        accepted_transactions = accepted_transactions.rename(
            columns=dict(zip(accepted_transactions.columns, correct_cols))
        )
        accepted_proofs = accepted_proofs.rename(
            columns=dict(zip(accepted_proofs.columns, correct_cols))
        )

        merged_transactions = unmatched_transactions.merge(
            accepted_transactions, how="left", indicator=True
        )
        remained_unmatched_transactions = merged_transactions[
            merged_transactions["_merge"] == "left_only"
        ].drop(columns=["_merge"])

        merged_proofs = unmatched_proofs.merge(
            accepted_proofs, how="left", indicator=True
        )
        remained_unmatched_proofs = merged_proofs[
            merged_proofs["_merge"] == "left_only"
        ].drop(columns=["_merge"])

        return remained_unmatched_transactions, remained_unmatched_proofs

    def validate(self) -> Results:
        """
        Run the full validation pipeline and return matched/unmatched results.

        Steps:
        1. Categorize both inputs concurrently via the LLM.
        2. Build normalised name and date keys for fuzzy matching.
        3. Generate candidate pairs (same date, fuzzy name similarity ≥ 80).
        4. Greedily resolve conflicts: highest similarity first, then lowest delta.
        5. Split pairs into validated (zero delta) and discrepancies (non-zero delta).
        6. Collect unmatched rows from both sides.

        Returns:
            A ``Results`` object with validated transactions, discrepancies,
            unmatched transactions, and unmatched proofs.
        """
        start = time()
        self.transactions = self.transactions.copy()
        self.proofs = self.proofs.copy()

        try:
            self._categorize_inputs()
        except Exception as e:
            print(f"Warning: Categorization failed and was skipped. Error: {e}")
            self.categorize_cost = {
                "model": "unavailable",
                "rowsProcessed": 0,
                "llmCalls": 0,
                "fallbackCalls": 0,
                "inputTokens": 0,
                "outputTokens": 0,
                "estimatedTotalCostUsd": 0.0,
                "latencySeconds": 0.0,
            }

        self.transactions["name_key"] = (
            self.transactions["business_name"].astype(str).str.strip().str.lower()
        )
        self.proofs["name_key"] = (
            self.proofs["business_name"].astype(str).str.strip().str.lower()
        )

        self.transactions["date_key"] = self.transactions["date"].apply(
            Validator._date_key
        )
        self.proofs["date_key"] = self.proofs["date"].apply(Validator._date_key)

        tx_totals = pd.to_numeric(self.transactions["total"], errors="coerce")
        pr_totals = pd.to_numeric(self.proofs["total"], errors="coerce")

        # Build all candidate (tx, proof) pairs that share a date and meet the name threshold
        candidates: list[tuple[int, int, int, float]] = []
        for tx_idx, tx_row in self.transactions.iterrows():
            same_date_proofs = self.proofs[
                self.proofs["date_key"] == tx_row["date_key"]
            ]
            if same_date_proofs.empty:
                continue

            for pr_idx, pr_row in same_date_proofs.iterrows():
                score = Validator._name_similarity(
                    tx_row["name_key"], pr_row["name_key"]
                )
                if score < 80:
                    continue

                total_delta = (
                    abs(float(tx_totals.loc[tx_idx]) - float(pr_totals.loc[pr_idx]))
                    if pd.notna(tx_totals.loc[tx_idx])
                    and pd.notna(pr_totals.loc[pr_idx])
                    else float("inf")
                )
                candidates.append((tx_idx, pr_idx, score, total_delta))

        # Sort: best similarity first, then smallest total delta to break ties
        candidates.sort(key=lambda item: (-item[2], item[3]))

        used_tx_indices: set[int] = set()
        used_proof_indices: set[int] = set()
        matched_pairs: list[tuple[int, int]] = []

        # Greedy one-to-one assignment: once an index is used, skip it
        for tx_idx, pr_idx, _, _ in candidates:
            if tx_idx in used_tx_indices or pr_idx in used_proof_indices:
                continue

            used_tx_indices.add(tx_idx)
            used_proof_indices.add(pr_idx)
            matched_pairs.append((tx_idx, pr_idx))

        end = time()
        print(f"Time taken to match {round(end - start, 3)}s")

        if matched_pairs:
            tx_idx_vec = [tx_idx for tx_idx, _ in matched_pairs]
            pr_idx_vec = [pr_idx for _, pr_idx in matched_pairs]
            merged_df = (
                self.transactions.loc[tx_idx_vec]
                .reset_index(drop=True)
                .add_suffix("_transaction")
                .join(
                    self.proofs.loc[pr_idx_vec]
                    .reset_index(drop=True)
                    .add_suffix("_proof")
                )
            )
        else:
            merged_df = pd.DataFrame(
                columns=[
                    "business_name_transaction",
                    "total_transaction",
                    "date_transaction",
                    "currency_transaction",
                    "business_name_proof",
                    "total_proof",
                    "date_proof",
                    "currency_proof",
                ]
            )

        merged_df = merged_df.drop(
            columns=[
                "name_key_transaction",
                "name_key_proof",
                "date_key_transaction",
                "date_key_proof",
            ],
            errors="ignore",
        )

        validated_transactions, discrepancies = Validator.validate_totals(merged_df)

        unmatched_transactions = self.find_unmatched_transactions(used_tx_indices)
        unmatched_proofs = self.find_unmatched_proofs(used_proof_indices)

        discrepancies = discrepancies.rename(
            columns={
                "business_name_transaction": "Transaction Business Name",
                "total_transaction": "Transaction Total",
                "date_transaction": "Transaction Date",
                "business_name_proof": "Proof Business Name",
                "total_proof": "Proof Total",
                "date_proof": "Proof Date",
                "delta": "Delta",
                "category_transaction": "Transaction Category",
                "category_proof": "Proof Category",
            }
        )

        unmatched_cols = ["Business Name", "Total", "Date"]
        if "category" in unmatched_transactions.columns:
            unmatched_cols.append("Category")
        unmatched_transactions = unmatched_transactions.reset_index(drop=True)
        unmatched_proofs = unmatched_proofs.reset_index(drop=True)
        unmatched_transactions.columns = unmatched_cols
        unmatched_proofs.columns = unmatched_cols

        return Results(
            validated_transactions,
            discrepancies,
            unmatched_transactions,
            unmatched_proofs,
        )

    def analyze_unmatched_results(
        self, unmatched_transactions: pd.DataFrame, unmatched_proofs: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Recommend likely pairings for unmatched rows using name/date/amount signals.

        Candidates are generated by cross-join, then kept when *any* of these
        signals match:
        1. Fuzzy business-name similarity at or above threshold.
        2. Date proximity within +/- 1 day.
        3. Total proximity within +/- amount threshold.

        Results are de-duplicated to one-to-one pairings and sorted to prefer
        stronger matches (more matching factors), then tighter date/amount and
        higher name similarity.

        Args:
            unmatched_transactions: Unmatched transaction rows with ``Business Name``,
                ``Total``, and ``Date`` columns.
            unmatched_proofs: Unmatched proof rows with the same schema.

        Returns:
            A DataFrame of recommended pairings with ``Transaction *``, ``Proof *``,
            and ``Reason`` columns, or an empty DataFrame if no candidates exist.
        """
        if unmatched_transactions.empty or unmatched_proofs.empty:
            return pd.DataFrame([])

        name_similarity_threshold = float(
            self.recommendation_matching["name_similarity_threshold"]
        )
        date_window_days = int(self.recommendation_matching["date_window_days"])
        amount_threshold = float(self.recommendation_matching["amount_threshold"])

        user_facing_reason = "Similar names, dates or amount"

        tx = unmatched_transactions.copy()
        pr = unmatched_proofs.copy()

        tx["__tx_id"] = tx.index
        pr["__pr_id"] = pr.index

        tx["__match_date"] = tx["Date"].apply(Validator._normalize_date_value)
        pr["__match_date"] = pr["Date"].apply(Validator._normalize_date_value)

        tx["__match_total"] = pd.to_numeric(tx["Total"], errors="coerce").round(2)
        pr["__match_total"] = pd.to_numeric(pr["Total"], errors="coerce").round(2)

        candidate_pairs = tx.merge(
            pr,
            how="cross",
            suffixes=("_tx", "_pr"),
        )

        if not candidate_pairs.empty:
            candidate_pairs["__name_similarity"] = (
                candidate_pairs.apply(
                    lambda row: Validator._name_similarity(
                        row["Business Name_tx"], row["Business Name_pr"]
                    ),
                    axis=1,
                )
                / 100.0
            )

            candidate_pairs["__date_distance"] = (
                (
                    candidate_pairs["__match_date_tx"]
                    - candidate_pairs["__match_date_pr"]
                )
                .abs()
                .dt.days
            )
            candidate_pairs["__date_distance"] = candidate_pairs[
                "__date_distance"
            ].fillna(float("inf"))

            candidate_pairs["__total_delta"] = (
                candidate_pairs["__match_total_tx"]
                - candidate_pairs["__match_total_pr"]
            ).abs()
            candidate_pairs["__total_delta"] = candidate_pairs["__total_delta"].fillna(
                float("inf")
            )

            candidate_pairs["__name_match"] = (
                candidate_pairs["__name_similarity"] >= name_similarity_threshold
            )
            candidate_pairs["__date_match"] = (
                candidate_pairs["__date_distance"] <= date_window_days
            )
            candidate_pairs["__amount_match"] = (
                candidate_pairs["__total_delta"] <= amount_threshold
            )

            # Keep rows where any matching signal is strong enough.
            candidate_pairs = candidate_pairs[
                candidate_pairs[["__name_match", "__date_match", "__amount_match"]].any(
                    axis=1
                )
            ]

            if not candidate_pairs.empty:
                candidate_pairs["__match_count"] = (
                    candidate_pairs["__name_match"].astype(int)
                    + candidate_pairs["__date_match"].astype(int)
                    + candidate_pairs["__amount_match"].astype(int)
                )

                def build_reason(row: pd.Series) -> str:
                    if (
                        bool(row["__name_match"])
                        or bool(row["__date_match"])
                        or bool(row["__amount_match"])
                    ):
                        return user_facing_reason
                    return ""

                candidate_pairs["__reason"] = candidate_pairs.apply(
                    build_reason, axis=1
                )

                candidate_pairs = candidate_pairs.sort_values(
                    by=[
                        "__match_count",
                        "__date_distance",
                        "__total_delta",
                        "__name_similarity",
                    ],
                    ascending=[False, True, True, False],
                )

                used_tx_ids: set[int] = set()
                used_pr_ids: set[int] = set()
                selected_rows: list[pd.Series] = []
                for _, row in candidate_pairs.iterrows():
                    tx_id = int(row["__tx_id"])
                    pr_id = int(row["__pr_id"])
                    if tx_id in used_tx_ids or pr_id in used_pr_ids:
                        continue
                    used_tx_ids.add(tx_id)
                    used_pr_ids.add(pr_id)
                    selected_rows.append(row)

                candidate_pairs = pd.DataFrame(selected_rows)

        recommendations = pd.DataFrame([])
        if not candidate_pairs.empty:
            recommendations = pd.DataFrame(
                {
                    "Transaction Business Name": candidate_pairs["Business Name_tx"],
                    "Transaction Total": candidate_pairs["Total_tx"],
                    "Transaction Date": candidate_pairs["Date_tx"],
                    "Transaction Category": candidate_pairs.get("Category_tx", ""),
                    "Proof Business Name": candidate_pairs["Business Name_pr"],
                    "Proof Total": candidate_pairs["Total_pr"],
                    "Proof Date": candidate_pairs["Date_pr"],
                    "Proof Category": candidate_pairs.get("Category_pr", ""),
                    "Reason": candidate_pairs["__reason"],
                }
            )

        if recommendations.empty:
            return pd.DataFrame([])

        recommendations = recommendations.drop_duplicates(
            subset=[
                "Transaction Business Name",
                "Transaction Total",
                "Transaction Date",
                "Proof Business Name",
                "Proof Total",
                "Proof Date",
            ]
        )

        return recommendations.reset_index(drop=True)

    def analyze_results(self, results: Results) -> tuple[str, pd.DataFrame]:
        """
        Produce a human-readable analysis summary and pairing recommendations.

        Args:
            results: The ``Results`` object returned by ``validate()``.

        Returns:
            A tuple ``(analysis, recommendations)`` where *analysis* is a plain-text
            summary string and *recommendations* is a DataFrame of suggested pairings
            (may be empty).
        """
        unmatched_transactions, unmatched_proofs = (
            results.unmatched_transactions,
            results.unmatched_proofs,
        )
        if unmatched_transactions.empty and unmatched_proofs.empty:
            analysis = (
                "Everything was validated. Great job keeping track of your spending!"
            )
            recommendations = pd.DataFrame([])
        elif unmatched_transactions.empty:
            analysis = "No unmatched transactions detected, so no recommendations were generated."
            recommendations = pd.DataFrame([])
        elif unmatched_proofs.empty:
            analysis = "Unmatched transactions found, but no unmatched proofs are available to recommend pairings."
            recommendations = pd.DataFrame([])
        else:
            recommendations = self.analyze_unmatched_results(
                unmatched_transactions, unmatched_proofs
            )
            if recommendations.empty:
                analysis = "I finished the validation process."
            else:
                analysis = "I finished the validation process and provided some recommendations for you."

        return analysis, recommendations
