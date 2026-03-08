import numpy as np
import pandas as pd
from time import time
from dataclasses import dataclass
import re

from fuzzywuzzy import process, fuzz

pd.set_option("display.max_columns", None)


@dataclass
class Results:
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
    def __init__(
        self,
        transactions: pd.DataFrame,
        proofs: pd.DataFrame,
        setup_client: bool = True,
    ):
        self.transactions = transactions
        self.proofs = proofs

    @staticmethod
    def match_business_names(transaction_name: str, proofs: list[str], threshold=80):
        """
        Given a list of business names from receipts,
        match the transaction to the closest name from the receipts.

        :param transaction_name: name of transaction to match
        :param receipts_names: business names from the receipts
        :param threshold: threshold of similarity
        :return: the match found
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
        """Parse raw/noisy date text into a normalized date or NaT."""
        raw = str(value).strip()
        parsed = pd.to_datetime(raw, errors="coerce")
        if pd.notna(parsed):
            return parsed.normalize()

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
        """Build a stable YYYY-MM-DD key for matching dates with format noise."""
        parsed = Validator._normalize_date_value(value)
        if pd.notna(parsed):
            return parsed.strftime("%Y-%m-%d")
        return str(value).strip().lower()

    @staticmethod
    def _name_similarity(left: object, right: object) -> int:
        """Compute fuzzy similarity for business names in a stable way."""
        return int(
            fuzz.partial_ratio(str(left).strip().lower(), str(right).strip().lower())
        )

    @staticmethod
    def validate_totals(merged_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Find any discrepancy in the list of transactions."""
        merged_df["delta"] = merged_df["total_transaction"] - merged_df["total_proof"]

        discrepancies: pd.DataFrame = np.round(merged_df[merged_df["delta"] != 0.0], 2)
        validated = merged_df[merged_df["delta"] == 0.0]

        validated = validated.drop(
            columns=["delta", "currency_proof", "currency_transaction"],
            errors="ignore",
        )
        discrepancies = discrepancies.drop(
            columns=["currency_proof", "currency_transaction"],
            errors="ignore",
        )

        validated.columns = [
            "Transaction Business Name",
            "Transaction Total",
            "Transaction Date",
            "Proof Business Name",
            "Proof Total",
            "Proof Date",
        ]
        validated["Result"] = ["Validated"] * len(validated)

        return validated, discrepancies

    def find_unmatched_transactions(self, used_tx_indices: set[int]) -> pd.DataFrame:
        """Identify transactions not used in any matched pair."""
        unmatched = self.transactions.loc[
            ~self.transactions.index.isin(list(used_tx_indices))
        ]

        return unmatched.drop(
            columns=["name_key", "date_key", "currency"], errors="ignore"
        )

    def find_unmatched_proofs(self, used_proof_indices: set[int]) -> pd.DataFrame:
        """Identify proofs not used in any matched pair."""
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
        # Update & render unmatched transactions and proofs with accepted recommendations
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
        """Run the validation process."""
        start = time()
        self.transactions = self.transactions.copy()
        self.proofs = self.proofs.copy()

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

        candidates.sort(key=lambda item: (-item[2], item[3]))

        used_tx_indices: set[int] = set()
        used_proof_indices: set[int] = set()
        matched_pairs: list[tuple[int, int]] = []

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

        discrepancies.columns = [
            "Transaction Business Name",
            "Transaction Total",
            "Transaction Date",
            "Proof Business Name",
            "Proof Total",
            "Proof Date",
            "Delta",
        ]

        unmatched_cols = ["Business Name", "Total", "Date"]
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
        """Recommend unmatched pairs when totals and dates are close."""
        if unmatched_transactions.empty or unmatched_proofs.empty:
            return pd.DataFrame([])

        tx = unmatched_transactions.copy()
        pr = unmatched_proofs.copy()

        tx["__match_date"] = tx["Date"].apply(Validator._normalize_date_value)
        pr["__match_date"] = pr["Date"].apply(Validator._normalize_date_value)

        tx["__match_total"] = pd.to_numeric(tx["Total"], errors="coerce").round(2)
        pr["__match_total"] = pd.to_numeric(pr["Total"], errors="coerce").round(2)

        tx = tx.dropna(subset=["__match_date", "__match_total"])
        pr = pr.dropna(subset=["__match_date", "__match_total"])

        if tx.empty or pr.empty:
            return pd.DataFrame([])

        candidate_pairs = tx.merge(
            pr,
            how="cross",
            suffixes=("_tx", "_pr"),
        )

        if not candidate_pairs.empty:
            candidate_pairs = candidate_pairs[
                (
                    candidate_pairs["__match_date_tx"]
                    - candidate_pairs["__match_date_pr"]
                ).abs()
                <= pd.Timedelta(days=2)
            ]
            candidate_pairs = candidate_pairs[
                (
                    candidate_pairs["__match_total_tx"]
                    - candidate_pairs["__match_total_pr"]
                ).abs()
                <= 0.01
            ]

            if not candidate_pairs.empty:
                candidate_pairs["__date_distance"] = (
                    (
                        candidate_pairs["__match_date_tx"]
                        - candidate_pairs["__match_date_pr"]
                    )
                    .abs()
                    .dt.days
                )
                candidate_pairs["__total_delta"] = (
                    candidate_pairs["__match_total_tx"]
                    - candidate_pairs["__match_total_pr"]
                ).abs()
                candidate_pairs["__name_similarity"] = candidate_pairs.apply(
                    lambda row: Validator._name_similarity(
                        row["Business Name_tx"], row["Business Name_pr"]
                    ),
                    axis=1,
                )

                candidate_pairs = candidate_pairs.sort_values(
                    by=["__date_distance", "__total_delta", "__name_similarity"],
                    ascending=[True, True, False],
                )

                used_tx: set[str] = set()
                used_pr: set[str] = set()
                selected_rows: list[pd.Series] = []
                for _, row in candidate_pairs.iterrows():
                    tx_key = (
                        str(row["Business Name_tx"]),
                        float(row["__match_total_tx"]),
                        str(row["Date_tx"]),
                    )
                    pr_key = (
                        str(row["Business Name_pr"]),
                        float(row["__match_total_pr"]),
                        str(row["Date_pr"]),
                    )
                    if tx_key in used_tx or pr_key in used_pr:
                        continue
                    used_tx.add(tx_key)
                    used_pr.add(pr_key)
                    selected_rows.append(row)

                candidate_pairs = pd.DataFrame(selected_rows)

        recommendations = pd.DataFrame([])
        if not candidate_pairs.empty:
            recommendations = pd.DataFrame(
                {
                    "Transaction Business Name": candidate_pairs["Business Name_tx"],
                    "Transaction Total": candidate_pairs["Total_tx"],
                    "Transaction Date": candidate_pairs["Date_tx"],
                    "Proof Business Name": candidate_pairs["Business Name_pr"],
                    "Proof Total": candidate_pairs["Total_pr"],
                    "Proof Date": candidate_pairs["Date_pr"],
                    "Reason": "Similar dates and amount",
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
        """Analyze the results & provide recommendations for unmatched rows."""
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
            analysis = "I finished the validation process and provided some recommendations for you."
            recommendations = self.analyze_unmatched_results(
                unmatched_transactions, unmatched_proofs
            )

        return analysis, recommendations
