import numpy as np
import pandas as pd
from time import time
from dataclasses import dataclass

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
        :param threshold: threshold of similiarity
        :return: the match found
        """
        match, score = process.extractOne(
            transaction_name, proofs, scorer=fuzz.partial_ratio
        )

        return match if score >= threshold else None

    @staticmethod
    def validate_totals(merged_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Find any discrepancy in the list of transactions
        """
        merged_df["delta"] = merged_df["total_transaction"] - merged_df["total_proof"]
        # Identify discrepancies
        # If delta > 0.0, then transaction > proof. Elif delta < 0.0, transaction < proof.
        # Else, we're good.
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

    def find_unmatched_transactions(self, merged_df: pd.DataFrame) -> pd.DataFrame:
        """
        Identify any transaction that had no match
        """
        unmatched = self.transactions[
            ~self.transactions["business_name"].isin(
                merged_df["business_name_transaction"]
            )
        ]

        return unmatched.drop(
            columns=["matched_name", "matched_date", "currency"],
            errors="ignore",
        )

    def find_unmatched_proofs(self, merged_df: pd.DataFrame) -> pd.DataFrame:
        """
        Identify any proof that had no match
        """
        unmatched = self.proofs[
            ~self.proofs["business_name"].isin(merged_df["business_name_proof"])
        ]

        return unmatched.drop(columns=["currency"], errors="ignore")

    @staticmethod
    def update_unmatched_dataframes(
        accepted_recommendations: pd.DataFrame,
        unmatched_transactions: pd.DataFrame,
        unmatched_proofs: pd.DataFrame,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        # Update & render unmatched transactions and proofs with accepted recommendations
        if accepted_recommendations.empty:
            return unmatched_transactions, unmatched_proofs

        # Update the unmatched transactions and proofs
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
        Run the validation process
        """
        start = time()
        self.transactions["matched_name"] = self.transactions["business_name"].apply(
            lambda x: self.match_business_names(x, self.proofs["business_name"].values)
        )

        self.transactions["matched_date"] = self.transactions["date"].apply(
            lambda x: self.match_business_names(x, self.proofs["date"].values)
        )

        end = time()
        print(f"Time taken to match {round(end - start, 3)}s")
        # Merge based on matched names and totals
        merged_df = self.transactions.merge(
            self.proofs,
            left_on=["matched_name", "matched_date"],
            right_on=["business_name", "date"],
            how="inner",
            suffixes=("_transaction", "_proof"),
        )

        merged_df = merged_df.drop(columns=["matched_name", "matched_date"])

        validated_transactions, discrepancies = Validator.validate_totals(merged_df)
        unmatched_transactions = self.find_unmatched_transactions(merged_df)
        unmatched_proofs = self.find_unmatched_proofs(merged_df)

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
        Provide deterministic recommendations using:
        1) totals match and dates are within +/- 1 day, but business names differ
        2) business names are similar and dates are within +/- 1 day, but totals differ
        """
        if unmatched_transactions.empty or unmatched_proofs.empty:
            return pd.DataFrame([])

        tx = unmatched_transactions.copy()
        pr = unmatched_proofs.copy()

        tx["__match_date"] = pd.to_datetime(tx["Date"], errors="coerce").dt.normalize()
        pr["__match_date"] = pd.to_datetime(pr["Date"], errors="coerce").dt.normalize()

        tx["__match_total"] = pd.to_numeric(tx["Total"], errors="coerce").round(2)
        pr["__match_total"] = pd.to_numeric(pr["Total"], errors="coerce").round(2)

        tx = tx.dropna(subset=["__match_date", "__match_total"])
        pr = pr.dropna(subset=["__match_date", "__match_total"])

        if tx.empty or pr.empty:
            return pd.DataFrame([])

        # Rule 1: same total and date within +/- 1 day, but business names differ.
        exact_total_date = tx.merge(
            pr,
            on=["__match_total"],
            how="inner",
            suffixes=("_tx", "_pr"),
        )

        if not exact_total_date.empty:
            exact_total_date = exact_total_date[
                (
                    exact_total_date["__match_date_tx"]
                    - exact_total_date["__match_date_pr"]
                ).abs()
                <= pd.Timedelta(days=1)
            ]
            exact_total_date = exact_total_date[
                exact_total_date["Business Name_tx"].str.strip().str.lower()
                != exact_total_date["Business Name_pr"].str.strip().str.lower()
            ]

        rule1 = pd.DataFrame([])
        if not exact_total_date.empty:
            rule1 = pd.DataFrame(
                {
                    "Transaction Business Name": exact_total_date["Business Name_tx"],
                    "Transaction Total": exact_total_date["Total_tx"],
                    "Transaction Date": exact_total_date["Date_tx"],
                    "Proof Business Name": exact_total_date["Business Name_pr"],
                    "Proof Total": exact_total_date["Total_pr"],
                    "Proof Date": exact_total_date["Date_pr"],
                    "Reason": "Matching totals and dates within one day, but different business names.",
                }
            )

        # Rule 2: similar business names + dates within +/- 1 day, but different totals.
        same_date_pairs = tx.merge(pr, how="cross", suffixes=("_tx", "_pr"))

        rule2 = pd.DataFrame([])
        if not same_date_pairs.empty:
            same_date_pairs["__name_similarity"] = same_date_pairs.apply(
                lambda row: fuzz.partial_ratio(
                    str(row["Business Name_tx"]), str(row["Business Name_pr"])
                ),
                axis=1,
            )

            similar_name_diff_total = same_date_pairs[
                (
                    (
                        same_date_pairs["__match_date_tx"]
                        - same_date_pairs["__match_date_pr"]
                    ).abs()
                    <= pd.Timedelta(days=1)
                )
                &
                (same_date_pairs["__name_similarity"] >= 80)
                & (
                    same_date_pairs["__match_total_tx"]
                    != same_date_pairs["__match_total_pr"]
                )
            ]

            if not similar_name_diff_total.empty:
                rule2 = pd.DataFrame(
                    {
                        "Transaction Business Name": similar_name_diff_total[
                            "Business Name_tx"
                        ],
                        "Transaction Total": similar_name_diff_total["Total_tx"],
                        "Transaction Date": similar_name_diff_total["Date_tx"],
                        "Proof Business Name": similar_name_diff_total["Business Name_pr"],
                        "Proof Total": similar_name_diff_total["Total_pr"],
                        "Proof Date": similar_name_diff_total["Date_pr"],
                        "Reason": "Similar business names and dates within one day, but different totals.",
                    }
                )

        recommendations = pd.concat([rule1, rule2], ignore_index=True)

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
        Analyze the results & provide recommendations
        for unmatched transactions & proofs
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
        else:
            analysis = "I finished the validation process and provided deterministic recommendations based on totals with date proximity (+/- 1 day) and business-name similarity with date proximity (+/- 1 day)."
            recommendations = self.analyze_unmatched_results(
                unmatched_transactions, unmatched_proofs
            )

        return analysis, recommendations
