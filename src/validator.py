import numpy as np
import pandas as pd
from time import time
from io import StringIO

from fuzzywuzzy import process, fuzz
from openai import OpenAI
from src.utils import setup_openai, GPT_MODEL


pd.set_option("display.max_columns", None)

setup_openai()
client = OpenAI()


class Validator:
    def __init__(self, transactions: pd.DataFrame, proofs: pd.DataFrame):
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
    def find_discrepancies(merged_df: pd.DataFrame) -> pd.DataFrame:
        """
        Find any discrepancy in the list of transactions
        """
        merged_df["delta"] = merged_df["total_transaction"] - merged_df["total_proof"]
        # Identify discrepancies
        # If delta > 0.0, then transaction > proof. Elif delta < 0.0, transaction < proof.
        # Else, we're good.
        discrepancies = np.round(merged_df[merged_df["delta"] != 0.0], 2)

        return discrepancies

    def find_unmatched_transactions(self, merged_df: pd.DataFrame) -> pd.DataFrame:
        """
        Identify any transaction that had no match
        """
        unmatched = self.transactions[
            ~self.transactions["business_name"].isin(
                merged_df["business_name_transaction"]
            )
        ]

        return unmatched.drop(columns=["matched_name", "matched_date"])

    def find_unmatched_proofs(self, merged_df: pd.DataFrame) -> pd.DataFrame:
        """
        Identify any proof that had no match
        """
        unmatched = self.proofs[
            ~self.proofs["business_name"].isin(merged_df["business_name_proof"])
        ]

        return unmatched

    def validate(self) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
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

        discrepancies = Validator.find_discrepancies(merged_df)
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

        return discrepancies, unmatched_transactions, unmatched_proofs

    @staticmethod
    def analyze_unmatched_results(
        unmatched_transactions: pd.DataFrame, unmatched_proofs: pd.DataFrame
    ) -> str:
        """
        Use LLM to provide recommendations for unmatched results
        """
        unmatched_trans_txt = unmatched_transactions.to_string(index=False)
        unmatched_proofs_txt = unmatched_proofs.to_string(index=False)
        full_prompt = """
            Your job is to analyze these unmatched transactions 
            and provide recommendations if there are potential matches that were not matched.
    
            The columns for both dataframes are:
            - Business Name (str): name of the business
            - Total (float): total amount of the transaction
            - Date (str): transaction date

            A reason behind unmatches is business names from transactions and proofs
            don't fully match even though there are matching Total and Date.
            ---------------------------
            Example:
            unmatched_transactions:
             Business Name      Total        Date
            Ikkousha Irvine         143.62       2023-01-01
              Jones LLC         230.45       2023-02-15
              Smith Inc         312.67       2023-03-30
            
            unmatched_proofs:
             Business Name      Total        Date
           Ikkousha Ramen      143.62       2023-01-01
              Taco Bell        230.45       2023-02-15
              AMC Movies       312.67       2023-03-30
    
            Recommendations Example:

            Transaction Business Name,Transaction Total,Transaction Date,Proof Business Name,Proof Total,Proof Date,Reason
            Ikkousha Long Beach, 143.62, 2023-01-01, Ikkousha Ramen, 143.62, 2023-01-01, Same business with matching
            dates and transaction totals.
            ---------------------------
            Only output the recommendations
        """
        # Send the prompt to ChatGPT
        response = client.chat.completions.create(
            model=GPT_MODEL,
            messages=[
                {"role": "system", "content": full_prompt},
                {
                    "role": "user",
                    "content": f"Data: {unmatched_trans_txt} {unmatched_proofs_txt}",
                },
            ],
        )

        return response.choices[0].message.content

    def analyze_results(
        self, validation_results: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
    ) -> tuple[str, pd.DataFrame]:
        """
        Analyze the results & provide recommendations
        for unmatched transactions & proofs
        """
        _, unmatched_transactions, unmatched_proofs = validation_results
        if unmatched_transactions.empty and unmatched_proofs.empty:
            analysis = "Everything was validated. Great job keeping track of your spending!"
            recommendations = pd.DataFrame([])
        else:
            analysis = "I finished the validation process and provided some recommendations below."
            recommendations = Validator.analyze_unmatched_results(
                unmatched_transactions, unmatched_proofs
                )
            recommendations = pd.read_csv(StringIO(recommendations))

        return analysis, recommendations
