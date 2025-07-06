import numpy as np
import pandas as pd
from time import time
from io import StringIO
from dataclasses import dataclass

from fuzzywuzzy import process, fuzz
from openai import OpenAI
from src.utils import setup_openai, GPT_MODEL

pd.set_option("display.max_columns", None)


@dataclass
class Results:
    def __init__(self, validated_transactions: pd.DataFrame = None, discrepancies: pd.DataFrame = None,
                 unmatched_transactions: pd.DataFrame = None, unmatched_proofs: pd.DataFrame = None):
        self.validated_transactions = validated_transactions
        self.discrepancies = discrepancies
        self.unmatched_transactions = unmatched_transactions
        self.unmatched_proofs = unmatched_proofs


class Validator:
    def __init__(self, transactions: pd.DataFrame,
                 proofs: pd.DataFrame, setup_client: bool = True):
        self.transactions = transactions
        self.proofs = proofs

        if setup_client:
            self.setup_client()

    def setup_client(self):
        """
        Sets up the OpenAI client by initializing the necessary configurations.

        This method calls the setup_openai function to configure the OpenAI environment
        and then creates an instance of the OpenAI client.
        """
        setup_openai()
        self.client = OpenAI()

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
        discrepancies = np.round(merged_df[merged_df["delta"] != 0.0], 2)
        validated = merged_df[merged_df["delta"] == 0.0]

        validated = validated.drop(columns=["delta"])
        validated.columns = ["Transaction Business Name",
                             "Transaction Total",
                             "Transaction Date",
                             "Proof Business Name",
                             "Proof Total",
                             "Proof Date"]
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

        return unmatched.drop(columns=["matched_name", "matched_date"])

    def find_unmatched_proofs(self, merged_df: pd.DataFrame) -> pd.DataFrame:
        """
        Identify any proof that had no match
        """
        unmatched = self.proofs[
            ~self.proofs["business_name"].isin(merged_df["business_name_proof"])
        ]

        return unmatched

    @staticmethod
    def update_unmatched_dataframes(accepted_recommendations: pd.DataFrame,
                                    unmatched_transactions: pd.DataFrame,
                                    unmatched_proofs: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        # Update & render unmatched transactions and proofs with accepted recommendations
        if accepted_recommendations.empty:
            return unmatched_transactions, unmatched_proofs

        # Update the unmatched transactions and proofs
        accepted_transactions: pd.DataFrame = accepted_recommendations.iloc[:, :3]
        accepted_transactions = accepted_transactions.map(lambda x: x.strip() if isinstance(x, str) else x)
        accepted_proofs: pd.DataFrame = accepted_recommendations.iloc[:, 3:-1]
        accepted_proofs = accepted_proofs.map(lambda x: x.strip() if isinstance(x, str) else x)

        correct_cols = unmatched_transactions.columns

        accepted_transactions = accepted_transactions.rename(columns=dict(zip(accepted_transactions.columns,
                                                                                correct_cols)))

        accepted_proofs = accepted_proofs.rename(columns=dict(zip(accepted_proofs.columns,
                                                                    correct_cols)))
        
        merged_transactions = unmatched_transactions.merge(accepted_transactions, how="left",
                                                            indicator=True)
        remained_unmatched_transactions = merged_transactions[merged_transactions["_merge"]=="left_only"]\
                                                                                .drop(columns=["_merge"])
        
        merged_proofs = unmatched_proofs.merge(accepted_proofs, how="left",
                                    indicator=True)
        remained_unmatched_proofs = merged_proofs[merged_proofs["_merge"]=="left_only"]\
                                                                .drop(columns=["_merge"])
        
        return remained_unmatched_transactions, remained_unmatched_proofs

    def validate(self, state: dict) -> Results:
        """
        Run the validation process
        """
        start = time()
        state["transactions"]["matched_name"] = state["transactions"]["business_name"].apply(
            lambda x: self.match_business_names(x, state["proofs"]["business_name"].values)
        )

        state["transactions"]["matched_date"] = state["transactions"]["date"].apply(
            lambda x: self.match_business_names(x, state["proofs"]["date"].values)
        )

        end = time()
        print(f"Time taken to match {round(end - start, 3)}s")
        # Merge based on matched names and totals
        merged_df = state["transactions"].merge(
            state["proofs"],
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

        return Results(validated_transactions, discrepancies, unmatched_transactions, unmatched_proofs)

    def analyze_unmatched_results(self,
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

            One reason behind unmatches is business names from transactions and proofs
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
            Another reason behind unmatches is business name can be different on transactions and proofs,
            even though the totals and dates are the exact same.

            ---------------------------
            Example:
            unmatched_transactions:
             Business Name      Total        Date
             Ikkousha Irvine    143.62       2023-01-01
              Jones LLC         230.45       2023-02-15
              Smith Inc         312.67       2023-03-30
            
            unmatched_proofs:
             Business Name      Total        Date
             Kiosk             143.62       2023-01-01
              Taco Bell        230.45       2023-02-15
              AMC Movies       312.67       2023-03-30

            Recommendations Example:

            Transaction Business Name,Transaction Total,Transaction Date,Proof Business Name,Proof Total,Proof Date,Reason
            Ikkousha Long Beach, 143.62, 2023-01-01, Kiosk, 143.62, 2023-01-01, Matching Totals and Dates.

            ---------------------------
            Only output the recommendations
        """
        # Send the prompt to ChatGPT
        response = self.client.chat.completions.create(
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
        self, results: Results
    ) -> tuple[str, pd.DataFrame]:
        """
        Analyze the results & provide recommendations
        for unmatched transactions & proofs
        """
        unmatched_transactions, unmatched_proofs = results.unmatched_transactions, results.unmatched_proofs
        if unmatched_transactions.empty and unmatched_proofs.empty:
            analysis = "Everything was validated. Great job keeping track of your spending!"
            recommendations = pd.DataFrame([])
        else:
            analysis = "I finished the validation process and provided some recommendations below."
            recommendations = self.analyze_unmatched_results(
                unmatched_transactions, unmatched_proofs
                )
            recommendations = pd.read_csv(StringIO(recommendations))
            recommendations.columns = [col.strip() for col in recommendations.columns]

        return analysis, recommendations
