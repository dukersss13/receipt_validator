import numpy as np
import pandas as pd
from time import time
from fuzzywuzzy import process, fuzz


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
        :param threshold: threshold of similiarity, defaults to 80
        :return: the match found
        """
        match, score = process.extractOne(transaction_name, proofs, scorer=fuzz.partial_ratio)

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
        unmatched = self.transactions[~self.transactions["business_name"].isin(merged_df["business_name_transaction"])]

        return unmatched.drop(columns=['matched_name', 'matched_date'])

    def find_unmatched_proofs(self, merged_df: pd.DataFrame) -> pd.DataFrame:
        """
        Identify any proof that had no match
        """
        unmatched = self.proofs[~self.proofs["business_name"].isin(merged_df["business_name_proof"])]

        return unmatched

    def validate(self) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        # Add a new column in df_bank for the best match from df_receipts
        start = time()
        self.transactions["matched_name"] = self.transactions["business_name"].apply(lambda x: \
                                                              self.match_business_names(x, self.proofs["business_name"].values))

        self.transactions["matched_date"] = self.transactions["date"].apply(lambda x: \
                                                              self.match_business_names(x, self.proofs["date"].values))

        end = time()
        print(f'Time taken to match {round(end - start, 3)}s')
        # Merge based on matched names and totals
        merged_df = self.transactions.merge(self.proofs, left_on=["matched_name", "matched_date"], 
                                right_on=["business_name", "date"], how="inner", suffixes=("_transaction", "_proof"))

        merged_df = merged_df.drop(columns=["matched_name", "matched_date"])

        discrepancies = Validator.find_discrepancies(merged_df)
        unmatched_transactions = self.find_unmatched_transactions(merged_df)
        unmatched_proofs = self.find_unmatched_proofs(merged_df)

        discrepancies.columns = ["Transaction Business Name", "Total Transaction", "Transaction Date",
                                 "Receipt Business Name", "Total on Receipt", "Receipt Date", "Delta"]

        unmatched_cols = ["Business Name", "Total", "Date"]
        unmatched_transactions.columns = unmatched_cols
        unmatched_proofs.columns = unmatched_cols

        return discrepancies, unmatched_transactions, unmatched_proofs
