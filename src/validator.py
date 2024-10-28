import numpy as np
import pandas as pd
from time import time
from fuzzywuzzy import process, fuzz


pd.set_option('display.max_columns', None)  # This will show all columns


class Validator:
    def __init__(self, transactions: pd.DataFrame, proofs: pd.DataFrame):
        self.transactions = transactions
        self.proofs = proofs
    
    @staticmethod
    def match_business_names(transaction_name: str, proofs: list[str], threshold=75):
        """
        Given a list of business names from receipts,
        match the transaction to the closest name from the receipts.

        :param transaction_name: name of transaction to match
        :param receipts_names: business names from the receipts
        :param threshold: threshold of similiarity
        :return: the match found
        """
        match, score = process.extractOne(transaction_name, proofs, scorer=fuzz.partial_ratio)

        return match if score >= threshold else None
    
    def find_discrepancies(self) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Find any discrepancy in the list of transactions
        """   
        # Difference (rows in df1 not in df2)
        self.proofs["business_name"], self.proofs["matched_name"] = self.proofs["matched_name"], self.proofs["business_name"]
        trans_dis = self.transactions[~self.transactions.isin(self.proofs).all(axis=1)]
        # Difference (rows in df2 not in df1)
        proofs_dis = self.proofs[~self.proofs.isin(self.transactions).all(axis=1)]

        if not len(trans_dis) and not len(proofs_dis):
            discrepancies = pd.DataFrame([], columns=["Business Name", "Transaction Total",
                                                      "Date", "Proof Total", "Delta"])
        else:
            discrepancies = trans_dis.merge(proofs_dis, on=["business_name", "date"],
                                            suffixes=["_transaction", "_proofs"])
            if len(discrepancies):
                discrepancies["delta"] = np.round(discrepancies["total_transaction"] - discrepancies["total_proofs"], 2)

            discrepancies = discrepancies.drop(columns=["matched_name_transaction", "matched_name_proofs"])
            discrepancies = Validator.cleanup_results(discrepancies, discrepancy_df=True)

        return discrepancies

    def find_unmatched_transactions(self) -> pd.DataFrame:
        """
        Identify any transaction with no matches
        """
        self.transactions["matched_name"] = self.transactions["business_name"].apply(lambda x: \
                                                              self.match_business_names(x, self.proofs["business_name"].values))
        unmatched_transactions = self.transactions[self.transactions["matched_name"].isnull()]

        if not unmatched_transactions.empty:
            self.transactions = self.transactions[self.transactions['matched_name'].notnull()]

        unmatched_transactions = Validator.cleanup_results(unmatched_transactions.drop(columns=["matched_name"]))

        return unmatched_transactions

    def find_unmatched_proofs(self) -> pd.DataFrame:
        """
        Identify any proof with no matches
        """
        self.proofs["matched_name"] = self.proofs["business_name"].apply(lambda x: \
                                                  self.match_business_names(x, self.transactions["business_name"].values))

        unmatched_proofs = self.proofs[self.proofs["matched_name"].isnull()]
        if not unmatched_proofs.empty:
            self.proofs = self.proofs[self.proofs['matched_name'].notnull()]

        unmatched_proofs = Validator.cleanup_results(unmatched_proofs.drop(columns=["matched_name"]))

        return unmatched_proofs

    @staticmethod
    def cleanup_results(df: pd.DataFrame, discrepancy_df: bool=False) -> pd.DataFrame:
        # Cosmetic clean up
        if discrepancy_df:
            df.columns = ["Business Name", "Transaction Total", "Date", "Proof Total", "Delta"]
            df = df[["Business Name", "Date", "Transaction Total", "Proof Total", "Delta"]]
        else:
            df.columns = ["Business Name", "Total", "Date"]

        return df

    def validate(self) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        start = time()
        unmatched_transactions = self.find_unmatched_transactions()
        unmatched_proofs = self.find_unmatched_proofs()
        discrepancies = self.find_discrepancies()
        end = time()
        print(f'Time taken to match {round(end - start, 2)}s')

        return discrepancies, unmatched_transactions, unmatched_proofs
