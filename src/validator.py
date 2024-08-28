import pandas as pd
from fuzzywuzzy import process, fuzz


class Validator:
    def __init__(self, bank_statement: pd.DataFrame, receipts: pd.DataFrame):
        self.bank_statement = bank_statement
        self.receipts = receipts
    
    @staticmethod
    def match_business_names(transaction_name, receipts_names, threshold=80):
        match, score = process.extractOne(transaction_name, receipts_names, scorer=fuzz.partial_ratio)
        return match if score >= threshold else None

    def validate(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        # Add a new column in df_bank for the best match from df_receipts
        self.bank_statement["matched_name"] = self.bank_statement["business_name"].apply(lambda x: \
                                                                 self.match_business_names(x, self.receipts["business_name"].values))

        self.bank_statement["matched_date"] = self.bank_statement["date"].apply(lambda x: \
                                              self.match_business_names(x, self.receipts["date"].values))

        # Merge based on matched names and totals
        merged_df = self.bank_statement.merge(self.receipts, left_on=["matched_name", "matched_date"], 
                                right_on=["business_name", "date"], how="inner", suffixes=("_bank", "_receipt"))

        merged_df = merged_df.drop(columns=["matched_name", "matched_date"])

        discrepancies = Validator.find_discrepancies(merged_df)
        unmatched = self.find_unmatched_transactions(merged_df)

        return discrepancies, unmatched

    
    @staticmethod
    def find_discrepancies(merged_df: pd.DataFrame) -> pd.DataFrame:
        """_summary_

        :param merged_df: _description_
        :return: _description_
        """
        merged_df["delta"] = abs(merged_df["total_receipt"] - merged_df["total_bank"])

        # Identify discrepancies
        discrepancies = merged_df[merged_df["delta"] > 0.01]

        return discrepancies

    def find_unmatched_transactions(self, merged_df: pd.DataFrame) -> pd.DataFrame:
        # Identify unmatched rows in df_bank
        unmatched = self.bank_statement[~self.bank_statement["business_name"].isin(merged_df["business_name_bank"])]

        return unmatched
