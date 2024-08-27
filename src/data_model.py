import numpy as np
import pandas as pd
from llm import analyze_receipts, create_image_payload


image_payload = create_image_payload(data_path="data/receipts")
receipts_info = analyze_receipts(image_payload)
receipts_list = eval(receipts_info)
receipts_vec = np.asarray(receipts_list)

df_bank = pd.DataFrame({"business_name": ["Taco Bell", "Cider Cellar", "Mogu mogu - Costa Mesa"],
                        "total": [24.39, 4.50, 57.0]})
df_receipts = pd.DataFrame(receipts_vec, columns=["business_name", "total"])
df_receipts["business_name"] = df_receipts["business_name"].astype(str)
df_receipts["total"] = df_receipts["total"].astype(float)


from fuzzywuzzy import process, fuzz

def match_business_names(transaction_name, receipts_names, threshold=80):
    match, score = process.extractOne(transaction_name, receipts_names, scorer=fuzz.partial_ratio)
    return match if score >= threshold else None

# Add a new column in df_bank for the best match from df_receipts
df_bank["matched_name"] = df_bank["business_name"].apply(lambda x: match_business_names(x, df_receipts["business_name"].values))

# Merge based on matched names and totals
matched_df = pd.merge(df_bank, df_receipts, left_on=["matched_name", "total"], 
                      right_on=["business_name", "total"], how="inner", suffixes=("_bank", "_receipt"))

# Identify unmatched rows in df_bank
unmatched = df_bank[~df_bank["business_name"].isin(matched_df["business_name_bank"])]

# Extract DATES & match with totals
