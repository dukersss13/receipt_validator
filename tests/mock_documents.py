import pandas as pd
import numpy as np
from faker import Faker

# Initialize Faker for generating random business names
fake = Faker()

def create_mock_documents(num: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    # Generate random data
    business_names = [fake.company() for _ in range(num)]
    totals = np.random.uniform(
        low=0.01, high=1000.0, size=num
    )  # random floats (0, 1000]
    dates = pd.to_datetime(
        np.random.randint(
            pd.Timestamp("2000-01-01").value, pd.Timestamp("2024-12-31").value, num
        ),
        unit="ns",
    )  # random dates from 2000 to 2024

    date_strings = dates.strftime('%Y-%m-%d')

    # Create the DataFrame
    transactions_df = pd.DataFrame(
        {"business_name": business_names, "total": totals, "date": date_strings}
    )

    # Create the second copy with business names in all caps
    proofs_df = transactions_df.copy()
    proofs_df["business_name"] = proofs_df["business_name"].str.upper()

    return transactions_df, proofs_df
