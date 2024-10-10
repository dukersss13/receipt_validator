import pandas as pd
import numpy as np
from faker import Faker

# Initialize Faker for generating random business names
fake = Faker()

def create_mock_documents(n_unique: int, n_similar: int=0) -> tuple[pd.DataFrame, pd.DataFrame]:
    # Generate random data
    business_names = [fake.company() for _ in range(n_unique)]
    base_totals = np.random.uniform(low=0.01, high=1000.0, size=n_unique)
    dates = pd.to_datetime(
        np.random.randint(
            pd.Timestamp("2000-01-01").value, pd.Timestamp("2024-12-31").value, n_unique
        ),
        unit="ns",
    )
    date_strings = dates.strftime('%Y-%m-%d')

    # Create lists to store the expanded data
    expanded_business_names = []
    expanded_totals = []
    expanded_dates = []

    for i in range(n_unique):
        # Add the original transaction
        expanded_business_names.append(business_names[i])
        expanded_totals.append(base_totals[i])
        expanded_dates.append(date_strings[i])

        # Add similar transactions
        for _ in range(n_similar - 1):
            expanded_business_names.append(business_names[i])
            # Generate a new total that's within ±10% of the original
            new_total = base_totals[i] * np.random.uniform(0.9, 1.1)
            expanded_totals.append(new_total)
            expanded_dates.append(date_strings[i])

    # Create the DataFrame
    transactions_df = pd.DataFrame({
        "business_name": expanded_business_names,
        "total": expanded_totals,
        "date": expanded_dates
    })

    # Create the second copy with business names in all caps
    proofs_df = transactions_df.copy()
    proofs_df["business_name"] = proofs_df["business_name"].str.upper()

    return transactions_df, proofs_df
