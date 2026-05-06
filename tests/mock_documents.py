import pandas as pd


def create_mock_documents(num: int = 3) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Create mock transaction and proof DataFrames for testing.

    Args:
        num: Number of rows to generate in each DataFrame.

    Returns:
        Tuple of (transactions_df, proofs_df).
    """
    transactions = pd.DataFrame(
        {
            "business_name": [f"Store_{i}" for i in range(num)],
            "total": [10.0 * (i + 1) for i in range(num)],
            "date": ["2024-01-01"] * num,
            "currency": ["USD"] * num,
        }
    )
    proofs = pd.DataFrame(
        {
            "business_name": [f"Store_{i}" for i in range(num)],
            "total": [10.0 * (i + 1) for i in range(num)],
            "date": ["2024-01-01"] * num,
            "currency": ["USD"] * num,
        }
    )
    return transactions, proofs
