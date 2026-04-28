import os

import pytest

from src.intelligence.categorize import TransactionCategorizer


@pytest.mark.skipif(
    os.getenv("GITHUB_ACTIONS") == "true",
    reason="Skipping real-client categorization test on GitHub Actions",
)
def test_categorizer_uses_business_name_only_for_transactions_and_proofs(
    sample_validated_transactions_df,
    sample_transactions_df,
):
    has_gemini_key = bool(os.getenv("GEMINI_API_KEY", "").strip()) or os.path.exists(
        "secrets/google_gemini_api_key"
    )
    if not has_gemini_key:
        pytest.skip("Skipping real-client categorization test: no API key configured")

    categorizer = TransactionCategorizer(
        config={"categorize.enabled": True, "categorize.chunk_size": 100}
    )

    tx_input = sample_transactions_df.copy()
    tx_input = tx_input.drop(columns=["category"], errors="ignore")

    tx_result = categorizer.categorize_dataframe(tx_input)
    ground_truth = sample_validated_transactions_df["Transaction Category"].astype(str)
    predicted = tx_result.frame["category"].astype(str)

    correct = (predicted == ground_truth).sum()
    overall_accuracy = correct / len(ground_truth)
    print(f"overall_accuracy={overall_accuracy:.2%}")

    assert tx_result.summary["rowsProcessed"] == len(ground_truth)
    assert overall_accuracy > 0.9
