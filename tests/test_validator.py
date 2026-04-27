import pandas as pd
import pytest

from intelligence.validator import Results, Validator
import os


@pytest.fixture
def mock_documents(sample_transactions_df, sample_proofs_df):
    return sample_transactions_df.copy(), sample_proofs_df.copy()


def test_validator(mock_documents):
    # Testing the Validator's validate function
    transactions, proofs = mock_documents
    validator = Validator(transactions, proofs)
    results: Results = validator.validate()

    validated_transactions = results.validated_transactions
    discrepancies = results.discrepancies
    unmatched_transactions = results.unmatched_transactions
    unmatched_proofs = results.unmatched_proofs

    assert len(validated_transactions)
    assert not len(discrepancies)
    assert not len(unmatched_transactions)
    assert not len(unmatched_proofs)


def test_find_discrepancies():
    # Test validation when there are discrepancies in totals
    transactions = pd.DataFrame(
        {
            "business_name": ["Ikkousha Craft Ramen", "Taco Bell"],
            "total": [12.30, 15.00],
            "date": ["2023-01-01", "2021-10-12"],
        }
    )
    proofs = pd.DataFrame(
        {
            "business_name": ["Ikkousha Craft Ramen", "Taco Bell"],
            "total": [12.30, 14.50],
            "date": ["2023-01-01", "2021-10-12"],
        }
    )

    validator = Validator(transactions, proofs)
    results = validator.validate()
    discrepancies = results.discrepancies

    assert len(discrepancies) == 1
    assert discrepancies["Transaction Business Name"].iloc[0] == "Taco Bell"
    assert discrepancies["Transaction Total"].iloc[0] == 15.00
    assert discrepancies["Proof Total"].iloc[0] == 14.50
    assert discrepancies["Delta"].iloc[0] == 0.50


def test_unmatched_transactions():
    # Test validation when there are unmatched transactions
    transactions = pd.DataFrame(
        {
            "business_name": ["Ikkousha Craft Ramen", "Taco Bell", "Starbucks"],
            "total": [12.30, 15.00, 5.00],
            "date": ["2023-01-01", "2021-10-12", "2022-05-15"],
        }
    )
    proofs = pd.DataFrame(
        {
            "business_name": ["Ikkousha Craft Ramen", "Taco Bell"],
            "total": [12.30, 15.00],
            "date": ["2023-01-01", "2021-10-12"],
        }
    )

    validator = Validator(transactions, proofs)
    results: Results = validator.validate()

    discrepancies = results.discrepancies
    unmatched_transactions = results.unmatched_transactions
    unmatched_proofs = results.unmatched_proofs

    assert not len(discrepancies)
    assert len(unmatched_transactions) == 1
    assert unmatched_transactions["Business Name"].iloc[0] == "Starbucks"
    assert unmatched_transactions["Total"].iloc[0] == 5.00
    assert unmatched_transactions["Date"].iloc[0] == "2022-05-15"
    assert not len(unmatched_proofs)


def test_unmatched_proofs():
    # Test validation when there are unmatched proofs
    transactions = pd.DataFrame(
        {
            "business_name": ["Ikkousha Craft Ramen", "Taco Bell"],
            "total": [12.30, 15.00],
            "date": ["2023-01-01", "2021-10-12"],
        }
    )
    proofs = pd.DataFrame(
        {
            "business_name": ["Ikkousha Craft Ramen", "Taco Bell", "Starbucks"],
            "total": [12.30, 15.00, 5.00],
            "date": ["2023-01-01", "2021-10-12", "2022-05-15"],
        }
    )

    validator = Validator(transactions, proofs)
    results = validator.validate()

    discrepancies = results.discrepancies
    unmatched_transactions = results.unmatched_transactions
    unmatched_proofs = results.unmatched_proofs

    assert not len(discrepancies)
    assert not len(unmatched_transactions)
    assert len(unmatched_proofs) == 1
    assert unmatched_proofs["Business Name"].iloc[0] == "Starbucks"
    assert unmatched_proofs["Total"].iloc[0] == 5.00
    assert unmatched_proofs["Date"].iloc[0] == "2022-05-15"


def test_validate_handles_empty_proofs_without_crashing():
    transactions = pd.DataFrame(
        {
            "business_name": ["Coffee Shop", "Book Store"],
            "total": [4.5, 19.2],
            "date": ["2024-01-03", "2024-01-03"],
        }
    )
    proofs = pd.DataFrame(columns=["business_name", "total", "date"])

    validator = Validator(transactions, proofs)
    results = validator.validate()

    assert len(results.validated_transactions) == 0
    assert len(results.discrepancies) == 0
    assert len(results.unmatched_transactions) == 2
    assert len(results.unmatched_proofs) == 0


@pytest.mark.skipif(
    os.getenv("GITHUB_ACTIONS") == "true", reason="Skipping on GitHub Actions"
)
def test_similar_business_names():
    # Test validation when there are similar business names
    transactions = pd.DataFrame(
        {
            "business_name": ["Ikkousha Craft Ramen", "Taco Bell"],
            "total": [12.30, 14.50],
            "date": ["2023-01-01", "2021-10-12"],
        }
    )
    proofs = pd.DataFrame(
        {
            "business_name": ["Ikkousha Ramen Costa Mesa", "Taco Bell"],
            "total": [12.30, 14.50],
            "date": ["2023-01-01", "2021-10-12"],
        }
    )

    validator = Validator(transactions, proofs)
    results = validator.validate()
    _, recommendations = validator.analyze_results(results)

    assert len(results.validated_transactions) == 1
    assert len(recommendations) >= 1


@pytest.mark.skipif(
    os.getenv("GITHUB_ACTIONS") == "true", reason="Skipping on GitHub Actions"
)
def test_different_name_same_totals_and_dates():
    # Test validation when there are similar business names
    transactions = pd.DataFrame(
        {
            "business_name": ["Boba Place", "Taco Bell"],
            "total": [12.30, 14.50],
            "date": ["2023-01-01", "2021-10-12"],
        }
    )
    proofs = pd.DataFrame(
        {
            "business_name": ["Kiosk Barcelona", "Taco Bell"],
            "total": [12.30, 14.50],
            "date": ["2023-01-01", "2021-10-12"],
        }
    )

    validator = Validator(transactions, proofs)
    results = validator.validate()
    _, recommendations = validator.analyze_results(results)

    assert len(results.validated_transactions) == 1
    assert len(recommendations) >= 1


def test_validate_matches_when_date_formats_differ():
    transactions = pd.DataFrame(
        {
            "business_name": ["Coffee Shop"],
            "total": [4.50],
            "date": ["03/08/2026"],
        }
    )
    proofs = pd.DataFrame(
        {
            "business_name": ["Coffee Shop"],
            "total": [4.50],
            "date": ["2026-03-08"],
        }
    )

    validator = Validator(transactions, proofs)
    results = validator.validate()

    assert len(results.validated_transactions) == 1
    assert len(results.unmatched_transactions) == 0
    assert len(results.unmatched_proofs) == 0


def test_no_recommendations_without_unmatched_transactions():
    validator = Validator(pd.DataFrame([]), pd.DataFrame([]))
    results = Results(
        validated_transactions=pd.DataFrame([]),
        discrepancies=pd.DataFrame([]),
        unmatched_transactions=pd.DataFrame([]),
        unmatched_proofs=pd.DataFrame(
            [{"Business Name": "Store", "Total": 10.0, "Date": "2024-01-01"}]
        ),
    )

    analysis, recommendations = validator.analyze_results(results)

    assert recommendations.empty
    assert "no recommendations" in analysis.lower()


def test_validate_matches_with_noisy_date_text():
    transactions = pd.DataFrame(
        {
            "business_name": ["Coffee Shop"],
            "total": [4.50],
            "date": ["2026-03-08"],
        }
    )
    proofs = pd.DataFrame(
        {
            "business_name": ["Coffee Shop"],
            "total": [4.50],
            "date": ["Date: 2026-03-08"],
        }
    )

    validator = Validator(transactions, proofs)
    results = validator.validate()

    assert len(results.validated_transactions) == 1
    assert len(results.unmatched_transactions) == 0
    assert len(results.unmatched_proofs) == 0


def test_recommend_when_unmatched_date_and_totals_match_even_if_names_differ():
    validator = Validator(pd.DataFrame([]), pd.DataFrame([]))
    unmatched_transactions = pd.DataFrame(
        [
            {
                "Business Name": "Completely Different Tx Name",
                "Total": 44.10,
                "Date": "2024-02-20",
            }
        ]
    )
    unmatched_proofs = pd.DataFrame(
        [
            {
                "Business Name": "Totally Different Proof Name",
                "Total": 44.10,
                "Date": "2024-02-21",
            }
        ]
    )

    recommendations = validator.analyze_unmatched_results(
        unmatched_transactions,
        unmatched_proofs,
    )

    assert len(recommendations) == 1
    assert recommendations["Transaction Total"].iloc[0] == 44.10
    assert recommendations["Proof Total"].iloc[0] == 44.10


def test_recommend_when_unmatched_dates_within_two_days_and_totals_within_cent():
    validator = Validator(pd.DataFrame([]), pd.DataFrame([]))
    unmatched_transactions = pd.DataFrame(
        [
            {
                "Business Name": "Merchant A",
                "Total": 10.00,
                "Date": "2024-02-20",
            }
        ]
    )
    unmatched_proofs = pd.DataFrame(
        [
            {
                "Business Name": "Merchant B",
                "Total": 10.01,
                "Date": "2024-02-22",
            }
        ]
    )

    recommendations = validator.analyze_unmatched_results(
        unmatched_transactions,
        unmatched_proofs,
    )

    assert len(recommendations) == 1
    assert recommendations["Transaction Total"].iloc[0] == 10.00
    assert recommendations["Proof Total"].iloc[0] == 10.01
