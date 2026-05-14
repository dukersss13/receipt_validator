import pandas as pd
import pytest

from src.agents.validator import Results, Validator
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


def test_analysis_does_not_claim_recommendations_when_none_generated(monkeypatch):
    validator = Validator(pd.DataFrame([]), pd.DataFrame([]))
    results = Results(
        validated_transactions=pd.DataFrame([]),
        discrepancies=pd.DataFrame([]),
        unmatched_transactions=pd.DataFrame(
            [{"Business Name": "Store A", "Total": 10.0, "Date": "2024-01-01"}]
        ),
        unmatched_proofs=pd.DataFrame(
            [{"Business Name": "Store B", "Total": 11.0, "Date": "2024-01-02"}]
        ),
    )

    monkeypatch.setattr(
        validator,
        "analyze_unmatched_results",
        lambda unmatched_transactions, unmatched_proofs: pd.DataFrame([]),
    )

    analysis, recommendations = validator.analyze_results(results)

    assert recommendations.empty
    assert "provided some recommendations" not in analysis.lower()


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
                "Business Name": "XQZ-718-ALPHA",
                "Total": 44.10,
                "Date": "2024-02-20",
                "Category": "Food",
            }
        ]
    )
    unmatched_proofs = pd.DataFrame(
        [
            {
                "Business Name": "MNR-004-BETA",
                "Total": 44.10,
                "Date": "2024-02-21",
                "Category": "Food",
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
    assert recommendations["Transaction Category"].iloc[0] == "Food"
    assert recommendations["Proof Category"].iloc[0] == "Food"
    assert recommendations["Reason"].iloc[0] == "Similar dates, totals"


def test_recommend_when_unmatched_dates_within_one_day_and_totals_within_cent():
    validator = Validator(pd.DataFrame([]), pd.DataFrame([]))
    unmatched_transactions = pd.DataFrame(
        [
            {
                "Business Name": "XQZ-718-ALPHA",
                "Total": 10.00,
                "Date": "2024-02-20",
            }
        ]
    )
    unmatched_proofs = pd.DataFrame(
        [
            {
                "Business Name": "MNR-004-BETA",
                "Total": 10.01,
                "Date": "2024-02-21",
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
    assert recommendations["Reason"].iloc[0] == "Similar dates, totals"


def test_recommend_when_only_name_is_similar():
    validator = Validator(pd.DataFrame([]), pd.DataFrame([]))
    unmatched_transactions = pd.DataFrame(
        [
            {
                "Business Name": "Starbucks Costa Mesa",
                "Total": 20.00,
                "Date": "2024-02-20",
            }
        ]
    )
    unmatched_proofs = pd.DataFrame(
        [
            {
                "Business Name": "Starbucks",
                "Total": 90.00,
                "Date": "2024-04-20",
            }
        ]
    )

    recommendations = validator.analyze_unmatched_results(
        unmatched_transactions,
        unmatched_proofs,
    )

    assert len(recommendations) == 1
    assert recommendations["Reason"].iloc[0] == "Similar names"


def test_recommend_when_only_date_is_similar():
    validator = Validator(pd.DataFrame([]), pd.DataFrame([]))
    unmatched_transactions = pd.DataFrame(
        [{"Business Name": "XQZ-718-ALPHA", "Total": 10.00, "Date": "2024-02-20"}]
    )
    unmatched_proofs = pd.DataFrame(
        [{"Business Name": "MNR-004-BETA", "Total": 99.00, "Date": "2024-02-21"}]
    )

    recommendations = validator.analyze_unmatched_results(
        unmatched_transactions,
        unmatched_proofs,
    )

    assert len(recommendations) == 1
    assert recommendations["Reason"].iloc[0] == "Similar dates"


def test_recommend_when_only_amount_is_similar():
    validator = Validator(pd.DataFrame([]), pd.DataFrame([]))
    unmatched_transactions = pd.DataFrame(
        [{"Business Name": "XQZ-718-ALPHA", "Total": 10.00, "Date": "2024-02-20"}]
    )
    unmatched_proofs = pd.DataFrame(
        [{"Business Name": "MNR-004-BETA", "Total": 10.03, "Date": "2024-03-20"}]
    )

    recommendations = validator.analyze_unmatched_results(
        unmatched_transactions,
        unmatched_proofs,
    )

    assert len(recommendations) == 1
    assert recommendations["Reason"].iloc[0] == "Similar totals"


def test_recommend_reason_includes_all_three_matching_factors():
    validator = Validator(pd.DataFrame([]), pd.DataFrame([]))
    unmatched_transactions = pd.DataFrame(
        [
            {
                "Business Name": "Whole Foods Market",
                "Total": 45.00,
                "Date": "2024-02-20",
            }
        ]
    )
    unmatched_proofs = pd.DataFrame(
        [{"Business Name": "Whole Foods", "Total": 45.02, "Date": "2024-02-21"}]
    )

    recommendations = validator.analyze_unmatched_results(
        unmatched_transactions,
        unmatched_proofs,
    )

    assert len(recommendations) == 1
    assert recommendations["Reason"].iloc[0] == "Similar names, dates, totals"


def test_recommendation_thresholds_are_configurable():
    validator = Validator(
        pd.DataFrame([]),
        pd.DataFrame([]),
        parsed_config={
            "recommendation_matching": {
                "name_similarity_threshold": 0.95,
                "date_window_days": 0,
                "amount_threshold": 2.0,
            }
        },
    )
    unmatched_transactions = pd.DataFrame(
        [{"Business Name": "XQZ-718-ALPHA", "Total": 10.00, "Date": "2024-02-20"}]
    )
    unmatched_proofs = pd.DataFrame(
        [{"Business Name": "MNR-004-BETA", "Total": 11.50, "Date": "2024-02-22"}]
    )

    recommendations = validator.analyze_unmatched_results(
        unmatched_transactions,
        unmatched_proofs,
    )

    assert len(recommendations) == 1
    assert recommendations["Reason"].iloc[0] == "Similar totals"


def test_recommendation_categories_map_from_lowercase_category_columns():
    validator = Validator(pd.DataFrame([]), pd.DataFrame([]))
    unmatched_transactions = pd.DataFrame(
        [
            {
                "Business Name": "Bakery North",
                "Total": 12.34,
                "Date": "2024-07-01",
                "category": "Food",
            }
        ]
    )
    unmatched_proofs = pd.DataFrame(
        [
            {
                "Business Name": "Bakery N.",
                "Total": 12.34,
                "Date": "2024-07-02",
                "category": "Groceries",
            }
        ]
    )

    recommendations = validator.analyze_unmatched_results(
        unmatched_transactions,
        unmatched_proofs,
    )

    assert len(recommendations) == 1
    assert recommendations["Transaction Category"].iloc[0] == "Food"
    assert recommendations["Proof Category"].iloc[0] == "Groceries"


def test_recommendation_categories_become_empty_strings_when_missing():
    validator = Validator(pd.DataFrame([]), pd.DataFrame([]))
    unmatched_transactions = pd.DataFrame(
        [
            {
                "Business Name": "Store A",
                "Total": 15.00,
                "Date": "2024-01-10",
                "Category": None,
            }
        ]
    )
    unmatched_proofs = pd.DataFrame(
        [
            {
                "Business Name": "Store B",
                "Total": 15.00,
                "Date": "2024-01-11",
                "Category": None,
            }
        ]
    )

    recommendations = validator.analyze_unmatched_results(
        unmatched_transactions,
        unmatched_proofs,
    )

    assert len(recommendations) == 1
    assert recommendations["Transaction Category"].iloc[0] == ""
    assert recommendations["Proof Category"].iloc[0] == ""


def test_unmatched_column_rename_uses_dict_not_positional():
    """Ensure unmatched DataFrames are renamed by column name, not position,
    so categories survive even when the DataFrame has extra columns."""
    transactions = pd.DataFrame(
        {
            "business_name": ["Coffee House", "Pet Store"],
            "total": [5.50, 22.00],
            "date": ["2024-06-01", "2024-06-02"],
            "category": ["Food", "Shopping"],
        }
    )
    proofs = pd.DataFrame(
        {
            "business_name": ["Deli Corner"],
            "total": [9.99],
            "date": ["2024-06-03"],
            "category": ["Food"],
        }
    )

    validator = Validator(transactions, proofs)
    results = validator.validate()

    # Unmatched DataFrames must have the renamed "Category" column
    if not results.unmatched_transactions.empty:
        assert "Category" in results.unmatched_transactions.columns
        for val in results.unmatched_transactions["Category"]:
            assert val in ("Food", "Shopping", "Other", "")

    if not results.unmatched_proofs.empty:
        assert "Category" in results.unmatched_proofs.columns
