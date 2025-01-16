import pandas as pd
import pytest

from mock_documents import create_mock_documents
from src.validator import Validator
import os


@pytest.fixture
def mock_documents():
    transactions, proofs = create_mock_documents(num=3)

    return transactions, proofs


def test_validator(mock_documents):
    # Testing the Validator's validate function
    transactions, proofs = mock_documents
    validator = Validator(transactions, proofs, setup_client=False)
    discrepancies, unmatched_transactions, unmatched_proofs = validator.validate()

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

    validator = Validator(transactions, proofs, setup_client=False)
    discrepancies, unmatched_transactions, unmatched_proofs = validator.validate()

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

    validator = Validator(transactions, proofs, setup_client=False)
    discrepancies, unmatched_transactions, unmatched_proofs = validator.validate()

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

    validator = Validator(transactions, proofs, setup_client=False)
    discrepancies, unmatched_transactions, unmatched_proofs = validator.validate()

    assert not len(discrepancies)
    assert not len(unmatched_transactions)
    assert len(unmatched_proofs) == 1
    assert unmatched_proofs["Business Name"].iloc[0] == "Starbucks"
    assert unmatched_proofs["Total"].iloc[0] == 5.00
    assert unmatched_proofs["Date"].iloc[0] == "2022-05-15"


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

    assert len(recommendations)
    assert (
        recommendations["Transaction Business Name"][0].strip()
        == "Ikkousha Craft Ramen"
    )
    assert (
        recommendations["Proof Business Name"][0].strip() == "Ikkousha Ramen Costa Mesa"
    )


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

    assert len(recommendations)
    assert recommendations["Transaction Business Name"][0].strip() == "Boba Place"
    assert recommendations["Proof Business Name"][0].strip() == "Kiosk Barcelona"
    assert recommendations["Transaction Total"][0] == 12.30
    assert recommendations["Proof Total"][0] == 12.30
