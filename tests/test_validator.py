import pandas as pd
import pytest

from mock_documents import create_mock_documents
from src.validator import Validator


@pytest.fixture
def mock_documents():
    transactions, proofs = create_mock_documents(num=5)

    return transactions, proofs


def test_validator(mock_documents):
    # Testing the Validator's validate function
    transactions, proofs = mock_documents
    validator = Validator(transactions, proofs)
    discrepancies, unmatched_transactions, unmatched_proofs = validator.validate()

    assert not len(discrepancies)
    assert not len(unmatched_transactions)
    assert not len(unmatched_proofs)


def test_similar_business_names():
    # Test validation when there are similar business names
    transactions = pd.DataFrame({"business_name": ["Ikkousha Craft Ramen", "Taco Bell"],
                                 "total": [12.30, 14.50],
                                 "date": ["2023-01-01", "2021-10-12"]})
    proofs = pd.DataFrame({"business_name": ["Ikkousha Ramen Costa Mesa", "Taco Bell"],
                            "total": [12.30, 14.50],
                            "date": ["2023-01-01", "2021-10-12"]})

    validator = Validator(transactions, proofs)
    results = validator.validate()
    recommendations = validator.analyze_results(results)
