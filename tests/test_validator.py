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
