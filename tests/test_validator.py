import pytest
from mock_documents import create_mock_documents
from src.validator import Validator


@pytest.fixture
def mock_documents():
    transactions, proofs = create_mock_documents(n_unique=5, n_similar=0)

    return transactions, proofs


@pytest.fixture
def similar_documents():
    transactions, proofs = create_mock_documents(n_unique=1, n_similar=3)

    return transactions, proofs


def test_validate_function(mock_documents):
    # Testing the Validator's validate function
    transactions, proofs = mock_documents
    validator = Validator(transactions, proofs)
    discrepancies, unmatched_transactions, unmatched_proofs = validator.validate()

    assert not len(discrepancies)
    assert not len(unmatched_transactions)
    assert not len(unmatched_proofs)


def test_similar_transactions(similar_documents):
    # Test to see the maching process when
    # we have multiple transactions on the same date
    # from the same business
    transactions, proofs = similar_documents
    proofs["total"] = [*proofs["total"].values[:-1], 100.0]
    validator = Validator(transactions, proofs)

    discrepancies, unmatched_transactions, unmatched_proofs = validator.validate()

    assert len(discrepancies)
    assert not len(unmatched_transactions)
    assert not len(unmatched_proofs)
