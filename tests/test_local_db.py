from tests.test_validator import mock_documents

from src.data.db_schema import Session
from data.database import (
    setup_db,
    append_transactions_to_db,
    append_proofs_to_db,
)

local_db = setup_db(engine_name="tests/data/db/test_db")
session_id = Session(user_id="local_test")


def test_setup_db(mock_documents):
    """
    Test the setup_db function to ensure it initializes the database correctly.
    """
    transactions, proofs = mock_documents

    # Append transactions and proofs to the local database
    append_transactions_to_db(local_db, session_id, transactions)
    append_proofs_to_db(local_db, session_id, proofs)

    # Verify that the data was added correctly
    session_obj = local_db.query(Session).first()
    assert session_obj is not None
    assert len(session_obj.transactions) == len(transactions)
    assert len(session_obj.proofs) == len(proofs)

    print("✅ Database setup test passed.")
