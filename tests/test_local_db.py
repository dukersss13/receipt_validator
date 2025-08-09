from tests.test_validator import mock_documents

from src.data.db_schema import Session
from src.data.database import DataBase

local_db = DataBase(engine_name="tests/data/db/test_db")
session_obj = Session(user_id="local_test")


def test_setup_db(mock_documents):
    """
    Test the setup_db function to ensure it initializes the database correctly.
    """
    transactions, proofs = mock_documents

    # Append transactions and proofs to the local database
    local_db.append_transactions(session_obj, transactions)
    local_db.append_proofs(session_obj, proofs)

    # Verify that the data was added correctly
    db_session = local_db.db.query(Session).first()
    assert db_session is not None
    assert len(db_session.transactions) == len(transactions)
    assert len(db_session.proofs) == len(proofs)

    print("✅ Database setup test passed.")
