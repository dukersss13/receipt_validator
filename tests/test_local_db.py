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

    db_transactions, db_proofs = local_db.load_session_history(session_obj.id)
    assert len(db_transactions) == len(transactions)
    assert len(db_proofs) == len(proofs)

    print("✅ Database setup test passed.")
