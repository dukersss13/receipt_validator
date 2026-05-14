import pytest
from werkzeug.security import generate_password_hash

from tests.mock_documents import create_mock_documents
from src.data.db_schema import Session
from src.data.database import DataBase

local_db = DataBase(engine_name="tests/data/db/test_db", reset_db=True)
session_obj = Session(session_id="test-local-session", user_id="local_test")


@pytest.fixture
def mock_documents():
    return create_mock_documents(num=3)


def test_setup_db(mock_documents):
    """
    Test the setup_db function to ensure it initializes the database correctly.
    """
    transactions, proofs = mock_documents

    # Append transactions and proofs to the local database
    local_db.append_transactions(session_obj, transactions)
    local_db.append_proofs(session_obj, proofs)

    db_transactions, db_proofs = local_db.load_session_history(session_obj.session_id)
    assert len(db_transactions) == len(transactions)
    assert len(db_proofs) == len(proofs)

    print("✅ Database setup test passed.")


def test_create_or_link_google_user_marks_provider_google() -> None:
    db = DataBase(engine_name="tests/data/db/test_google_link", reset_db=True)
    user = db.create_user_auth(
        "linkme@example.com",
        generate_password_hash("StrongPass123!"),
    )
    assert user.provider == "email"

    linked = db.create_or_link_google_user(
        email="linkme@example.com",
        provider_id="google-sub-999",
        password_hash_fallback=generate_password_hash("fallback"),
    )

    assert linked.email == "linkme@example.com"
    assert linked.provider == "google"
    assert linked.provider_id == "google-sub-999"
