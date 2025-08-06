# Set up the SQLite database
import os
import pandas as pd

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from src.data.db_schema import Base, Session, Transaction, Proof


def append_transactions_to_db(
    db: Session, session_obj: Session, transaction_data: pd.DataFrame
):
    """
    Append transaction data from a DataFrame to the given database session.
    Links all transactions to the provided session object.

    Parameters:
    - db: SQLAlchemy database session
    - session_obj: an instance of the Session model (already added to db)
    - data: a pandas DataFrame with columns ['business_name', 'total', 'date', 'currency']
    """

    # Convert string dates to datetime.date (if needed)
    if transaction_data["date"].dtype == object:
        transaction_data["date"] = pd.to_datetime(transaction_data["date"]).dt.date

    # Ensure session_obj is flushed so it gets an ID
    if session_obj.id is None:
        db.add(session_obj)
        db.flush()

    # Create Transaction objects and add to DB
    for _, row in transaction_data.iterrows():
        txn = Transaction(
            business_name=row["business_name"],
            total=row["total"],
            date=row["date"],
            currency=row["currency"],
            session_id=session_obj.id,  # foreign key
        )
        db.add(txn)

    db.commit()
    print(
        f"✅ {len(transaction_data)} transactions committed to session ID {session_obj.id}"
    )

def append_proofs_to_db(db: Session, session_obj: Session, proof_data: pd.DataFrame):
    """
    Append proof data from a DataFrame to the given database session.
    Links all proof records to the provided session object.

    Parameters:
    - db: SQLAlchemy database session
    - session_obj: an instance of the Session model (already added to db)
    - data: a pandas DataFrame with columns ['business_name', 'total', 'date', 'currency']
    """

    # Convert string dates to datetime.date (if needed)
    if proof_data["date"].dtype == object:
        proof_data["date"] = pd.to_datetime(proof_data["date"]).dt.date

    # Ensure session_obj is flushed so it has an ID
    if session_obj.id is None:
        db.add(session_obj)
        db.flush()

    # Create Proof objects and add to DB
    for _, row in proof_data.iterrows():
        proof = Proof(
            business_name=row["business_name"],
            total=row["total"],
            date=row["date"],
            currency=row["currency"],
            session_id=session_obj.id,  # foreign key
        )
        db.add(proof)

    db.commit()
    print(f"✅ {len(proof_data)} proofs committed to session ID {session_obj.id}")


def setup_db(engine_name: str, local_db: bool = True) -> Session:
    """
    Set up the database with the given engine name.
    If the database file already exists, it connects to it.
    Otherwise, it creates the database and initializes tables.
    
    Returns a SQLAlchemy Session instance.
    """
    if local_db:
        db_path = f"{engine_name}.db"
        db_exists = os.path.exists(db_path)

        engine = create_engine(f"sqlite:///{db_path}", echo=True)
        Base.metadata.drop_all(bind=engine)
        Base.metadata.create_all(bind=engine)

        if db_exists:
            print(f"📂 Found existing database '{db_path}'.")
        else:
            print(f"🆕 Creating new database '{db_path}'.")

    # Create DB session
    SessionLocal = sessionmaker(bind=engine)
    db = SessionLocal()

    return db