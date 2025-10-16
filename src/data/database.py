# Set up the SQLite database
import os
import pandas as pd

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from src.data.db_schema import Base, Session, Transaction, Proof


class DataBase:
    def __init__(self, engine_name: str, local_db: bool = True):
        """
        Initialize the database connection and create tables if needed.
        """
        self.db_path = f"{engine_name}.db"
        self.local_db = local_db

        if local_db:
            db_exists = os.path.exists(self.db_path)
            self.engine = create_engine(f"sqlite:///{self.db_path}", echo=True)
            Base.metadata.drop_all(bind=self.engine)
            Base.metadata.create_all(bind=self.engine)

            if db_exists:
                print(f"📂 Found existing database '{self.db_path}'.")
            else:
                print(f"🆕 Creating new database '{self.db_path}'.")
        else:
            self.engine = create_engine(engine_name, echo=True)

        SessionLocal = sessionmaker(bind=self.engine)
        self.db = SessionLocal()
    

    def load_session_history(self, session_id: int) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Load both transaction and proof history for a given session ID.
        Returns (transactions_df, proofs_df).
        """
        # --- Load Transactions ---
        txn_query = self.db.query(Transaction).filter(Transaction.session_id == session_id)
        transactions_df = pd.read_sql(txn_query.statement, self.db.bind)

        # --- Load Proofs ---
        proof_query = self.db.query(Proof).filter(Proof.session_id == session_id)
        proofs_df = pd.read_sql(proof_query.statement, self.db.bind)

        print(f"Loaded {len(transactions_df)} transactions and {len(proofs_df)} proofs for session {session_id}")

        return transactions_df, proofs_df


    def append_transactions(self, session_obj: Session, transaction_data: pd.DataFrame):
        """
        Append transaction data from a DataFrame to the database session.
        Links all transactions to the provided session object.
        """
        if transaction_data["date"].dtype == object:
            transaction_data["date"] = pd.to_datetime(transaction_data["date"]).dt.date

        if session_obj.id is None:
            self.db.add(session_obj)
            self.db.flush()

        for _, row in transaction_data.iterrows():
            txn = Transaction(
                business_name=row["business_name"],
                total=row["total"],
                date=row["date"],
                currency=row["currency"],
                session_id=session_obj.id,
            )
            self.db.add(txn)

        self.db.commit()
        print(
            f"✅ {len(transaction_data)} transactions committed to session ID {session_obj.id}"
        )

    def append_proofs(self, session_obj: Session, proof_data: pd.DataFrame):
        """
        Append proof data from a DataFrame to the database session.
        Links all proof records to the provided session object.
        """
        if proof_data["date"].dtype == object:
            proof_data["date"] = pd.to_datetime(proof_data["date"]).dt.date

        if session_obj.id is None:
            self.db.add(session_obj)
            self.db.flush()

        for _, row in proof_data.iterrows():
            proof = Proof(
                business_name=row["business_name"],
                total=row["total"],
                date=row["date"],
                currency=row["currency"],
                session_id=session_obj.id,
            )
            self.db.add(proof)

        self.db.commit()
        print(f"✅ {len(proof_data)} proofs committed to session ID {session_obj.id}")
