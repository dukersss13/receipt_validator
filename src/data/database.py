import os

import pandas as pd

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from src.data.db_schema import Base, Session, Transaction, Proof


class DataBase:
    def __init__(
        self,
        engine_name: str,
        local_db: bool = True,
        reset_db: bool = False,
        echo: bool = False,
    ):
        """
        Initialize the database connection and create tables if needed.
        """
        self.local_db = local_db

        if local_db:
            self.db_path = (
                f"{engine_name}.db" if not engine_name.endswith(".db") else engine_name
            )
            parent_dir = os.path.dirname(self.db_path)
            if parent_dir:
                os.makedirs(parent_dir, exist_ok=True)

            db_exists = os.path.exists(self.db_path)
            self.engine = create_engine(f"sqlite:///{self.db_path}", echo=echo)

            if reset_db:
                Base.metadata.drop_all(bind=self.engine)

            Base.metadata.create_all(bind=self.engine)

            if db_exists:
                print(f"📂 Found existing database '{self.db_path}'.")
            else:
                print(f"🆕 Creating new database '{self.db_path}'.")
        else:
            self.engine = create_engine(engine_name, echo=echo)
            if reset_db:
                Base.metadata.drop_all(bind=self.engine)
            Base.metadata.create_all(bind=self.engine)

        self.SessionLocal = sessionmaker(bind=self.engine, expire_on_commit=False)

    @staticmethod
    def _normalize_input_df(frame: pd.DataFrame) -> pd.DataFrame:
        """Normalize incoming transaction/proof rows before persistence."""
        if frame is None or frame.empty:
            return pd.DataFrame(
                [], columns=["business_name", "total", "date", "currency"]
            )

        normalized = frame.copy()
        normalized.columns = [str(col).strip().lower() for col in normalized.columns]

        required = ["business_name", "total", "date", "currency"]
        missing = [col for col in required if col not in normalized.columns]
        if missing:
            raise ValueError(f"Missing required columns: {', '.join(missing)}")

        normalized = normalized[required]
        normalized["business_name"] = (
            normalized["business_name"].astype(str).str.strip()
        )
        normalized["total"] = pd.to_numeric(normalized["total"], errors="raise")
        normalized["date"] = pd.to_datetime(normalized["date"], errors="raise").dt.date
        normalized["currency"] = (
            normalized["currency"]
            .astype(str)
            .str.upper()
            .str.strip()
            .replace("", "USD")
        )

        return normalized

    def get_or_create_session(
        self, session_id: str, user_id: str | None = None
    ) -> Session:
        """Fetch a session by external session_id or create it when missing."""
        normalized_session_id = str(session_id).strip()
        if not normalized_session_id:
            raise ValueError("session_id cannot be empty.")

        with self.SessionLocal() as db:
            session_obj = (
                db.query(Session)
                .filter(Session.session_id == normalized_session_id)
                .first()
            )

            if session_obj is None:
                session_obj = Session(session_id=normalized_session_id, user_id=user_id)
                db.add(session_obj)
                db.commit()
                db.refresh(session_obj)

            return session_obj

    def save_session_inputs(
        self,
        session_id: str,
        transaction_data: pd.DataFrame,
        proof_data: pd.DataFrame,
        replace_existing: bool = True,
    ) -> None:
        """
        Save transaction/proof inputs for a session_id.
        If replace_existing=True, existing rows for this session are replaced.
        """
        normalized_session_id = str(session_id).strip()
        if not normalized_session_id:
            raise ValueError("session_id cannot be empty.")

        transactions_df = self._normalize_input_df(transaction_data)
        proofs_df = self._normalize_input_df(proof_data)

        with self.SessionLocal() as db:
            session_obj = (
                db.query(Session)
                .filter(Session.session_id == normalized_session_id)
                .first()
            )

            if session_obj is None:
                session_obj = Session(session_id=normalized_session_id)
                db.add(session_obj)
                db.flush()

            if replace_existing:
                db.query(Proof).filter(Proof.session_ref_id == session_obj.id).delete()
                db.query(Transaction).filter(
                    Transaction.session_ref_id == session_obj.id
                ).delete()

            for _, row in transactions_df.iterrows():
                db.add(
                    Transaction(
                        session_ref_id=session_obj.id,
                        business_name=row["business_name"],
                        total=float(row["total"]),
                        currency=row["currency"],
                        date=row["date"],
                    )
                )

            for _, row in proofs_df.iterrows():
                db.add(
                    Proof(
                        session_ref_id=session_obj.id,
                        business_name=row["business_name"],
                        total=float(row["total"]),
                        currency=row["currency"],
                        date=row["date"],
                    )
                )

            db.commit()

    def load_session_history(
        self, session_id: str
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Load both transaction and proof history for a given external session_id.
        Returns (transactions_df, proofs_df).
        """
        normalized_session_id = str(session_id).strip()
        if not normalized_session_id:
            raise ValueError("session_id cannot be empty.")

        with self.SessionLocal() as db:
            session_obj = (
                db.query(Session)
                .filter(Session.session_id == normalized_session_id)
                .first()
            )

            if session_obj is None:
                raise ValueError(f"Session '{normalized_session_id}' not found")

            txn_query = db.query(Transaction).filter(
                Transaction.session_ref_id == session_obj.id
            )
            transactions_df = pd.read_sql(txn_query.statement, db.bind)

            proof_query = db.query(Proof).filter(Proof.session_ref_id == session_obj.id)
            proofs_df = pd.read_sql(proof_query.statement, db.bind)

        for frame in (transactions_df, proofs_df):
            if "date" in frame.columns and not frame.empty:
                frame["date"] = pd.to_datetime(frame["date"], errors="coerce").dt.date

        print(
            f"Loaded {len(transactions_df)} transactions and {len(proofs_df)} proofs for session {normalized_session_id}"
        )

        return transactions_df, proofs_df

    def append_transactions(self, session_obj: Session, transaction_data: pd.DataFrame):
        """
        Append transaction data from a DataFrame to the database session.
        Links all transactions to the provided session object.
        """
        if not session_obj.session_id:
            raise ValueError("session_obj.session_id is required")

        empty_proofs = pd.DataFrame(
            [], columns=["business_name", "total", "date", "currency"]
        )
        self.save_session_inputs(
            session_id=session_obj.session_id,
            transaction_data=transaction_data,
            proof_data=empty_proofs,
            replace_existing=False,
        )

        print(
            f"✅ {len(transaction_data)} transactions committed to session ID {session_obj.session_id}"
        )

    def append_proofs(self, session_obj: Session, proof_data: pd.DataFrame):
        """
        Append proof data from a DataFrame to the database session.
        Links all proof records to the provided session object.
        """
        if not session_obj.session_id:
            raise ValueError("session_obj.session_id is required")

        empty_transactions = pd.DataFrame(
            [], columns=["business_name", "total", "date", "currency"]
        )
        self.save_session_inputs(
            session_id=session_obj.session_id,
            transaction_data=empty_transactions,
            proof_data=proof_data,
            replace_existing=False,
        )

        print(
            f"✅ {len(proof_data)} proofs committed to session ID {session_obj.session_id}"
        )
