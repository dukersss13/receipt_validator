import os
import json
from datetime import datetime

import pandas as pd

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from src.data.db_schema import Base, Session, Transaction, Proof, SessionState


class DataBase:
    """
    SQLAlchemy-backed persistence layer for sessions, transactions, and proofs.

    Supports both SQLite (local) and any SQLAlchemy-compatible remote engine.
    All public methods operate within managed ``SessionLocal`` contexts so
    callers never need to handle raw database sessions.
    """

    def __init__(
        self,
        engine_name: str,
        local_db: bool = True,
        reset_db: bool = False,
        echo: bool = False,
    ):
        """
        Initialize the database connection and create tables if needed.

        Args:
            engine_name: For local SQLite, the path to the ``.db`` file (the
                ``.db`` suffix is added automatically when absent). For remote
                databases, a full SQLAlchemy connection URL.
            local_db: When ``True``, creates a local SQLite database and ensures
                the parent directory exists.
            reset_db: When ``True``, drops all existing tables before recreating
                them. Use with caution — all persisted data will be lost.
            echo: Pass ``True`` to enable SQLAlchemy query logging (useful for
                debugging).
        """
        self.local_db = local_db

        if local_db:
            # Ensure the .db extension is present
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
    def _normalize_date_series(series: pd.Series) -> pd.Series:
        """
        Normalise a Series of potentially noisy date strings to ``datetime.date`` values.

        Tries ``pd.to_datetime`` first, then falls back to extracting a date-shaped
        substring via regex for any values that could not be parsed directly.

        Args:
            series: Pandas Series of date values (strings, datetimes, etc.).

        Returns:
            Series of ``datetime.date`` objects.

        Raises:
            ValueError: If one or more values cannot be parsed after both attempts.
        """
        parsed = pd.to_datetime(series, errors="coerce")

        # Attempt regex-based extraction for values that failed the first parse
        unresolved = parsed.isna()
        if unresolved.any():
            extracted = series.astype(str).str.extract(
                r"(\d{4}[-/]\d{1,2}[-/]\d{1,2}|\d{1,2}[-/]\d{1,2}[-/]\d{2,4})",
                expand=False,
            )
            reparsed = pd.to_datetime(extracted, errors="coerce")
            parsed = parsed.where(~unresolved, reparsed)

        if parsed.isna().any():
            raise ValueError("Unable to parse one or more date values from inputs.")

        return parsed.dt.date

    @staticmethod
    def _normalize_input_df(frame: pd.DataFrame) -> pd.DataFrame:
        """
        Normalise incoming transaction or proof rows before persistence.

        Lowercases column names, enforces the required four columns, coerces
        numeric totals, normalises date strings, and uppercases currency codes.

        Args:
            frame: Raw input DataFrame that must contain ``business_name``,
                ``total``, ``date``, and ``currency`` columns (case-insensitive).

        Returns:
            A clean, four-column DataFrame ready for database insertion.

        Raises:
            ValueError: If any required column is missing after name normalisation.
        """
        if frame is None or frame.empty:
            return pd.DataFrame(
                [], columns=["business_name", "total", "date", "currency"]
            )

        normalized = frame.copy()
        # Lowercase column names to handle inconsistent casing from callers
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
        normalized["date"] = DataBase._normalize_date_series(normalized["date"])
        # Normalise currency codes to uppercase; replace blank values with USD
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
        """
        Fetch a session record by external session ID, creating it when absent.

        Args:
            session_id: External session identifier string. Must be non-empty.
            user_id: Optional user identifier to associate with a newly created session.

        Returns:
            The existing or newly created ``Session`` ORM object.

        Raises:
            ValueError: If *session_id* is empty.
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
        Persist transaction and proof DataFrames for a given session.

        When *replace_existing* is ``True`` (default) any existing rows linked to
        this session are deleted before the new rows are inserted, making the
        operation idempotent for re-validation runs.

        Args:
            session_id: External session identifier string.
            transaction_data: DataFrame of transaction rows to persist.
            proof_data: DataFrame of proof rows to persist.
            replace_existing: When ``True``, existing transaction and proof rows
                for this session are replaced. Set to ``False`` to append.

        Raises:
            ValueError: If *session_id* is empty or a required column is missing.
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
        Load both transaction and proof history for a given external session ID.

        Args:
            session_id: External session identifier string.

        Returns:
            A tuple ``(transactions_df, proofs_df)`` with date columns cast to
            ``datetime.date``.

        Raises:
            ValueError: If *session_id* is empty or the session does not exist.
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
        Append transaction rows to an existing session without replacing current data.

        Args:
            session_obj: The ``Session`` ORM object to link new transactions to.
            transaction_data: DataFrame of transaction rows to append.

        Raises:
            ValueError: If ``session_obj.session_id`` is empty.
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
        Append proof rows to an existing session without replacing current data.

        Args:
            session_obj: The ``Session`` ORM object to link new proof records to.
            proof_data: DataFrame of proof rows to append.

        Raises:
            ValueError: If ``session_obj.session_id`` is empty.
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

    def clear_all_data(self) -> None:
        """
        Remove all persisted sessions, transactions, proofs, and session states.

        This operation is irreversible. Intended for test teardown and
        administrative resets only.
        """
        with self.SessionLocal() as db:
            db.query(SessionState).delete()
            db.query(Proof).delete()
            db.query(Transaction).delete()
            db.query(Session).delete()
            db.commit()

    def save_session_state(self, session_id: str, state: dict) -> None:
        """
        Persist frontend/UI state for a session to support resume flows.

        Upserts a ``SessionState`` record: creates one if absent, otherwise
        overwrites the existing payload.

        Args:
            session_id: External session identifier string.
            state: Arbitrary JSON-serialisable dict of UI state.

        Raises:
            ValueError: If *session_id* is empty or *state* is not a dict.
        """
        normalized_session_id = str(session_id).strip()
        if not normalized_session_id:
            raise ValueError("session_id cannot be empty.")
        if not isinstance(state, dict):
            raise ValueError("state must be an object.")

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

            state_obj = (
                db.query(SessionState)
                .filter(SessionState.session_ref_id == session_obj.id)
                .first()
            )

            payload = json.dumps(state)
            if state_obj is None:
                state_obj = SessionState(
                    session_ref_id=session_obj.id,
                    payload=payload,
                    updated_at=datetime.utcnow(),
                )
                db.add(state_obj)
            else:
                state_obj.payload = payload
                state_obj.updated_at = datetime.utcnow()

            db.commit()

    def load_session_state(self, session_id: str) -> dict | None:
        """
        Load previously saved frontend/UI state for a session.

        Args:
            session_id: External session identifier string.

        Returns:
            The deserialised state dict, or ``None`` if no state has been saved
            for this session.

        Raises:
            ValueError: If *session_id* is empty or the session does not exist.
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

            state_obj = (
                db.query(SessionState)
                .filter(SessionState.session_ref_id == session_obj.id)
                .first()
            )

            if state_obj is None:
                return None

            try:
                return json.loads(state_obj.payload)
            except json.JSONDecodeError:
                return None
