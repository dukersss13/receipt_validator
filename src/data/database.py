import os
import json
from datetime import datetime

import pandas as pd

from sqlalchemy import create_engine
from sqlalchemy import inspect
from sqlalchemy.orm import sessionmaker

from src.data.db_schema import (
    Base,
    Proof,
    Session,
    SessionState,
    Transaction,
    UserAuth,
)


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
            self.engine = create_engine(
                f"sqlite:///{self.db_path}",
                echo=echo,
                connect_args={"timeout": 30, "check_same_thread": False},
            )

            if reset_db:
                Base.metadata.drop_all(bind=self.engine)

            Base.metadata.create_all(bind=self.engine)

            # Improve read/write concurrency for SQLite in local/dev mode.
            with self.engine.begin() as conn:
                conn.exec_driver_sql("PRAGMA journal_mode=WAL;")
                conn.exec_driver_sql("PRAGMA synchronous=NORMAL;")
                conn.exec_driver_sql("PRAGMA foreign_keys=ON;")

            if db_exists:
                print(f"📂 Found existing database '{self.db_path}'.")
            else:
                print(f"🆕 Creating new database '{self.db_path}'.")
        else:
            self.engine = create_engine(
                engine_name,
                echo=echo,
                pool_size=20,
                max_overflow=40,
                pool_pre_ping=True,
                pool_recycle=3600,
            )
            if reset_db:
                Base.metadata.drop_all(bind=self.engine)
            Base.metadata.create_all(bind=self.engine)

        self.SessionLocal = sessionmaker(bind=self.engine, expire_on_commit=False)
        self._ensure_auth_schema_compatibility()

    def _ensure_auth_schema_compatibility(self) -> None:
        """Ensure new auth columns/indexes exist for backward-compatible upgrades."""
        try:
            dialect = self.engine.dialect.name
            with self.engine.begin() as conn:
                if dialect == "sqlite":
                    rows = conn.exec_driver_sql(
                        "PRAGMA table_info(user_auth);"
                    ).fetchall()
                    existing_cols = {str(row[1]) for row in rows}
                else:
                    inspector = inspect(self.engine)
                    existing_cols = {
                        str(col.get("name", ""))
                        for col in inspector.get_columns("user_auth")
                    }

                if "provider" not in existing_cols:
                    conn.exec_driver_sql(
                        "ALTER TABLE user_auth ADD COLUMN provider VARCHAR(32) DEFAULT 'email'"
                    )
                if "provider_id" not in existing_cols:
                    conn.exec_driver_sql(
                        "ALTER TABLE user_auth ADD COLUMN provider_id VARCHAR(255)"
                    )

                conn.exec_driver_sql(
                    "CREATE UNIQUE INDEX IF NOT EXISTS ux_user_auth_provider_provider_id "
                    "ON user_auth(provider, provider_id)"
                )
        except Exception:
            # Keep startup resilient for fresh databases and environments where
            # the schema is already current.
            pass

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
            elif user_id and session_obj.user_id and session_obj.user_id != user_id:
                raise ValueError(
                    f"Session '{normalized_session_id}' does not belong to user '{user_id}'."
                )
            elif user_id and not session_obj.user_id:
                session_obj.user_id = user_id
                db.commit()
                db.refresh(session_obj)

            return session_obj

    @staticmethod
    def _normalize_email(email: str) -> str:
        normalized = str(email).strip().lower()
        if not normalized or "@" not in normalized:
            raise ValueError("A valid email address is required.")
        return normalized

    def create_user_auth(self, email: str, password_hash: str) -> UserAuth:
        """Create an auth account and return the stored user row."""
        normalized_email = self._normalize_email(email)
        if not str(password_hash).strip():
            raise ValueError("password_hash cannot be empty.")

        with self.SessionLocal() as db:
            existing = (
                db.query(UserAuth).filter(UserAuth.email == normalized_email).first()
            )
            if existing is not None:
                raise ValueError("An account with that email already exists.")

            user = UserAuth(
                email=normalized_email,
                password_hash=password_hash,
                provider="email",
                provider_id=None,
            )
            db.add(user)
            db.commit()
            db.refresh(user)
            return user

    def set_user_auth_password(self, email: str, password_hash: str) -> UserAuth:
        """Set/replace a user's password hash and return the stored row."""
        normalized_email = self._normalize_email(email)
        if not str(password_hash).strip():
            raise ValueError("password_hash cannot be empty.")

        with self.SessionLocal() as db:
            user = db.query(UserAuth).filter(UserAuth.email == normalized_email).first()
            if user is None:
                raise ValueError("No account found for that email.")

            user.password_hash = password_hash
            if not str(user.provider or "").strip():
                user.provider = "email"
            db.commit()
            db.refresh(user)
            return user

    def create_or_link_google_user(
        self,
        email: str,
        provider_id: str,
        password_hash_fallback: str,
    ) -> UserAuth:
        """Find or create a Google-authenticated user and return the stored row."""
        normalized_email = self._normalize_email(email)
        normalized_provider_id = str(provider_id).strip()
        if not normalized_provider_id:
            raise ValueError("provider_id cannot be empty.")
        if not str(password_hash_fallback).strip():
            raise ValueError("password_hash_fallback cannot be empty.")

        with self.SessionLocal() as db:
            provider_user = (
                db.query(UserAuth)
                .filter(
                    UserAuth.provider == "google",
                    UserAuth.provider_id == normalized_provider_id,
                )
                .first()
            )
            if provider_user is not None:
                if provider_user.email != normalized_email:
                    provider_user.email = normalized_email
                    db.commit()
                    db.refresh(provider_user)
                return provider_user

            existing_by_email = (
                db.query(UserAuth).filter(UserAuth.email == normalized_email).first()
            )
            if existing_by_email is not None:
                if not str(existing_by_email.password_hash or "").strip():
                    existing_by_email.password_hash = password_hash_fallback
                # Keep provider metadata consistent once an account is linked to Google.
                existing_by_email.provider = "google"
                existing_by_email.provider_id = normalized_provider_id
                db.commit()
                db.refresh(existing_by_email)
                return existing_by_email

            user = UserAuth(
                email=normalized_email,
                password_hash=password_hash_fallback,
                provider="google",
                provider_id=normalized_provider_id,
            )
            db.add(user)
            db.commit()
            db.refresh(user)
            return user

    def get_user_auth(self, email: str) -> UserAuth | None:
        """Fetch an auth account by normalized email."""
        normalized_email = self._normalize_email(email)
        with self.SessionLocal() as db:
            return db.query(UserAuth).filter(UserAuth.email == normalized_email).first()

    def save_session_inputs(
        self,
        session_id: str,
        transaction_data: pd.DataFrame,
        proof_data: pd.DataFrame,
        replace_existing: bool = True,
        user_id: str | None = None,
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
                session_obj = Session(session_id=normalized_session_id, user_id=user_id)
                db.add(session_obj)
                db.flush()
            elif user_id and session_obj.user_id and session_obj.user_id != user_id:
                raise ValueError(
                    f"Session '{normalized_session_id}' does not belong to user '{user_id}'."
                )
            elif user_id and not session_obj.user_id:
                session_obj.user_id = user_id
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
        self, session_id: str, user_id: str | None = None
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
            query = db.query(Session).filter(
                Session.session_id == normalized_session_id
            )
            if user_id is not None:
                query = query.filter(Session.user_id == user_id)

            session_obj = query.first()

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
            db.query(UserAuth).delete()
            db.commit()

    def save_session_state(
        self, session_id: str, state: dict, user_id: str | None = None
    ) -> None:
        """
        Persist frontend/UI state for a session to support resume flows.

        Upserts a ``SessionState`` record: creates one if absent, otherwise
        overwrites the existing payload.

        Args:
            session_id: External session identifier string.
            state: Arbitrary JSON-serialisable dict of UI state.
            user_id: Optional owning user for session scoping.

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
                session_obj = Session(session_id=normalized_session_id, user_id=user_id)
                db.add(session_obj)
                db.flush()
            elif user_id and session_obj.user_id and session_obj.user_id != user_id:
                raise ValueError(
                    f"Session '{normalized_session_id}' does not belong to user '{user_id}'."
                )
            elif user_id and not session_obj.user_id:
                session_obj.user_id = user_id
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

    def load_session_state(
        self, session_id: str, user_id: str | None = None
    ) -> dict | None:
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
            query = db.query(Session).filter(
                Session.session_id == normalized_session_id
            )
            if user_id is not None:
                query = query.filter(Session.user_id == user_id)

            session_obj = query.first()

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
