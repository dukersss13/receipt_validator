from datetime import datetime

from sqlalchemy import Column, Date, DateTime, Float, ForeignKey, Integer, String, Text
from sqlalchemy.orm import relationship, declarative_base


Base = declarative_base()


class Session(Base):
    """
    ORM model representing a user validation session.

    A session groups a set of transaction and proof records uploaded together
    for a single validation run. It also optionally stores UI state to support
    session resume.
    """

    __tablename__ = "sessions"

    id = Column(Integer, primary_key=True)
    session_id = Column(String(64), unique=True, nullable=False, index=True)
    user_id = Column(String, nullable=True)  # Optional for multi-user
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)

    transactions = relationship(
        "Transaction", back_populates="session", cascade="all, delete-orphan"
    )
    proofs = relationship(
        "Proof", back_populates="session", cascade="all, delete-orphan"
    )
    state = relationship(
        "SessionState",
        back_populates="session",
        cascade="all, delete-orphan",
        uselist=False,
    )


class Transaction(Base):
    """
    ORM model representing a single transaction row from a bank or card statement.
    """

    __tablename__ = "transactions"

    id = Column(Integer, primary_key=True)
    session_ref_id = Column(
        Integer, ForeignKey("sessions.id"), nullable=False, index=True
    )
    business_name = Column(String, nullable=False)
    total = Column(Float, nullable=False)
    currency = Column(String, nullable=False)
    date = Column(Date, nullable=False)

    session = relationship("Session", back_populates="transactions")
    proofs = relationship(
        "Proof", back_populates="transaction", cascade="all, delete-orphan"
    )


class Proof(Base):
    """
    ORM model representing a single receipt or proof-of-purchase row.

    A proof may optionally be linked to a matched ``Transaction`` after validation.
    """

    __tablename__ = "proofs"

    id = Column(Integer, primary_key=True)
    transaction_id = Column(Integer, ForeignKey("transactions.id"), nullable=True)
    session_ref_id = Column(
        Integer, ForeignKey("sessions.id"), nullable=False, index=True
    )
    business_name = Column(String, nullable=False)
    total = Column(Float, nullable=False)
    currency = Column(String, nullable=False)
    date = Column(Date, nullable=False)

    transaction = relationship("Transaction", back_populates="proofs")
    session = relationship("Session", back_populates="proofs")


class SessionState(Base):
    """
    ORM model storing serialised frontend/UI state for a session.

    The ``payload`` column holds a JSON-encoded dict that the web app uses to
    restore the UI when a user resumes a previous session.
    """

    __tablename__ = "session_states"

    id = Column(Integer, primary_key=True)
    session_ref_id = Column(
        Integer, ForeignKey("sessions.id"), nullable=False, unique=True, index=True
    )
    payload = Column(Text, nullable=False)
    updated_at = Column(DateTime, nullable=False, default=datetime.utcnow)

    session = relationship("Session", back_populates="state")
