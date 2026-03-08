from datetime import datetime

from sqlalchemy import Column, Date, DateTime, Float, ForeignKey, Integer, String, Text
from sqlalchemy.orm import relationship, declarative_base


Base = declarative_base()


class Session(Base):
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
    __tablename__ = "session_states"

    id = Column(Integer, primary_key=True)
    session_ref_id = Column(
        Integer, ForeignKey("sessions.id"), nullable=False, unique=True, index=True
    )
    payload = Column(Text, nullable=False)
    updated_at = Column(DateTime, nullable=False, default=datetime.utcnow)

    session = relationship("Session", back_populates="state")
