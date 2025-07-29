from sqlalchemy import Column, Integer, String, Float, Date, ForeignKey
from sqlalchemy.orm import relationship, declarative_base


Base = declarative_base()


class Session(Base):
    __tablename__ = 'sessions'

    id = Column(Integer, primary_key=True)
    user_id = Column(String, nullable=True)  # Optional for multi-user
    created_at = Column(Date)

    transactions = relationship("Transaction", back_populates="session", cascade="all, delete-orphan")
    proofs = relationship("Proof", back_populates="session", cascade="all, delete-orphan")


class Transaction(Base):
    __tablename__ = 'transactions'

    id = Column(Integer, primary_key=True)
    session_id = Column(Integer, ForeignKey('sessions.id'))
    business_name = Column(String)
    total = Column(Float)
    currency = Column(String)
    date = Column(Date)

    session = relationship("Session", back_populates="transactions")
    proofs = relationship("Proof", back_populates="transaction", cascade="all, delete-orphan")


class Proof(Base):
    __tablename__ = 'proofs'

    id = Column(Integer, primary_key=True)
    transaction_id = Column(Integer, ForeignKey('transactions.id'), nullable=True)
    session_id = Column(Integer, ForeignKey('sessions.id'))  # NEW: direct link to session
    business_name = Column(String)
    total = Column(Float)
    currency = Column(String)
    date = Column(Date)

    transaction = relationship("Transaction", back_populates="proofs")
    session = relationship("Session", back_populates="proofs")
