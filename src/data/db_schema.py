import datetime
from sqlalchemy import Column, Integer, String, Float, Date, ForeignKey, create_engine
from sqlalchemy.orm import relationship, declarative_base, sessionmaker



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
    file_path = Column(String)  # URL or local path to the receipt or file
    uploaded_at = Column(Date)

    transaction = relationship("Transaction", back_populates="proofs")
    session = relationship("Session", back_populates="proofs")


# Set up the SQLite database
engine = create_engine("sqlite:///app.db", echo=True)  # `echo=True` prints SQL to terminal

# Create all tables
Base.metadata.create_all(engine)

# Optional: create a test session
SessionLocal = sessionmaker(bind=engine)
db = SessionLocal()

# Create and commit example data
session_obj = Session(user_id="user_123")
txn = Transaction(business_name="Starbucks", total=5.25, currency="USD", date=datetime.date.today(), session=session_obj)
proof = Proof(file_path="receipt.jpg", transaction=txn, session=session_obj)

db.add(session_obj)
db.commit()

print("🎉 Database created and sample data added.")
