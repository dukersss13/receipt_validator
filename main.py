from src.data_reader import DataReader, DataType
from validator import Validator


dr = DataReader()
transactions = dr.read_data(DataType.TRANSACTIONS)
proofs = dr.read_data(DataType.PROOFS)

validator = Validator(transactions, proofs)
