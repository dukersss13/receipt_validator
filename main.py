from src.data_reader import DataReader, DataType
from src.validator import Validator


dr = DataReader()
transactions = dr.load_data(DataType.TRANSACTIONS)
proofs = dr.load_data(DataType.PROOFS)

validator = Validator(transactions, proofs)
discrepancies, unmatched = validator.validate()
