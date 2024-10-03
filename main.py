import gradio as gr
import pandas as pd
from src.data_reader import DataReader, DataType
from src.validator import Validator
from src.utils import load_empty_dataframe


def validate(state, transactions, proofs) -> tuple: 
    print('Total transactions uploaded:', len(transactions if transactions else []))
    print('Total proofs uploaded:', len(proofs if proofs else []))
    dr = DataReader(transactions=transactions, proofs=proofs)

    if transactions:
        transactions_data = dr.load_data(DataType.TRANSACTIONS)
    else:
        transactions_data = state['transactions']

    if proofs:
        proofs_data = dr.load_data(DataType.PROOFS)
    else:
        proofs_data = state['proofs']

    validator = Validator(transactions_data, proofs_data)
    discrepancies, unmatched_transactions, unmatched_proofs = validator.validate()
    
    # Clear current file uploads to prepare for next turn (if any)
    transactions = proofs = None

    note = 'Ran Validation\n'
    if len(unmatched_proofs) > 0:
        note += 'Upload more transactions\n'

    if len(unmatched_transactions) > 0:
        note += 'Upload more proofs\n'

    state['discrepancies'] = pd.concat([state['discrepancies'], discrepancies], ignore_index=True).drop_duplicates()
    state['unmatched_transactions'] = pd.concat([state['unmatched_transactions'], unmatched_transactions], ignore_index=True).drop_duplicates()
    state['unmatched_proofs'] = pd.concat([state['unmatched_proofs'], unmatched_proofs], ignore_index=True).drop_duplicates()
    state['transactions'] = pd.concat([state['transactions'], transactions_data], ignore_index=True).drop_duplicates()
    state['proofs'] = pd.concat([state['proofs'], proofs_data], ignore_index=True).drop_duplicates()
    
    return (
        state['discrepancies'],
        state['unmatched_transactions'],
        state['unmatched_proofs'],
        note,
        transactions,
        proofs,
        state
    )


with gr.Blocks() as ui:
    state = gr.State({
        'unmatched_transactions': load_empty_dataframe(),
        'unmatched_proofs': load_empty_dataframe(),
        'discrepancies': load_empty_dataframe(columns=['']),
        'transactions': load_empty_dataframe(),
        'proofs': load_empty_dataframe()
    })
    transactions_dir = gr.File(label='Upload Transactions', file_count='multiple', file_types=['image'])
    proofs_dir = gr.File(label='Upload Transactions', file_count='multiple', file_types=['image'])
    run_btn = gr.Button(value='Validate')

    note = gr.Text(value='', label='Note(s)')
    discrepancies = gr.DataFrame(state.value['discrepancies'], label='Discrepancies')
    unmatched_transactions = gr.DataFrame(state.value['unmatched_transactions'], label='Unmatched Transactions')
    unmatched_proofs = gr.DataFrame(state.value['unmatched_proofs'], label='Unmatched Proofs')
     
    run_btn.click(fn=validate,
                  inputs=[state, transactions_dir, proofs_dir],
                  outputs=[discrepancies, unmatched_transactions, unmatched_proofs, note, transactions_dir, proofs_dir, state])
    print()

if __name__ == '__main__':
    ui.launch()
