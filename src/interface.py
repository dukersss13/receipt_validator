import gradio as gr
import pandas as pd

from src.data_reader import DataReader, DataType
from src.validator import Validator
from src.style.css import button_custom_css, type_writer_effect



class Interface:
    @staticmethod
    def create_empty_df(columns=["Business Name", "Total", "Date"]) -> pd.DataFrame:
        # Create empty placeholder df
        return pd.DataFrame([], columns=columns)

    def run_validation(
        self, state, transactions: pd.DataFrame, proofs: pd.DataFrame
    ) -> tuple:
        """
        Run the validation process

        :param state: Gradio State
        :param transactions: list of transactions
        :param proofs: list of proofs

        :return the results
        """
        print("Total transactions uploaded:", len(transactions if transactions else []))
        print("Total proofs uploaded:", len(proofs if proofs else []))
        self.data_reader = DataReader(transactions=transactions, proofs=proofs)

        if transactions:
            transactions_data = self.data_reader.load_data(DataType.TRANSACTIONS)
        else:
            transactions_data = state["transactions"]

        if proofs:
            proofs_data = self.data_reader.load_data(DataType.PROOFS)
        else:
            proofs_data = state["proofs"]

        validator = Validator(transactions_data, proofs_data)
        validation_results = validator.validate()
        results = validator.analyze_results(validation_results)
        discrepancies, unmatched_transactions, unmatched_proofs = validation_results

        # Clear current file uploads to prepare for next turn (if any)
        transactions = proofs = None

        state["discrepancies"] = pd.concat(
            [state["discrepancies"], discrepancies], ignore_index=True
        ).drop_duplicates()

        state["unmatched_transactions"] = pd.concat(
            [state["unmatched_transactions"], unmatched_transactions], ignore_index=True
        ).drop_duplicates()

        state["unmatched_proofs"] = pd.concat(
            [state["unmatched_proofs"], unmatched_proofs], ignore_index=True
        ).drop_duplicates()

        state["transactions"] = pd.concat(
            [state["transactions"], transactions_data], ignore_index=True
        ).drop_duplicates()

        state["proofs"] = pd.concat(
            [state["proofs"], proofs_data], ignore_index=True
        ).drop_duplicates()

        return (
            state["discrepancies"],
            state["unmatched_transactions"],
            state["unmatched_proofs"],
            results,
            transactions,
            proofs,
            state,
        )

    def run_interface(self):
        """
        Start the gradio UI interface
        """
        with gr.Blocks(css=button_custom_css) as ui:
            state = gr.State(
                {
                    "unmatched_transactions": self.create_empty_df(),
                    "unmatched_proofs": self.create_empty_df(),
                    "discrepancies": self.create_empty_df(columns=[""]),
                    "transactions": self.create_empty_df(),
                    "proofs": self.create_empty_df(),
                }
            )
            transactions_dir = gr.File(
                label="Upload Transactions", file_count="multiple", file_types=["image"]
            )
            proofs_dir = gr.File(
                label="Upload Proofs", file_count="multiple", file_types=["image"]
            )
            run_btn = gr.Button(value="Validate", variant="primary")

            results = gr.Textbox(value="", label="Results", render=True)
            # results = gr.HTML("<div id='result_field'></div>", label="Results")

            discrepancies = gr.DataFrame(
                state.value["discrepancies"],
                headers=["Transaction Business Name", "Total Transaction", "Transaction Date",
                         "Receipt Business Name", "Total on Receipt", "Receipt Date", "Delta"],
                label="Discrepancies"
            )

            unmatched_transactions = gr.DataFrame(
                state.value["unmatched_transactions"],
                headers=["Business Name", "Total", "Date"],
                label="Unmatched Transactions"
            )
            unmatched_proofs = gr.DataFrame(
                state.value["unmatched_proofs"],
                headers=["Business Name", "Total", "Date"],
                label="Unmatched Proofs"
            )

            run_btn.click(
                fn=self.run_validation,
                inputs=[state, transactions_dir, proofs_dir],
                outputs=[
                    discrepancies,
                    unmatched_transactions,
                    unmatched_proofs,
                    results,
                    transactions_dir,
                    proofs_dir,
                    state,
                ],
            )
            print()

        ui.launch()
