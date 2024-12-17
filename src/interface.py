import gradio as gr
import pandas as pd
from time import sleep

from src.data_reader import DataReader, DataType
from src.validator import Validator
from src.style.css import interface_theme



class Interface:
    @staticmethod
    def create_empty_df(columns=["Business Name", "Total", "Date"]) -> pd.DataFrame:
        # Create empty placeholder df
        return pd.DataFrame([], columns=columns)

    def run_validation(
        self, state, transactions: pd.DataFrame, proofs: pd.DataFrame,
        progress=gr.Progress()
    ) -> tuple:
        """
        Run the validation process.

        :param state: Gradio State
        :param transactions: list of transactions
        :param proofs: list of proofs

        :return: the results
        """
        total_steps = 8
        step_increment = 1 / total_steps

        progress(0, desc="Starting Validation")
        print("Total transactions uploaded:", len(transactions if transactions else []))
        print("Total proofs uploaded:", len(proofs if proofs else []))
        
        progress(step_increment, desc="Initializing Data Reader")
        self.data_reader = DataReader(transactions=transactions, proofs=proofs)

        sleep(0.5) # Simulate Progress

        # Load Transactions
        progress(2 * step_increment, desc="Loading Transactions")
        if transactions:
            transactions_data = self.data_reader.load_data(DataType.TRANSACTIONS)
        else:
            transactions_data = state["transactions"]

        # Load Proofs
        progress(3 * step_increment, desc="Loading Proofs")
        if proofs:
            proofs_data = self.data_reader.load_data(DataType.PROOFS)
        else:
            proofs_data = state["proofs"]

        progress(4 * step_increment, desc="Initializing Validator")
        validator = Validator(transactions_data, proofs_data)
        
        progress(5 * step_increment, desc="Running Validation")
        validation_results = validator.validate()
        
        progress(6 * step_increment, desc="Analyzing Results & Providing Recommendations")
        analysis, recommendations = validator.analyze_results(validation_results)
        
        discrepancies, unmatched_transactions, unmatched_proofs = validation_results

        # Clear current file uploads to prepare for next turn (if any)
        transactions = proofs = None
        progress(7 * step_increment, desc="Preparing Results")
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

        progress(8 * step_increment, desc="Displaying Results")

        return (
            state["discrepancies"],
            state["unmatched_transactions"],
            state["unmatched_proofs"],
            analysis,
            recommendations,
            transactions,
            proofs,
            state,
        )

    def run_interface(self):
        """
        Start the gradio UI interface
        """
        custom_theme = gr.themes.Base()

        with gr.Blocks(css=interface_theme, theme=custom_theme) as ui:
            # Header Text
            gr.Markdown("# Receipt Validator", elem_classes=["header-text"])

            state = gr.State(
                {
                    "unmatched_transactions": self.create_empty_df(),
                    "unmatched_proofs": self.create_empty_df(),
                    "discrepancies": self.create_empty_df(columns=[""]),
                    "recommendations": self.create_empty_df(columns=[""]),
                    "transactions": self.create_empty_df(),
                    "proofs": self.create_empty_df(),
                }
            )
            with gr.Row():
                transactions_dir = gr.File(
                    label="Upload Transactions", file_count="multiple", file_types=["image"]
                )
                proofs_dir = gr.File(
                    label="Upload Proofs", file_count="multiple", file_types=["image"]
                )

            with gr.Row():
                run_btn = gr.Button(value="Validate", variant="primary", elem_classes="custom_button")
                clear_btn = gr.Button(value="Clear", variant="secondary", elem_classes="custom_button")

            results = gr.Textbox(value="", label="Results", render=True)

            recommendations = gr.DataFrame(
                state.value["recommendations"],
                headers=["Transaction Business Name", "Transaction Total", "Transaction Date",
                         "Proof Business Name", "Proof Total", "Proof Date", "Reason"],
                label="Recommendations"
            )

            discrepancies = gr.DataFrame(
                state.value["discrepancies"],
                headers=["Transaction Business Name", "Transaction Total", "Transaction Date",
                         "Proof Business Name", "Proof Total", "Proof Date", "Delta"],
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
                    recommendations,
                    transactions_dir,
                    proofs_dir,
                    state,
                ],
            )
            # Clear button functionality
            clear_btn.click(
                fn=lambda: [None] * 8,
                outputs=[
                    discrepancies,
                    unmatched_transactions,
                    unmatched_proofs,
                    results,
                    recommendations,
                    transactions_dir,
                    proofs_dir,
                    state,
                ]
            )
            print()

        ui.launch()
