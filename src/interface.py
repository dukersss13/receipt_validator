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
    ) -> tuple[gr.Dataframe]:
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
        data_reader = DataReader(transactions=transactions, proofs=proofs)

        sleep(0.5) # Simulate Progress

        # Load Transactions
        progress(2 * step_increment, desc="Loading Transactions")
        if transactions:
            transactions_data = data_reader.load_data(DataType.TRANSACTIONS)
        else:
            transactions_data = state["transactions"]

        # Load Proofs
        progress(3 * step_increment, desc="Loading Proofs")
        if proofs:
            proofs_data = data_reader.load_data(DataType.PROOFS)
        else:
            proofs_data = state["proofs"]

        progress(4 * step_increment, desc="Initializing Validator")
        validator = Validator(transactions_data, proofs_data)
        
        progress(5 * step_increment, desc="Running Validation")
        validation_results = validator.validate()
        
        progress(6 * step_increment, desc="Analyzing Results & Providing Recommendations")
        results, recommendations = validator.analyze_results(validation_results)
        
        discrepancies, unmatched_transactions, unmatched_proofs = validation_results

        # Clear current file uploads to prepare for next turn (if any)
        transactions = proofs = None
        progress(7 * step_increment, desc="Preparing Results")
        state["discrepancies"] = discrepancies

        state["unmatched_transactions"] = unmatched_transactions
        state["unmatched_proofs"] = unmatched_proofs

        state["recommendations"] = recommendations

        progress(8 * step_increment, desc="Displaying Results")

        return (
            gr.Dataframe(state["discrepancies"], visible=Interface.is_not_empty(discrepancies)),
            gr.Dataframe(state["unmatched_transactions"], visible=Interface.is_not_empty(unmatched_transactions)),
            gr.Dataframe(state["unmatched_proofs"], visible=Interface.is_not_empty(unmatched_proofs)),
            results,
            gr.Dataframe(state["recommendations"], visible=False)
        ) 

    @staticmethod
    def is_not_empty(df: pd.DataFrame) -> bool:
        """
        Check if the df is empty
        """
        return not df.empty

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
                    "proofs": self.create_empty_df()
                }
            )
            with gr.Row():
                transactions_input = gr.File(
                    label="Upload Transactions", file_count="multiple", file_types=["image"]
                )
                proofs_input = gr.File(
                    label="Upload Proofs", file_count="multiple", file_types=["image"]
                )

            with gr.Row():
                validate_btn = gr.Button(value="Validate", variant="primary", elem_classes="custom_button")
                clear_btn = gr.Button(value="Clear", variant="secondary", elem_classes="custom_button")

            results = gr.Textbox(value="", label="Results", render=True)
            
            discrepancies = gr.Dataframe(value=state.value["discrepancies"],
                                         label="Discrepancies", render=True,
                                         visible=False)

            with gr.Row():
                unmatched_transactions = gr.Dataframe(value=state.value["unmatched_transactions"],
                                                      label="Unmatched Transactions", render=True,
                                                      visible=False)

                unmatched_proofs = gr.Dataframe(value=state.value["unmatched_proofs"],
                                                label="Unmatched Proofs", render=True,
                                                visible=False)

            recommendations = gr.Dataframe(value=state.value["recommendations"],
                                           label="Recommendations",
                                           visible=False)
            output = gr.Textbox("You accepted recommendations for:", label="Notes", visible=False)

            validate_btn.click(
                fn=self.run_validation,
                inputs=[state, transactions_input, proofs_input],
                outputs=[
                    discrepancies,
                    unmatched_transactions,
                    unmatched_proofs,
                    results,
                    recommendations
                ],
            )

            # Clear button functionality
            clear_btn.click(
                fn=lambda: [None] * 5,
                outputs=[
                    discrepancies,
                    unmatched_transactions,
                    unmatched_proofs,
                    results,
                    recommendations
                ]
            )

            @gr.render(inputs=[recommendations, output], triggers=[recommendations.change, output.change])
            def display_recommendations(rec: pd.DataFrame, out):
                if len(rec.columns) < 7:
                    return 

                checkboxes = []
                with gr.Column():
                    gr.Markdown("### Recommendations")
                    if rec.empty:
                        gr.Textbox("No recommendations", show_label=False)
                        return

                    for _, row in rec.iterrows():
                        with gr.Row():
                            gr.Text(row["Transaction Business Name"], show_label=True, label="Trans Business Name")
                            gr.Text(row["Transaction Total"], show_label=True, label="Trans Total")
                            gr.Text(row["Transaction Date"], show_label=True, label="Trans Date")
                            gr.Text(row["Proof Business Name"], show_label=True, label="Proof Business Name")
                            gr.Text(row["Proof Total"], show_label=True, label="Proof Total")
                            gr.Text(row["Proof Date"], show_label=True, label="Proof Date")
                            gr.Text(row["Reason"], show_label=True, label="Reason")

                            checkbox = gr.Checkbox(label="Accept", value=False)
                            checkboxes.append(checkbox)

                    submit_btn = gr.Button("Accept Recommendations")

                    def accept_recommendation(*checkbox_values):
                        selected_indices = [i for i, value in enumerate(checkbox_values) if value]

                        output_txt = process_recommendations(selected_indices)
                        rec.drop(selected_indices, inplace=True)
                        rec.reset_index(drop=True, inplace=True)
                        return (
                            gr.Textbox(out + '\n' + output_txt, label="Notes", visible=True),
                            rec
                        )

                    def process_recommendations(selected_indices):
                        rows = []
                        for index in selected_indices:
                            dropped_row = rec.iloc[index]
                            rows.append(f'{dropped_row["Transaction Business Name"]} - {dropped_row["Proof Business Name"]}')
                        return '\n'.join(map(str, rows)) if selected_indices else "No recommendations accepted."

                    submit_btn.click(fn=accept_recommendation, inputs=checkboxes, outputs=[output, recommendations])

        ui.launch()

