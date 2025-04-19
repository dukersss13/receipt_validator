import gradio as gr
import pandas as pd
import tempfile

from concurrent.futures import ThreadPoolExecutor

from src.data_reader import DataReader, DataType
from src.validator import Validator
from src.style.css import interface_theme


class Interface:
    @staticmethod
    def create_empty_df(columns=["Business Name", "Total", "Date"]) -> pd.DataFrame:
        # Create empty placeholder df
        return pd.DataFrame([], columns=columns)

    def run_validation(
        self,
        state,
        transactions: pd.DataFrame,
        proofs: pd.DataFrame,
        progress=gr.Progress(),
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

        # Load Transactions
        progress(2 * step_increment, desc="Loading Transactions")
        from time import time

        s = time()

        def load_transactions():
            return (
                data_reader.load_data(DataType.TRANSACTIONS)
                if transactions
                else state["transactions"]
            )

        def load_proofs():
            return (
                data_reader.load_data(DataType.PROOFS)
                if proofs
                else state["proofs"]
            )

        progress(3 * step_increment, desc="Loading Proofs")

        with ThreadPoolExecutor(
            max_workers=2
        ) as executor:  # Use ProcessPoolExecutor if CPU-bound
            future_transactions = executor.submit(load_transactions)
            future_proofs = executor.submit(load_proofs)

            transactions_data = future_transactions.result()
            proofs_data = future_proofs.result()

        e = time()
        print(f"Total data reading time: {round(e - s, 2)}s")

        progress(4 * step_increment, desc="Initializing Validator")
        validator = Validator(transactions_data, proofs_data)

        progress(5 * step_increment, desc="Running Validation")
        validation_results = validator.validate()

        progress(
            6 * step_increment, desc="Analyzing Results & Providing Recommendations"
        )
        results, recommendations = validator.analyze_results(validation_results)

        (
            validated_transactions,
            discrepancies,
            unmatched_transactions,
            unmatched_proofs,
        ) = (
            validation_results.validated_transactions,
            validation_results.discrepancies,
            validation_results.unmatched_transactions,
            validation_results.unmatched_proofs,
        )

        # Clear current file uploads to prepare for next turn (if any)
        transactions = proofs = None
        progress(7 * step_increment, desc="Preparing Results")

        state["validated_transactions"] = validated_transactions
        state["discrepancies"] = discrepancies
        state["unmatched_transactions"] = unmatched_transactions
        state["unmatched_proofs"] = unmatched_proofs
        state["recommendations"] = recommendations

        progress(8 * step_increment, desc="Displaying Results")

        return (
            gr.Dataframe(
                state["validated_transactions"],
                visible=Interface.is_not_empty(validated_transactions),
            ),
            gr.Dataframe(
                state["discrepancies"], visible=Interface.is_not_empty(discrepancies)
            ),
            gr.Dataframe(
                state["unmatched_transactions"],
                visible=Interface.is_not_empty(unmatched_transactions),
            ),
            gr.Dataframe(
                state["unmatched_proofs"],
                visible=Interface.is_not_empty(unmatched_proofs),
            ),
            results,
            gr.Dataframe(state["recommendations"], visible=False),
        )

    @staticmethod
    def is_not_empty(df: pd.DataFrame) -> bool:
        """
        Check if the df is empty
        """
        return not df.empty

    @staticmethod
    def save_df_as_csv(df: pd.DataFrame):
        """
        Save a DataFrame as a CSV file in a temporary location.
        Parameters:
        df (pandas.DataFrame): The DataFrame to be saved as a CSV file.
        Returns:
        str: The file path of the saved CSV file.
        """
        with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as temp_file:
            df.to_csv(temp_file.name, index=False)
            return temp_file.name

    @staticmethod
    def download_records(validated_records: pd.DataFrame):
        """
        Save the given DataFrame as a CSV file and return the file path.
        Args:
            validated_records (pandas.DataFrame): The DataFrame to be saved as a CSV file.
        Returns:
            str: The file path of the saved CSV file.
        """
        csv_path = Interface.save_df_as_csv(validated_records)

        return csv_path

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
                    "validated_transactions": self.create_empty_df(),
                    "unmatched_transactions": self.create_empty_df(),
                    "unmatched_proofs": self.create_empty_df(),
                    "discrepancies": self.create_empty_df(columns=[""]),
                    "recommendations": self.create_empty_df(columns=[""]),
                    "transactions": self.create_empty_df(),
                    "proofs": self.create_empty_df(),
                }
            )
            with gr.Row():
                transactions_input = gr.File(
                    label="Upload Transactions",
                    file_count="multiple",
                )
                proofs_input = gr.File(
                    label="Upload Proofs", file_count="multiple", file_types=["image"]
                )

            with gr.Row():
                validate_btn = gr.Button(
                    value="Validate", variant="primary", elem_classes="custom_button"
                )
                clear_btn = gr.Button(
                    value="Clear", variant="secondary", elem_classes="custom_button"
                )

            results = gr.Textbox(value="", label="Results", render=True)

            def update_download_button_visibility(df: pd.DataFrame):
                if "1" in df.columns:
                    visible = False
                else:
                    visible = not df.empty
                return gr.update(visible=visible)

            validated_transactions = gr.Dataframe(
                value=state.value["validated_transactions"],
                label="Validated Transactions",
                visible=False,
            )

            download_btn = gr.DownloadButton(
                label="Download Records",
                variant="primary",
                elem_classes="custom_button",
                visible=False,
            )

            validated_transactions.change(
                fn=update_download_button_visibility,
                inputs=validated_transactions,
                outputs=download_btn,
            )

            download_btn.click(
                fn=self.download_records,
                inputs=validated_transactions,
                outputs=download_btn,
            )

            discrepancies = gr.Dataframe(
                value=state.value["discrepancies"],
                label="Discrepancies",
                visible=False,
            )

            with gr.Row():
                unmatched_transactions = gr.Dataframe(
                    value=state.value["unmatched_transactions"],
                    label="Unmatched Transactions",
                    visible=False,
                )
                unmatched_proofs = gr.Dataframe(
                    value=state.value["unmatched_proofs"],
                    label="Unmatched Proofs",
                    visible=False,
                )

            recommendations = gr.Dataframe(
                value=state.value["recommendations"],
                label="Recommendations",
                visible=False,
            )

            output = gr.Textbox(
                "You accepted recommendations for:", label="Notes", visible=False
            )

            validate_btn.click(
                fn=self.run_validation,
                inputs=[state, transactions_input, proofs_input],
                outputs=[
                    validated_transactions,
                    discrepancies,
                    unmatched_transactions,
                    unmatched_proofs,
                    results,
                    recommendations,
                ],
            )

            # Clear button functionality
            clear_btn.click(
                fn=lambda: [
                    gr.File(
                        value=None,
                        label="Upload Transactions",
                        file_count="multiple",
                        file_types=None,
                    ),
                    gr.File(
                        value=None,
                        label="Upload Proofs",
                        file_count="multiple",
                        file_types=["image"],
                    ),
                    gr.Dataframe(visible=False),
                    gr.DownloadButton(
                        label="Download Records",
                        variant="primary",
                        elem_classes="custom_button",
                        visible=False,
                    ),
                    gr.Dataframe(visible=False),
                    gr.Dataframe(visible=False),
                    gr.Dataframe(visible=False),
                    "",
                    gr.Textbox("", visible=False),
                    gr.Dataframe(visible=False),
                ],
                outputs=[
                    transactions_input,
                    proofs_input,
                    validated_transactions,
                    download_btn,
                    discrepancies,
                    unmatched_transactions,
                    unmatched_proofs,
                    results,
                    output,
                    recommendations,
                ],
            )

            @gr.render(
                inputs=[
                    recommendations,
                    validated_transactions,
                    unmatched_proofs,
                    unmatched_transactions,
                    output,
                ],
                triggers=[
                    recommendations.change,
                    validated_transactions.change,
                    unmatched_proofs.change,
                    unmatched_transactions.change,
                    output.change,
                ],
            )
            def display_recommendations(
                rec: pd.DataFrame,
                validated_transactions_df: pd.DataFrame,
                unmatched_proofs_df: pd.DataFrame,
                unmatched_transactions_df: pd.DataFrame,
                out: str,
            ):
                if len(rec.columns) < 7:
                    return

                checkboxes = []
                with gr.Column():
                    for _, row in rec.iterrows():
                        with gr.Row():
                            gr.Text(
                                row["Transaction Business Name"],
                                show_label=True,
                                label="Trans Business Name",
                            )
                            gr.Text(
                                row["Transaction Total"],
                                show_label=True,
                                label="Trans Total",
                            )
                            gr.Text(
                                row["Transaction Date"],
                                show_label=True,
                                label="Trans Date",
                            )
                            gr.Text(
                                row["Proof Business Name"],
                                show_label=True,
                                label="Proof Business Name",
                            )
                            gr.Text(
                                row["Proof Total"], show_label=True, label="Proof Total"
                            )
                            gr.Text(
                                row["Proof Date"], show_label=True, label="Proof Date"
                            )
                            gr.Text(row["Reason"], show_label=True, label="Reason")

                            checkbox = gr.Checkbox(label="Accept", value=False)
                            checkboxes.append(checkbox)

                    submit_btn = gr.Button(
                        "Accept Recommendations", visible=Interface.is_not_empty(rec)
                    )

                    def accept_recommendation(*checkbox_values):
                        selected_indices = [
                            i for i, value in enumerate(checkbox_values) if value
                        ]
                        output_txt = process_recommendations(selected_indices)
                        accepted_recommendations_df = rec.iloc[selected_indices]

                        rec.drop(selected_indices, inplace=True)
                        rec.reset_index(drop=True, inplace=True)

                        unmatched_trans, unmatched_pr = (
                            Validator.update_unmatched_dataframes(
                                accepted_recommendations_df,
                                unmatched_transactions_df,
                                unmatched_proofs_df,
                            )
                        )

                        accepted_recommendations_df["Result"] = ["Recommended"] * len(
                            accepted_recommendations_df
                        )
                        validated = pd.concat(
                            [validated_transactions_df, accepted_recommendations_df],
                            axis=0,
                        )

                        return (
                            gr.Textbox(
                                out + "\n" + output_txt, label="Notes", visible=True
                            ),
                            gr.Dataframe(
                                unmatched_trans,
                                visible=Interface.is_not_empty(unmatched_trans),
                            ),
                            gr.Dataframe(
                                unmatched_pr,
                                visible=Interface.is_not_empty(unmatched_pr),
                            ),
                            gr.Dataframe(
                                validated, visible=Interface.is_not_empty(validated)
                            ),
                            gr.Dataframe(rec, visible=Interface.is_not_empty(rec)),
                        )

                    def process_recommendations(selected_indices):
                        rows = []
                        for index in selected_indices:
                            dropped_row = rec.iloc[index]
                            rows.append(
                                f'{dropped_row["Transaction Business Name"]} - {dropped_row["Proof Business Name"]}'
                            )

                        return (
                            "\n".join(map(str, rows))
                            if selected_indices
                            else "No recommendations accepted."
                        )

                    submit_btn.click(
                        fn=accept_recommendation,
                        inputs=checkboxes,
                        outputs=[
                            output,
                            unmatched_transactions,
                            unmatched_proofs,
                            validated_transactions,
                            recommendations,
                        ],
                    )

        ui.launch()
