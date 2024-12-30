import gradio as gr
import pandas as pd

def process_recommendations(selected_indices):
    return f"You accepted recommendations for rows: {', '.join(map(str, selected_indices))}" if selected_indices else "No recommendations accepted."

def create_interface():
    # Sample recommendations DataFrame
    recommendations_df = pd.DataFrame({
        "Transaction Business Name": ["Business A", "Business B", "Business C"],
        "Transaction Total": [100, 200, 300],
        "Transaction Date": ["2022-01-01", "2022-01-02", "2022-01-03"],
        "Proof Business Name": ["Proof A", "Proof B", "Proof C"],
        "Proof Total": [100, 200, 300],
        "Proof Date": ["2022-01-01", "2022-01-02", "2022-01-03"],
        "Reason": ["Reason A", "Reason B", "Reason C"]
    })

    # Create the Gradio interface
    with gr.Blocks() as demo:
        gr.Markdown("# Recommendations with Accept Checkboxes")

        # Create a list to hold checkbox states
        checkboxes = []

        # Create a single column for both DataFrame and checkboxes
        with gr.Column():
            for index, row in recommendations_df.iterrows():
                with gr.Row():
                    # Display each row of the DataFrame
                    gr.Text(row["Transaction Business Name"])
                    gr.Text(row["Transaction Total"])
                    gr.Text(row["Transaction Date"])
                    gr.Text(row["Proof Business Name"])
                    gr.Text(row["Proof Total"])
                    gr.Text(row["Proof Date"])
                    gr.Text(row["Reason"])
                    
                    # Add a checkbox for each row
                    checkbox = gr.Checkbox(label="Accept", value=False)
                    checkboxes.append(checkbox)

        submit_btn = gr.Button("Submit")
        output = gr.Textbox(label="Output")

        # Define the action on button click
        def on_submit(*checkbox_values):
            selected_indices = [i for i, value in enumerate(checkbox_values) if value]
            return process_recommendations(selected_indices)

        submit_btn.click(fn=on_submit, inputs=checkboxes, outputs=output)

    demo.launch()

create_interface()
