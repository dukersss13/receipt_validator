"""
Used as a testing template
"""

import gradio as gr
import pandas as pd
import tempfile

def save_df_as_csv(df):
    with tempfile.NamedTemporaryFile(delete=False, suffix='.csv') as temp_file:
        df.to_csv(temp_file.name, index=False)
        return temp_file.name

def download_csv(df):
    csv_path = save_df_as_csv(df)
    return csv_path

# Assuming you have your DataFrame 'df' already defined
df = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})

with gr.Blocks() as demo:
    gr.DataFrame(df)
    download_button = gr.DownloadButton("Download CSV")
    download_button.click(fn=download_csv, inputs=[gr.State(df)], outputs=download_button)

demo.launch()
