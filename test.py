import gradio as gr
import pandas as pd

def validate_and_recommend(input_text):
    # Example: create dataframe if text meets criteria
    if "recommend" in input_text.lower():
        df = pd.DataFrame({
            "Item": ["A", "B"],
            "Score": [0.9, 0.8]
        })
        return gr.update(value=df, visible=True)
    else:
        # Hide DataFrame if empty
        return gr.update(value=None, visible=False)

with gr.Blocks() as demo:
    text_input = gr.Textbox(label="Enter something")
    recommendations_df = gr.Dataframe(
        headers=["Item", "Score"], visible=False, interactive=False
    )

    text_input.change(
        fn=validate_and_recommend,
        inputs=text_input,
        outputs=recommendations_df
    )

demo.launch()
