import gradio as gr

def create_session_id():
    import uuid
    return str(uuid.uuid4())

def generate_new_session():
    new_id = create_session_id()
    return new_id, new_id

with gr.Blocks() as demo:
    session_id_state = gr.State()

    with gr.Row():
        live_session_box = gr.Textbox(value="", label="Live Session ID", interactive=False)
        create_button = gr.Button("Create New Session")

    with gr.Row():
        textbox = gr.Textbox(value="", label="Enter Past Session ID")
        submit_btn = gr.Button("Submit")

    create_button.click(
        fn=generate_new_session,
        inputs=[],
        outputs=[live_session_box, session_id_state]
    )

    # Simulate loading history
    def load_history(session_id):
        return f"Loaded history for session: {session_id}"

    history_box = gr.Textbox(label="History Output", interactive=False)

    submit_btn.click(
        fn=load_history,
        inputs=textbox,
        outputs=history_box
    )

demo.launch()
