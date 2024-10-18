
# Custom typewriting effect script
def type_writer_effect(note_text, speed=200):
    js_code = f'''
        <script>
            var i = 0;
            var txt = "{note_text}"; /* The text to type */
            var speed = {speed}; /* Speed in milliseconds */

            function typeWriter() {{
              if (i < txt.length) {{
                document.getElementById("note_field").innerHTML += txt.charAt(i);
                i++;
                setTimeout(typeWriter, speed);
              }}
            }}

            document.getElementById("note_field").innerHTML = "";
            typeWriter();
        </script>
    '''
    return js_code


button_custom_css = """
.gradio-button.primary {
    background: linear-gradient(to bottom right, #FFE4B5, #FFDAB9)
    color: orange
    border: 1px solid orange
}
.gradio-button.primary:hover {
    background: orange
    color: white
}
"""


