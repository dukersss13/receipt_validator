
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

header_css = """
    .header-text {
        font-size: 42px;
        font-weight: bold;
        text-align: center;
        margin-bottom: 10px;
    }
"""

interface_theme = header_css
