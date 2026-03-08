import threading
import webbrowser

from webui.app import app


if __name__ == "__main__":
    url = "http://127.0.0.1:7860"

    # Launch browser shortly after server startup.
    threading.Timer(1.0, lambda: webbrowser.open(url)).start()

    app.run(host="0.0.0.0", port=7860, debug=True, use_reloader=False)
