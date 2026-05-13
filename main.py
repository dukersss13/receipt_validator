import threading
import webbrowser
from urllib.parse import urlparse

from webui.app import app

if __name__ == "__main__":
    host = "0.0.0.0"
    port = 7860
    frontend_url = f"http://127.0.0.1:{port}"
    parsed_frontend = urlparse(frontend_url)

    # Flask serves both API (backend) and web UI (frontend) on the same port.
    print(f"Backend: {port}")
    print(f"Frontend: {parsed_frontend.port}")

    # Launch browser shortly after server startup.
    threading.Timer(1.0, lambda: webbrowser.open(frontend_url)).start()

    app.run(host=host, port=port, debug=True, use_reloader=False)
