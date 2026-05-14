import threading
import webbrowser
import os
from urllib.parse import urlparse

from backend_app import app

if __name__ == "__main__":
    host = os.getenv("ARVEE_HOST", "0.0.0.0")
    port = int(os.getenv("ARVEE_PORT", "7860"))
    debug = str(os.getenv("ARVEE_DEBUG", "false")).strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    auto_open_browser = str(
        os.getenv("ARVEE_OPEN_BROWSER", "true")
    ).strip().lower() in {"1", "true", "yes", "on"}
    frontend_url = os.getenv("ARVEE_FRONTEND_URL", f"http://127.0.0.1:{port}")
    parsed_frontend = urlparse(frontend_url)

    # Flask serves both API (backend) and web UI (frontend) on the same port.
    print(f"Backend: {port}")
    print(f"Frontend: {parsed_frontend.port}")

    # Launch browser shortly after server startup when enabled.
    if auto_open_browser:
        threading.Timer(1.0, lambda: webbrowser.open(frontend_url)).start()

    app.run(host=host, port=port, debug=debug, use_reloader=False)
