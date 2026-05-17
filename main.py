import functools
import http.server
import os
import socket
import socketserver
import threading
import webbrowser

from backend_app import app

DEFAULT_BACKEND_PORT = 7860
DEFAULT_FRONTEND_PORT = 8000
DEFAULT_WEBUI_DIR = os.path.join(os.path.dirname(__file__), "arvee_web_ui")


def _find_available_port(preferred: int) -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        if probe.connect_ex(("127.0.0.1", preferred)) != 0:
            return preferred

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as fallback:
        fallback.bind(("127.0.0.1", 0))
        return int(fallback.getsockname()[1])


class _ThreadingServer(socketserver.ThreadingTCPServer):
    allow_reuse_address = True


def _start_static_server(directory: str, default_port: int = DEFAULT_FRONTEND_PORT) -> tuple[int, socketserver.TCPServer]:
    port = _find_available_port(default_port)
    handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=directory)
    server = _ThreadingServer(("127.0.0.1", port), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    print(f"Static web UI server listening on http://127.0.0.1:{port}")
    return port, server


if __name__ == "__main__":
    host = os.getenv("ARVEE_HOST", "127.0.0.1")
    backend_port = int(os.getenv("ARVEE_PORT", str(DEFAULT_BACKEND_PORT)))
    frontend_port = int(os.getenv("ARVEE_FRONTEND_PORT", str(DEFAULT_FRONTEND_PORT)))
    debug = str(os.getenv("ARVEE_DEBUG", "false")).strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    auto_open_browser = str(
        os.getenv("ARVEE_OPEN_BROWSER", "true")
    ).strip().lower() in {"1", "true", "yes", "on"}

    backend_port = _find_available_port(backend_port)
    static_port, _static_server = _start_static_server(DEFAULT_WEBUI_DIR, frontend_port)

    os.environ["ARVEE_CORS_ORIGINS"] = (
        f"http://127.0.0.1:{static_port},http://localhost:{static_port}"
    )
    os.environ.setdefault("ARVEE_REQUIRE_USER_ID", "0")

    backend_base_url = f"http://127.0.0.1:{backend_port}"
    frontend_url = (
        f"http://127.0.0.1:{static_port}/index.html?apiBaseUrl={backend_base_url}"
    )

    print(f"Backend listening on http://{host}:{backend_port}")
    print(f"Frontend listening on {frontend_url}")

    if auto_open_browser:
        threading.Timer(1.0, lambda: webbrowser.open(frontend_url)).start()

    app.run(host=host, port=backend_port, debug=debug, use_reloader=False)
