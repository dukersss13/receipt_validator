"""Headless backend server for the ArVee iOS app.

Starts the Flask API on 0.0.0.0:7860 without launching a browser window.
The iOS app connects to http://<your-ip>:7860 (configured in Settings).
"""

import os
import socket

from webui.app import app


def detect_lan_ip() -> str:
    """Best-effort LAN IP detection used for startup instructions."""
    override = os.getenv("ARVEE_LAN_IP", "192.168.4.54").strip()
    if override:
        return override

    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
            # No packets are sent; connect selects the outbound interface.
            sock.connect(("8.8.8.8", 80))
            detected = sock.getsockname()[0]
            if detected:
                return detected
    except OSError:
        pass

    return "192.168.4.54"


if __name__ == "__main__":
    host = "0.0.0.0"
    port = 7860
    lan_ip = detect_lan_ip()
    ios_api_base_url = f"http://{lan_ip}:{port}"

    print(f"ArVee backend listening on http://{host}:{port}")
    print("Connect the iOS app via Settings -> API Base URL")
    print(f"Set API Base URL to: {ios_api_base_url}")

    if lan_ip.startswith("172.17."):
        print(
            "Note: Running in a container. iPhone may need the host Mac LAN IP instead."
        )
        print("Set ARVEE_LAN_IP=<your-mac-ip> when starting backend.py to override.")

    app.run(host=host, port=port, debug=True, use_reloader=False)
