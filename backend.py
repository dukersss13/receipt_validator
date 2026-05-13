"""Headless backend server for the ArVee iOS app.

Starts the Flask API on 0.0.0.0:7860 without launching a browser window.
The iOS app connects to http://<your-ip>:7860 (configured in Settings).
"""

from webui.app import app

if __name__ == "__main__":
    host = "0.0.0.0"
    port = 7860

    print(f"ArVee backend listening on http://{host}:{port}")
    print("Connect the iOS app via Settings → API Base URL")

    app.run(host=host, port=port, debug=True, use_reloader=False)
