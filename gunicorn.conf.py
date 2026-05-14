import os

bind = f"0.0.0.0:{os.getenv('ARVEE_PORT', '7860')}"
worker_class = "gthread"
workers = int(os.getenv("ARVEE_GUNICORN_WORKERS", "2"))
threads = int(os.getenv("ARVEE_GUNICORN_THREADS", "8"))
timeout = int(os.getenv("ARVEE_GUNICORN_TIMEOUT", "180"))
graceful_timeout = int(os.getenv("ARVEE_GUNICORN_GRACEFUL_TIMEOUT", "30"))
keepalive = int(os.getenv("ARVEE_GUNICORN_KEEPALIVE", "5"))
accesslog = "-"
errorlog = "-"
loglevel = os.getenv("ARVEE_GUNICORN_LOG_LEVEL", "info")
