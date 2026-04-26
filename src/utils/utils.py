import os
import uuid

OPENAI_MODEL = "gpt-4o-mini"
GEMINI_FLASH_LITE_MODEL = "gemini-2.5-flash-lite"
# Backward-compatible alias used across the codebase.
GPT_MODEL = OPENAI_MODEL


def load_secret_file(name: str) -> str:
    """
    Loads a secret file
    """
    with open(name, "r", encoding="utf-8") as f:
        return f.read().strip()


def load_openai_key() -> str:
    """
    Loads the openai API key
    """
    return load_secret_file("secrets/openai_api_key")


def setup_openai():
    """
    Sets the env variable to the api access key
    """
    api_key = load_openai_key()
    os.environ["OPENAI_API_KEY"] = api_key


def setup_gemini():
    """
    Sets env vars for Gemini API access.
    """
    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    if not api_key:
        try:
            api_key = load_google_gemini_api_key()
        except OSError:
            api_key = ""

    if api_key:
        os.environ["GEMINI_API_KEY"] = api_key
        # OpenAI-compatible Gemini endpoints accept api_key explicitly,
        # but also honoring GOOGLE_API_KEY can help external tooling.
        os.environ.setdefault("GOOGLE_API_KEY", api_key)


def load_google_cse() -> str:
    """
    Loads the google search engine ID
    """
    return load_secret_file("secrets/google_cse")


def load_google_api_key():
    """
    Loads the google search engine ID
    """
    return load_secret_file("secrets/google_api_key")


def load_google_gemini_api_key() -> str:
    """
    Loads the Gemini API key.
    """
    return load_secret_file("secrets/google_gemini_api_key")


def setup_google_search():
    """
    Sets env variables for google search
    """
    cse_key = load_google_cse()
    api_key = load_google_api_key()
    os.environ["GOOGLE_CSE_ID"] = cse_key
    os.environ["GOOGLE_API_KEY"] = api_key


def load_tavily_api_key():
    # Load Tavily Search API key
    return load_secret_file("secrets/tavily_api_key")


def setup_tavily_search():
    api_key = load_tavily_api_key()
    os.environ["TAVILY_API_KEY"] = api_key


def load_exchange_rate_key():
    # Load Tavily Search API key
    return load_secret_file("secrets/exchange_rate_key")


def create_session_id() -> str:
    """
    Generates a random UUID for session identification.
    """
    return str(uuid.uuid4())
