import os
import uuid


def load_secret_file(name: str) -> str:
    """
    Load and strip a secret value from a local plaintext file.

    Args:
        name: Relative or absolute path to the secrets file.

    Returns:
        The file contents with leading/trailing whitespace removed.
    """
    with open(name, "r", encoding="utf-8") as f:
        return f.read().strip()


def setup_gemini() -> None:
    """
    Populate ``GEMINI_API_KEY`` and ``GOOGLE_API_KEY`` environment variables.

    Reads the key from the environment first; if absent, falls back to the
    local secrets file. When a key is found it is written to both env vars so
    that both the native Gemini SDK and any Google-API-compatible tooling can
    pick it up without additional configuration.
    """
    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    if not api_key:
        try:
            api_key = load_google_gemini_api_key()
        except OSError:
            # Secrets file is absent in CI/test environments; key stays empty
            api_key = ""

    if api_key:
        os.environ["GEMINI_API_KEY"] = api_key
        # OpenAI-compatible Gemini endpoints accept api_key explicitly,
        # but also honoring GOOGLE_API_KEY can help external tooling.
        os.environ.setdefault("GOOGLE_API_KEY", api_key)


def load_google_cse() -> str:
    """
    Return the Google Custom Search Engine ID from the local secrets directory.

    Returns:
        The CSE ID string.
    """
    return load_secret_file("secrets/google_cse")


def load_google_api_key() -> str:
    """
    Return the Google API key used by search integrations.

    Returns:
        The API key string.
    """
    return load_secret_file("secrets/google_api_key")


def load_google_gemini_api_key() -> str:
    """
    Return the Gemini API key from the local secrets directory.

    Returns:
        The Gemini API key string.
    """
    return load_secret_file("secrets/google_gemini_api_key")


def setup_google_search() -> None:
    """
    Populate ``GOOGLE_CSE_ID`` and ``GOOGLE_API_KEY`` environment variables.

    Reads both values from the local secrets directory and writes them into the
    process environment so that Google Search integrations can discover them.
    """
    cse_key = load_google_cse()
    api_key = load_google_api_key()
    os.environ["GOOGLE_CSE_ID"] = cse_key
    os.environ["GOOGLE_API_KEY"] = api_key


def load_tavily_api_key() -> str:
    """
    Return the Tavily API key from the local secrets directory.

    Returns:
        The Tavily API key string.
    """
    return load_secret_file("secrets/tavily_api_key")


def setup_tavily_search() -> None:
    """
    Populate ``TAVILY_API_KEY`` in the process environment.

    Reads the key from the local secrets directory and writes it into the
    process environment for the Tavily search integration to discover.
    """
    api_key = load_tavily_api_key()
    os.environ["TAVILY_API_KEY"] = api_key


def load_exchange_rate_key() -> str:
    """
    Return the Exchange Rate API key from the local secrets directory.

    Returns:
        The API key string.
    """
    return load_secret_file("secrets/exchange_rate_key")


def create_session_id() -> str:
    """
    Generate a random UUID string for session identification.

    Returns:
        A new UUID4 string in hyphenated format (e.g. ``"xxxxxxxx-xxxx-..."``).
    """
    return str(uuid.uuid4())
