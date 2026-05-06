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


def load_google_gemini_api_key() -> str:
    """
    Return the Gemini API key from the local secrets directory.

    Returns:
        The Gemini API key string.
    """
    return load_secret_file("secrets/google_gemini_api_key")


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
