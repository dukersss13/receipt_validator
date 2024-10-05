import os
import pandas as pd


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
