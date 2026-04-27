import os
from types import SimpleNamespace

from intelligence.helper_agent import HelperAgent
from ui_test import _mock_validated_transactions


def _init_terminal_chat_agent() -> HelperAgent:
    helper = HelperAgent()

    # If no API key exists, use local fake graph so terminal mode still works.
    has_key = bool(os.getenv("GEMINI_API_KEY", "").strip()) or bool(
        os.getenv("GOOGLE_API_KEY", "").strip()
    )

    return helper


def _run_terminal_chat() -> None:
    validated_rows = _mock_validated_transactions(30)
    helper = _init_terminal_chat_agent()
    chat_history: list[dict[str, str]] = []

    print("Terminal chat mode started.")
    print("Dataset: 30 mock validated rows from ui_test.py")
    print("Type your question and press Enter.")
    print("Type 'exit' to quit.")

    while True:
        try:
            user_text = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nTerminal chat ended.")
            break

        if not user_text:
            continue

        if user_text.lower() in {"exit", "quit"}:
            print("Terminal chat ended.")
            break

        question = user_text
        result = helper.ask(
            question,
            validated_rows,
            chat_history=chat_history,
        )
        answer = str(result.get("answer", "") or "")
        print(f"Arvee: {answer}")
        print()

        chat_history.append({"role": "user", "text": question})
        chat_history.append({"role": "assistant", "text": answer})


if __name__ == "__main__":
    _run_terminal_chat()
