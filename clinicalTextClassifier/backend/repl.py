"""Interactive terminal REPL for the clinical text clarifier agent.

Run from the backend directory:
    python repl.py

Enter clinical text at the prompt. Press Enter on an empty line to exit.
"""

import json

from app.agent.agent import ClinicalTextClarifierAgent
from app.agent.schema import Agent1Request


def main() -> None:
    agent = ClinicalTextClarifierAgent()
    session_id = None

    print("Clinical Text Clarifier Terminal Test")
    print("Enter clinical text. Press Enter on an empty line to exit.\n")

    while True:
        text = input("Input: ").strip()
        if not text:
            print("Exiting.")
            break

        print("Processing with Ollama...", flush=True)
        response = agent.process(
            Agent1Request(text=text, session_id=session_id)
        )
        session_id = response.session_id

        print("\nJSON output:")
        print(json.dumps(
            response.model_dump(mode="json"),
            indent=2,
            ensure_ascii=False,
        ))

        if response.requires_clarification and response.clarification_question:
            print(f"\nNext question: {response.clarification_question}")
        else:
            print("\nNo further clarification needed.")
        print()


if __name__ == "__main__":
    main()
