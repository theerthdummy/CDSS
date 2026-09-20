"""Interactive demo for clinical text extraction.

Usage:
    cd backend
    python demo.py

This script lets you enter a clinical prompt, then answer the agent's clarification
questions one by one until it decides the case is complete.
"""

from pprint import pprint

from app.agent.agent import ClinicalTextClarifierAgent
from app.agent.schema import Agent1Request


def run_case(text: str) -> None:
    """Run a single case and show the result, then loop on clarification answers."""
    agent = ClinicalTextClarifierAgent()
    current_text = text.strip()

    while True:
        print("\nCURRENT CASE:")
        print(current_text)
        print("\nOUTPUT:")
        response = agent.process(Agent1Request(text=current_text))
        pprint(response.model_dump())

        if not response.requires_clarification or not response.clarification_question:
            print("\nFINAL STATUS: no further clarification needed.")
            break

        print("\nCLARIFICATION QUESTION:")
        print(response.clarification_question)

        answer = input("\nYour answer: ").strip()
        if not answer:
            print("No answer entered. Ending demo.")
            break

        current_text = (
            current_text + " " + response.clarification_question + " " + answer
        )


def main() -> None:
    print("Clinical Text Clarifier Demo")
    print("Type a clinical prompt and answer follow-up clarification questions.")
    print("Press Enter on an empty prompt to exit.\n")

    while True:
        prompt = input("Enter clinical prompt: ").strip()
        if not prompt:
            print("Exiting demo.")
            break

        run_case(prompt)


if __name__ == "__main__":
    main()
