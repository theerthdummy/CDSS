"""API Routes for clinical text processing."""

from fastapi import APIRouter
from app.agent.agent import ClinicalTextClarifierAgent
from app.agent.schema import Agent1Request

router = APIRouter()
_agent = ClinicalTextClarifierAgent()


def _build_user_message(response) -> str:
    """Build a human-friendly confirmation message."""
    return "Info noted."


@router.post("/clarify")
def clarify(request: Agent1Request):
    """Extract and structure clinical information from patient text."""
    response = _agent.process(request)
    result = response.model_dump()
    result["message"] = _build_user_message(response)
    result["session_id"] = response.session_id

    print("Final structured patient data:", result)

    return result