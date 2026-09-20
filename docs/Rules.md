# CDSS — Development Rules

## 1. Boundary & Immutability Rules
* **Agent 1 is Immutable:** Do NOT modify the output schema of Agent 1. It is fixed. Transformations must occur in the orchestrator.
* **Knowledge Graph Exclusion:** We are strictly implementing the research paper's core features. There is no Knowledge Graph. All mappings to Agent 4 must pass empty arrays (`[]`) for `kg_evidence`.
* **Agent 4 & 5 Contracts:** The interface between Agent 4 (`FusionResponse.unified_context`) and Agent 5 (`UnifiedClinicalContext`) is 100% compatible. Do not modify these schemas.
* **No Database Overwrites:** `vectorDbIngestion` is a builder tool and should not be modified to act as a runtime service. `vectorRetrieval` is the sole runtime read-only access layer.

## 2. Orchestration Rules
* **Tooling:** Use **CrewAI** for the master orchestrator.
* **Feedback Loop:** The CrewAI sequential process (`Process.sequential`) will handle the linear task progression (Tasks 1 through 5). However, the adaptive feedback loop (Confidence < 0.70) must be handled by the FastAPI `/analyze` route dynamically wrapping the CrewAI kickoff, explicitly checking the output, and re-running the crew if Agent 5 yields a `FeedbackRequest`.
* **Adapters:** Schema adaptation must be strictly isolated to an `adapters.py` utility within the `orchestrator` folder.

## 3. Library & Tech Rules
* **Framework:** Use `FastAPI` for all agent service wrappers. Use `pydantic` for schema definition.
* **Async HTTP:** Use `httpx` or `requests` for agent-to-agent communication in the Orchestrator's CrewAI tools.
* **LLM Provider:** Prioritize the Gemini SDK (`google-genai`) for primary LLM calls. Groq is the designated fallback.

## 4. Error Handling & Fallbacks
* **Graceful Degradation:** The AI must never crash the pipeline due to an LLM timeout or 500 error. Every agent MUST implement a deterministic fallback (e.g., Agent 5's Level 3 Python rule-engine).
* **Validation:** All inputs and outputs across HTTP boundaries must be strictly validated using Pydantic. If an LLM returns malformed JSON, catch the validation error and trigger the fallback tier.

## 5. Security & Environment
* **No Hardcoded Secrets:** API keys (Gemini, Tavily, Groq, NCBI) must only exist in `.env` files. Read them using `os.getenv` or Pydantic `BaseSettings`.
* **Sanitization:** Log outputs must strip or mask sensitive tokens (`AIzaSy...`, `gsk_...`).
