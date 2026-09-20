# CDSS — Implementation Phases

## Phase 1: Agent 3 (Evidence Scanner) Integration
* **Goal:** Make the newly added `evidenceScanner` executable as a microservice.
* **Tasks:**
  1. Add a `FastAPI` wrapper (`app.py`) to `evidenceScanner/`.
  2. Implement `POST /api/v1/scan` that accepts an `EvidenceRequest` and returns a `ScannerResponse`.
  3. Ensure environment variables (`.env`) for Tavily, Gemini, and NCBI are configured.
  4. Test Agent 3 in isolation.

## Phase 2: Orchestrator Adapters
* **Goal:** Create the translation layer connecting the disparate agent schemas.
* **Tasks:**
  1. Create `orchestrator/adapters.py`.
  2. Write `agent1_to_search_query()` for Agent 2.
  3. Write `agent1_to_evidence_request()` for Agent 3.
  4. Build mapping functions to handle `kg_evidence = []` rules for Agent 4's input.
  5. Build query refiners for Agent 5's `FeedbackRequest` loop.

## Phase 3: Orchestrator CrewAI Rewrite
* **Goal:** Rebuild the Orchestrator to support all 5 agents and the adaptive feedback loop.
* **Tasks:**
  1. Update `orchestrator/config.py` with URLs for Agents 1-5.
  2. Write `orchestrator/tools.py` with HTTP client `BaseTool` wrappers for Agents 3, 4, and 5.
  3. Rewrite `orchestrator/crew.py` to sequence Tasks 1 through 5.
  4. Rewrite `orchestrator/app.py` `POST /analyze` to wrap the CrewAI kickoff in an iterative `while` loop (max 3 retries) checking Agent 5's confidence score/feedback.

## Phase 4: Full System Deployment & Testing
* **Goal:** Run the system end-to-end and prove the feedback loop.
* **Tasks:**
  1. Generate a root `docker-compose.yml` to orchestrate Qdrant and the 6 API services.
  2. Execute a clear-cut case (e.g., STEMI) to verify a 1-pass success.
  3. Execute an ambiguous case to verify the Agent 5 feedback loop successfully triggers re-queries.
  4. Update `Memory.md` and `Walkthrough.md` with final documentation.
