# CDSS — Potential Risk Analysis

This document outlines the technical, architectural, and operational risks identified across all 6 project folders, categorized by severity, along with mitigation strategies.

---

## 1. `clinicalTextClassifier` (Agent 1)
**Role:** Clinical Text Parsing & SNOMED CT Mapping

* 🔴 **High Risk: Memory Leak in Session Handling**
  * **Issue:** Conversation state is maintained in an unbounded, in-memory class dictionary (`_session_store` in `agent.py`). There is no TTL (Time-To-Live) or cleanup mechanism.
  * **Impact:** In a long-running deployment, this will eventually exhaust server memory (OOM crash) as more sessions are created.
  * **Mitigation:** Implement a periodic cleanup task for sessions older than 24 hours, or move state to Redis.
* 🟡 **Medium Risk: Frontend Dependency Hell**
  * **Issue:** `frontend/package.json` specifies impossible future versions (`vite: ^8.2.0`, `axios: ^1.19.0`).
  * **Impact:** Running `npm install` will fail for developers/deployments.
  * **Mitigation:** Downgrade versions to stable current releases (`vite ^5.0.0`, `axios ^1.7.0`).
* 🟡 **Medium Risk: Silent LLM Failures**
  * **Issue:** If the LLM generates completely malformed JSON that Pydantic rejects, the agent falls back to deterministic extraction without explicitly alerting the orchestrator of the degraded state.

## 2. `vectorDbIngestion` (Database Builder)
**Role:** Ingesting medical literature into Qdrant

* 🔴 **High Risk: Destructive Docker Commands**
  * **Issue:** The ingestion script runs `docker rm -f` on the Qdrant container on every build.
  * **Impact:** Accidental triggering of this script in a production or staging environment will permanently delete all vector embeddings.
  * **Mitigation:** Add a strict safeguard (e.g., `--force` flag or environment check) before tearing down the database.
* 🟡 **Medium Risk: HuggingFace Authentication Block**
  * **Issue:** Downloading restricted biomedical models requires a manual `huggingface-cli login`.
  * **Impact:** Automated CI/CD or Docker builds will hang or fail without an injected `HF_TOKEN`.

## 3. `vectorRetrieval` (Agent 2)
**Role:** BioBERT Vector Database Search

* 🟡 **Medium Risk: Synchronous ML Model Loading**
  * **Issue:** The `SentenceTransformer` BioBERT model is loaded synchronously during FastAPI startup.
  * **Impact:** The API will be unresponsive and block health checks for 5-15 seconds during boot. In cloud environments (like Kubernetes), this can trigger premature readiness probe failures and restart loops.
  * **Mitigation:** Load the model asynchronously or in a separate background thread, and return `503 Service Unavailable` on the `/search` route until loading completes.
* ⚪ **Low Risk: Unimplemented Schema Filters**
  * **Issue:** `evidence_level` and `disease_category` exist in the Pydantic schema but aren't actively filtering Qdrant payloads in the core search logic.

## 4. `evidenceScanner` (Agent 3)
**Role:** PubMed & Web Guideline Search

* 🔴 **Critical Risk: Flawed `@retry` Logic in Web Scraper**
  * **Issue:** In `web_scraper.py`, if the Tavily API returns a 500 or 429 status code, the code does `if resp.status_code != 200: return None`.
  * **Impact:** Because it returns `None` instead of raising an `Exception`, the `tenacity` `@retry` decorator assumes the function succeeded. It will *never* actually retry failed network calls.
  * **Mitigation:** Change `return None` to `raise Exception(f"API Error {resp.status_code}")` to trigger the retry block.
* 🔴 **High Risk: Missing Application Server**
  * **Issue:** The core logic exists, but there is no `app.py` or FastAPI router. It cannot be orchestrated over HTTP yet.

## 5. `dataFusion` (Agent 4)
**Role:** RAG + Web Intelligence Fusion

* 🟡 **Medium Risk: Groq Fallback State Mutation**
  * **Issue:** When the primary LLM fails and falls back to a secondary Groq model, the internal payload payload/history is not cleanly reset on the second retry.
  * **Impact:** The fallback might send malformed or truncated context, leading to inaccurate conflict resolution.
  * **Mitigation:** Deep copy the request payload before initiating the fallback LLM call.

## 6. `clinicalReasoning` (Agent 5)
**Role:** Clinical Decision & Adaptive Loop Gatekeeper

* 🟡 **Medium Risk: Network Dependency & Timeout Propagation**
  * **Issue:** Agent 5 depends on Google Gemini and Groq Cloud. If both experience high latency (e.g., 20+ seconds), the total wait time multiplies.
  * **Impact:** The Orchestrator's HTTP client might time out before Agent 5 finishes its cascade.
  * **Mitigation:** Ensure the Orchestrator's HTTP client timeout is > 60 seconds for Agent 5.
* ⚪ **Low Risk: Model Name Discrepancy**
  * **Issue:** Documentation claims `gemini-2.5-flash`, but code/config targets `gemini-3.5-flash`.

## 7. `orchestrator` (CrewAI Master Controller)
**Role:** Pipeline Management & Feedback Loop

* 🔴 **Critical Risk: CrewAI Latency Multiplier**
  * **Issue:** CrewAI uses an LLM to "think" about what tool to call next. In a 5-agent pipeline, the LLM will spend 5-10 seconds just *reasoning about calling* Agent 1, Agent 2, etc., before the actual Agent does any work.
  * **Impact:** Total pipeline time could exceed 60–90 seconds per pass. With a 3-loop feedback cycle, a single patient request could take 3–5 minutes.
  * **Mitigation:** Use CrewAI's `Process.sequential` with highly rigid, constrained task descriptions so it doesn't get stuck in "thought loops", or pre-bind tools directly to tasks.
* 🔴 **High Risk: Opaque Error Swallowing**
  * **Issue:** If Agent 4 goes down and returns a 500 error, CrewAI's agent might try to "guess" or hallucinate a FusionResponse instead of halting the pipeline.
  * **Impact:** The physician receives a hallucinated clinical fusion instead of a clear system error.
  * **Mitigation:** The custom HTTP `BaseTool` wrappers in the orchestrator MUST raise hard exceptions that abort the CrewAI process, rather than returning stringified errors.
