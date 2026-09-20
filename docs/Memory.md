# CDSS — Memory & Progress Tracker

## Current Status
**Date:** August 30, 2026
**State:** Phases 1-3 implemented and reviewed. All bugs fixed. Starting Phase 4.

## Bug Fixes Applied (Aug 30)
- [x] **Phase 1 Fix:** `evidenceScanner/requirements.txt` — fixed missing newline between `python-dateutil` and `fastapi`, removed duplicate `tenacity` entry.
- [x] **Phase 3 Fix:** `orchestrator/requirements.txt` — removed unnecessary heavy ML deps (`torch`, `sentence-transformers`, `qdrant-client`, `transformers`). Orchestrator only makes HTTP calls.
- [x] **Phase 3 Fix:** `orchestrator/tools.py` — changed `type[BaseModel]` to `Type[BaseModel]` for Python <3.12 compatibility. Increased httpx timeout to 90s.
- [x] **Phase 3 Fix (CRITICAL):** `orchestrator/crew.py` — split the single evidence-gathering task (which asked LLM to call 2 tools and return 2 JSONs in 1 string) into 2 separate tasks (Task 2: RAG, Task 3: Scanner). Added a 5th CrewAI agent role (`web_scanner`). Each task now has unambiguous single-tool instructions.

## What Has Been Done
- [x] **Research Phase:** Audited all 6 project folders (`clinicalTextClassifier`, `vectorDbIngestion`, `vectorRetrieval`, `evidenceScanner`, `dataFusion`, `clinicalReasoning`, `orchestrator`).
- [x] **Schema Alignment:** Mapped exact Pydantic I/O models across all agents to identify integration gaps.
- [x] **Agent 3 Discovery:** Confirmed the new `evidenceScanner` folder exists and contains valid logic, but lacks a FastAPI server.
- [x] **Agent 4 Adapters:** Confirmed Agent 4 inherently supports raw JSON inputs from Agent 2/3 and can handle missing Knowledge Graph data gracefully.
- [x] **Documentation:** Created `PRD.md`, `Architecture.md`, `Rules.md`, `Phases.md`, `Design.md`, `Risk_Analysis.md`, and this `Memory.md`.
- [x] **User Decisions Locked:**
  - Keep CrewAI for Orchestration.
  - Drop Knowledge Graph (`kg_evidence=[]`).
  - Do not modify Agent 1's output schema (transform in Orchestrator).

## What Is Pending (Phases)
- [x] **Phase 1: Evidence Scanner Server**
  - [x] Fixed `tenacity` retry bug in `web_scraper.py`
  - [x] Added `app.py` FastAPI wrapper for `evidenceScanner`.
- [x] **Phase 2: Orchestrator Adapters**
  - [x] Create `orchestrator/adapters.py`.
- [x] **Phase 3: Orchestrator Rewrite**
  - [x] Update `crew.py` to wire all 5 agents.
  - [x] Update `app.py` to handle the `while` loop (feedback loop) checking confidence ≥ 0.70.
- [x] **Phase 4: Full System Test**
  - [x] Created `start_all.ps1` startup script.
  - [x] Syntax-verified all new/modified files.
  - [ ] Launch all ports (8000, 8002, 8003, 8004, 8005, 9000).
  - [ ] Test `POST /analyze`.

## Known Issues / Gotchas
- `evidenceScanner` currently has a bug in its `web_scraper.py` retry logic where non-200 HTTP codes return `None` instead of raising exceptions, which bypasses `tenacity`.
- Need to ensure `top_k=5` is dynamically passed through to `SearchRequest` during Agent 1 → Agent 2 transformation.
