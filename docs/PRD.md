# CDSS — Project Requirements Document

## Project Name
**Adaptive Multi-Agent LLM-Based Clinical Decision Support System (CDSS)**

## Reference Paper
*"An Adaptive Multi-Agent LLM-Based Clinical Decision Support System Integrating Biomedical RAG and Web Intelligence"*

---

## 1. Project Overview

This system is a multi-agent clinical decision support system that processes unstructured clinical text (from a physician) and produces structured clinical reasoning output — including differential diagnoses, treatment recommendations, and safety analysis — by orchestrating 5 specialized AI agents in a sequential pipeline with an adaptive feedback loop.

The system is **not** an autonomous diagnostic tool. It is a **decision-support assistant** for licensed healthcare professionals.

---

## 2. Target Users

| User | Role |
|---|---|
| **Primary** | Physicians, clinicians, and healthcare professionals entering patient data |
| **Secondary** | Medical researchers evaluating RAG + Web Intelligence approaches |
| **Tertiary** | Developers extending the multi-agent CDSS framework |

---

## 3. Core Features

### 3.1 Clinical Text Processing (Agent 1)
- Accept raw, unstructured clinical text (e.g., "55 year old male with chest pain for 2 hours")
- Extract structured entities: symptoms, demographics, measurements, medications, medical history
- Map clinical concepts to SNOMED CT codes
- Multi-turn conversation with clarification questions for missing information
- Deterministic-first processing with conditional LLM fallback

### 3.2 Biomedical RAG Retrieval (Agent 2)
- Semantic search over a curated medical knowledge base using BioBERT embeddings
- Qdrant vector database with 768-dimensional dense vectors
- 3-level fallback search strategy (strict → relaxed → simplified)
- Source-priority re-ranking (clinical trials > papers > datasets > preprints)
- 5 ingestion sources: PubMed Central, Europe PMC, ClinicalTrials.gov, HuggingFace, medRxiv/bioRxiv

### 3.3 Evidence-Based Web Intelligence (Agent 3)
- Real-time evidence scanning from PubMed and Tavily (clinical guidelines)
- LLM-powered clinical profile extraction (acute symptoms, chronic conditions, condition hints)
- Dynamic search routing: comorbid patients → guidelines only; acute-only → PubMed + guidelines
- Multi-stage fallback (1-year → 10-year window, pairwise entity combinations)

### 3.4 Clinical Data Fusion (Agent 4)
- Cross-validates RAG evidence (Agent 2) with web evidence (Agent 3)
- Terminology normalization and entity deduplication
- Conflict detection (directional contradictions, treatment contraindications)
- Source-priority-weighted confidence scoring (bounded [0.10, 0.98])
- Rule-based baseline with optional LLM semantic enhancement (Llama 3.1)

### 3.5 Adaptive Clinical Reasoning (Agent 5)
- Confidence threshold gate (τ = 0.70)
- 3-tier reasoning cascade: Gemini 2.5 Flash → Groq Llama 3.1 8B → Deterministic Python Engine
- Adaptive feedback loop: if confidence < 0.70, request more evidence (max 3 iterations)
- Structured output: primary diagnosis, differential diagnoses, treatment recommendations, safety analysis
- Clinical output validation at every cascade level

### 3.6 CrewAI Orchestration
- Sequential pipeline: Agent 1 → Agent 2 + 3 (parallel) → Agent 4 → Agent 5
- Adaptive feedback loop managed by the orchestrator
- Single unified API endpoint: `POST /analyze`
- Health monitoring for all downstream agents

---

## 4. Non-Functional Requirements

| Requirement | Target |
|---|---|
| **Latency** | < 60 seconds for full pipeline (without feedback loops) |
| **Availability** | Graceful degradation — deterministic fallbacks at every agent |
| **Scalability** | Microservice architecture — each agent independently deployable |
| **Security** | API keys masked in logs, no PHI stored persistently |
| **Testability** | Each agent independently testable via its own REST API |

---

## 5. Out of Scope

- Frontend UI (Agent 1 has a React frontend, but the full pipeline is API-only)
- Knowledge Graph construction or querying
- Real-time patient monitoring
- EHR/FHIR integration
- Multi-language support
- User authentication / RBAC

---

## 6. Success Criteria

1. ✅ `POST /analyze` with clinical text returns a structured `ClinicalDecisionSupportResponse`
2. ✅ Ambiguous inputs trigger the adaptive feedback loop (up to 3 iterations)
3. ✅ Clear-cut cases (e.g., STEMI) resolve in a single pass with confidence ≥ 0.70
4. ✅ Each agent is independently testable via Thunder Client
5. ✅ The system never crashes — all failures cascade to deterministic fallbacks
