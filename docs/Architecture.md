# CDSS — Architecture Document

## 1. System Architecture Flow

The system is designed as a loosely coupled microservices architecture. The central Orchestrator (CrewAI-based) manages the sequential flow and conditional loops across 5 specialized independent Agents.

```mermaid
sequenceDiagram
    participant User
    participant Orch as Orchestrator (Port 9000)
    participant A1 as Agent 1: Clarifier (Port 8000)
    participant A2 as Agent 2: Vector RAG (Port 8002)
    participant A3 as Agent 3: Web Scanner (Port 8003)
    participant A4 as Agent 4: Data Fusion (Port 8004)
    participant A5 as Agent 5: Reasoning (Port 8005)

    User->>Orch: POST /analyze {text}
    Orch->>A1: POST /clarify
    A1-->>Orch: Structured Entities
    
    par Parallel Evidence Gathering
        Orch->>A2: POST /search (Search Query)
        Orch->>A3: POST /api/v1/scan (Entities + Conditions)
    end
    
    A2-->>Orch: Vector RAG Evidence
    A3-->>Orch: PubMed/Tavily Guidelines
    
    loop Adaptive Feedback (Max 3)
        Orch->>A4: POST /api/v1/fuse (Agent2 + Agent3 data)
        A4-->>Orch: UnifiedClinicalContext
        
        Orch->>A5: POST /api/v1/reason (Unified Context)
        
        alt Confidence ≥ 0.70
            A5-->>Orch: ClinicalDecisionSupportResponse (Success)
            Orch-->>User: Final Output
        else Confidence < 0.70
            A5-->>Orch: FeedbackRequest (Gaps identified)
            Note over Orch: Re-query A2 & A3 based on feedback
        end
    end
```

## 2. Technical Stack

| Component | Technology |
|---|---|
| **API Framework** | FastAPI, Uvicorn |
| **Data Validation** | Pydantic |
| **Orchestration** | CrewAI, Python `asyncio` |
| **LLMs (Primary)** | Google Gemini (via `google-genai`), Llama 3.1 8B (via Groq) |
| **Vector Database** | Qdrant (Docker) |
| **Embeddings** | HuggingFace Sentence Transformers (`pritamdeka/BioBERT-...`) |
| **Data Ingestion** | BioPython (NCBI), Tavily API, HuggingFace Datasets |
| **Frontend (Agent 1)**| React, Vite, Tailwind CSS |

## 3. Directory Structure

```text
c:\Users\Karth\Documents\CDSS\
├── PRD.md
├── Architecture.md
├── Rules.md
├── Phases.md
├── Design.md
├── Memory.md
├── clinicalTextClassifier/   # Agent 1 (Entities & SNOMED CT)
├── vectorDbIngestion/        # RAG DB Builder (CLI)
├── vectorRetrieval/          # Agent 2 (BioBERT RAG Search)
├── evidenceScanner/          # Agent 3 (PubMed & Tavily Scanner)
├── dataFusion/               # Agent 4 (Merge, Detect Conflicts)
├── clinicalReasoning/        # Agent 5 (Decision & Feedback)
└── orchestrator/             # Master Controller (CrewAI)
```

## 4. Contract Schema Adapters
Because the agents were developed independently, the **Orchestrator** acts as the Anti-Corruption Layer, performing schema transformations:
* **Agent 1 Output** is mapped to Agent 2 `SearchRequest` and Agent 3 `EvidenceRequest`.
* **Agent 2 & 3 Outputs** are directly passed into Agent 4's `FusionRequest`, as Agent 4 inherently implements adapters for upstream payload variants.
* **Agent 4 Output** exactly matches Agent 5 Input (`UnifiedClinicalContext`).
* **Knowledge Graph** is intentionally stubbed out (`kg_evidence = []`) during transformation.
