---
title: Clinical Decision Support System (CDSS)
emoji: 🏥
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
---

# Clinical Decision Support System (CDSS)

A Multi-Agent Biomedical AI system that implements an adaptive clinical decision support pipeline integrating retrieval-augmented generation (RAG) and live web intelligence.

## System Architecture

The CDSS is composed of 5 distinct agents orchestrated by a CrewAI central controller. The system processes raw patient notes and autonomously reasons through a differential diagnosis, checking local vector databases and live PubMed/clinical guidelines to support its findings.

### 1. Agent 1: Clinical Text Clarifier (`clinicalTextClassifier`)
* **Role**: Entity Extraction and Structuring
* **Port**: `8000`
* **Description**: Takes unstructured human input (e.g., "55yo male with chest pain") and uses `gemini-3.5-flash-lite` to extract standardized entities (age, gender, symptoms, medical history).

### 2. Agent 2: Vector Retrieval (`vectorRetrieval`)
* **Role**: Local Biomedical RAG Search
* **Port**: `8002`
* **Description**: Searches a local Qdrant Vector DB containing ingested medical literature and epidemiological data based on Agent 1's symptoms.

### 3. Agent 3: Evidence Scanner (`evidenceScanner`)
* **Role**: Live Web Intelligence
* **Port**: `8003`
* **Description**: Translates symptoms into advanced boolean MeSH queries to search PubMed and uses Tavily to search clinical guidelines on the live internet using `gemini-3.6-flash`.

### 4. Agent 4: Data Fusion (`dataFusion`)
* **Role**: Evidence Synthesis
* **Port**: `8004`
* **Description**: Merges, deduplicates, and synthesizes the flat vector records from Agent 2 and the live web evidence from Agent 3 into a single cohesive "Unified Clinical Context".

### 5. Agent 5: Clinical Reasoning Engine (`clinicalReasoning`)
* **Role**: Diagnosis & Adaptive Optimization
* **Port**: `8005`
* **Description**: Evaluates the unified clinical context against the patient's profile to formulate a differential diagnosis, safety flags, and recommended actions using `gemini-3.5-flash-lite`. Calculates a confidence score.

### 6. The Orchestrator (`orchestrator`)
* **Role**: CrewAI Pipeline Manager
* **Port**: `9000`
* **Description**: Manages the flow of data between agents. Implements an **Adaptive Feedback Loop**: if Agent 5's confidence score is < 0.70, it triggers Agents 2 and 3 to perform a secondary, highly-targeted search based on the missing information gaps identified by Agent 5.

### 7. The Frontend (`frontend`)
* **Role**: User Interface
* **Port**: `5173` (Vite dev server)
* **Description**: A React-based chat dashboard that communicates directly with the Orchestrator to trigger the pipeline and visualize the final diagnosis.

## Data Preparation (`vectorDbIngestion`)
Contains scripts for embedding and ingesting medical literature (e.g., from NCBI) into the local Qdrant Docker container.

## Technologies Used
* **AI/LLMs**: Google Gemini API, CrewAI, sentence-transformers
* **Backend APIs**: FastAPI, Python 3.10+
* **Frontend**: React, Vite
* **Database**: Qdrant Vector Database (Docker)
