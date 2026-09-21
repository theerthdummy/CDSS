"""app.py - FastAPI application for the CDSS Orchestrator.

Exposes:
- POST /chat: Stateful multi-turn clinical conversational assistant powered by hosted OpenAI GPT.
- POST /analyze: Executes the multi-agent clinical reasoning pipeline with response validation.
- POST /upload: Upload a medical document (PDF/image/DOCX) for direct OCR-based diagnosis & prognosis.
- GET /health: Health check auditing downstream agents and OpenAI GPT availability.
"""

import asyncio
import json
import logging
import sys
import time
from pathlib import Path
from typing import Optional, Any, Dict, List

# Ensure parent and orchestrator root are on sys.path
_ORCHESTRATOR_DIR = Path(__file__).resolve().parent
_PROJECT_DIR = _ORCHESTRATOR_DIR.parent
for _p in [str(_ORCHESTRATOR_DIR), str(_PROJECT_DIR)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import httpx

import config
import adapters
from clinical_assistant.assistant import ClinicalConversationalAssistant
from clinical_assistant.llm_client import OpenAIGPTClient, LLMProvider
from clinical_assistant.document_ocr import (
    extract_text_from_file,
    parse_prescription,
    build_clinical_context_from_document,
    TextExtractionError,
    SUPPORTED_EXTENSIONS,
)

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(name)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI(
    title="CDSS Orchestrator & Conversational Assistant",
    description="Multi-Agent CDSS Orchestrator with Hosted OpenAI GPT Conversational Assistant",
    version="3.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Hosted OpenAI GPT Client & Conversational Assistant
gpt_client = OpenAIGPTClient(
    api_key=config.OPENAI_API_KEY,
    base_url=config.OPENAI_BASE_URL,
    model_name=config.OPENAI_MODEL,
)

assistant = ClinicalConversationalAssistant(
    gpt_client=gpt_client,
    agent2_url=config.AGENT2_URL,
    agent3_url=config.AGENT3_URL,
)


# ── Request / Response Models ──────────────────────────────────────────────

class ChatRequest(BaseModel):
    message: str = Field(..., description="Patient / Clinician input message", min_length=1)
    session_id: Optional[str] = Field(None, description="Optional session ID for multi-turn state maintenance")
    conversation_history: Optional[List[Dict[str, Any]]] = Field(None, description="Optional conversation history for state hydration")


class ChatResponse(BaseModel):
    session_id: str
    response: str
    patient_state: Dict[str, Any]
    clinical_assessment: Dict[str, Any]
    follow_up_questions: List[str]
    urgent_flag: bool
    evidence: List[Dict[str, Any]]
    validation: Dict[str, Any]
    llm_metadata: Dict[str, Any]


class AnalyzeRequest(BaseModel):
    text: str = Field(..., description="Raw clinical text from the patient/physician", min_length=1)
    session_id: Optional[str] = Field(None, description="Optional session ID for conversational memory")


class AnalyzeResponse(BaseModel):
    patient_text: str
    final_output: Any
    iterations: int
    success: bool = True
    session_id: Optional[str] = None


async def post_json(client: httpx.AsyncClient, url: str, payload: dict, timeout: float = 30.0) -> dict:
    """Helper to post JSON and parse the response with timeout."""
    try:
        response = await client.post(url, json=payload, timeout=timeout)
        response.raise_for_status()
        return response.json()
    except httpx.HTTPStatusError as e:
        logger.error(f"HTTP {e.response.status_code} from {url}: {e.response.text}")
        raise
    except Exception as e:
        logger.error(f"Network error calling {url}: {str(e)}")
        raise


# ── Conversational Endpoint (Multi-Turn) ───────────────────────────────────

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    """Execute a multi-turn conversation turn with the hosted OpenAI GPT clinical assistant.

    Maintains long-term patient entities and short-term message history across turns.
    Enforces progressive reasoning, urgent red-flag triage, and anti-hallucination validation.
    """
    try:
        result = await assistant.chat(
            message=request.message,
            session_id=request.session_id,
            conversation_history=request.conversation_history,
        )
        return ChatResponse(**result)
    except Exception as exc:
        logger.error("Chat assistant error: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Clinical Assistant error: {str(exc)}")


# ── Pipeline Analysis Endpoint ─────────────────────────────────────────────

@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze(request: AnalyzeRequest):
    """Executes the full clinical pipeline with multi-turn memory integration and response validation."""
    t_start = time.perf_counter()
    logger.info("[TIMESTAMP] Analyze request received: %s...", request.text[:80])

    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(60.0, connect=10.0)) as client:
            # First, execute a conversation turn to update memory and check validation
            chat_result = await assistant.chat(
                message=request.text,
                session_id=request.session_id,
            )
            session_id = chat_result["session_id"]

            logger.info("[Iteration 0] Kicking off Initial Pipeline...")
            
            # Task 1: Agent 1 (Clarifier)
            agent1_payload = {"text": request.text}
            if session_id:
                agent1_payload["session_id"] = session_id
            agent1_json = await post_json(client, f"{config.AGENT1_URL}/clarify", agent1_payload, timeout=20.0)
            
            # Domain Guardrail: Reject non-medical queries
            symptoms = agent1_json.get("symptoms", [])
            history = agent1_json.get("medical_history", [])
            meds = agent1_json.get("medications", [])
            
            if not symptoms and not history and not meds and not chat_result["patient_state"]["confirmed_symptoms"]:
                logger.warning("Query rejected by domain guardrail: No medical entities detected.")
                return AnalyzeResponse(
                    patient_text=request.text,
                    final_output={
                        "clinical_assessment": {
                            "primary_interpretation": "This query does not appear to contain any clinical symptoms, medical history, or medications. I am a specialized Clinical Decision Support System. Please provide a relevant medical scenario."
                        }
                    },
                    iterations=0,
                    success=False,
                    session_id=session_id,
                )
            
            # PARALLEL RETRIEVAL: Agent 2 (Biomedical RAG) + Agent 3 (Web Scanner)
            agent2_query = adapters.agent1_to_search_query(agent1_json)
            agent3_req = adapters.agent1_to_evidence_request(agent1_json, raw_patient_text=request.text)

            async def call_agent2():
                t0 = time.perf_counter()
                logger.info("[TIMESTAMP] Agent 2 START")
                try:
                    res = await post_json(client, f"{config.AGENT2_URL}/search", {"query": agent2_query, "top_k": 5}, timeout=15.0)
                    logger.info("[TIMESTAMP] Agent 2 END (dur=%.2fs)", time.perf_counter() - t0)
                    return res
                except Exception as e:
                    logger.warning("[TIMESTAMP] Agent 2 FAILED (dur=%.2fs): %s", time.perf_counter() - t0, e)
                    return {"query": agent2_query, "results": [], "status": "unavailable"}

            async def call_agent3():
                t0 = time.perf_counter()
                logger.info("[TIMESTAMP] Agent 3 START")
                try:
                    res = await post_json(client, f"{config.AGENT3_URL}/api/v1/scan", agent3_req, timeout=15.0)
                    logger.info("[TIMESTAMP] Agent 3 END (dur=%.2fs)", time.perf_counter() - t0)
                    return res
                except Exception as e:
                    logger.warning("[TIMESTAMP] Agent 3 FAILED (dur=%.2fs): %s", time.perf_counter() - t0, e)
                    return {"acute_symptoms": [], "evidence": [], "summary": "Web evidence unavailable", "status": "unavailable"}

            # Concurrently execute independent Agent 2 & Agent 3
            agent2_json, agent3_json = await asyncio.gather(call_agent2(), call_agent3())

            # Task 4: Agent 4 (Fusion) - Strictly depends on Agent 2 & Agent 3
            t_a4_start = time.perf_counter()
            logger.info("[TIMESTAMP] Agent 4 START")
            fusion_payload = {
                "agent2_output": adapters.agent2_to_fusion_input(agent2_json),
                "agent3_output": adapters.agent3_to_fusion_input(agent3_json)
            }
            agent4_json = await post_json(client, f"{config.AGENT4_URL}/api/v1/fuse", fusion_payload, timeout=20.0)
            logger.info("[TIMESTAMP] Agent 4 END (dur=%.2fs)", time.perf_counter() - t_a4_start)
            
            # Task 5: Agent 5 (Reasoning) - Strictly depends on Agent 4
            t_a5_start = time.perf_counter()
            logger.info("[TIMESTAMP] Agent 5 START")
            unified_context = agent4_json.get("unified_context", {})
            current_result = await post_json(client, f"{config.AGENT5_URL}/api/v1/reason?retry_count=0", unified_context, timeout=30.0)
            logger.info("[TIMESTAMP] Agent 5 END (dur=%.2fs)", time.perf_counter() - t_a5_start)

            # Enrich Agent 5 output with conversational assistant response & memory
            if isinstance(current_result, dict):
                current_result["conversational_response"] = chat_result["response"]
                current_result["patient_state"] = chat_result["patient_state"]
                current_result["urgent_flag"] = chat_result["urgent_flag"]
                current_result["follow_up_questions"] = chat_result["follow_up_questions"]
                current_result["validation"] = chat_result["validation"]
                current_result["agent_metadata"] = {
                    "agent": "orchestrator",
                    "provider": gpt_client.provider.value,
                    "model": gpt_client.model_name,
                    "purpose": "Clinical Conversational Assistant",
                    "status": "success",
                }

            logger.info("[TIMESTAMP] Analyze pipeline COMPLETED in %.2fs", time.perf_counter() - t_start)

            return AnalyzeResponse(
                patient_text=request.text,
                final_output=current_result,
                iterations=1,
                success=True,
                session_id=session_id,
            )

    except Exception as e:
        logger.error("Pipeline failed: %s", e, exc_info=True)
        # Safe fallback response utilizing the conversational assistant output
        return AnalyzeResponse(
            patient_text=request.text,
            final_output={
                "clinical_assessment": {
                    "primary_interpretation": chat_result["response"] if "chat_result" in locals() else "Assessment completed.",
                },
                "conversational_response": chat_result["response"] if "chat_result" in locals() else "",
                "patient_state": chat_result["patient_state"] if "chat_result" in locals() else {},
            },
            iterations=1,
            success=True,
            session_id=chat_result["session_id"] if "chat_result" in locals() else None,
        )


# ── Document Upload → OCR → Clinical Diagnosis ─────────────────────────────

import os, shutil, tempfile, uuid as _uuid

@app.post("/upload")
async def upload_document(
    file: UploadFile = File(...),
    session_id: Optional[str] = Form(None),
    user_message: Optional[str] = Form(None),
):
    """
    Accept a medical document (PDF, JPEG, PNG, DOCX, TXT) and:
    1. Save to a temp file.
    2. Run OCR / text extraction (adapted from PilotMaster DocPilot pipeline).
    3. Parse prescription fields (medications, diagnosis, vitals, etc.).
    4. Build a structured clinical context prompt.
    5. Route through the conversational assistant for diagnosis & prognosis.
    6. Return the same shape as /chat.
    """
    ext = os.path.splitext(file.filename or "")[1].lower()
    allowed = SUPPORTED_EXTENSIONS  # from document_ocr module
    if ext not in allowed:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported file type '{ext}'. Supported: {', '.join(allowed)}",
        )

    tmp_dir = tempfile.mkdtemp()
    tmp_path = os.path.join(tmp_dir, f"{_uuid.uuid4()}{ext}")

    try:
        # Save upload to temp file
        with open(tmp_path, "wb") as fh:
            shutil.copyfileobj(file.file, fh)

        logger.info("[UPLOAD] Saved '%s' (%d bytes) to %s", file.filename, os.path.getsize(tmp_path), tmp_path)

        # OCR / text extraction
        try:
            raw_text = extract_text_from_file(tmp_path, mime_type=file.content_type)
        except TextExtractionError as exc:
            raise HTTPException(status_code=422, detail=str(exc))

        # Parse structured prescription fields
        parsed = parse_prescription(raw_text)
        clinical_context = build_clinical_context_from_document(parsed)

        # Build the message the assistant will process.
        # Combine OCR context with any optional user instruction (e.g. "give prognosis")
        follow_on = (user_message or "").strip()
        if follow_on:
            assistant_prompt = (
                f"{clinical_context}\n\n"
                f"[CLINICIAN INSTRUCTION]\n{follow_on}"
            )
        else:
            assistant_prompt = (
                f"{clinical_context}\n\n"
                "[CLINICIAN INSTRUCTION]\n"
                "Based on the prescription / medical document above, provide:\n"
                "1. Clinical Impression & Diagnosis\n"
                "2. Prognosis\n"
                "3. Recommended further investigations or management"
            )

        # Route through the existing conversational assistant pipeline
        chat_result = await assistant.chat(
            message=assistant_prompt,
            session_id=session_id,
        )

        return {
            **chat_result,
            "document_info": {
                "filename": file.filename,
                "file_type": ext,
                "extracted_chars": len(raw_text),
                "parsed_fields": {
                    k: v for k, v in parsed.items() if k != "raw_text" and v
                },
            },
        }

    finally:
        # Clean up temp files securely
        try:
            shutil.rmtree(tmp_dir, ignore_errors=True)
        except Exception:
            pass


# ── Health Check (Non-blocking Async) ──────────────────────────────────────

@app.get("/health")
async def health_check():
    """Check connectivity to all downstream agents and hosted OpenAI GPT status."""
    agents = {
        "agent1_clarifier": config.AGENT1_URL,
        "agent2_retrieval": config.AGENT2_URL,
        "agent3_scanner": config.AGENT3_URL,
        "agent4_fusion": config.AGENT4_URL,
        "agent5_reasoning": config.AGENT5_URL,
    }
    
    status = {}
    async with httpx.AsyncClient(timeout=5.0) as client:
        for name, url in agents.items():
            try:
                r = await client.get(url)
                status[name] = {"url": url, "status": "healthy", "code": r.status_code}
            except Exception as e:
                status[name] = {"url": url, "status": "unreachable", "error": str(e)}

    # Verify Hosted OpenAI GPT Client
    openai_status = {
        "provider": gpt_client.provider.value,
        "model": gpt_client.model_name,
        "configured": gpt_client.is_available,
        "base_url": gpt_client.base_url,
    }

    all_healthy = all(s["status"] == "healthy" for s in status.values())
    return {
        "overall": "healthy" if all_healthy else "degraded",
        "agents": status,
        "openai_gpt": openai_status,
    }

