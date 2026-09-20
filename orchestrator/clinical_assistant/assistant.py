"""Clinical Conversational Assistant Master Coordinator.

Orchestrates the complete multi-turn conversational loop:
LISTEN -> EXTRACT & VALIDATE -> UPDATE HYBRID MEMORY -> RETRIEVE VECTOR RAG & WEB EVIDENCE (PARALLEL) ->
PROGRESSIVE REASONING (HOSTED OPENAI GPT ASYNC) -> RESPONSE VALIDATION -> ASSISTANT RESPONSE
"""

import asyncio
import logging
import time
import uuid
from typing import Any, Dict, List, Optional
import httpx

from orchestrator.clinical_assistant.llm_client import OpenAIGPTClient
from orchestrator.clinical_assistant.memory.hybrid_memory import (
    FactStatus,
    HybridMemoryManager,
    PatientEntityMemory,
    ShortTermMessageBuffer,
)
from orchestrator.clinical_assistant.extraction.state_extractor import ClinicalStateExtractor
from orchestrator.clinical_assistant.grounding.evidence_grounder import (
    EvidenceGrounder,
    GroundedEvidenceContext,
)
from orchestrator.clinical_assistant.progressive_reasoning.reasoning_controller import (
    ProgressiveReasoningController,
)
from orchestrator.clinical_assistant.validation.domain_guardrail import DomainGuardrail
from orchestrator.clinical_assistant.validation.response_validator import (
    ClinicalResponseValidator,
    ValidationResult,
)

logger = logging.getLogger("orchestrator.clinical_assistant.assistant")


SYSTEM_PROMPT = """You are an expert, calm, and empathetic Clinical Decision Support Conversational Assistant.

YOUR CORE MANDATE:
1. STRICT TRUTH-TELLING & ANTI-FABRICATION:
   - You MUST NEVER state, imply, or assume that the patient has any symptom, medical history, or condition that has NOT been explicitly confirmed by the patient.
   - If the patient reported ONLY fever, you MUST NOT say they have headache, cough, neck stiffness, rash, or vomiting unless they confirmed it.
   - UNKNOWN does NOT mean TRUE. UNKNOWN does NOT mean FALSE.
   - Clearly separate:
     * What the patient has CONFIRMED.
     * What the patient has DENIED.
     * What is UNKNOWN / pending inquiry.
     * Clinical HYPOTHESES / differentials (which are possibilities, NEVER established facts).

2. BULLET-POINT-FIRST, CONVERSATIONAL UX:
   - Keep responses concise, scannable, and conversational. Avoid large paragraphs or medical textbook style.
   - Default structure:
     **So far**
     - Concise confirmed point (e.g. Fever: 102°F for 2 days)
     - Denied points if any

     **What this could mean**
     - Concise point explaining possibilities simply without academic jargon.

     **A few questions**
     1. Short, high-value question?
     2. Short, high-value question?
     3. Short, high-value question?
   - Individual bullets must contain ONE idea only and be 1 short sentence.
   - Ask 2 to 5 high-value questions at a time. Never dump 15-20 questions.
   - DO NOT repeat the entire patient history on every turn. Acknowledge newly provided facts naturally.
   - NO Markdown tables (NEVER use |---|---|).
   - NO HTML tags (NEVER use <br>, <table>, etc.).
   - NO backslash-escaped Markdown (NEVER use \\*\\* or \\<br>).

3. URGENT CONDITIONS EXCEPTION:
   - If red flags are detected (e.g. fever with altered mental status/hallucinations, acute chest pain, severe breathing difficulty), prioritize safety immediately:
     **Important**
     - Clear warning explaining the urgent risk.
     - Advise immediate in-person evaluation (Emergency Department / Urgent Care).

4. EVIDENCE GROUNDING:
   - Ground your clinical reasoning in the provided Biomedical Vector RAG and Live Web Evidence.
   - Explain conditions in clear, lay-accessible terms (e.g. "trouble breathing" instead of "respiratory compromise").

5. STRICT IN-DOMAIN BOUNDARY:
   - You are strictly a clinical and healthcare decision support assistant.
   - Under NO circumstances may you answer non-medical questions (e.g. geography, distances, travel, coding, trivia, sports, cooking, weather).
   - If a user asks an irrelevant or non-medical question, politely refuse and instruct them to provide clinical or healthcare concerns.
"""


class ClinicalConversationalAssistant:
    """Master Multi-Turn Conversational Clinical Assistant with Parallel Retrieval and Async LLM."""

    def __init__(
        self,
        gpt_client: Optional[OpenAIGPTClient] = None,
        agent2_url: str = "http://localhost:8002",
        agent3_url: str = "http://localhost:8003",
    ):
        self.gpt_client = gpt_client or OpenAIGPTClient()
        self.agent2_url = agent2_url
        self.agent3_url = agent3_url
        self._asked_questions_by_session: Dict[str, set] = {}

    async def chat(
        self,
        message: str,
        session_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Execute one complete multi-turn conversational cycle with parallel retrieval and async GPT."""
        t_req_start = time.perf_counter()
        session_id = session_id or str(uuid.uuid4())
        session_data = HybridMemoryManager.get_or_create(session_id)
        memory: PatientEntityMemory = session_data["long_term"]
        buffer: ShortTermMessageBuffer = session_data["short_term"]
        asked_questions = self._asked_questions_by_session.setdefault(session_id, set())

        logger.info("[TIMESTAMP] Request received | session: %s | input: %s", session_id, message[:80])

        # Step 0: Out-of-Domain Guardrail Check
        is_ood, ood_reason = DomainGuardrail.is_out_of_domain(message, memory)
        if is_ood:
            logger.info("Out-of-domain query intercepted: %s (reason: %s)", message[:60], ood_reason)
            refusal_text = DomainGuardrail.get_refusal_response()
            buffer.add_message("user", message)
            buffer.add_message("assistant", refusal_text)
            return {
                "session_id": session_id,
                "response": refusal_text,
                "patient_state": {
                    "age": memory.age,
                    "gender": memory.gender,
                    "confirmed_symptoms": list(memory.confirmed_symptoms.keys()),
                    "denied_symptoms": sorted(list(memory.denied_symptoms)),
                    "unknown_symptoms": sorted(list(memory.unknown_symptoms)),
                    "measurements": memory.measurements,
                    "medical_history": memory.medical_history,
                    "medical_history_status": memory.medical_history_status,
                    "suspected_hypotheses": [h.condition for h in memory.suspected_conditions],
                },
                "clinical_assessment": {
                    "stage": "OUT_OF_DOMAIN",
                    "is_urgent": False,
                    "priority_level": "LOW",
                    "red_flags": [],
                    "differential_considerations": [],
                    "reasoning_summary": "Query is outside the clinical/medical domain. Non-medical questions are politely refused.",
                },
                "follow_up_questions": [
                    "What symptoms or medical concerns are you or the patient experiencing?"
                ],
                "urgent_flag": False,
                "evidence": [],
                "validation": {
                    "is_valid": True,
                    "action_taken": "domain_guardrail_intercepted",
                    "issue_count": 0,
                    "issues": [],
                },
                "llm_metadata": {
                    "provider": self.gpt_client.provider.value,
                    "model": self.gpt_client.model_name,
                    "purpose": "Clinical Domain Guardrail Interception",
                },
            }

        # Step 1: Extract patient state with strict anti-fabrication
        extracted = ClinicalStateExtractor.extract_from_turn(message, memory)
        logger.info(
            "Extracted turn facts: confirmed=%s, denied=%s, measurements=%s",
            extracted["confirmed_symptoms"],
            extracted["denied_symptoms"],
            extracted["measurements"],
        )

        # Step 2: Add user turn to strictly truncated message buffer
        buffer.add_message("user", message)

        # Step 3: Progressive Clinical Reasoning & Emergency Triage
        triage = ProgressiveReasoningController.evaluate(memory, asked_questions=asked_questions)
        for q in triage.suggested_follow_up_questions:
            asked_questions.add(q)

        # Step 4: Retrieve Biomedical Vector RAG (Agent 2) and Web Evidence (Agent 3) CONCURRENTLY
        confirmed_symptoms = list(memory.confirmed_symptoms.keys())
        query_text = " ".join(confirmed_symptoms) if confirmed_symptoms else message

        async def fetch_agent2(client: httpx.AsyncClient):
            t0 = time.perf_counter()
            logger.info("[TIMESTAMP] Agent 2 (Biomedical RAG) START")
            try:
                res2 = await client.post(
                    f"{self.agent2_url}/search",
                    json={"query": query_text, "top_k": 3},
                    timeout=8.0,
                )
                dur = time.perf_counter() - t0
                if res2.status_code == 200:
                    logger.info("[TIMESTAMP] Agent 2 (Biomedical RAG) END (status=200, dur=%.2fs)", dur)
                    return ("agent2", EvidenceGrounder.process_agent2_rag(res2.json()))
                logger.warning("[TIMESTAMP] Agent 2 (Biomedical RAG) END (status=%d, dur=%.2fs)", res2.status_code, dur)
                return ("agent2", [])
            except asyncio.TimeoutError:
                dur = time.perf_counter() - t0
                logger.warning("[TIMESTAMP] Agent 2 (Biomedical RAG) TIMEOUT after %.2fs", dur)
                return ("agent2", [])
            except Exception as e:
                dur = time.perf_counter() - t0
                logger.warning("[TIMESTAMP] Agent 2 (Biomedical RAG) FAILED after %.2fs: %s", dur, e)
                return ("agent2", [])

        async def fetch_agent3(client: httpx.AsyncClient):
            t0 = time.perf_counter()
            logger.info("[TIMESTAMP] Agent 3 (Web Evidence) START")
            try:
                res3 = await client.post(
                    f"{self.agent3_url}/api/v1/scan",
                    json={
                        "raw_text": message,
                        "entities": confirmed_symptoms,
                        "chronic_conditions": memory.medical_history,
                        "max_results": 2,
                    },
                    timeout=10.0,
                )
                dur = time.perf_counter() - t0
                if res3.status_code == 200:
                    logger.info("[TIMESTAMP] Agent 3 (Web Evidence) END (status=200, dur=%.2fs)", dur)
                    return ("agent3", EvidenceGrounder.process_agent3_web(res3.json()))
                logger.warning("[TIMESTAMP] Agent 3 (Web Evidence) END (status=%d, dur=%.2fs)", res3.status_code, dur)
                return ("agent3", [])
            except asyncio.TimeoutError:
                dur = time.perf_counter() - t0
                logger.warning("[TIMESTAMP] Agent 3 (Web Evidence) TIMEOUT after %.2fs", dur)
                return ("agent3", [])
            except Exception as e:
                dur = time.perf_counter() - t0
                logger.warning("[TIMESTAMP] Agent 3 (Web Evidence) FAILED after %.2fs: %s", dur, e)
                return ("agent3", [])

        rag_items = []
        web_items = []

        async with httpx.AsyncClient() as http_client:
            results = await asyncio.gather(
                fetch_agent2(http_client),
                fetch_agent3(http_client),
                return_exceptions=True,
            )

        for res in results:
            if isinstance(res, tuple):
                source, items = res
                if source == "agent2":
                    rag_items = items
                elif source == "agent3":
                    web_items = items
            elif isinstance(res, Exception):
                logger.error("Concurrent retrieval exception: %s", res)

        evidence_ctx = GroundedEvidenceContext(
            biomedical_rag_evidence=rag_items,
            web_evidence=web_items,
        )

        # Step 5: Assemble Prompt Context
        patient_summary = memory.retrieve_relevant_context(message)
        context_prompt = (
            f"[PATIENT CLINICAL STATE - AUTHORITATIVE FACTS ONLY]\n"
            f"- Age: {patient_summary.get('age') or 'Unspecified'}\n"
            f"- Gender: {patient_summary.get('gender') or 'Unspecified'}\n"
            f"- Confirmed Symptoms: {', '.join(patient_summary.get('confirmed_symptoms', [])) or 'None confirmed'}\n"
            f"- Explicitly Denied Symptoms: {', '.join(patient_summary.get('denied_symptoms', [])) or 'None reported'}\n"
            f"- Unknown / Unasked Findings: {', '.join(patient_summary.get('unknown_symptoms', [])[:8]) or 'None'}\n"
            f"- Measurements / Vitals: {patient_summary.get('measurements') or 'None'}\n"
            f"- Medical History: {', '.join(patient_summary.get('medical_history', [])) or patient_summary.get('medical_history_status')}\n\n"
            f"[CLINICAL TRIAGE ASSESSMENT]\n"
            f"- Priority Level: {triage.priority_level} (Urgent: {triage.is_urgent})\n"
            f"- Red Flags: {', '.join(triage.red_flags_detected) or 'None'}\n"
            f"- Potential Differentials: {', '.join(triage.differential_considerations)}\n"
            f"- Triage Summary: {triage.reasoning_summary}\n\n"
            f"{evidence_ctx.to_prompt_context()}\n"
        )

        # Construct messages for OpenAI GPT model
        prompt_messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "system", "content": context_prompt},
        ]
        # Include truncated message buffer
        prompt_messages.extend(buffer.get_messages())

        # Step 6: Invoke Hosted OpenAI GPT Model (Asynchronous)
        raw_gpt_content = ""
        model_used = self.gpt_client.model_name
        t_gpt_start = time.perf_counter()
        logger.info("[TIMESTAMP] GPT request START (model: %s)", model_used)
        try:
            llm_result = await self.gpt_client.generate_async(
                messages=prompt_messages,
                temperature=0.2,
                max_tokens=600,
            )
            raw_gpt_content = llm_result.get("content", "")
            model_used = llm_result.get("model", model_used)
            t_gpt_dur = time.perf_counter() - t_gpt_start
            logger.info("[TIMESTAMP] GPT response RECEIVED in %.2fs (model: %s)", t_gpt_dur, model_used)
        except Exception as exc:
            logger.error("[FALLBACK WARNING] Primary OpenAI GPT model failed: %s. Using safe clinical fallback.", exc)
            # Safe clinical fallback obeying all anti-fabrication rules
            raw_gpt_content = self._build_deterministic_clinical_response(memory, triage)

        # Step 7: Post-Generation Response Validation
        t_val_start = time.perf_counter()
        validation: ValidationResult = ClinicalResponseValidator.validate_response(
            response_text=raw_gpt_content,
            memory=memory,
            retrieved_evidence_text=evidence_ctx.to_prompt_context(),
        )
        logger.info(
            "[TIMESTAMP] Validation COMPLETE in %.3fs (action: %s, issues: %d)",
            time.perf_counter() - t_val_start,
            validation.action_taken,
            len(validation.issues),
        )

        final_response_text = validation.sanitized_response

        # Step 8: Update short-term buffer with assistant response
        buffer.add_message("assistant", final_response_text)

        # Update clinical hypotheses in long-term memory (strictly isolated)
        for diff in triage.differential_considerations:
            memory.record_hypothesis(diff, confidence=0.6 if triage.is_urgent else 0.4)

        t_total = time.perf_counter() - t_req_start
        logger.info("[TIMESTAMP] Final response ready in %.2fs", t_total)

        return {
            "session_id": session_id,
            "response": final_response_text,
            "patient_state": {
                "age": memory.age,
                "gender": memory.gender,
                "confirmed_symptoms": list(memory.confirmed_symptoms.keys()),
                "denied_symptoms": sorted(list(memory.denied_symptoms)),
                "unknown_symptoms": sorted(list(memory.unknown_symptoms)),
                "measurements": memory.measurements,
                "medical_history": memory.medical_history,
                "medical_history_status": memory.medical_history_status,
                "suspected_hypotheses": [h.condition for h in memory.suspected_conditions],
            },
            "clinical_assessment": {
                "stage": triage.stage,
                "is_urgent": triage.is_urgent,
                "priority_level": triage.priority_level,
                "red_flags": triage.red_flags_detected,
                "differential_considerations": triage.differential_considerations,
                "reasoning_summary": triage.reasoning_summary,
            },
            "follow_up_questions": triage.suggested_follow_up_questions,
            "urgent_flag": triage.is_urgent,
            "evidence": [item.model_dump() for item in evidence_ctx.get_all_items()],
            "validation": {
                "is_valid": validation.is_valid,
                "action_taken": validation.action_taken,
                "issue_count": len(validation.issues),
                "issues": [issue.model_dump() for issue in validation.issues],
            },
            "llm_metadata": {
                "provider": self.gpt_client.provider.value,
                "model": model_used,
                "purpose": "Clinical Conversational Assistant",
            },
        }

    def _build_deterministic_clinical_response(
        self,
        memory: PatientEntityMemory,
        triage: Any,
    ) -> str:
        """Deterministic safe clinical response generator obeying strict anti-fabrication rules."""
        confirmed = list(memory.confirmed_symptoms.keys())
        confirmed_str = ", ".join(confirmed) if confirmed else "reported symptoms"
        temp = memory.measurements.get("temperature", "")
        temp_str = f": {temp}" if temp else ""

        lines = [
            "**So far**",
            f"- Reported: {confirmed_str}{temp_str}",
        ]

        if memory.denied_symptoms:
            lines.append(f"- Denied: {', '.join(sorted(list(memory.denied_symptoms)))}")

        if triage.is_urgent:
            lines.extend([
                "",
                "**Important**",
                f"- {triage.urgency_reason}",
                "- Please seek prompt, in-person medical assessment at an emergency department or urgent care.",
                f"- Conditions requiring emergency rule-out: {', '.join(triage.differential_considerations[:3])}.",
            ])

        if triage.suggested_follow_up_questions:
            lines.extend([
                "",
                "**A few questions**",
            ])
            for idx, q in enumerate(triage.suggested_follow_up_questions[:4], 1):
                lines.append(f"{idx}. {q}")

        return "\n".join(lines)

