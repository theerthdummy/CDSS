"""Clinical Information Extraction Agent (Agent 1).

Workflow (per turn — every message is independently evaluated):
1. Patient Input → Lightweight Normalization (Python)
2. Input Complexity Router → classify_input_complexity() → "level1" | "level2"
   - Level 1 (atomic/single-field): Deterministic extraction only, 0 LLM calls.
   - Level 2 (multi-field/narrative): Deterministic pre-pass + one LLM call.
3. Authoritative State Manager (single source of truth, avoids infinite loops)
4. Mandatory Field Validation & Dynamic Recalculation
5. SNOMED CT Concept Mapping (Staged Snowstorm lookup)
6. Structured JSON & Prioritized Next Question
"""

import json
import uuid
from typing import Any, Dict, List, Optional

from app.agent.extractor import classify_input_complexity, extract_deterministic_entities
from app.agent.normalizer import normalize_clinical_text
from app.agent.prompt import SYSTEM_PROMPT_CLINICAL_INTERPRETATION
from app.agent.schema import Agent1Request, Agent1Response
from app.agent.state import (
    create_empty_patient_state,
    determine_next_clarification,
    merge_patient_state,
)
from app.services import gemini, ollama, snomed


class ClinicalTextClarifierAgent:
    """Extracts clinical information with deterministic-first processing and conditional LLM interpretation."""

    # In-memory session store: {session_id: {"accumulated_state": {...}, "pending_question": "...", "history": [...]}}
    _session_store: Dict[str, Dict[str, Any]] = {}

    def process(self, request: Agent1Request) -> Agent1Response:
        """Process patient input through the streamlined, optimized pipeline."""
        original_text = request.text or ""
        normalized_text = normalize_clinical_text(original_text)

        # 1. Initialize or load session
        session_id = request.session_id or str(uuid.uuid4())
        session = self._session_store.setdefault(
            session_id,
            {
                "accumulated_state": create_empty_patient_state(),
                "pending_question": None,
                "history": [],
            },
        )

        accumulated_state = session.get("accumulated_state", create_empty_patient_state())
        pending_question = session.get("pending_question")
        history = session.get("history", [])

        # 2. Route input: classify complexity independently on every turn
        used_llm = False
        if not normalized_text:
            new_info = create_empty_patient_state()
        else:
            route = classify_input_complexity(
                normalized_text,
                pending_question=pending_question,
            )

            if route == "level1":
                # --- Level 1: Single atomic value — deterministic only, 0 LLM ---
                det_extracted, is_fully_understood = extract_deterministic_entities(
                    normalized_text,
                    pending_question=pending_question,
                    current_state=accumulated_state,
                )
                if is_fully_understood:
                    new_info = det_extracted
                    used_llm = False
                else:
                    # Rare edge-case: router said level1 but extractor can't fully
                    # parse (e.g. a new condition name). Re-route to level2.
                    llm_extracted = self._interpret_with_llm(
                        normalized_text=normalized_text,
                        pending_question=pending_question,
                        accumulated_state=accumulated_state,
                        history=history,
                        deterministic_fallback=det_extracted,
                    )
                    new_info = llm_extracted
                    used_llm = True
            else:
                # --- Level 2: Multi-field / narrative — LLM semantic interpretation ---
                det_extracted, _ = extract_deterministic_entities(
                    normalized_text,
                    pending_question=pending_question,
                    current_state=accumulated_state,
                )
                llm_extracted = self._interpret_with_llm(
                    normalized_text=normalized_text,
                    pending_question=pending_question,
                    accumulated_state=accumulated_state,
                    history=history,
                    deterministic_fallback=det_extracted,
                )
                new_info = llm_extracted
                used_llm = True

        # 3. Merge into authoritative patient state
        merged_state = merge_patient_state(accumulated_state, new_info)

        # 4. Record history turn
        history.append({
            "user_input": original_text,
            "normalized_input": normalized_text,
            "pending_question": pending_question,
            "extracted": new_info,
            "used_llm": used_llm,
        })

        # 5. Determine next prioritized question & dynamically recalculate missing info
        if not normalized_text:
            clarification_q, needs_clarification, missing_info = None, False, []
        else:
            clarification_q, needs_clarification, missing_info = determine_next_clarification(
                merged_state,
                pending_question=pending_question,
            )

        # 6. Update session state
        session["accumulated_state"] = merged_state
        session["pending_question"] = clarification_q if needs_clarification else None
        session["history"] = history

        # 7. Map clinical concepts (symptoms, conditions, medications) to SNOMED CT
        terminology_mappings = snomed.map_clinical_concepts(
            symptoms=merged_state.get("symptoms", []),
            medical_history=merged_state.get("medical_history", []),
            medications=merged_state.get("medications", []),
        )

        # 8. Build validated response
        return Agent1Response(
            session_id=session_id,
            original_input=original_text,
            normalized_input=normalized_text,
            age=merged_state.get("age"),
            gender=merged_state.get("gender"),
            medical_history=merged_state.get("medical_history", []),
            medical_history_status=merged_state.get("medical_history_status", "UNKNOWN"),
            medications=merged_state.get("medications", []),
            medications_status=merged_state.get("medications_status", "UNKNOWN"),
            symptoms=merged_state.get("symptoms", []),
            denied_symptoms=merged_state.get("denied_symptoms", []),
            measurements=merged_state.get("measurements", {}),
            duration=merged_state.get("duration"),
            severity=merged_state.get("severity"),
            other_information=merged_state.get("other_information", []),
            relationships=merged_state.get("relationships", []),
            missing_information=missing_info,
            terminology_mappings=terminology_mappings,
            clarification_question=clarification_q,
            requires_clarification=needs_clarification,
        )

    def _interpret_with_llm(
        self,
        normalized_text: str,
        pending_question: Optional[str],
        accumulated_state: Dict[str, Any],
        history: List[Dict[str, Any]],
        deterministic_fallback: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Perform exactly one LLM call for semantic interpretation when deterministic extraction is insufficient."""
        # Build context block
        context_lines = []
        if any(accumulated_state.get(k) for k in ["age", "gender", "medical_history", "medications", "symptoms", "measurements"]):
            context_lines.append("WHAT WE ALREADY KNOW:")
            if accumulated_state.get("age"):
                context_lines.append(f"  Age: {accumulated_state['age']}")
            if accumulated_state.get("gender"):
                context_lines.append(f"  Gender: {accumulated_state['gender']}")
            if accumulated_state.get("medical_history"):
                context_lines.append(f"  Medical history: {', '.join(accumulated_state['medical_history'])}")
            if accumulated_state.get("medications"):
                context_lines.append(f"  Medications: {', '.join(accumulated_state['medications'])}")
            if accumulated_state.get("symptoms"):
                context_lines.append(f"  Symptoms: {', '.join(accumulated_state['symptoms'])}")
            if accumulated_state.get("measurements"):
                for k, v in accumulated_state["measurements"].items():
                    context_lines.append(f"  {k.title()}: {v}")

        if pending_question:
            context_lines.append(f"\nPENDING QUESTION ASKED BY ASSISTANT:\n  {pending_question}")

        context_block = "\n".join(context_lines)

        user_prompt = f"""Context:
{context_block}

Current Patient Input:
{normalized_text}

Extract all clinical facts from the Current Patient Input into JSON."""

        try:
            parsed = None
            # Try Groq round-robin first, then Gemini fallback
            if gemini.is_groq_available() or gemini.is_gemini_available():
                parsed = gemini.call_llm_json(
                    prompt=user_prompt,
                    system_prompt=SYSTEM_PROMPT_CLINICAL_INTERPRETATION,
                    temperature=0.0,
                    timeout=15,
                )
            elif ollama.is_ollama_available():
                raw_response = ollama.call_ollama_text_only(
                    prompt=user_prompt,
                    system_prompt=SYSTEM_PROMPT_CLINICAL_INTERPRETATION,
                    temperature=0.0,
                    timeout=30,
                    num_predict=400,
                )
                cleaned_json = raw_response.strip()
                if cleaned_json.startswith("```json"):
                    cleaned_json = cleaned_json[7:]
                if cleaned_json.startswith("```"):
                    cleaned_json = cleaned_json[3:]
                if cleaned_json.endswith("```"):
                    cleaned_json = cleaned_json[:-3]
                parsed = json.loads(cleaned_json.strip())

            if not isinstance(parsed, dict):
                return deterministic_fallback

            # Validate and supplement with high-confidence deterministic extractions
            result = create_empty_patient_state()
            # Age: always prefer deterministic fallback (pure digit); strip non-digits from LLM output
            if deterministic_fallback.get("age"):
                result["age"] = deterministic_fallback["age"]
            elif parsed.get("age"):
                raw_age = str(parsed["age"])
                import re as _re
                age_digits = _re.search(r"\d{1,3}", raw_age)
                result["age"] = age_digits.group(0) if age_digits else raw_age

            if parsed.get("gender"):
                result["gender"] = str(parsed["gender"]).lower()
            elif deterministic_fallback.get("gender"):
                result["gender"] = deterministic_fallback["gender"]

            if isinstance(parsed.get("medical_history"), list):
                result["medical_history"] = [str(x) for x in parsed["medical_history"]]
                result["medical_history_status"] = "PRESENT" if result["medical_history"] else parsed.get("medical_history_status", "UNKNOWN")
            if deterministic_fallback.get("medical_history"):
                result["medical_history"] = list(set(result["medical_history"] + deterministic_fallback["medical_history"]))
                result["medical_history_status"] = "PRESENT"

            if isinstance(parsed.get("medications"), list):
                result["medications"] = [str(x) for x in parsed["medications"]]
                result["medications_status"] = "PRESENT" if result["medications"] else parsed.get("medications_status", "UNKNOWN")
            if deterministic_fallback.get("medications"):
                result["medications"] = list(set(result["medications"] + deterministic_fallback["medications"]))
                result["medications_status"] = "PRESENT"

            if isinstance(parsed.get("symptoms"), list):
                result["symptoms"] = [str(x) for x in parsed["symptoms"]]
            if deterministic_fallback.get("symptoms"):
                result["symptoms"] = list(set(result["symptoms"] + deterministic_fallback["symptoms"]))

            if isinstance(parsed.get("denied_symptoms"), list):
                result["denied_symptoms"] = [str(x) for x in parsed["denied_symptoms"]]
            if deterministic_fallback.get("denied_symptoms"):
                result["denied_symptoms"] = list(set(result["denied_symptoms"] + deterministic_fallback["denied_symptoms"]))

            if isinstance(parsed.get("measurements"), dict):
                result["measurements"] = parsed["measurements"]
            if deterministic_fallback.get("measurements"):
                result["measurements"].update(deterministic_fallback["measurements"])

            if parsed.get("duration"):
                result["duration"] = str(parsed["duration"])
            elif deterministic_fallback.get("duration"):
                result["duration"] = deterministic_fallback["duration"]

            if parsed.get("severity"):
                result["severity"] = str(parsed["severity"])
            elif deterministic_fallback.get("severity"):
                result["severity"] = deterministic_fallback["severity"]

            if isinstance(parsed.get("other_information"), list):
                result["other_information"] = [str(x) for x in parsed["other_information"]]
            if deterministic_fallback.get("other_information"):
                result["other_information"] = list(set(result["other_information"] + deterministic_fallback["other_information"]))

            if isinstance(parsed.get("relationships"), list):
                result["relationships"] = [str(x) for x in parsed["relationships"]]
            if deterministic_fallback.get("relationships"):
                result["relationships"] = list(set(result["relationships"] + deterministic_fallback["relationships"]))

            if deterministic_fallback.get("denied_history"):
                result["denied_history"] = deterministic_fallback["denied_history"]
            if deterministic_fallback.get("denied_medications"):
                result["denied_medications"] = deterministic_fallback["denied_medications"]

            return result

        except Exception as e:
            print(f"LLM extraction error: {e}")
            return deterministic_fallback

