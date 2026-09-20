"""Authoritative Patient State Manager for Agent 1.

Maintains the single authoritative source of truth for the patient's clinical state:
- Distinguishes UNKNOWN vs NONE_REPORTED vs PRESENT for medical history and medications.
- Supports both global and entity-specific negation (e.g. negating hypertension specifically without clearing unrelated history).
- Keeps medical history (diagnoses/conditions) and medications (drugs/treatments) completely separate.
- Distinguishes positive symptoms from explicitly denied symptoms.
- Recalculates missing information dynamically from the UPDATED patient state.
- Generates the next prioritized clarification question strictly avoiding redundant questions for already-provided fields.
"""

from typing import Dict, Any, List, Optional, Tuple


def create_empty_patient_state() -> Dict[str, Any]:
    """Create a pristine, structured patient state."""
    return {
        "age": None,
        "gender": None,
        "medical_history": [],
        "medical_history_status": "UNKNOWN",  # UNKNOWN | NONE_REPORTED | PRESENT
        "medications": [],
        "medications_status": "UNKNOWN",      # UNKNOWN | NONE_REPORTED | PRESENT
        "symptoms": [],
        "denied_symptoms": [],
        "measurements": {},
        "duration": None,
        "severity": None,
        "other_information": [],
        "relationships": [],
    }


def merge_patient_state(
    current_state: Dict[str, Any],
    new_info: Dict[str, Any]
) -> Dict[str, Any]:
    """Merge newly extracted information into the authoritative patient state with idempotent set reconciliation."""
    merged = {
        "age": current_state.get("age"),
        "gender": current_state.get("gender"),
        "medical_history": list(current_state.get("medical_history", [])),
        "medical_history_status": current_state.get("medical_history_status", "UNKNOWN"),
        "medications": list(current_state.get("medications", [])),
        "medications_status": current_state.get("medications_status", "UNKNOWN"),
        "symptoms": list(current_state.get("symptoms", [])),
        "denied_symptoms": list(current_state.get("denied_symptoms", [])),
        "measurements": dict(current_state.get("measurements", {})),
        "duration": current_state.get("duration"),
        "severity": current_state.get("severity"),
        "other_information": list(current_state.get("other_information", [])),
        "relationships": list(current_state.get("relationships", [])),
    }

    # 1. Age & Gender (clean update / correction)
    if new_info.get("age"):
        merged["age"] = str(new_info["age"])
    if new_info.get("gender"):
        merged["gender"] = new_info["gender"]

    # 2. Medical History (Global vs. Entity-Specific Negation)
    new_hist_status = new_info.get("medical_history_status")
    denied_history = new_info.get("denied_history", [])

    if new_hist_status == "NONE_REPORTED":
        # Global denial: clear medical history
        merged["medical_history_status"] = "NONE_REPORTED"
        merged["medical_history"] = []
    else:
        hist_set = set(merged["medical_history"])
        # Remove entity-specific denied conditions
        for denied_cond in denied_history:
            hist_set.discard(denied_cond)

        # Add newly reported conditions
        if new_info.get("medical_history"):
            hist_set.update(new_info["medical_history"])
            merged["medical_history_status"] = "PRESENT"
        elif hist_set:
            merged["medical_history_status"] = "PRESENT"
        elif new_hist_status in ["PRESENT", "KNOWN"]:
            merged["medical_history_status"] = "PRESENT"

        merged["medical_history"] = sorted(list(hist_set))

    # 3. Medications (Global vs. Entity-Specific Negation)
    new_meds_status = new_info.get("medications_status")
    denied_medications = new_info.get("denied_medications", [])

    if new_meds_status == "NONE_REPORTED":
        # Global denial: clear medications
        merged["medications_status"] = "NONE_REPORTED"
        merged["medications"] = []
    else:
        meds_set = set(merged["medications"])
        # Remove entity-specific denied medications
        for denied_med in denied_medications:
            meds_set.discard(denied_med)

        # Add newly reported medications
        if new_info.get("medications"):
            meds_set.update(new_info["medications"])
            merged["medications_status"] = "PRESENT"
        elif meds_set:
            merged["medications_status"] = "PRESENT"
        elif new_meds_status in ["PRESENT", "KNOWN"]:
            merged["medications_status"] = "PRESENT"

        merged["medications"] = sorted(list(meds_set))

    # 4. Denied Symptoms
    if new_info.get("denied_symptoms"):
        denied_set = set(merged["denied_symptoms"])
        denied_set.update(new_info["denied_symptoms"])
        # If any newly asserted positive symptom is in denied_symptoms, remove it from denied
        if new_info.get("symptoms"):
            denied_set.difference_update(new_info["symptoms"])
        merged["denied_symptoms"] = sorted(list(denied_set))

    # 5. Positive Symptoms (exclude any currently denied symptom)
    if new_info.get("symptoms"):
        sym_set = set(merged["symptoms"])
        sym_set.update(new_info["symptoms"])
        sym_set.difference_update(merged["denied_symptoms"])
        merged["symptoms"] = sorted(list(sym_set))
    else:
        merged["symptoms"] = [s for s in merged["symptoms"] if s not in merged["denied_symptoms"]]

    # 6. Measurements (idempotent overwrite for updated vital keys)
    if new_info.get("measurements"):
        merged["measurements"].update(new_info["measurements"])

    # 7. Duration & Severity
    if new_info.get("duration"):
        merged["duration"] = new_info["duration"]
    if new_info.get("severity"):
        merged["severity"] = new_info["severity"]

    # 8. Other Information & Relationships
    if new_info.get("other_information"):
        other_set = set(merged["other_information"])
        other_set.update(new_info["other_information"])
        merged["other_information"] = sorted(list(other_set))

    if new_info.get("relationships"):
        rel_set = set(merged["relationships"])
        rel_set.update(new_info["relationships"])
        merged["relationships"] = sorted(list(rel_set))

    return merged


def calculate_missing_information(state: Dict[str, Any]) -> List[str]:
    """Calculate missing clinical fields strictly from the authoritative state."""
    missing = []

    if not state.get("age"):
        missing.append("age")
    if not state.get("gender"):
        missing.append("gender")
    if state.get("medical_history_status") == "UNKNOWN":
        missing.append("medical history, previous diagnoses, treatments, and medications")

    # Clinical details
    if not state.get("symptoms") and not state.get("denied_symptoms"):
        missing.append("symptoms or chief complaint")
    elif "fever" in state.get("symptoms", []) and "temperature" not in state.get("measurements", {}):
        missing.append("temperature measurement")

    return missing


def determine_next_clarification(
    state: Dict[str, Any],
    pending_question: Optional[str] = None
) -> Tuple[Optional[str], bool, List[str]]:
    """Determine next prioritized clarification question based on UPDATED state.

    Priority Hierarchy (strictly for genuinely missing information):
    1. Age (if missing)
    2. Gender (if missing)
    3. Medical History & Medications (if status UNKNOWN and not already queried)
    4. Current Medications (if history known but medication status still UNKNOWN and not already queried)
    5. Chief Complaint / Symptoms (if completely empty)
    6. Temperature (if fever present in symptoms but temperature missing from measurements)
    7. Duration (if symptoms present but duration missing)
    """
    missing_info = calculate_missing_information(state)
    pending_lower = pending_question.lower() if pending_question else ""

    # 1. Mandatory Priority: Age (only if not already provided)
    if not state.get("age"):
        return "What is the patient's age?", True, ["age"]

    # 2. Mandatory Priority: Gender (only if not already provided)
    if not state.get("gender"):
        return "What is the patient's gender?", True, ["gender"]

    # 3. Mandatory Priority: Medical History & Medications (only if UNKNOWN and not previously asked)
    if state.get("medical_history_status") == "UNKNOWN":
        # Avoid infinite loop if pending question was already asking about medical history
        already_asked_hist = any(k in pending_lower for k in ["medical condition", "medical history", "previous diagnos"])
        if not already_asked_hist:
            return (
                "Do you have any existing medical conditions or previous diagnoses, "
                "and are you currently taking any medications or treatments?",
                True,
                ["medical history, previous diagnoses, treatments, and medications"]
            )

    # 4. Medications if medical history was reported but medication status still unknown
    if state.get("medications_status") == "UNKNOWN":
        already_asked_meds = any(k in pending_lower for k in ["taking any", "prescription or over-the-counter", "medication"])
        if not already_asked_meds:
            return (
                "Are you currently taking or have you recently taken any prescription or over-the-counter medications?",
                True,
                ["current medications and treatments"]
            )

    # 5. Chief complaint / Symptoms if completely empty
    if not state.get("symptoms") and not state.get("denied_symptoms"):
        already_asked_sym = "symptom" in pending_lower or "complaint" in pending_lower
        if not already_asked_sym:
            return "What are the patient's main symptoms or chief complaint?", True, ["symptoms"]

    # 6. Specific Measurement: Fever without temperature
    if "fever" in state.get("symptoms", []) and "temperature" not in state.get("measurements", {}):
        already_asked_temp = "temperature" in pending_lower or "temp" in pending_lower
        if not already_asked_temp:
            return "What is the patient's current temperature?", True, ["temperature measurement"]

    # 7. Duration if symptoms present but duration missing
    if state.get("symptoms") and not state.get("duration"):
        already_asked_dur = "how long" in pending_lower or "duration" in pending_lower
        if not already_asked_dur:
            return "How long have these symptoms been present?", True, ["duration of symptoms"]

    # Case is complete or all mandatory fields addressed
    return None, False, []

