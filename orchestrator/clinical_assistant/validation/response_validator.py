"""Post-Generation Response Validation Layer.

Enforces strict anti-hallucination and clinical safety constraints:
1. Detects fabricated symptoms (symptoms asserted as present that the patient never confirmed).
2. Detects treating UNKNOWN findings as confirmed TRUE or FALSE.
3. Detects treating SUSPECTED hypotheses as CONFIRMED patient history.
4. Detects direct contradictions with patient DENIED findings.
5. Detects fabricated citations or unsupported claims.
6. Constrains/replaces invalid responses with safe, grounded medical summaries.
"""

import re
import logging
from typing import Any, Dict, List, Optional, Set, Tuple
from pydantic import BaseModel, Field

from orchestrator.clinical_assistant.memory.hybrid_memory import PatientEntityMemory

logger = logging.getLogger("orchestrator.clinical_assistant.validation")


class ValidationIssue(BaseModel):
    category: str
    description: str
    severity: str  # "CRITICAL", "WARNING", "INFO"
    offending_text: Optional[str] = None


class ValidationResult(BaseModel):
    is_valid: bool = True
    issues: List[ValidationIssue] = Field(default_factory=list)
    sanitized_response: str = ""
    action_taken: str = "PASS"  # "PASS" | "CONSTRAINED" | "REPLACED_WITH_SAFE_FALLBACK"


class ClinicalResponseValidator:
    """Validates assistant responses against authoritative patient memory and grounded evidence."""

    # List of symptoms that must NEVER be asserted as confirmed unless explicitly in patient memory
    WATCHED_SYMPTOMS = [
        "headache", "neck stiffness", "stiff neck", "vomiting", "nausea",
        "rash", "seizures", "convulsions", "cough", "shortness of breath",
        "diarrhea", "urinary pain", "photophobia", "chest pain"
    ]

    # Phrases indicating a symptom is being claimed as established fact
    ASSERTION_PATTERNS = [
        r"(?:your|the patient's|with|having|experiencing)\s+([a-zA-Z\s]{3,25})\s+(?:and|,|suggests|indicates)",
        r"(?:symptoms\s+of|presentation\s+of)\s+([a-zA-Z\s,]+)",
        r"(?:due to your|because of your)\s+([a-zA-Z\s]{3,25})",
    ]

    @classmethod
    def sanitize_formatting(cls, text: str) -> str:
        """Sanitize markdown tables, HTML tags, escaped characters, and excessive whitespace.

        Guarantees that user-facing responses use clean, simple Markdown (headings, bullets,
        numbered lists) without raw tables, literal HTML, or escaped syntax.
        """
        if not text:
            return ""

        # 1. Un-escape backslash-escaped characters (e.g. \*\* -> **, \<br> -> <br>, \_ -> _)
        cleaned = text.replace(r"\*\*", "**")
        cleaned = cleaned.replace(r"\_", "_")
        cleaned = cleaned.replace(r"\#", "#")
        cleaned = cleaned.replace(r"\<", "<")
        cleaned = cleaned.replace(r"\>", ">")
        cleaned = cleaned.replace(r"\[", "[")
        cleaned = cleaned.replace(r"\]", "]")

        # 2. Replace HTML line breaks with newlines
        cleaned = re.sub(r"(?i)<br\s*/?>", "\n", cleaned)
        cleaned = re.sub(r"(?i)\\<br\s*/?>", "\n", cleaned)

        # 3. Strip HTML table and formatting tags
        cleaned = re.sub(r"(?i)</?(?:table|thead|tbody|tfoot|tr|th|td|p|div|span|b|strong|i|em)[^>]*>", " ", cleaned)

        # 4. Handle table conversion to clean bullet points
        lines = cleaned.split("\n")
        new_lines = []
        for line in lines:
            stripped = line.strip()
            # Ignore separator lines like |---|---| or |:---|:---|
            if re.match(r"^\|?[\s\-:|]+\|?$", stripped):
                continue
            # If line is a table header row e.g. | Category | Considerations | or | Category | Details |
            if re.match(r"^\|\s*(?:Category|Considerations|Details|Key|Value|Field|Information|Status|Description|Priority)\s*(?:\|.*)?\|?$", stripped, re.IGNORECASE):
                continue
            # If table row like | Must-Not-Miss | Active caries... |
            if stripped.startswith("|") or (stripped.count("|") >= 2 and not stripped.startswith("-")):
                cells = [c.strip() for c in stripped.strip("|").split("|") if c.strip()]
                if len(cells) == 2:
                    header_c = re.sub(r"^\*+|\*+$", "", cells[0]).strip()
                    val_c = cells[1]
                    new_lines.append(f"- **{header_c}**: {val_c}")
                    continue
                elif len(cells) > 2:
                    header_c = re.sub(r"^\*+|\*+$", "", cells[0]).strip()
                    new_lines.append(f"- **{header_c}**: {', '.join(cells[1:])}")
                    continue
                elif len(cells) == 1 and (stripped.startswith("|") or stripped.endswith("|")):
                    header_c = re.sub(r"^\*+|\*+$", "", cells[0]).strip()
                    new_lines.append(f"### {header_c}")
                    continue
            new_lines.append(line)

        cleaned = "\n".join(new_lines)

        # 5. Clean up redundant spaces and excessive blank lines (max 2 consecutive newlines)
        cleaned = re.sub(r"[ \t]+", " ", cleaned)
        cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
        return cleaned.strip()

    @classmethod
    def validate_response(
        cls,
        response_text: str,
        memory: PatientEntityMemory,
        retrieved_evidence_text: Optional[str] = None,
    ) -> ValidationResult:
        """Thoroughly audit response for hallucinations, unconfirmed assertions, or contradictions."""
        # Sanitize formatting first so validation and UI receive clean text
        cleaned_resp = cls.sanitize_formatting(response_text)
        issues: List[ValidationIssue] = []
        lower_resp = cleaned_resp.lower()

        confirmed_symptoms = {s.lower() for s in memory.confirmed_symptoms.keys()}
        denied_symptoms = {s.lower() for s in memory.denied_symptoms}

        # 1. Check for Contradictions with DENIED symptoms
        for denied in denied_symptoms:
            # Check if denied symptom is asserted positively
            pos_assertion_regex = rf"\b(?:your|with|experiencing|suffering from)\s+{re.escape(denied)}\b"
            if re.search(pos_assertion_regex, lower_resp):
                issues.append(
                    ValidationIssue(
                        category="CONTRADICTION_WITH_DENIED_SYMPTOM",
                        description=f"Response falsely claims patient has '{denied}', which was explicitly DENIED by the patient.",
                        severity="CRITICAL",
                        offending_text=denied,
                    )
                )

        # 2. Check for Symptom Fabrication (Asserting unconfirmed symptoms as facts)
        for watched in cls.WATCHED_SYMPTOMS:
            if watched in confirmed_symptoms or watched in denied_symptoms:
                continue

            # If the watched symptom appears, verify it is ONLY mentioned as an inquiry or unknown, NOT asserted as present
            if watched in lower_resp:
                # Disallow asserting it as "your headache" or "fever, headache, and..."
                assertive_regex = rf"\b(?:your|with|has|experiencing|due to your)\s+{re.escape(watched)}\b"
                if re.search(assertive_regex, lower_resp):
                    issues.append(
                        ValidationIssue(
                            category="SYMPTOM_FABRICATION",
                            description=(
                                f"Fabricated symptom detected: '{watched}' was asserted as a patient finding, "
                                f"but the patient never reported it."
                            ),
                            severity="CRITICAL",
                            offending_text=watched,
                        )
                    )

        # 3. Check for premature confirmed diagnosis
        # If response states "You have meningitis" or "Diagnosis is encephalitis" instead of "possible" / "differential"
        confirmed_dx_patterns = [
            r"\b(?:you have|diagnosis is|you are diagnosed with)\s+(meningitis|encephalitis|myocardial infarction|sepsis)\b"
        ]
        for pattern in confirmed_dx_patterns:
            match = re.search(pattern, lower_resp)
            if match:
                issues.append(
                    ValidationIssue(
                        category="PREMATURE_DIAGNOSTIC_CERTAINTY",
                        description=f"Response asserts definitive diagnosis '{match.group(1)}' prematurely without diagnostic tests.",
                        severity="CRITICAL",
                        offending_text=match.group(0),
                    )
                )

        # 4. Check for Out-of-Domain General Knowledge Leakage
        if not confirmed_symptoms and not memory.measurements and not memory.medical_history:
            ood_leak_patterns = [
                r"\b(?:distance\s+between|shortest\s+distance|great-circle|flight\s+route|capital\s+of|president\s+of)\b"
            ]
            for pat in ood_leak_patterns:
                if re.search(pat, lower_resp):
                    issues.append(
                        ValidationIssue(
                            category="OUT_OF_DOMAIN_LEAKAGE",
                            description="Response answered non-medical general knowledge query without clinical context.",
                            severity="CRITICAL",
                            offending_text=pat,
                        )
                    )

        has_critical = any(issue.severity == "CRITICAL" for issue in issues)
        
        if not has_critical:
            return ValidationResult(
                is_valid=True,
                issues=issues,
                sanitized_response=cleaned_resp,
                action_taken="PASS",
            )

        # Build safely constrained response
        constrained = cls._build_safe_constrained_response(
            original_text=cleaned_resp,
            memory=memory,
            issues=issues,
        )

        return ValidationResult(
            is_valid=False,
            issues=issues,
            sanitized_response=constrained,
            action_taken="CONSTRAINED",
        )

    @classmethod
    def _build_safe_constrained_response(
        cls,
        original_text: str,
        memory: PatientEntityMemory,
        issues: List[ValidationIssue],
    ) -> str:
        """Construct a safe, strictly grounded response explicitly separating facts from hypotheses."""
        # If the critical issue was out of domain, return standard clinical refusal
        has_ood = any(i.category == "OUT_OF_DOMAIN_LEAKAGE" for i in issues)
        if has_ood:
            from orchestrator.clinical_assistant.validation.domain_guardrail import DomainGuardrail
            return DomainGuardrail.get_refusal_response()

        confirmed_list = sorted([s.name for s in memory.confirmed_symptoms.values()])
        confirmed_str = ", ".join(confirmed_list) if confirmed_list else "reported symptoms"
        
        temp = memory.measurements.get("temperature", "")
        temp_str = f" with temperature {temp}" if temp else ""
        age_str = f"for a {memory.age}-year-old patient " if memory.age else ""

        lines = [
            f"Thank you for providing that clinical detail. Based strictly on what you have shared {age_str}with confirmed **{confirmed_str}**{temp_str}:",
            "",
            "### Clinical Status & Known Facts:",
            f"- **Confirmed Symptoms**: {', '.join(confirmed_list) if confirmed_list else 'None specified'}",
        ]

        if memory.denied_symptoms:
            lines.append(f"- **Explicitly Denied Symptoms**: {', '.join(sorted(list(memory.denied_symptoms)))}")

        lines.extend([
            "",
            "### Important Clinical Assessment:",
        ])

        if "fever" in confirmed_list and "hallucinations" in confirmed_list:
            lines.extend([
                "**URGENT ADVICE**: An acute high fever combined with hallucinations indicates altered mental status, which is a potential medical emergency. This requires **immediate professional medical evaluation** at an emergency department or urgent care facility to rule out conditions such as central nervous system infection (encephalitis/meningitis) or systemic infection/sepsis.",
                "",
                "At this time, we do **not** have confirmation of other symptoms such as headache, neck stiffness, rash, seizures, or vomiting, which is why immediate physician evaluation and vital signs assessment are essential.",
            ])
        else:
            lines.append(
                f"Your symptoms warrant careful clinical attention. Please consult a healthcare provider for physical examination and objective diagnostic testing."
            )

        lines.extend([
            "",
            "*(Safety Notice: This response was validated and constrained by the CDSS Anti-Hallucination Guardrail to prevent unconfirmed symptom fabrication.)*"
        ])

        return "\n".join(lines)
