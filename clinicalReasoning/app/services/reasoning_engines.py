"""Reasoning engine abstractions and implementations for Agent 5.

Defines the common ReasoningEngine interface and the three hierarchy levels:
Level 1: Local Mistral via Ollama (Primary Orchestration Engine)
Level 2: Groq model fallback for non-orchestration agent flows
Level 3: Deterministic Python Fallback Engine
Plus DevelopmentPlaceholderEngine for testing.
"""

import json
import logging
import time
from abc import ABC, abstractmethod
from typing import Any, List, Optional

import httpx
from pydantic import ValidationError

from app.config import settings
from app.models.input_models import UnifiedClinicalContext
from app.models.output_models import (
    AgentMetadata,
    ClinicalAssessment,
    ClinicalDecisionSupportResponse,
    ClinicalReasoningDetail,
    DecisionSupport,
    ReasoningPolicy,
    SafetyAnalysis,
    TraceabilityLink,
    UncertaintyAssessment,
)
from app.services.prompts import SYSTEM_INSTRUCTIONS, build_clinical_reasoning_prompt

logger = logging.getLogger("agent5.reasoning_engines")

try:
    from google import genai
    from google.genai import types
    GENAI_SDK_AVAILABLE = True
except ImportError:
    GENAI_SDK_AVAILABLE = False


def _strip_markdown_fences(raw_text: str) -> str:
    text = raw_text.strip()
    if text.startswith("```json"):
        text = text[7:]
    elif text.startswith("```"):
        text = text[3:]
    if text.endswith("```"):
        text = text[:-3]
    return text.strip()


def _extract_json_object(raw_text: str, provider_name: str) -> dict:
    text = _strip_markdown_fences(raw_text)
    decoder = json.JSONDecoder()

    try:
        parsed, _ = decoder.raw_decode(text)
        if isinstance(parsed, dict):
            return parsed
    except Exception:
        pass

    for index, character in enumerate(text):
        if character != "{":
            continue
        try:
            parsed, _ = decoder.raw_decode(text[index:])
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            continue

    raise EngineExecutionError(f"Failed to parse {provider_name} response as JSON object.")


class EngineExecutionError(Exception):
    """Raised when a reasoning engine fails during execution or schema validation."""

    pass


class EngineUnavailableError(EngineExecutionError):
    """Raised when a reasoning engine API or local model is unreachable or unavailable."""

    pass


class ReasoningEngine(ABC):
    """Abstract Base Class defining the interface for all clinical reasoning engines."""

    @abstractmethod
    def reason(
        self,
        context: UnifiedClinicalContext,
        policy: Optional[ReasoningPolicy] = None,
    ) -> ClinicalDecisionSupportResponse:
        """Execute clinical reasoning over a validated UnifiedClinicalContext."""
        pass


class GeminiReasoningEngine(ReasoningEngine):
    """Level 1 Primary Reasoning Engine using Gemini 2.5 Flash via google-genai SDK."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: Optional[str] = None,
        timeout: Optional[float] = None,
        client: Optional[Any] = None,
    ) -> None:
        self.api_key = api_key if api_key is not None else settings.gemini_api_key
        self.model_name = model_name or settings.primary_reasoning_model
        self.timeout = timeout or settings.gemini_timeout_seconds
        self._client = client

    def _get_client(self) -> Any:
        if self._client is not None:
            return self._client

        if not GENAI_SDK_AVAILABLE:
            raise EngineUnavailableError("The 'google-genai' SDK package is not installed.")

        if not self.api_key:
            raise EngineUnavailableError(
                "GEMINI_API_KEY environment variable is not configured."
            )

        try:
            return genai.Client(api_key=self.api_key)
        except Exception as exc:
            error_msg = f"Failed to initialize Gemini Client: {str(exc)}"
            logger.error(error_msg)
            raise EngineUnavailableError(error_msg) from exc

    def reason(
        self,
        context: UnifiedClinicalContext,
        policy: Optional[ReasoningPolicy] = None,
    ) -> ClinicalDecisionSupportResponse:
        """Execute semantic clinical reasoning using Gemini 2.5 Flash."""
        client = self._get_client()
        user_prompt = build_clinical_reasoning_prompt(context, policy=policy)

        logger.info(f"Invoking Gemini model '{self.model_name}' for clinical reasoning...")

        config = types.GenerateContentConfig(
            system_instruction=SYSTEM_INSTRUCTIONS,
            response_mime_type="application/json",
            response_schema=ClinicalDecisionSupportResponse,
            temperature=0.1,
        )

        response = None
        last_exc = None
        for attempt in range(2):
            try:
                response = client.models.generate_content(
                    model=self.model_name,
                    contents=user_prompt,
                    config=config,
                )
                break
            except Exception as exc:
                last_exc = exc
                if "503" in str(exc) or "UNAVAILABLE" in str(exc):
                    logger.warning(f"Gemini API returned 503 UNAVAILABLE on attempt {attempt + 1}. Retrying in 1.5s...")
                    time.sleep(1.5)
                else:
                    break

        if response is None:
            error_msg = f"Gemini API request failed: {str(last_exc)}"
            logger.error(error_msg)
            raise EngineExecutionError(error_msg) from last_exc

        if not response or not hasattr(response, "text") or not response.text:
            error_msg = "Gemini API returned an empty or missing response payload."
            logger.error(error_msg)
            raise EngineExecutionError(error_msg)

        raw_text = response.text.strip()

        try:
            json_data = _extract_json_object(raw_text, "Gemini")
        except EngineExecutionError as json_err:
            error_msg = f"Failed to parse Gemini response as JSON: {str(json_err)}"
            logger.error(error_msg)
            raise EngineExecutionError(error_msg) from json_err

        json_data["agent_metadata"] = {
            "agent": "agent_5",
            "model": self.model_name,
            "reasoning_mode": "primary",
            "status": "success",
        }

        try:
            validated_response = ClinicalDecisionSupportResponse.model_validate(json_data)
            logger.info("Gemini clinical reasoning execution and Pydantic validation successful.")
            return validated_response
        except ValidationError as val_err:
            error_msg = f"Gemini response failed Pydantic schema validation: {str(val_err)}"
            logger.error(error_msg)
            raise EngineExecutionError(error_msg) from val_err


class GroqReasoningEngine(ReasoningEngine):
    """Level 2 Secondary Fallback Reasoning Engine: Llama 3.1 8B Instruct via Groq Cloud API.

    Uses Groq's OpenAI-compatible /chat/completions endpoint over httpx.
    Requires GROQ_API_KEY configured in environment. Zero local dependencies.
    """

    GROQ_API_BASE = "https://api.groq.com/openai/v1"

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: Optional[str] = None,
        timeout: Optional[float] = None,
        http_client: Optional[httpx.Client] = None,
    ) -> None:
        self.api_key = api_key if api_key is not None else settings.groq_api_key
        self.model_name = model_name or settings.groq_model
        self.timeout = timeout or settings.groq_timeout_seconds
        self._http_client = http_client
        # Build ordered key pool: GROQ_API_KEY_1..4 first, then fallback single GROQ_API_KEY
        self._api_key_pool: list = []
        if api_key is None:
            # Use pool keys from config first
            self._api_key_pool = list(settings.groq_api_key_pool)
            # Add the single GROQ_API_KEY as final fallback if not already in pool
            if settings.groq_api_key and settings.groq_api_key not in self._api_key_pool:
                self._api_key_pool.append(settings.groq_api_key)
        else:
            # Explicit key passed in — use only that
            self._api_key_pool = [api_key] if api_key else []

    def _get_http_client(self) -> httpx.Client:
        if self._http_client is not None:
            return self._http_client
        return httpx.Client(timeout=self.timeout)

    def reason(
        self,
        context: UnifiedClinicalContext,
        policy: Optional[ReasoningPolicy] = None,
    ) -> ClinicalDecisionSupportResponse:
        """Execute secondary clinical reasoning using Llama 3.1 8B Instruct via Groq Cloud API.
        
        Attempts each API key in the pool (GROQ_API_KEY_1..4, then GROQ_API_KEY) sequentially.
        On 401/429/connection errors, rotates to the next key. Only raises EngineUnavailableError
        if ALL keys are exhausted.
        """
        if not self._api_key_pool:
            raise EngineUnavailableError(
                "No Groq API keys configured. Set GROQ_API_KEY_1..4 or GROQ_API_KEY in .env."
            )

        client = self._get_http_client()
        user_prompt = build_clinical_reasoning_prompt(context, policy=policy)
        endpoint = f"{self.GROQ_API_BASE}/chat/completions"

        last_error: Optional[Exception] = None

        for key_index, current_key in enumerate(self._api_key_pool, start=1):
            logger.info(
                f"Invoking Groq model '{self.model_name}' with API key {key_index}/{len(self._api_key_pool)}..."
            )

            resp_format = {
                "type": "json_schema",
                "json_schema": {
                    "name": "clinical_decision_support_response",
                    "schema": ClinicalDecisionSupportResponse.model_json_schema()
                }
            }
            payload = {
                "model": self.model_name,
                "messages": [
                    {"role": "system", "content": SYSTEM_INSTRUCTIONS},
                    {"role": "user", "content": user_prompt},
                ],
                "temperature": 0.1,
                "response_format": resp_format,
            }

            try:
                response = client.post(endpoint, headers=headers, json=payload)
                if response.status_code == 400 and "json_schema" in response.text:
                    payload["response_format"] = {"type": "json_object"}
                    response = client.post(endpoint, headers=headers, json=payload)
            except httpx.ConnectError as conn_err:
                logger.warning(f"Groq API unreachable with key {key_index}: {conn_err}")
                last_error = EngineUnavailableError(f"Groq API unreachable: {str(conn_err)}")
                continue
            except httpx.TimeoutException as timeout_err:
                logger.warning(f"Groq API timed out with key {key_index}: {timeout_err}")
                last_error = EngineExecutionError(f"Groq API request timed out after {self.timeout}s.")
                continue
            except Exception as exc:
                logger.warning(f"Groq HTTP request failed with key {key_index}: {exc}")
                last_error = EngineExecutionError(f"Groq HTTP request failed: {str(exc)}")
                continue

            if response.status_code == 401:
                logger.warning(f"Groq API key {key_index} authentication failed (HTTP 401). Trying next key...")
                last_error = EngineUnavailableError("Groq API authentication failed (HTTP 401).")
                continue
            elif response.status_code == 429:
                logger.warning(f"Groq API key {key_index} rate limited (HTTP 429). Trying next key...")
                last_error = EngineUnavailableError("Groq API rate limit exceeded (HTTP 429).")
                continue
            elif response.status_code == 404:
                error_msg = f"Groq model '{self.model_name}' not found (HTTP 404)."
                logger.warning(error_msg)
                raise EngineUnavailableError(error_msg)
            elif response.status_code != 200:
                error_msg = f"Groq API returned HTTP {response.status_code}: {response.text[:200]}"
                logger.error(error_msg)
                last_error = EngineExecutionError(error_msg)
                continue

            # --- Successful response: parse and validate ---
            try:
                groq_resp_data = response.json()
                raw_text = groq_resp_data["choices"][0]["message"]["content"].strip()
            except (KeyError, IndexError, Exception) as parse_err:
                error_msg = f"Failed to parse Groq API response structure: {str(parse_err)}"
                logger.error(error_msg)
                raise EngineExecutionError(error_msg) from parse_err

            if not raw_text:
                error_msg = "Groq response content is empty."
                logger.error(error_msg)
                raise EngineExecutionError(error_msg)

            try:
                json_data = _extract_json_object(raw_text, "Groq")
            except EngineExecutionError as json_err:
                error_msg = f"Failed to parse Groq JSON output: {str(json_err)}"
                logger.error(error_msg)
                raise EngineExecutionError(error_msg) from json_err

            json_data["agent_metadata"] = {
                "agent": "agent_5",
                "model": self.model_name,
                "reasoning_mode": "primary" if settings.primary_reasoning_provider == "groq" else "fallback_model",
                "status": "success",
            }

            try:
                validated_response = ClinicalDecisionSupportResponse.model_validate(json_data)
                logger.info(
                    f"Groq fallback reasoning succeeded with API key {key_index}/{len(self._api_key_pool)}."
                )
                return validated_response
            except ValidationError as val_err:
                error_msg = f"Groq response failed Pydantic schema validation: {str(val_err)}"
                logger.error(error_msg)
                raise EngineExecutionError(error_msg) from val_err

        # All keys exhausted
        raise EngineUnavailableError(
            f"All {len(self._api_key_pool)} Groq API keys exhausted. Last error: {last_error}"
        )


class OllamaReasoningEngine(ReasoningEngine):
    """Offline Local Fallback Reasoning Engine using Ollama.

    Uses Ollama's /api/generate endpoint. Since 7B models struggle with massive 
    complex JSON schemas, we ask Mistral for a plain-text clinical assessment 
    and construct the Pydantic model deterministically around it.
    """

    OLLAMA_API_BASE = f"{settings.ollama_base_url.rstrip('/')}/api"

    def __init__(
        self,
        model_name: Optional[str] = None,
        timeout: Optional[float] = None,
        http_client: Optional[httpx.Client] = None,
    ) -> None:
        self.model_name = model_name or settings.ollama_model or settings.primary_reasoning_model
        self.timeout = timeout or settings.ollama_timeout_seconds
        self._http_client = http_client

    def _get_http_client(self) -> httpx.Client:
        if self._http_client is not None:
            return self._http_client
        return httpx.Client(timeout=self.timeout)

    def reason(
        self,
        context: UnifiedClinicalContext,
        policy: Optional[ReasoningPolicy] = None,
    ) -> ClinicalDecisionSupportResponse:
        client = self._get_http_client()
        endpoint = f"{self.OLLAMA_API_BASE}/generate"
        
        # Build a highly simplified prompt for Mistral 7B
        findings = ", ".join([mf.finding for mf in context.merged_findings])
        evidence = " ".join([se.finding for se in context.supporting_evidence])
        
        prompt = (
            f"You are a medical AI assistant. Analyze the following clinical data and write a concise, "
            f"professional clinical interpretation paragraph (max 3-4 sentences) outlining the most likely diagnosis "
            f"and clinical significance.\n\n"
            f"Key Findings: {findings}\n"
            f"Evidence: {evidence}\n\n"
            f"Interpretation:"
        )

        logger.info(f"Invoking local Ollama model '{self.model_name}' (Plain text mode)...")

        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.2
            }
        }

        try:
            response = client.post(endpoint, json=payload)
            response.raise_for_status()
            resp_data = response.json()
            mistral_text = resp_data.get("response", "").strip()
        except Exception as exc:
            logger.warning(f"Ollama execution failed: {exc}")
            raise EngineExecutionError(f"Ollama execution failed: {str(exc)}")
            
        if not mistral_text:
            raise EngineExecutionError("Mistral returned empty text.")

        # Construct the complex JSON safely in Python
        key_findings = [mf.finding for mf in context.merged_findings]
        recommended_actions = [
            f"Review priority finding from {ep.primary_source}: {ep.finding}"
            for ep in context.evidence_priority
        ]
        if not recommended_actions:
            recommended_actions = ["Conduct comprehensive clinical evaluation."]

        return ClinicalDecisionSupportResponse(
            clinical_assessment=ClinicalAssessment(
                primary_interpretation=mistral_text,
                clinical_significance="Moderate",
                differential_considerations=key_findings[:3] or ["None"],
            ),
            reasoning=ClinicalReasoningDetail(
                key_findings=key_findings or ["None"],
                supporting_factors=["Mistral 7B Analysis applied to context."],
                conflicting_factors=[ce.description for ce in context.conflicting_evidence],
                reasoning_summary="AI synthesis completed by Mistral offline.",
            ),
            decision_support=DecisionSupport(
                recommended_actions=recommended_actions,
                additional_information_needed=["Verify AI-generated interpretation clinically."],
                priority_level="Routine",
            ),
            safety=SafetyAnalysis(
                safety_flags=[],
                contraindications_or_concerns=[],
            ),
            uncertainty=UncertaintyAssessment(
                confidence_score=context.confidence_score,
                fusion_confidence=context.confidence_score,
                reasoning_confidence=context.confidence_score,
                uncertainty_factors=["Analysis performed by lightweight 7B model offline."],
            ),
            traceability=[],
            agent_metadata=AgentMetadata(
                agent="agent_5",
                model=self.model_name,
                reasoning_mode="fallback_model",
                status="success",
            ),
        )
class DeterministicFallbackEngine(ReasoningEngine):
    """Level 3 Final Safety Mechanism: Rule-Based Deterministic Python Engine (No LLM)."""

    def reason(
        self,
        context: UnifiedClinicalContext,
        policy: Optional[ReasoningPolicy] = None,
    ) -> ClinicalDecisionSupportResponse:
        """Generate conservative rule-based decision support from validated Agent 4 context adhering to ReasoningPolicy."""
        key_findings = [mf.finding for mf in context.merged_findings]
        supporting_factors = [se.finding for se in context.supporting_evidence]
        conflicting_factors = [ce.description for ce in context.conflicting_evidence]

        safety_flags: List[str] = []
        contraindications: List[str] = []
        for ce in context.conflicting_evidence:
            safety_flags.append(f"WARNING: {ce.type} detected regarding '{ce.finding}'.")
            contraindications.append(ce.description)

        recommended_actions = [
            f"Review priority finding from {ep.primary_source}: {ep.finding}"
            for ep in context.evidence_priority
        ]
        if not recommended_actions:
            recommended_actions = ["Conduct comprehensive clinical evaluation based on merged findings."]

        traceability_links = [
            TraceabilityLink(
                reasoning_item=st.finding,
                supported_by_findings=[st.finding],
                upstream_sources=st.sources,
            )
            for st in context.source_traceability
        ]

        cert_label = policy.certainty_level if policy else "LIKELY"

        uncertainty_factors = [
            "Output generated via deterministic fallback engine without LLM semantic synthesis."
        ]
        if policy and policy.uncertainty_required:
            uncertainty_factors.append("Explicit uncertainty required per ReasoningPolicy due to evidence conflicts or gaps.")

        return ClinicalDecisionSupportResponse(
            clinical_assessment=ClinicalAssessment(
                primary_interpretation=f"Deterministic Context Assessment ({cert_label}): {context.fusion_summary}",
                clinical_significance="High" if context.conflicting_evidence else "Moderate",
                differential_considerations=[mf.finding for mf in context.merged_findings[:3]],
            ),
            reasoning=ClinicalReasoningDetail(
                key_findings=key_findings,
                supporting_factors=supporting_factors,
                conflicting_factors=conflicting_factors,
                reasoning_summary="Conservative deterministic reasoning generated directly from Agent 4 fused context.",
            ),
            decision_support=DecisionSupport(
                recommended_actions=recommended_actions,
                additional_information_needed=[
                    "Missing patient demographics and specific lab/vital values in supplied context."
                ],
                priority_level="Urgent" if context.conflicting_evidence else "Routine",
            ),
            safety=SafetyAnalysis(
                safety_flags=safety_flags,
                contraindications_or_concerns=contraindications,
            ),
            uncertainty=UncertaintyAssessment(
                confidence_score=max(0.0, min(1.0, context.confidence_score * 0.9)),
                fusion_confidence=context.confidence_score,
                reasoning_confidence=max(0.0, min(1.0, context.confidence_score * 0.9)),
                uncertainty_factors=uncertainty_factors,
            ),
            traceability=traceability_links,
            agent_metadata=AgentMetadata(
                agent="agent_5",
                model="deterministic-fallback",
                reasoning_mode="deterministic_fallback",
                status="fallback_applied",
            ),
        )


class DevelopmentPlaceholderEngine(ReasoningEngine):
    """Development Placeholder Engine used during API testing prior to live LLM execution."""

    def reason(
        self,
        context: UnifiedClinicalContext,
        policy: Optional[ReasoningPolicy] = None,
    ) -> ClinicalDecisionSupportResponse:
        """Return valid ClinicalDecisionSupportResponse indicating development placeholder execution."""
        return ClinicalDecisionSupportResponse(
            clinical_assessment=ClinicalAssessment(
                primary_interpretation=f"Development Placeholder Analysis: {context.fusion_summary}",
                clinical_significance="Moderate",
                differential_considerations=[m.finding for m in context.merged_findings[:3]],
            ),
            reasoning=ClinicalReasoningDetail(
                key_findings=[m.finding for m in context.merged_findings],
                supporting_factors=[s.finding for s in context.supporting_evidence],
                conflicting_factors=[c.description for c in context.conflicting_evidence],
                reasoning_summary="Development placeholder reasoning confirming architecture execution.",
            ),
            decision_support=DecisionSupport(
                recommended_actions=[
                    f"Review finding: {p.finding}" for p in context.evidence_priority[:2]
                ],
                additional_information_needed=["Pending live LLM integration."],
                priority_level="Routine",
            ),
            safety=SafetyAnalysis(
                safety_flags=[c.description for c in context.conflicting_evidence],
                contraindications_or_concerns=[c.finding for c in context.conflicting_evidence],
            ),
            uncertainty=UncertaintyAssessment(
                confidence_score=context.confidence_score,
                fusion_confidence=context.confidence_score,
                reasoning_confidence=context.confidence_score,
                uncertainty_factors=["Development placeholder engine active."],
            ),
            traceability=[
                TraceabilityLink(
                    reasoning_item=st.finding,
                    supported_by_findings=[st.finding],
                    upstream_sources=st.sources,
                )
                for st in context.source_traceability
            ],
            agent_metadata=AgentMetadata(
                agent="agent_5",
                model="development-placeholder",
                reasoning_mode="development_placeholder",
                status="not_implemented",
            ),
        )
