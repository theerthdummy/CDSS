"""LLM semantic enhancement module using Groq GPT-OSS 120B with Ollama fallback handling."""

import json
import time
from typing import Any, Dict, Optional, Tuple

import httpx

from app.config import settings
from app.utils import logger

SYSTEM_PROMPT = """You are a Semantic Clinical Data Fusion Assistant in an Adaptive Multi-Agent Clinical Decision Support System.

YOUR ROLE:
You receive biomedical retrieval outputs from upstream agents (Agent 2 Biomedical RAG & Knowledge Graph, and Agent 3 Latest Medical Evidence) along with an intermediate rule-based baseline.
Your goal is to perform genuine multi-source clinical data fusion: synthesizing, semantically refining, organizing, deduplicating, and standardizing the evidence without discarding any source.

CRITICAL FUSION CONSTRAINTS:
1. GENUINE MULTI-SOURCE FUSION:
   - Independently process Agent 2 evidence and Agent 3 evidence.
   - Extract core claims and concepts from BOTH Agent 2 and Agent 3.
   - Identify findings supported by BOTH agents (set sources: ["Agent 2", "Agent 3"], upstream_agents: ["Agent 2", "Agent 3"]).
   - Identify findings unique to Agent 2 (sources: ["Agent 2"], upstream_agents: ["Agent 2"]).
   - Identify findings unique to Agent 3 (sources: ["Agent 3"], upstream_agents: ["Agent 3"]).
   - NEVER automatically treat Agent 3 as superior or discard Agent 2. Both streams are essential.

2. MEDICAL-TERM NORMALIZATION (CONCISE CONCEPTS ONLY):
   - The 'normalized_medical_terms' array MUST contain concise medical/biomedical concepts (e.g., ["Aspirin", "Acute Coronary Syndrome", "Mortality"] or ["Biomedical Association", "Therapeutic Intervention", "Outcome Analysis"]).
   - NEVER convert full sentences, study conclusions, or claims into normalized medical terms.
   - Standardize abbreviations (e.g., 'AMI' -> 'Acute Myocardial Infarction', 'ECG' -> 'Electrocardiogram (ECG)').

3. SEPARATE FINDING, EVIDENCE, AND SOURCE:
   - 'merged_findings': Concise finding/concept name with accurate 'sources' list.
   - 'normalized_medical_terms': Clean, concise biomedical concept terms.
   - 'supporting_evidence': Specific literature statements/findings with 'source_attribution' explicitly noting the originating agent ("Agent 2", "Agent 3", or "Agent 2, Agent 3").

4. CONFLICT DETECTION:
   - Explicitly compare claims between Agent 2 and Agent 3.
   - If Agent 2 and Agent 3 make contradictory claims regarding the same intervention, outcome, or association (e.g., one states positive effect/benefit, the other states no effect/negative/contraindication), populate 'conflicting_evidence' with 'type', 'finding', and 'description' explaining the opposing statements.
   - If no contradiction exists, keep 'conflicting_evidence' empty [].

5. EMPTY KNOWLEDGE GRAPH EVIDENCE:
   - An empty 'kg_evidence': [] indicates knowledge graph data was not provided/unavailable. It is NOT negative evidence. Do NOT invent graph relationships.

6. CALIBRATED CONFIDENCE:
   - Confidence must strictly reflect evidence quality, agreement across agents, completeness, and conflicts:
     * High (0.85 - 0.95): Strong agreement between Agent 2 and Agent 3 with complete, consistent evidence.
     * Moderate (0.70 - 0.84): Valid single-source evidence or minor limitations.
     * Low (< 0.70): Inconclusive/insufficient study findings, explicit uncertainty/limitations, unmitigated conflicts, or weak source support.
   - Do NOT automatically assign 0.90+ simply because an article is present.

7. JSON FORMAT:
   - Return valid JSON matching the EXACT UnifiedClinicalContext schema below with NO markdown wrappers or extra text.

JSON RESPONSE SCHEMA:
{
  "unified_context": {
    "merged_findings": [
      {
        "finding": "concise finding or concept name",
        "sources": ["Agent 2", "Agent 3"]
      }
    ],
    "normalized_medical_terms": ["Concise Term 1", "Concise Term 2"],
    "supporting_evidence": [
      {
        "finding": "full evidence statement from literature",
        "source_attribution": "Agent 2 / Agent 3 (Database Name)"
      }
    ],
    "conflicting_evidence": [
      {
        "type": "Evidence Contradiction / Treatment Conflict",
        "finding": "name of conflicting topic or intervention",
        "description": "detailed explanation of opposing evidence statements"
      }
    ],
    "evidence_priority": [
      {
        "finding": "finding name",
        "priority_score": 0.95,
        "primary_source": "Biomedical RAG / Latest Medical Evidence"
      }
    ],
    "source_traceability": [
      {
        "finding": "finding name",
        "sources": ["PubMed", "Biomedical RAG"],
        "upstream_agents": ["Agent 2", "Agent 3"]
      }
    ],
    "fusion_summary": "concise factual synthesis of fused evidence, agreement/disagreement, and provenance",
    "confidence_score": 0.88
  }
}
"""


def _clean_json_markdown_fences(content: str) -> str:
    """Remove markdown code block fences (e.g. ```json ... ```) if present in raw LLM output."""
    cleaned = content.strip()
    if cleaned.startswith("```"):
        lines = cleaned.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        cleaned = "\n".join(lines).strip()
    return cleaned


def enhance_clinical_context_with_llama(
    deterministic_context_dict: Dict[str, Any],
    raw_request_dict: Optional[Dict[str, Any]] = None,
    correlation_id: str = "N/A"
) -> Optional[Tuple[Dict[str, Any], Dict[str, Any]]]:
    """Send intermediate rule-based clinical fusion context and raw upstream evidence to LLM for semantic enhancement.

    Args:
        deterministic_context_dict: Rule-based baseline fusion context dictionary.
        raw_request_dict: Optional raw input payload containing agent2_output and agent3_output.
        correlation_id: Request correlation ID for tracing.

    Returns:
        Tuple of (parsed_json, execution_stats) if successful, or None if LLM is unconfigured, disabled,
        or encounters ANY error (enabling automatic rule-based fallback).
    """
    if not settings.llm_enabled:
        reason = "LLM enhancement is disabled in configuration settings (LLM_ENABLED=False)."
        logger.info(f"[{correlation_id}] {reason}")
        return None

    api_key = settings.llm_api_key
    if not settings.groq_api_key_pool and (not api_key or api_key == "mock_key_or_set_real_api_key"):
        reason = (
            f"LLM API key is unconfigured or set to default mock key (Provider: '{settings.llm_provider}', "
            f"Base URL: '{settings.llm_base_url}'). Set GROQ_API_KEY to enable Groq primary fusion."
        )
        logger.warning(f"[{correlation_id}] {reason}")
        return None

    prompt_parts = []
    if raw_request_dict:
        prompt_parts.append(f"""[UPSTREAM EVIDENCE PAYLOADS - AGENT 2 & AGENT 3]
Agent 2 Output (Biomedical RAG & Knowledge Graph):
{json.dumps(raw_request_dict.get('agent2_output', {}), indent=2)}

Agent 3 Output (Latest Medical Evidence & PubMed):
{json.dumps(raw_request_dict.get('agent3_output', {}), indent=2)}
""")

    prompt_parts.append(f"""[INTERMEDIATE RULE-BASED BASELINE FUSION]
{json.dumps(deterministic_context_dict, indent=2)}

Perform genuine multi-source fusion, extract concise normalized concepts, detect any conflicts between Agent 2 and Agent 3, preserve all source attribution, and output valid JSON matching the UnifiedClinicalContext schema.
""")

    user_prompt = "\n".join(prompt_parts)

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": settings.llm_model_name,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": settings.llm_temperature,
        "max_tokens": settings.llm_max_tokens,
        "response_format": {"type": "json_object"},
    }

    url = f"{settings.llm_base_url.rstrip('/')}/chat/completions"

    start_time = time.perf_counter()
    logger.info(
        f"[{correlation_id}] Calling LLM Provider '{settings.llm_provider}' with Model '{settings.llm_model_name}' "
        f"for primary semantic enhancement..."
    )

    try:
        # === GROQ PRIMARY ATTEMPT (before fallback) ===
        if settings.groq_api_key_pool:
            groq_endpoint = "https://api.groq.com/openai/v1/chat/completions"
            groq_model = settings.groq_model
            groq_payload = {
                "model": groq_model,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                "temperature": settings.llm_temperature,
                "max_tokens": settings.llm_max_tokens,
                "response_format": {"type": "json_object"},
            }

            for groq_idx, groq_key in enumerate(settings.groq_api_key_pool, start=1):
                logger.info(
                    f"[{correlation_id}] Trying Groq key {groq_idx}/{len(settings.groq_api_key_pool)} "
                    f"with model '{groq_model}'..."
                )
                groq_headers = {
                    "Authorization": f"Bearer {groq_key}",
                    "Content-Type": "application/json",
                }
                try:
                    with httpx.Client(timeout=settings.llm_timeout) as groq_client:
                        groq_response = groq_client.post(groq_endpoint, headers=groq_headers, json=groq_payload)

                    if groq_response.status_code == 401:
                        logger.warning(f"[{correlation_id}] Groq key {groq_idx} auth failed (401). Trying next...")
                        continue
                    if groq_response.status_code == 429:
                        logger.warning(f"[{correlation_id}] Groq key {groq_idx} rate limited (429). Trying next...")
                        time.sleep(1.0)
                        continue
                    if groq_response.status_code == 404 and groq_model in ("llama-3.1-8b-instruct", "llama-3.1-8b-instant"):
                        alt_model = "llama-3.1-8b-instant" if groq_model == "llama-3.1-8b-instruct" else "llama-3.3-70b-versatile"
                        logger.info(f"[{correlation_id}] Groq model 404. Retrying with '{alt_model}'...")
                        groq_payload["model"] = alt_model
                        with httpx.Client(timeout=settings.llm_timeout) as groq_client:
                            groq_response = groq_client.post(groq_endpoint, headers=groq_headers, json=groq_payload)
                    if groq_response.status_code != 200:
                        logger.warning(f"[{correlation_id}] Groq key {groq_idx} returned HTTP {groq_response.status_code}. Trying next...")
                        continue

                    groq_response.raise_for_status()
                    groq_latency = round((time.perf_counter() - start_time) * 1000, 2)
                    groq_res_data = groq_response.json()
                    groq_usage = groq_res_data.get("usage", {})
                    groq_choices = groq_res_data.get("choices", [])
                    if groq_choices and groq_choices[0].get("message", {}).get("content"):
                        groq_raw = groq_choices[0]["message"]["content"]
                        groq_cleaned = _clean_json_markdown_fences(groq_raw)
                        if groq_cleaned:
                            groq_parsed = json.loads(groq_cleaned)
                            if "unified_context" not in groq_parsed:
                                if "merged_findings" in groq_parsed:
                                    groq_parsed = {"unified_context": groq_parsed}
                                else:
                                    logger.warning(f"[{correlation_id}] Groq key {groq_idx} JSON missing 'unified_context'. Trying next...")
                                    continue
                            groq_stats = {
                                "latency_ms": groq_latency,
                                "token_usage": {
                                    "prompt_tokens": groq_usage.get("prompt_tokens", 0),
                                    "completion_tokens": groq_usage.get("completion_tokens", 0),
                                    "total_tokens": groq_usage.get("total_tokens", 0),
                                },
                            }
                            logger.info(
                                f"[{correlation_id}] Groq round-robin succeeded with key {groq_idx} in {groq_latency}ms."
                            )
                            return groq_parsed, groq_stats

                except (httpx.RequestError, httpx.TimeoutException) as groq_net_err:
                    logger.warning(f"[{correlation_id}] Groq key {groq_idx} network error: {groq_net_err}. Trying next...")
                    continue
                except (json.JSONDecodeError, KeyError) as groq_parse_err:
                    logger.warning(f"[{correlation_id}] Groq key {groq_idx} parse error: {groq_parse_err}. Trying next...")
                    continue

            logger.warning(f"[{correlation_id}] All Groq keys exhausted. Falling back to configured local Ollama Mistral...")

        ollama_endpoint = f"{settings.ollama_base_url.rstrip('/')}/api/generate"
        ollama_payload = {
            "model": settings.ollama_model,
            "prompt": f"{SYSTEM_PROMPT}\n\n{user_prompt}",
            "stream": False,
            "options": {"temperature": settings.llm_temperature},
        }

        with httpx.Client(timeout=settings.llm_timeout) as client:
            response = client.post(ollama_endpoint, json=ollama_payload)
            response.raise_for_status()

        latency_ms = round((time.perf_counter() - start_time) * 1000, 2)
        res_data = response.json()
        raw_content = res_data.get("response", "")
        cleaned_content = _clean_json_markdown_fences(raw_content)
        if not cleaned_content:
            logger.warning(f"[{correlation_id}] Ollama returned empty content.")
            return None

        parsed_json = json.loads(cleaned_content)
        if "unified_context" not in parsed_json:
            if "merged_findings" in parsed_json:
                parsed_json = {"unified_context": parsed_json}
            else:
                logger.warning(f"[{correlation_id}] Ollama JSON output missing required 'unified_context' root key.")
                return None

        stats = {
            "latency_ms": latency_ms,
            "token_usage": {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
            },
        }
        logger.info(f"[{correlation_id}] Ollama Mistral fallback succeeded in {latency_ms} ms.")
        return parsed_json, stats

    except (httpx.HTTPStatusError, httpx.RequestError, TimeoutError) as net_err:
        logger.warning(f"[{correlation_id}] LLM API network/HTTP error: {net_err}")
        return None

    except (json.JSONDecodeError, KeyError, TypeError, ValueError) as parse_err:
        logger.warning(f"[{correlation_id}] Failed to parse valid JSON from LLM output: {parse_err}")
        return None

    except Exception as exc:
        logger.error(
            f"[{correlation_id}] Unexpected error in LLM enhancement module: {exc}",
            exc_info=True,
        )
        return None
