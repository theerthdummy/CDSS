/**
 * CDSS API Contract Types (Derived from FastAPI OpenAPI Specification)
 * 
 * Source of truth: CDSS Orchestrator OpenAPI 3.0.0
 */

/**
 * @typedef {Object} ChatRequest
 * @property {string} message - Patient / Clinician input message (required, minLength: 1)
 * @property {string|null} [session_id] - Optional session ID for multi-turn state maintenance
 */

/**
 * @typedef {Object} PatientMeasurement
 * @property {string} [temperature] - Temperature string (e.g. "102 °F")
 * @property {string} [blood_pressure] - Blood pressure string
 * @property {string} [heart_rate] - Heart rate string
 */

/**
 * @typedef {Object} PatientState
 * @property {string|null} age - Patient age
 * @property {string|null} gender - Patient gender ("female" | "male" | null)
 * @property {string[]} confirmed_symptoms - Confirmed patient symptoms
 * @property {string[]} denied_symptoms - Explicitly denied patient symptoms
 * @property {string[]} unknown_symptoms - Unasked or unverified symptoms
 * @property {PatientMeasurement} measurements - Vitals and measurements
 * @property {string[]} medical_history - Known chronic conditions or medical history
 * @property {string} medical_history_status - "UNKNOWN" | "NONE_REPORTED" | "PRESENT"
 * @property {string[]} suspected_hypotheses - Hypotheses considered
 */

/**
 * @typedef {Object} ClinicalAssessment
 * @property {string} [stage] - Reasoning stage (e.g. "EXPLORATORY" | "INITIAL" | "EMERGENCY_TRIAGE")
 * @property {boolean} is_urgent - True if urgent medical assessment is indicated
 * @property {string} priority_level - "EMERGENCY" | "URGENT" | "ELEVATED" | "ROUTINE"
 * @property {string[]} red_flags - Detected critical findings or emergency red flags
 * @property {string[]} differential_considerations - Differential diagnoses under consideration
 * @property {string} reasoning_summary - Concise clinical reasoning summary
 * @property {string} [primary_interpretation] - High-level clinical interpretation
 */

/**
 * @typedef {Object} EvidenceItem
 * @property {string} [id] - Evidence ID
 * @property {string} source - Source type or origin ("Vector RAG" | "Web Scanner" | "literature")
 * @property {string} [title] - Document title or article heading
 * @property {string} [snippet] - Content excerpt or abstract snippet
 * @property {number} [score] - Similarity or relevance score (0.0 to 1.0)
 * @property {number} [confidence] - Confidence score
 * @property {string} [url] - Source URL if available
 * @property {string} [document_type] - Type of evidence document
 */

/**
 * @typedef {Object} ValidationResult
 * @property {boolean} is_valid - True if response passed all safety checks
 * @property {string} action_taken - "PASS" | "SANITIZED" | "CONSTRAINED"
 * @property {number} issue_count - Total issues caught
 * @property {Array<{ category: string, description: string, severity: string }>} issues - Detailed issues
 */

/**
 * @typedef {Object} LLMMetadata
 * @property {string} provider - LLM provider (e.g. "OpenAI")
 * @property {string} model - Hosted model name (e.g. "openai/gpt-oss-120b")
 * @property {string} purpose - Purpose of model invocation
 */

/**
 * @typedef {Object} ChatResponse
 * @property {string} session_id - Active conversation session UUID
 * @property {string} response - Rendered assistant response text
 * @property {PatientState} patient_state - Authoritative patient state
 * @property {ClinicalAssessment} clinical_assessment - Clinical assessment and triage
 * @property {string[]} follow_up_questions - 2 to 5 prioritized follow-up questions
 * @property {boolean} urgent_flag - Flag indicating red-flag urgency
 * @property {EvidenceItem[]} evidence - Retrieved biomedical RAG and web evidence
 * @property {ValidationResult} validation - Guardrail validation results
 * @property {LLMMetadata} llm_metadata - Hosted model metadata
 */

/**
 * @typedef {Object} AnalyzeRequest
 * @property {string} text - Raw clinical text from patient/physician (required)
 * @property {string|null} [session_id] - Optional session ID
 */

/**
 * @typedef {Object} AnalyzeResponse
 * @property {string} patient_text - Original input text
 * @property {any} final_output - Multi-agent fusion and reasoning output
 * @property {number} iterations - Number of refinement loops executed
 * @property {boolean} success - Pipeline execution success flag
 * @property {string|null} session_id - Associated session ID
 */

export const EMPTY_PATIENT_STATE = Object.freeze({
    age: null,
    gender: null,
    confirmed_symptoms: [],
    denied_symptoms: [],
    unknown_symptoms: [],
    measurements: {},
    medical_history: [],
    medical_history_status: "UNKNOWN",
    suspected_hypotheses: [],
});
