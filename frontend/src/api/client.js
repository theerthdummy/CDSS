import axios from "axios";

/**
 * Production-grade API client for the Clinical Decision Support System.
 * Matches FastAPI Orchestrator OpenAPI contract with zero endpoint discrepancies.
 */

const BASE_URL = import.meta.env.VITE_API_BASE_URL || "http://localhost:9000";

export const apiClient = axios.create({
    baseURL: BASE_URL,
    headers: {
        "Content-Type": "application/json",
    },
    timeout: 60000, // 60 seconds
});

/**
 * Normalize Axios and network errors into structured, user-friendly error objects.
 */
export function normalizeApiError(error) {
    if (!error) {
        return {
            type: "UNKNOWN",
            message: "An unexpected error occurred.",
            statusCode: null,
            detail: null,
        };
    }

    if (error.response) {
        const status = error.response.status;
        const data = error.response.data;

        if (status === 422) {
            return {
                type: "VALIDATION_ERROR",
                message: "The clinical input could not be validated. Please check the provided information.",
                statusCode: status,
                detail: data?.detail || null,
            };
        }

        if (status === 500) {
            return {
                type: "SERVER_ERROR",
                message: data?.detail || "The clinical engine encountered an internal issue. Please try again.",
                statusCode: status,
                detail: data?.detail || null,
            };
        }

        return {
            type: "HTTP_ERROR",
            message: data?.detail || `Server returned error (${status}).`,
            statusCode: status,
            detail: data,
        };
    }

    if (error.code === "ECONNABORTED" || error.message?.includes("timeout")) {
        return {
            type: "TIMEOUT",
            message: "The request timed out. The clinical knowledge engines may be busy. Please retry.",
            statusCode: 408,
            detail: error.message,
        };
    }

    if (error.request || error.code === "ERR_NETWORK") {
        return {
            type: "NETWORK_OFFLINE",
            message: "Cannot connect to the CDSS backend service. Please check your network connection or ensure the orchestrator service is running on port 9000.",
            statusCode: 0,
            detail: error.message,
        };
    }

    return {
        type: "CLIENT_ERROR",
        message: error.message || "Failed to process request.",
        statusCode: null,
        detail: String(error),
    };
}

/**
 * Execute a conversational turn with the hosted OpenAI GPT Clinical Assistant.
 * 
 * @param {string} message - Patient/Clinician input message
 * @param {string|null} sessionId - Optional session UUID for multi-turn state continuity
 * @param {Array<Object>|null} conversationHistory - Optional previous turn messages for state hydration
 * @returns {Promise<ChatResponse>}
 */
export async function chatWithAssistant(message, sessionId = null, conversationHistory = null) {
    try {
        const payload = {
            message: message.trim(),
            session_id: sessionId || null,
            conversation_history: conversationHistory || null,
        };
        const response = await apiClient.post("/chat", payload);
        return response.data;
    } catch (error) {
        throw normalizeApiError(error);
    }
}

/**
 * Execute the deep 5-Agent Clinical Diagnostic Pipeline (Agent 1 Clarifier -> Agent 2 RAG + Agent 3 Web -> Agent 4 Fusion -> Agent 5 Reasoning).
 * 
 * @param {string} text - Raw patient text or clinical notes
 * @param {string|null} sessionId - Optional session UUID
 * @returns {Promise<AnalyzeResponse>}
 */
export async function analyzeClinicalPipeline(text, sessionId = null) {
    try {
        const payload = {
            text: text.trim(),
            session_id: sessionId || null,
        };
        const response = await apiClient.post("/analyze", payload, {
            timeout: 90000, // 90s for deep multi-agent reasoning
        });
        return response.data;
    } catch (error) {
        throw normalizeApiError(error);
    }
}

/**
 * Check real-time connectivity to the orchestrator, downstream agents, and hosted OpenAI GPT model.
 * 
 * @returns {Promise<{ overall: string, agents: Record<string, any>, openai_gpt: Record<string, any> }>}
 */
export async function getSystemHealth() {
    try {
        const response = await apiClient.get("/health", { timeout: 8000 });
        return response.data;
    } catch (error) {
        throw normalizeApiError(error);
    }
}

/**
 * Upload a medical document (PDF / image / DOCX) for OCR-based diagnosis and prognosis.
 * The backend will extract text, parse clinical fields, and run the full clinical assistant pipeline.
 *
 * @param {File} file - The file object from an <input type="file"> or drag-and-drop.
 * @param {string|null} sessionId - Optional session UUID to maintain conversation context.
 * @param {string|null} userMessage - Optional clinician instruction (e.g. "give prognosis for this prescription").
 * @returns {Promise<ChatResponse & { document_info: object }>}
 */
export async function uploadDocument(file, sessionId = null, userMessage = null) {
    try {
        const formData = new FormData();
        formData.append("file", file);
        if (sessionId) formData.append("session_id", sessionId);
        if (userMessage) formData.append("user_message", userMessage);

        const response = await apiClient.post("/upload", formData, {
            headers: { "Content-Type": "multipart/form-data" },
            timeout: 90000, // OCR on large PDFs can take time
        });
        return response.data;
    } catch (error) {
        throw normalizeApiError(error);
    }
}

