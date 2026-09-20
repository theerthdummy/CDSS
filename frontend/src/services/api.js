/**
 * Backwards-compatibility bridge for services/api.js.
 * Delegates to the production typed client in api/client.js.
 */
import {
    apiClient,
    chatWithAssistant as clientChatWithAssistant,
    analyzeClinicalPipeline as clientAnalyzeClinicalPipeline,
    getSystemHealth,
} from "../api/client";

export const chatWithAssistant = clientChatWithAssistant;
export const clarifyPatientNotes = clientAnalyzeClinicalPipeline;

export async function checkBackendStatus() {
    try {
        const health = await getSystemHealth();
        return health?.overall === "healthy" || health?.overall === "degraded";
    } catch {
        return false;
    }
}

export default apiClient;