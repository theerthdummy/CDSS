import { useChatState } from "../context/ChatContext";

/**
 * useChat hook adapter connecting to the centralized ChatContext.
 */
export function useChat() {
    const {
        messages,
        isLoading,
        patientState,
        clinicalAssessment,
        systemHealth,
        sendMessage,
        resetConsultation,
    } = useChatState();

    return {
        messages,
        isLoading,
        patientData: {
            patient_state: patientState,
            clinical_assessment: clinicalAssessment,
        },
        connectionStatus: systemHealth?.overall === "healthy" ? "ready" : "offline",
        sendMessage,
        resetChat: resetConsultation,
    };
}

export default useChat;
