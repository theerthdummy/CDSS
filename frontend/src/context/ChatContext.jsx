/* eslint-disable react-refresh/only-export-components */
import { createContext, useContext, useCallback, useEffect, useState } from "react";
import { chatWithAssistant, analyzeClinicalPipeline, getSystemHealth } from "../api/client";
import { EMPTY_PATIENT_STATE } from "../api/types";

const ChatContext = createContext(null);

let messageCounter = 0;

function createMessage(sender, text, options = {}) {
    messageCounter += 1;
    return {
        id: `msg-${sender}-${messageCounter}-${Date.now()}`,
        sender,
        text,
        kind: options.kind || (sender === "user" ? "user" : "text"),
        timestamp: new Date(),
        isUrgent: options.isUrgent || false,
        validation: options.validation || null,
        evidence: options.evidence || [],
    };
}

export function ChatProvider({ children }) {
    const [messages, setMessages] = useState([]);
    const [sessionId, setSessionId] = useState(null);
    const [patientState, setPatientState] = useState(EMPTY_PATIENT_STATE);
    const [clinicalAssessment, setClinicalAssessment] = useState(null);
    const [evidence, setEvidence] = useState([]);
    const [validation, setValidation] = useState(null);
    const [followUpQuestions, setFollowUpQuestions] = useState([]);
    const [llmMetadata, setLlmMetadata] = useState(null);
    const [consultationMode, setConsultationMode] = useState("conversational"); // "conversational" | "deep_diagnostic"

    const [isLoading, setIsLoading] = useState(false);
    const [loadingStage, setLoadingStage] = useState("");
    const [error, setError] = useState(null);
    const [systemHealth, setSystemHealth] = useState(null);

    // Initial system health check
    const refreshSystemHealth = useCallback(async () => {
        try {
            const health = await getSystemHealth();
            setSystemHealth(health);
            return health;
        } catch (err) {
            console.warn("System health check failed:", err);
            setSystemHealth({
                overall: "unreachable",
                agents: {},
                openai_gpt: { configured: false },
            });
            return null;
        }
    }, []);

    useEffect(() => {
        refreshSystemHealth();
        const interval = setInterval(refreshSystemHealth, 30000); // 30s poll
        return () => clearInterval(interval);
    }, [refreshSystemHealth]);

    /**
     * Send message using the active consultation mode.
     */
    const sendMessage = useCallback(async (text, explicitMode = null) => {
        const trimmed = text ? text.trim() : "";
        if (!trimmed) return;

        const activeMode = explicitMode || consultationMode;

        // Clear error and append user message
        setError(null);
        setMessages((prev) => [...prev, createMessage("user", trimmed)]);
        setIsLoading(true);

        try {
            if (activeMode === "conversational") {
                setLoadingStage("Consulting clinical assistant & retrieving evidence...");
                const result = await chatWithAssistant(trimmed, sessionId);

                if (result.session_id) {
                    setSessionId(result.session_id);
                }

                if (result.patient_state) {
                    setPatientState(result.patient_state);
                }

                if (result.clinical_assessment) {
                    setClinicalAssessment(result.clinical_assessment);
                }

                if (Array.isArray(result.evidence)) {
                    setEvidence(result.evidence);
                }

                if (result.validation) {
                    setValidation(result.validation);
                }

                if (Array.isArray(result.follow_up_questions)) {
                    setFollowUpQuestions(result.follow_up_questions);
                }

                if (result.llm_metadata) {
                    setLlmMetadata(result.llm_metadata);
                }

                setMessages((prev) => [
                    ...prev,
                    createMessage("assistant", result.response, {
                        isUrgent: result.urgent_flag || result.clinical_assessment?.is_urgent,
                        validation: result.validation,
                        evidence: result.evidence,
                    }),
                ]);
            } else {
                // Deep Multi-Agent Clinical Analysis
                setLoadingStage("Executing 5-Agent Diagnostic Fusion Pipeline...");
                const result = await analyzeClinicalPipeline(trimmed, sessionId);

                if (result.session_id) {
                    setSessionId(result.session_id);
                }

                const out = result.final_output;
                if (out) {
                    if (out.patient_state) {
                        setPatientState(out.patient_state);
                    }
                    if (out.clinical_assessment) {
                        setClinicalAssessment(out.clinical_assessment);
                    }
                    if (Array.isArray(out.follow_up_questions)) {
                        setFollowUpQuestions(out.follow_up_questions);
                    }
                    if (out.validation) {
                        setValidation(out.validation);
                    }
                    if (out.agent_metadata) {
                        setLlmMetadata(out.agent_metadata);
                    }

                    const responseText =
                        out.conversational_response ||
                        out.clinical_assessment?.primary_interpretation ||
                        out.reasoning?.reasoning_summary ||
                        "Clinical analysis completed.";

                    setMessages((prev) => [
                        ...prev,
                        createMessage("assistant", responseText, {
                            isUrgent: out.urgent_flag || out.clinical_assessment?.is_urgent,
                            validation: out.validation,
                        }),
                    ]);
                }
            }
        } catch (err) {
            console.error("Clinical message execution failed:", err);
            setError(err);
            setMessages((prev) => [
                ...prev,
                createMessage("assistant", err.message || "An error occurred while processing your clinical request.", {
                    kind: "error",
                }),
            ]);
        } finally {
            setIsLoading(false);
            setLoadingStage("");
        }
    }, [consultationMode, sessionId]);

    /**
     * Reset the entire consultation state to start a fresh patient case.
     */
    const resetConsultation = useCallback(() => {
        setMessages([]);
        setSessionId(null);
        setPatientState(EMPTY_PATIENT_STATE);
        setClinicalAssessment(null);
        setEvidence([]);
        setValidation(null);
        setFollowUpQuestions([]);
        setError(null);
        setIsLoading(false);
        setLoadingStage("");
        refreshSystemHealth();
    }, [refreshSystemHealth]);

    return (
        <ChatContext.Provider
            value={{
                messages,
                sessionId,
                patientState,
                clinicalAssessment,
                evidence,
                validation,
                followUpQuestions,
                llmMetadata,
                consultationMode,
                setConsultationMode,
                isLoading,
                loadingStage,
                error,
                systemHealth,
                sendMessage,
                resetConsultation,
                refreshSystemHealth,
            }}
        >
            {children}
        </ChatContext.Provider>
    );
}

export function useChatState() {
    const context = useContext(ChatContext);
    if (!context) {
        throw new Error("useChatState must be used within a ChatProvider");
    }
    return context;
}
