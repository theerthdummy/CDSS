/* eslint-disable react-refresh/only-export-components */
import { createContext, useContext, useCallback, useEffect, useState, useRef } from "react";
import { chatWithAssistant, analyzeClinicalPipeline, getSystemHealth, uploadDocument } from "../api/client";
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
        timestamp: new Date().toISOString(),
        isUrgent: options.isUrgent || false,
        validation: options.validation || null,
        evidence: options.evidence || [],
    };
}

function generateSessionTitle(text) {
    if (!text) return "Clinical Consultation";
    const clean = text.replace(/\s+/g, " ").trim();
    const firstSentence = clean.split(/[.?!]\s/)[0];
    if (firstSentence && firstSentence.length <= 42) {
        return firstSentence;
    }
    if (clean.length <= 40) {
        return clean;
    }
    const truncated = clean.slice(0, 40);
    const lastSpace = truncated.lastIndexOf(" ");
    if (lastSpace > 20) {
        return `${truncated.slice(0, lastSpace)}...`;
    }
    return `${truncated}...`;
}

export function ChatProvider({ children }) {
    const [messages, setMessages] = useState([]);
    const [sessionId, setSessionId] = useState(null); // Backend session id
    const [activeSessionId, setActiveSessionId] = useState(null); // Local session key
    const [patientState, setPatientState] = useState(EMPTY_PATIENT_STATE);
    const [clinicalAssessment, setClinicalAssessment] = useState(null);
    const [evidence, setEvidence] = useState([]);
    const [validation, setValidation] = useState(null);
    const [followUpQuestions, setFollowUpQuestions] = useState([]);
    const [llmMetadata, setLlmMetadata] = useState(null);
    const [consultationMode, setConsultationMode] = useState("conversational"); // "conversational" | "deep_diagnostic"
    const [isTelemetryOpen, setIsTelemetryOpen] = useState(false);
    const [isSidebarOpen, setIsSidebarOpen] = useState(true);

    // Recent Consultations History with full message persistence
    const [recentSessions, setRecentSessions] = useState(() => {
        try {
            const saved = localStorage.getItem("cdss_recent_sessions_v2");
            return saved ? JSON.parse(saved) : [];
        } catch {
            return [];
        }
    });

    const [isLoading, setIsLoading] = useState(false);
    const [loadingStage, setLoadingStage] = useState("");
    const [error, setError] = useState(null);
    const [systemHealth, setSystemHealth] = useState(null);

    // Save recent sessions to localStorage whenever it changes
    useEffect(() => {
        try {
            localStorage.setItem("cdss_recent_sessions_v2", JSON.stringify(recentSessions));
        } catch (err) {
            console.warn("Failed to persist recent sessions to localStorage:", err);
        }
    }, [recentSessions]);

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
     * Load an existing saved consultation session from history.
     */
    const loadSession = useCallback((targetSessionId) => {
        const target = recentSessions.find((s) => s.id === targetSessionId);
        if (!target) return;

        setActiveSessionId(target.id);
        setSessionId(target.backendSessionId || null);
        setMessages(target.messages || []);
        setPatientState(target.patientState || EMPTY_PATIENT_STATE);
        setClinicalAssessment(target.clinicalAssessment || null);
        setEvidence(target.evidence || []);
        setValidation(target.validation || null);
        setFollowUpQuestions(target.followUpQuestions || []);
        setLlmMetadata(target.llmMetadata || null);
        setError(null);
    }, [recentSessions]);

    /**
     * Start a new fresh consultation.
     */
    const resetConsultation = useCallback(() => {
        setMessages([]);
        setSessionId(null);
        setActiveSessionId(null);
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

    /**
     * Delete a single session from history.
     */
    const deleteRecentSession = useCallback((idToDelete) => {
        setRecentSessions((prev) => prev.filter((s) => s.id !== idToDelete));
        if (activeSessionId === idToDelete) {
            resetConsultation();
        }
    }, [activeSessionId, resetConsultation]);

    /**
     * Clear all saved sessions.
     */
    const clearAllSessions = useCallback(() => {
        setRecentSessions([]);
        resetConsultation();
    }, [resetConsultation]);

    /**
     * Send message using the active consultation mode.
     */
    const sendMessage = useCallback(async (text, explicitMode = null) => {
        const trimmed = text ? text.trim() : "";
        if (!trimmed) return;

        const activeMode = explicitMode || consultationMode;
        const currentActiveId = activeSessionId || `session-${Date.now()}`;
        if (!activeSessionId) {
            setActiveSessionId(currentActiveId);
        }

        const userMsg = createMessage("user", trimmed);
        const updatedMessagesWithUser = [...messages, userMsg];

        // Update active messages state
        setMessages(updatedMessagesWithUser);
        setError(null);
        setIsLoading(true);

        const sessionTitle = generateSessionTitle(trimmed);

        // Update or create entry in recentSessions immediately with the user message
        setRecentSessions((prev) => {
            const existingIdx = prev.findIndex((s) => s.id === currentActiveId);
            const titleToUse = prev[existingIdx]?.title || sessionTitle;
            const updatedSession = {
                id: currentActiveId,
                backendSessionId: sessionId,
                title: titleToUse,
                timestamp: new Date().toISOString(),
                messages: updatedMessagesWithUser,
                patientState,
                clinicalAssessment,
                evidence,
                validation,
                followUpQuestions,
                llmMetadata,
            };

            if (existingIdx >= 0) {
                const copy = [...prev];
                copy[existingIdx] = updatedSession;
                return copy;
            }
            return [updatedSession, ...prev.slice(0, 29)];
        });

        // Prepare conversation history payload for state hydration
        const historyPayload = messages.map((m) => ({
            role: m.sender === "user" ? "user" : "assistant",
            content: m.text,
        }));

        try {
            if (activeMode === "conversational") {
                setLoadingStage("Consulting clinical assistant & retrieving evidence...");
                const result = await chatWithAssistant(trimmed, sessionId, historyPayload);

                const newBackendId = result.session_id || sessionId;
                if (result.session_id) {
                    setSessionId(result.session_id);
                }

                const newPatientState = result.patient_state || patientState;
                if (result.patient_state) {
                    setPatientState(result.patient_state);
                }

                const newAssessment = result.clinical_assessment || clinicalAssessment;
                if (result.clinical_assessment) {
                    setClinicalAssessment(result.clinical_assessment);
                }

                const newEvidence = Array.isArray(result.evidence) ? result.evidence : evidence;
                if (Array.isArray(result.evidence)) {
                    setEvidence(result.evidence);
                }

                const newValidation = result.validation || validation;
                if (result.validation) {
                    setValidation(result.validation);
                }

                const newFollowUps = Array.isArray(result.follow_up_questions) ? result.follow_up_questions : followUpQuestions;
                if (Array.isArray(result.follow_up_questions)) {
                    setFollowUpQuestions(result.follow_up_questions);
                }

                const newLlmMeta = result.llm_metadata || llmMetadata;
                if (result.llm_metadata) {
                    setLlmMetadata(result.llm_metadata);
                }

                const assistantMsg = createMessage("assistant", result.response, {
                    isUrgent: result.urgent_flag || result.clinical_assessment?.is_urgent,
                    validation: result.validation,
                    evidence: result.evidence,
                });

                const allMessagesFinal = [...updatedMessagesWithUser, assistantMsg];
                setMessages(allMessagesFinal);

                // Update session in recentSessions with the full assistant response and clinical state
                setRecentSessions((prev) => {
                    const existingIdx = prev.findIndex((s) => s.id === currentActiveId);
                    const updatedSession = {
                        id: currentActiveId,
                        backendSessionId: newBackendId,
                        title: prev[existingIdx]?.title || sessionTitle,
                        messages: allMessagesFinal,
                        patientState: newPatientState,
                        clinicalAssessment: newAssessment,
                        evidence: newEvidence,
                        validation: newValidation,
                        followUpQuestions: newFollowUps,
                        llmMetadata: newLlmMeta,
                        timestamp: new Date().toISOString(),
                    };

                    if (existingIdx < 0) {
                        return [updatedSession, ...prev.slice(0, 29)];
                    }
                    const copy = [...prev];
                    copy[existingIdx] = updatedSession;
                    return copy;
                });
            } else {
                // Deep Multi-Agent Clinical Analysis
                setLoadingStage("Executing 5-Agent Diagnostic Fusion Pipeline...");
                const result = await analyzeClinicalPipeline(trimmed, sessionId);

                const newBackendId = result.session_id || sessionId;
                if (result.session_id) {
                    setSessionId(result.session_id);
                }

                const out = result.final_output;
                let newPatientState = patientState;
                let newAssessment = clinicalAssessment;
                let newFollowUps = followUpQuestions;
                let newValidation = validation;
                let newLlmMeta = llmMetadata;

                if (out) {
                    if (out.patient_state) {
                        newPatientState = out.patient_state;
                        setPatientState(out.patient_state);
                    }
                    if (out.clinical_assessment) {
                        newAssessment = out.clinical_assessment;
                        setClinicalAssessment(out.clinical_assessment);
                    }
                    if (Array.isArray(out.follow_up_questions)) {
                        newFollowUps = out.follow_up_questions;
                        setFollowUpQuestions(out.follow_up_questions);
                    }
                    if (out.validation) {
                        newValidation = out.validation;
                        setValidation(out.validation);
                    }
                    if (out.agent_metadata) {
                        newLlmMeta = out.agent_metadata;
                        setLlmMetadata(out.agent_metadata);
                    }

                    const responseText =
                        out.conversational_response ||
                        out.clinical_assessment?.primary_interpretation ||
                        out.reasoning?.reasoning_summary ||
                        "Clinical analysis completed.";

                    const assistantMsg = createMessage("assistant", responseText, {
                        isUrgent: out.urgent_flag || out.clinical_assessment?.is_urgent,
                        validation: out.validation,
                    });

                    const allMessagesFinal = [...updatedMessagesWithUser, assistantMsg];
                    setMessages(allMessagesFinal);

                    // Update session in recentSessions with the full response
                    setRecentSessions((prev) => {
                        const existingIdx = prev.findIndex((s) => s.id === currentActiveId);
                        const updatedSession = {
                            id: currentActiveId,
                            backendSessionId: newBackendId,
                            title: prev[existingIdx]?.title || sessionTitle,
                            messages: allMessagesFinal,
                            patientState: newPatientState,
                            clinicalAssessment: newAssessment,
                            evidence,
                            validation: newValidation,
                            followUpQuestions: newFollowUps,
                            llmMetadata: newLlmMeta,
                            timestamp: new Date().toISOString(),
                        };

                        if (existingIdx < 0) {
                            return [updatedSession, ...prev.slice(0, 29)];
                        }
                        const copy = [...prev];
                        copy[existingIdx] = updatedSession;
                        return copy;
                    });
                }
            }
        } catch (err) {
            console.error("Clinical message execution failed:", err);
            setError(err);
            const errorMsg = createMessage("assistant", err.message || "An error occurred while processing your clinical request.", {
                kind: "error",
            });
            const allMessagesFinal = [...updatedMessagesWithUser, errorMsg];
            setMessages(allMessagesFinal);

            setRecentSessions((prev) => {
                const existingIdx = prev.findIndex((s) => s.id === currentActiveId);
                if (existingIdx < 0) return prev;
                const copy = [...prev];
                copy[existingIdx] = {
                    ...copy[existingIdx],
                    messages: allMessagesFinal,
                };
                return copy;
            });
        } finally {
            setIsLoading(false);
            setLoadingStage("");
        }
    }, [consultationMode, activeSessionId, messages, sessionId, patientState, clinicalAssessment, evidence, validation, followUpQuestions, llmMetadata]);

    /**
     * Upload a medical document and route through clinical assistant for direct diagnosis.
     */
    const uploadAndAnalyze = useCallback(async (file, userMessage = null) => {
        if (!file) return;

        const currentActiveId = activeSessionId || `session-${Date.now()}`;
        if (!activeSessionId) setActiveSessionId(currentActiveId);

        const fileName = file.name;
        const label = userMessage
            ? `📎 ${fileName} — ${userMessage}`
            : `📎 ${fileName}`;

        const userMsg = createMessage("user", label, { kind: "user" });
        const updatedWithUser = [...messages, userMsg];
        setMessages(updatedWithUser);
        setError(null);
        setIsLoading(true);
        setLoadingStage("Extracting text from document & running clinical analysis...");

        const sessionTitle = `Document: ${fileName}`;

        setRecentSessions((prev) => {
            const existingIdx = prev.findIndex((s) => s.id === currentActiveId);
            const updatedSession = {
                id: currentActiveId,
                backendSessionId: sessionId,
                title: prev[existingIdx]?.title || sessionTitle,
                timestamp: new Date().toISOString(),
                messages: updatedWithUser,
                patientState, clinicalAssessment, evidence, validation, followUpQuestions, llmMetadata,
            };
            if (existingIdx >= 0) {
                const copy = [...prev];
                copy[existingIdx] = updatedSession;
                return copy;
            }
            return [updatedSession, ...prev.slice(0, 29)];
        });

        try {
            const result = await uploadDocument(file, sessionId, userMessage);

            const newBackendId = result.session_id || sessionId;
            if (result.session_id) setSessionId(result.session_id);
            if (result.patient_state) setPatientState(result.patient_state);
            if (result.clinical_assessment) setClinicalAssessment(result.clinical_assessment);
            if (Array.isArray(result.evidence)) setEvidence(result.evidence);
            if (result.validation) setValidation(result.validation);
            if (Array.isArray(result.follow_up_questions)) setFollowUpQuestions(result.follow_up_questions);
            if (result.llm_metadata) setLlmMetadata(result.llm_metadata);

            const docInfo = result.document_info || {};
            const docSummaryLines = [];
            if (docInfo.parsed_fields?.diagnosis) docSummaryLines.push(`**Stated Diagnosis**: ${docInfo.parsed_fields.diagnosis}`);
            if (docInfo.parsed_fields?.medications?.length) docSummaryLines.push(`**Medications**: ${docInfo.parsed_fields.medications.join(", ")}`);
            const docSummary = docSummaryLines.length
                ? `> 📄 **${fileName}** parsed — ${docInfo.extracted_chars?.toLocaleString()} chars extracted.\n> ${docSummaryLines.join(" · ")}\n\n`
                : `> 📄 **${fileName}** — ${docInfo.extracted_chars?.toLocaleString()} chars extracted.\n\n`;

            const assistantMsg = createMessage("assistant", docSummary + result.response, {
                isUrgent: result.urgent_flag || result.clinical_assessment?.is_urgent,
                validation: result.validation,
                evidence: result.evidence,
            });

            const allMessages = [...updatedWithUser, assistantMsg];
            setMessages(allMessages);

            setRecentSessions((prev) => {
                const existingIdx = prev.findIndex((s) => s.id === currentActiveId);
                const updatedSession = {
                    id: currentActiveId,
                    backendSessionId: newBackendId,
                    title: prev[existingIdx]?.title || sessionTitle,
                    messages: allMessages,
                    patientState: result.patient_state || patientState,
                    clinicalAssessment: result.clinical_assessment || clinicalAssessment,
                    evidence: result.evidence || evidence,
                    validation: result.validation || validation,
                    followUpQuestions: result.follow_up_questions || followUpQuestions,
                    llmMetadata: result.llm_metadata || llmMetadata,
                    timestamp: new Date().toISOString(),
                };
                if (existingIdx < 0) return [updatedSession, ...prev.slice(0, 29)];
                const copy = [...prev];
                copy[existingIdx] = updatedSession;
                return copy;
            });
        } catch (err) {
            console.error("Document upload failed:", err);
            setError(err);
            const errMsg = createMessage("assistant", err.message || "Failed to process uploaded document.", { kind: "error" });
            const allMessages = [...updatedWithUser, errMsg];
            setMessages(allMessages);
        } finally {
            setIsLoading(false);
            setLoadingStage("");
        }
    }, [activeSessionId, messages, sessionId, patientState, clinicalAssessment, evidence, validation, followUpQuestions, llmMetadata]);

    return (
        <ChatContext.Provider
            value={{
                messages,
                sessionId,
                activeSessionId,
                patientState,
                clinicalAssessment,
                evidence,
                validation,
                followUpQuestions,
                llmMetadata,
                consultationMode,
                setConsultationMode,
                isTelemetryOpen,
                setIsTelemetryOpen,
                isSidebarOpen,
                setIsSidebarOpen,
                recentSessions,
                loadSession,
                deleteRecentSession,
                clearAllSessions,
                isLoading,
                loadingStage,
                error,
                systemHealth,
                sendMessage,
                uploadAndAnalyze,
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
