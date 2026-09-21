import { useEffect, useRef, useState, useCallback } from "react";
import MessageBubble from "./MessageBubble";
import LoadingSpinner from "./LoadingSpinner";
import { useChatState } from "../context/ChatContext";

const CLINICAL_EXAMPLES = [
    {
        title: "FUO in Immunocompetent Adult",
        text: "55 y.o female with temp 102°F for 2 days. Unrevealing initial exam, no focal source, negative UA.",
    },
    {
        title: "Toxic Encephalopathy vs. CNS Infection",
        text: "High fever 103°F with acute visual hallucinations, confusion, and mild nuchal discomfort.",
    },
    {
        title: "Undifferentiated Systemic Presentation",
        text: "Profound malaise, diffuse myalgias, intermittent low-grade fever, and unexplained transaminitis.",
    },
    {
        title: "Atypical Acute Coronary Syndrome",
        text: "Sharp retrosternal chest pain for 2h, diaphoresis, dyspnea, normal baseline ECG with diabetes history.",
    },
];

function ChatWindow({ onSelectPrompt, onShowToast }) {
    const { messages, isLoading, loadingStage, followUpQuestions, sendMessage } = useChatState();
    const messagesContainerRef = useRef(null);
    const lastUserMsgRef = useRef(null);
    const prevMessagesCountRef = useRef(messages.length);

    // Scroll to the latest user message when sent, but DO NOT auto-scroll down to bottom on assistant response
    useEffect(() => {
        const isNewMessage = messages.length > prevMessagesCountRef.current;
        if (isNewMessage) {
            const lastMsg = messages[messages.length - 1];
            if (lastMsg?.sender === "user" && lastUserMsgRef.current) {
                // When user sends a message, gently ensure the user message is in view
                lastUserMsgRef.current.scrollIntoView({ behavior: "smooth", block: "start" });
            }
            // When assistant response arrives, DO NOT scroll to bottom; let user scroll manually to read
        }
        prevMessagesCountRef.current = messages.length;
    }, [messages]);

    const hasMessages = messages && messages.length > 0;

    return (
        <div className="chatgpt-conversation-stream">
            <div
                ref={messagesContainerRef}
                className="chat-messages-scroll-area"
            >
                {!hasMessages ? (
                    <div className="chatgpt-hero-state">
                        <div className="hero-badge-pill">
                            <span className="hero-status-pulse" />
                            <span>Biomedical RAG & Web Intelligence Studio</span>
                        </div>

                        <h1 className="hero-gradient-title">
                            CDSS <span className="gradient-highlight">Clinical Co-Pilot</span>
                        </h1>
                        <p className="hero-subtext">
                            Evidence-grounded diagnostic reasoning, stratified differentials, and clinical co-pilot support for attending physicians.
                        </p>

                        <div className="hero-scenarios-grid">
                            {CLINICAL_EXAMPLES.map((example, idx) => (
                                <button
                                    key={idx}
                                    type="button"
                                    className="hero-scenario-card"
                                    onClick={() => onSelectPrompt && onSelectPrompt(example.text)}
                                >
                                    <div className="scenario-card-header">
                                        <span className="scenario-tag">CASE 0{idx + 1}</span>
                                        <span className="scenario-arrow-icon">↗</span>
                                    </div>
                                    <span className="scenario-headline">{example.title}</span>
                                    <span className="scenario-snippet">"{example.text}"</span>
                                </button>
                            ))}
                        </div>
                    </div>
                ) : (
                    <div className="messages-stream-list">
                        {messages.map((msg, index) => {
                            const isLastUser = msg.sender === "user" && index >= messages.length - 2;
                            return (
                                <div key={msg.id} ref={isLastUser ? lastUserMsgRef : null} className="message-wrapper-anchor">
                                    <MessageBubble
                                        sender={msg.sender}
                                        text={msg.text}
                                        kind={msg.kind}
                                        timestamp={msg.timestamp}
                                        isUrgent={msg.isUrgent}
                                        validation={msg.validation}
                                        onShowToast={onShowToast}
                                    />
                                </div>
                            );
                        })}

                        {/* Loading State with Progressive Stage */}
                        {isLoading && (
                            <div className="assistant-loading-row" aria-live="polite">
                                <LoadingSpinner stage={loadingStage} />
                            </div>
                        )}

                        {/* Suggested Follow-up Quick Clarifications */}
                        {!isLoading && followUpQuestions && followUpQuestions.length > 0 && (
                            <div className="followup-chips-section">
                                <span className="chips-title">High-Yield Clinical Discriminators:</span>
                                <div className="chips-list">
                                    <button
                                        type="button"
                                        className="chip-btn"
                                        onClick={() => sendMessage("No cough, sore throat, or respiratory symptoms.")}
                                    >
                                        No respiratory signs
                                    </button>
                                    <button
                                        type="button"
                                        className="chip-btn"
                                        onClick={() => sendMessage("Negative for meningismus, neck stiffness, or severe headache.")}
                                    >
                                        No nuchal rigidity / headache
                                    </button>
                                    <button
                                        type="button"
                                        className="chip-btn"
                                        onClick={() => sendMessage("No dysuria, hematuria, or flank tenderness.")}
                                    >
                                        No urinary / flank signs
                                    </button>
                                </div>
                            </div>
                        )}
                    </div>
                )}
            </div>
        </div>
    );
}

export default ChatWindow;

