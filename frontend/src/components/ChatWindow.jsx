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
    const [isUserScrolledUp, setIsUserScrolledUp] = useState(false);
    const [hasUnreadBelow, setHasUnreadBelow] = useState(false);
    const prevMessagesCountRef = useRef(messages.length);

    // Scroll to bottom smoothly within the internal container
    const scrollToBottom = useCallback((behavior = "smooth") => {
        const container = messagesContainerRef.current;
        if (!container) return;
        container.scrollTo({
            top: container.scrollHeight,
            behavior,
        });
        setIsUserScrolledUp(false);
        setHasUnreadBelow(false);
    }, []);

    // Handle user scrolling inside the message container
    const handleScroll = () => {
        const container = messagesContainerRef.current;
        if (!container) return;

        const { scrollTop, scrollHeight, clientHeight } = container;
        const distanceFromBottom = scrollHeight - scrollTop - clientHeight;
        const scrolledUp = distanceFromBottom > 100;

        setIsUserScrolledUp(scrolledUp);
        if (!scrolledUp) {
            setHasUnreadBelow(false);
        }
    };

    // Auto-scroll when new messages arrive or loading stages change
    useEffect(() => {
        if (messages.length > prevMessagesCountRef.current) {
            if (isUserScrolledUp) {
                setHasUnreadBelow(true);
            } else {
                scrollToBottom("smooth");
            }
        }
        prevMessagesCountRef.current = messages.length;
    }, [messages.length, isUserScrolledUp, scrollToBottom]);

    // Handle initial mount or reset
    useEffect(() => {
        scrollToBottom("auto");
    }, [scrollToBottom]);

    const hasMessages = messages && messages.length > 0;

    return (
        <section className="chat-window-card" aria-label="Clinical Decision Support Workspace">
            <div className="chat-window-header">
                <div className="chat-header-title">
                    <span className="chat-header-icon" aria-hidden="true">🩺</span>
                    <div>
                        <h2>Clinical Consultation Co-Pilot</h2>
                        <span className="chat-header-subtitle">Evidence-Grounded Physician Decision Support</span>
                    </div>
                </div>
                {hasMessages && (
                    <span className="message-count-pill" aria-label={`${messages.length} messages in consultation`}>
                        {messages.length} {messages.length === 1 ? "turn" : "turns"}
                    </span>
                )}
            </div>

            <div
                ref={messagesContainerRef}
                className="chat-messages-container"
                onScroll={handleScroll}
            >
                {!hasMessages ? (
                    <div className="empty-conversation-state">
                        <div className="empty-icon-shield" aria-hidden="true">
                            <svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round">
                                <path d="M22 12h-4l-3 9L9 3l-3 9H2" />
                            </svg>
                        </div>
                        <h3>Clinical Decision Support Co-Pilot</h3>
                        <p className="empty-description">
                            High-yield diagnostic reasoning, stratified differentials, and clinical decision support for attending physicians and medical specialists.
                        </p>
                        <p className="empty-sub-hint">
                            Select a clinical case presentation below or enter physician consultation notes:
                        </p>

                        <div className="example-scenarios-grid">
                            {CLINICAL_EXAMPLES.map((example, idx) => (
                                <button
                                    key={idx}
                                    type="button"
                                    className="scenario-card-btn"
                                    onClick={() => onSelectPrompt && onSelectPrompt(example.text)}
                                >
                                    <div className="scenario-card-top">
                                        <span className="scenario-badge">Scenario {idx + 1}</span>
                                        <span className="scenario-arrow">→</span>
                                    </div>
                                    <span className="scenario-title">{example.title}</span>
                                    <span className="scenario-text">"{example.text}"</span>
                                </button>
                            ))}
                        </div>
                    </div>
                ) : (
                    <div className="messages-flow">
                        {messages.map((msg) => (
                            <MessageBubble
                                key={msg.id}
                                sender={msg.sender}
                                text={msg.text}
                                kind={msg.kind}
                                timestamp={msg.timestamp}
                                isUrgent={msg.isUrgent}
                                validation={msg.validation}
                                onShowToast={onShowToast}
                            />
                        ))}

                        {/* Loading State with Progressive Clinical Stage */}
                        {isLoading && (
                            <div className="assistant-loading-row" aria-live="polite">
                                <LoadingSpinner stage={loadingStage} />
                            </div>
                        )}

                        {/* Suggested Follow-up Quick-Reply Chips */}
                        {!isLoading && followUpQuestions && followUpQuestions.length > 0 && (
                            <div className="followup-chips-section">
                                <span className="chips-title">Suggested Quick Clarifications:</span>
                                <div className="chips-list">
                                    <button
                                        type="button"
                                        className="chip-btn"
                                        onClick={() => sendMessage("No cough, sore throat, or shortness of breath.")}
                                    >
                                        No respiratory symptoms
                                    </button>
                                    <button
                                        type="button"
                                        className="chip-btn"
                                        onClick={() => sendMessage("No severe headache or neck stiffness.")}
                                    >
                                        No headache or stiff neck
                                    </button>
                                    <button
                                        type="button"
                                        className="chip-btn"
                                        onClick={() => sendMessage("No pain or burning when urinating.")}
                                    >
                                        No urinary discomfort
                                    </button>
                                </div>
                            </div>
                        )}
                    </div>
                )}
            </div>

            {/* Floating Scroll-to-Bottom Button (ChatGPT style) */}
            {isUserScrolledUp && (
                <button
                    type="button"
                    className={`btn-scroll-bottom ${hasUnreadBelow ? "btn-scroll-bottom-unread" : ""}`}
                    onClick={() => scrollToBottom("smooth")}
                    aria-label="Scroll to latest messages"
                >
                    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
                        <line x1="12" y1="5" x2="12" y2="19" />
                        <polyline points="19 12 12 19 5 12" />
                    </svg>
                    <span>{hasUnreadBelow ? "New messages below" : "Scroll to latest"}</span>
                </button>
            )}
        </section>
    );
}

export default ChatWindow;
