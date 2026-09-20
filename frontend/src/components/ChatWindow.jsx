import { useEffect, useRef, useState, useCallback } from "react";
import MessageBubble from "./MessageBubble";
import LoadingSpinner from "./LoadingSpinner";
import { useChatState } from "../context/ChatContext";

const CLINICAL_EXAMPLES = [
    {
        title: "Isolated Fever in Adult",
        text: "55 y.o female with temp 102°F for 2 days. No history, no medications.",
    },
    {
        title: "Emergency Red Flag (Altered Mental Status)",
        text: "I have had a high fever of 103 for 24 hours and I started seeing strange hallucinations.",
    },
    {
        title: "Vague Symptoms Presentation",
        text: "I feel really sick and uneasy today with some stomach discomfort.",
    },
    {
        title: "Acute Chest Pain Presentation",
        text: "Patient reports sharp substernal chest pain for 2 hours with shortness of breath.",
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
        const isNewMessage = messages.length > prevMessagesCountRef.current;
        const lastMessage = messages[messages.length - 1];
        const isUserSender = lastMessage?.sender === "user";

        if (isUserSender || !isUserScrolledUp) {
            // User sent a message or was already at the bottom -> auto-scroll smoothly
            scrollToBottom("smooth");
        } else if (isNewMessage && isUserScrolledUp) {
            // New message arrived while user was reading history -> notify without jumping
            setHasUnreadBelow(true);
        }

        prevMessagesCountRef.current = messages.length;
    }, [messages, isLoading, loadingStage, isUserScrolledUp, scrollToBottom]);

    const hasMessages = messages.length > 0;

    return (
        <section className="chat-window-card" aria-label="Clinical Conversation Area">
            <div className="chat-window-header">
                <div className="chat-header-title">
                    <span className="chat-header-icon" aria-hidden="true">💬</span>
                    <div>
                        <h2>Clinical Consultation</h2>
                        <span className="chat-header-subtitle">Evidence-Grounded Physician Assistant</span>
                    </div>
                </div>
                {hasMessages && (
                    <span className="message-count-pill">
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
                        <h3>Clinical Decision Support Assistant</h3>
                        <p className="empty-description">
                            Empathetic, evidence-grounded clinical reasoning powered by hosted OpenAI GPT and Biomedical Vector RAG.
                        </p>
                        <p className="empty-sub-hint">
                            Select a clinical scenario below or describe your patient's symptoms:
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
