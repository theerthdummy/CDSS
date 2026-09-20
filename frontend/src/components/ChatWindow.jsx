import { useEffect, useRef } from "react";
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
        title: "Acute Chest Pain Presentation",
        text: "Patient reports sharp substernal chest pain for 2 hours with shortness of breath.",
    },
];

function ChatWindow({ onSelectPrompt, onShowToast }) {
    const { messages, isLoading, loadingStage, followUpQuestions, sendMessage } = useChatState();
    const scrollBottomRef = useRef(null);

    useEffect(() => {
        scrollBottomRef.current?.scrollIntoView({ behavior: "smooth" });
    }, [messages, isLoading, loadingStage]);

    const hasMessages = messages.length > 0;

    return (
        <section className="chat-window-card" aria-label="Clinical Conversation Area">
            <div className="chat-window-header">
                <div className="chat-header-title">
                    <span className="chat-header-icon" aria-hidden="true">💬</span>
                    <h2>Clinical Consultation</h2>
                </div>
                {hasMessages && (
                    <span className="message-count-pill">
                        {messages.length} {messages.length === 1 ? "turn" : "turns"}
                    </span>
                )}
            </div>

            <div className="chat-messages-container">
                {!hasMessages ? (
                    <div className="empty-conversation-state">
                        <div className="empty-icon-shield" aria-hidden="true">
                            <svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                                <path d="M22 12h-4l-3 9L9 3l-3 9H2" />
                            </svg>
                        </div>
                        <h3>Clinical Decision Support Assistant</h3>
                        <p className="empty-description">
                            Evidence-grounded conversational reasoning powered by hosted OpenAI GPT and Biomedical Vector RAG.
                        </p>
                        <p className="empty-sub-hint">
                            Select an example scenario below or describe the patient's presentation:
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
                                <span className="chips-title">Suggested Quick Answers / Follow-ups:</span>
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
                                        No headache / neck stiffness
                                    </button>
                                    <button
                                        type="button"
                                        className="chip-btn"
                                        onClick={() => sendMessage("No pain or burning when urinating.")}
                                    >
                                        No urinary symptoms
                                    </button>
                                </div>
                            </div>
                        )}

                        <div ref={scrollBottomRef} />
                    </div>
                )}
            </div>
        </section>
    );
}

export default ChatWindow;
