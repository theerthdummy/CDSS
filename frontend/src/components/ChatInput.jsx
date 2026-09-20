import { useRef, useEffect } from "react";
import { useChatState } from "../context/ChatContext";

function ChatInput({ value = "", onChange, onSend, disabled }) {
    const { consultationMode, loadingStage } = useChatState();
    const textareaRef = useRef(null);

    // Auto-resize textarea based on content
    useEffect(() => {
        const textarea = textareaRef.current;
        if (!textarea) return;
        textarea.style.height = "auto";
        const newHeight = Math.min(Math.max(textarea.scrollHeight, 44), 160);
        textarea.style.height = `${newHeight}px`;
    }, [value]);

    const handleSubmit = (event) => {
        event.preventDefault();
        const trimmed = value.trim();
        if (!trimmed || disabled) return;
        onSend(trimmed);
        if (onChange) {
            onChange("");
        }
        if (textareaRef.current) {
            textareaRef.current.style.height = "44px";
        }
    };

    const handleKeyDown = (event) => {
        if (event.key === "Enter" && !event.shiftKey) {
            event.preventDefault();
            handleSubmit(event);
        }
    };

    const isDeepMode = consultationMode === "deep_diagnostic";

    return (
        <form className="chat-input-container" onSubmit={handleSubmit}>
            <div className="input-mode-indicator">
                <span className="indicator-dot" />
                <span className="indicator-text">
                    Active Mode: <strong>{isDeepMode ? "5-Agent Deep Diagnostic Pipeline" : "Conversational Consultation (Fast)"}</strong>
                </span>
            </div>

            <div className="input-box-wrapper">
                <textarea
                    ref={textareaRef}
                    value={value}
                    onChange={(e) => onChange && onChange(e.target.value)}
                    onKeyDown={handleKeyDown}
                    placeholder={
                        isDeepMode
                            ? "Enter comprehensive patient presentation or clinical note for 5-agent pipeline analysis..."
                            : "Type patient symptoms, history, or answers to follow-up questions..."
                    }
                    aria-label="Clinical message or symptom input"
                    rows={1}
                    disabled={disabled}
                />

                <div className="input-actions-row">
                    <span className="input-shortcut-hint">
                        Press <kbd>Enter ↵</kbd> to send · <kbd>Shift + Enter</kbd> for line break
                    </span>

                    <button
                        type="submit"
                        className={`btn-send-message ${isDeepMode ? "btn-send-deep" : ""}`}
                        disabled={disabled || !value.trim()}
                        aria-label={disabled ? "Processing clinical response" : "Send message"}
                    >
                        {disabled ? (
                            <>
                                <span className="btn-spinner" aria-hidden="true" />
                                <span>{loadingStage ? "Analyzing..." : "Processing..."}</span>
                            </>
                        ) : (
                            <>
                                <span>{isDeepMode ? "Run Analysis" : "Send"}</span>
                                <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
                                    <line x1="22" y1="2" x2="11" y2="13" />
                                    <polygon points="22 2 15 22 11 13 2 9 22 2" />
                                </svg>
                            </>
                        )}
                    </button>
                </div>
            </div>
        </form>
    );
}

export default ChatInput;
