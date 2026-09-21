import { useRef, useEffect } from "react";
import { useChatState } from "../context/ChatContext";

function ChatInput({ value = "", onChange, onSend, disabled }) {
    const {
        consultationMode,
        setConsultationMode,
        loadingStage,
        isTelemetryOpen,
        setIsTelemetryOpen,
    } = useChatState();
    const textareaRef = useRef(null);

    // Auto-resize textarea based on content
    useEffect(() => {
        const textarea = textareaRef.current;
        if (!textarea) return;
        textarea.style.height = "auto";
        const newHeight = Math.min(Math.max(textarea.scrollHeight, 48), 180);
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
            textareaRef.current.style.height = "48px";
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
        <div className="chatgpt-input-dock-wrapper">
            <form className="chatgpt-input-card" onSubmit={handleSubmit}>
                <div className="input-textarea-row">
                    <textarea
                        ref={textareaRef}
                        value={value}
                        onChange={(e) => onChange && onChange(e.target.value)}
                        onKeyDown={handleKeyDown}
                        placeholder={
                            isDeepMode
                                ? "Enter full patient clinical note for 5-Agent Diagnostic Pipeline..."
                                : "Ask anything about your patient case, differential, or clinical guidelines..."
                        }
                        aria-label="Clinical inquiry or case input"
                        rows={1}
                        disabled={disabled}
                    />

                    <button
                        type="submit"
                        className={`btn-chatgpt-send ${value.trim() ? "active" : ""}`}
                        disabled={disabled || !value.trim()}
                        aria-label={disabled ? "Processing clinical response" : "Send message"}
                    >
                        {disabled ? (
                            <span className="send-spinner" aria-hidden="true" />
                        ) : (
                            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
                                <line x1="12" y1="19" x2="12" y2="5" />
                                <polyline points="5 12 12 5 19 12" />
                            </svg>
                        )}
                    </button>
                </div>

                {/* Bottom Model & Parameter Pills Bar (DocPilot / ChatGPT Studio Style) */}
                <div className="input-pills-bar">
                    <div className="pills-scroll-group">
                        <div className="model-pill-badge" title="Active Reasoning Engine">
                            <span className="pill-dot-green" />
                            <span className="pill-label">GPT-OSS 120B</span>
                            <span className="pill-sub">MODEL</span>
                        </div>

                        <button
                            type="button"
                            className={`mode-toggle-pill ${isDeepMode ? "pill-deep" : "pill-fast"}`}
                            onClick={() => setConsultationMode(isDeepMode ? "conversational" : "deep_diagnostic")}
                            title="Click to toggle between Conversational and 5-Agent Deep Diagnostic Mode"
                        >
                            <span className="pill-label">
                                {isDeepMode ? "5-Agent Deep Diagnostic" : "Conversational Co-Pilot"}
                            </span>
                            <span className="pill-sub">MODE ▾</span>
                        </button>

                        <div className="model-pill-badge" title="Hybrid Biomedical RAG & PubMed Intelligence Active">
                            <span className="pill-label">BioBERT + PubMed</span>
                            <span className="pill-sub">RETRIEVER</span>
                        </div>

                        <button
                            type="button"
                            className={`telemetry-toggle-pill ${isTelemetryOpen ? "telemetry-active" : ""}`}
                            onClick={() => setIsTelemetryOpen(!isTelemetryOpen)}
                            title="Toggle Live Patient Demographics, Vitals & Hypotheses Drawer"
                        >
                            <span className="pill-label">📊 Patient Telemetry</span>
                            <span className="pill-sub">{isTelemetryOpen ? "OPEN" : "INSPECT"}</span>
                        </button>
                    </div>

                    <span className="input-hint-micro">
                        <kbd>↵</kbd> send · <kbd>⇧↵</kbd> line
                    </span>
                </div>
            </form>
        </div>
    );
}

export default ChatInput;
