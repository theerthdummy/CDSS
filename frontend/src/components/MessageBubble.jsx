import { useState } from "react";
import MarkdownRenderer from "./MarkdownRenderer";

function MessageBubble({ sender, text, kind = "text", timestamp, isUrgent, validation, onShowToast }) {
    const isUser = sender === "user";
    const isError = kind === "error";
    const [copied, setCopied] = useState(false);

    const handleCopy = async () => {
        try {
            await navigator.clipboard.writeText(text);
            setCopied(true);
            onShowToast && onShowToast("Copied response to clipboard", "success");
            setTimeout(() => setCopied(false), 2000);
        } catch {
            onShowToast && onShowToast("Failed to copy to clipboard", "error");
        }
    };

    const formattedTime = timestamp
        ? new Date(timestamp).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })
        : null;

    return (
        <article
            className={`message-row ${isUser ? "message-row-user" : "message-row-assistant"}`}
            aria-label={`${isUser ? "Clinician" : "Assistant"} message`}
        >
            <div className={`message-bubble ${isUrgent ? "message-bubble-urgent" : ""} ${isError ? "message-bubble-error" : ""}`}>
                {/* Header Row */}
                <div className="message-header-row">
                    <div className="message-sender-group">
                        <span className="sender-avatar" aria-hidden="true">
                            {isUser ? "👤" : "🩺"}
                        </span>
                        <span className="message-sender">
                            {isUser ? "Attending Clinician / Physician" : "CDSS Clinical Co-Pilot"}
                        </span>
                    </div>

                    <div className="message-meta-group">
                        {isUrgent && (
                            <span className="badge-urgent-tag">
                                ⚠️ Urgent Safety Alert
                            </span>
                        )}
                        {validation?.is_valid && !isUser && (
                            <span className="badge-verified-tag" title="Verified by Anti-Hallucination Guardrail">
                                ✓ Evidence-Grounded
                            </span>
                        )}
                        {formattedTime && <span className="message-time">{formattedTime}</span>}
                        <button
                            type="button"
                            className="btn-copy-msg"
                            onClick={handleCopy}
                            title={copied ? "Copied!" : "Copy message"}
                            aria-label="Copy message text"
                        >
                            {copied ? (
                                <span style={{ color: "var(--color-primary)", fontWeight: "bold" }}>✓ Copied</span>
                            ) : (
                                <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
                                    <rect x="9" y="9" width="13" height="13" rx="2" ry="2" />
                                    <path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1" />
                                </svg>
                            )}
                        </button>
                    </div>
                </div>

                {/* Content */}
                <div className="message-text">
                    {isUser ? (
                        <p className="user-plain-text">{text}</p>
                    ) : (
                        <MarkdownRenderer content={text} />
                    )}
                </div>
            </div>
        </article>
    );
}

export default MessageBubble;
