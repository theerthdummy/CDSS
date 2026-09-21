import { useRef, useEffect, useState } from "react";
import { useChatState } from "../context/ChatContext";

function ChatInput({ value = "", onChange, onSend, disabled }) {
    const {
        consultationMode,
        setConsultationMode,
        loadingStage,
        isTelemetryOpen,
        setIsTelemetryOpen,
        uploadAndAnalyze,
        isLoading,
    } = useChatState();
    const textareaRef = useRef(null);
    const fileInputRef = useRef(null);
    const [isDragOver, setIsDragOver] = useState(false);
    const [pendingFile, setPendingFile] = useState(null); // file staged for upload

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

        // If there's a staged file, upload it (optionally with the typed message as instruction)
        if (pendingFile) {
            const instruction = value.trim() || null;
            uploadAndAnalyze(pendingFile, instruction);
            setPendingFile(null);
            if (onChange) onChange("");
            if (textareaRef.current) textareaRef.current.style.height = "48px";
            return;
        }

        const trimmed = value.trim();
        if (!trimmed || disabled) return;
        onSend(trimmed);
        if (onChange) onChange("");
        if (textareaRef.current) textareaRef.current.style.height = "48px";
    };

    const handleKeyDown = (event) => {
        if (event.key === "Enter" && !event.shiftKey) {
            event.preventDefault();
            handleSubmit(event);
        }
        // Escape clears the staged file
        if (event.key === "Escape" && pendingFile) {
            setPendingFile(null);
        }
    };

    const handleFileChange = (e) => {
        const file = e.target.files?.[0];
        if (file) setPendingFile(file);
        // Reset so the same file can be selected again
        e.target.value = "";
    };

    // Drag & drop handlers on the whole input card
    const handleDragOver = (e) => { e.preventDefault(); setIsDragOver(true); };
    const handleDragLeave = () => setIsDragOver(false);
    const handleDrop = (e) => {
        e.preventDefault();
        setIsDragOver(false);
        const file = e.dataTransfer.files?.[0];
        if (file) setPendingFile(file);
    };

    const isDeepMode = consultationMode === "deep_diagnostic";
    const hasFile = !!pendingFile;
    const canSend = (hasFile || value.trim()) && !disabled;

    // Decide placeholder text based on state
    let placeholder;
    if (hasFile) {
        placeholder = `Optionally add an instruction (e.g. "give prognosis"), or press ↵ to analyze…`;
    } else if (isDeepMode) {
        placeholder = "Enter full patient clinical note for 5-Agent Diagnostic Pipeline...";
    } else {
        placeholder = "Ask about your patient case, or drop a prescription/report file...";
    }

    return (
        <div
            className="chatgpt-input-dock-wrapper"
            onDragOver={handleDragOver}
            onDragLeave={handleDragLeave}
            onDrop={handleDrop}
        >
            {/* Hidden native file picker — must NOT be display:none for .click() to work */}
            <input
                ref={fileInputRef}
                type="file"
                accept=".pdf,.png,.jpg,.jpeg,.webp,.docx,.txt,.md"
                style={{
                    position: "absolute",
                    opacity: 0,
                    width: 0,
                    height: 0,
                    pointerEvents: "none",
                }}
                onChange={handleFileChange}
                aria-label="Upload medical document"
            />

            {/* Drag overlay indicator */}
            {isDragOver && (
                <div className="drop-overlay" aria-hidden="true">
                    <span className="drop-overlay-icon">📎</span>
                    <span className="drop-overlay-label">Drop to analyze</span>
                </div>
            )}

            {/* Staged file badge */}
            {hasFile && (
                <div className="staged-file-badge">
                    <span className="staged-file-icon">📄</span>
                    <span className="staged-file-name" title={pendingFile.name}>
                        {pendingFile.name.length > 38
                            ? `${pendingFile.name.slice(0, 35)}…`
                            : pendingFile.name}
                    </span>
                    <span className="staged-file-size">
                        ({(pendingFile.size / 1024).toFixed(0)} KB)
                    </span>
                    <button
                        type="button"
                        className="staged-file-remove"
                        onClick={() => setPendingFile(null)}
                        title="Remove file"
                        aria-label="Remove staged file"
                    >
                        ✕
                    </button>
                </div>
            )}

            <form
                className={`chatgpt-input-card${isDragOver ? " drag-active" : ""}${hasFile ? " has-file" : ""}`}
                onSubmit={handleSubmit}
            >
                <div className="input-textarea-row">
                    {/* Upload / paperclip button */}
                    <button
                        type="button"
                        className={`btn-upload-attachment${hasFile ? " has-file" : ""}`}
                        onClick={() => fileInputRef.current?.click()}
                        disabled={disabled}
                        title="📎 Upload prescription, lab report, or scan (PDF, image, DOCX) for direct OCR-based diagnosis"
                        aria-label="Upload medical document"
                    >
                        <svg width="17" height="17" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                            <path d="M21.44 11.05l-9.19 9.19a6 6 0 0 1-8.49-8.49l9.19-9.19a4 4 0 0 1 5.66 5.66L9.41 17.41a2 2 0 0 1-2.83-2.83l8.49-8.48" />
                        </svg>
                        {!hasFile && <span className="upload-btn-pulse" aria-hidden="true" />}
                    </button>

                    <textarea
                        ref={textareaRef}
                        value={value}
                        onChange={(e) => onChange && onChange(e.target.value)}
                        onKeyDown={handleKeyDown}
                        placeholder={placeholder}
                        aria-label="Clinical inquiry or case input"
                        rows={1}
                        disabled={disabled}
                    />

                    <button
                        type="submit"
                        className={`btn-chatgpt-send ${canSend ? "active" : ""}`}
                        disabled={!canSend}
                        aria-label={disabled ? "Processing clinical response" : hasFile ? "Analyze document" : "Send message"}
                    >
                        {disabled ? (
                            <span className="send-spinner" aria-hidden="true" />
                        ) : hasFile ? (
                            /* Upload/analyze icon when file staged */
                            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
                                <polyline points="16 16 12 12 8 16" />
                                <line x1="12" y1="12" x2="12" y2="21" />
                                <path d="M20.39 18.39A5 5 0 0 0 18 9h-1.26A8 8 0 1 0 3 16.3" />
                            </svg>
                        ) : (
                            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
                                <line x1="12" y1="19" x2="12" y2="5" />
                                <polyline points="5 12 12 5 19 12" />
                            </svg>
                        )}
                    </button>
                </div>

                {/* Bottom Model & Parameter Pills Bar */}
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

                        {/* Upload pill indicator */}
                        <button
                            type="button"
                            className={`mode-toggle-pill pill-upload${hasFile ? " pill-upload-active" : ""}`}
                            onClick={() => fileInputRef.current?.click()}
                            disabled={disabled}
                            title="Upload prescription, report, or scan for direct OCR-based diagnosis"
                        >
                            <span className="pill-label">📎 {hasFile ? pendingFile.name.slice(0, 20) + (pendingFile.name.length > 20 ? "…" : "") : "Upload Doc"}</span>
                            <span className="pill-sub">OCR</span>
                        </button>
                    </div>

                    <span className="input-hint-micro">
                        <kbd>↵</kbd> send · <kbd>⇧↵</kbd> line · drop file to analyze
                    </span>
                </div>
            </form>
        </div>
    );
}

export default ChatInput;
