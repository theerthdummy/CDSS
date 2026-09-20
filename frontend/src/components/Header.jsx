import ThemeToggle from "./ThemeToggle";
import { useChatState } from "../context/ChatContext";

function Header({ onOpenSystemStatus, onReset, onShowToast }) {
    const { systemHealth, consultationMode, setConsultationMode, isLoading } = useChatState();

    const isHealthy = systemHealth?.overall === "healthy";
    const statusLabel = isHealthy ? "System Ready" : systemHealth?.overall === "degraded" ? "Degraded" : "Connecting...";

    const handleReset = () => {
        if (isLoading) return;
        onReset && onReset();
        onShowToast && onShowToast("New clinical consultation started. Patient context reset.", "info");
    };

    return (
        <header className="app-header">
            <div className="header-container">
                {/* Brand & Title */}
                <div className="header-brand">
                    <div className="header-mark" aria-hidden="true">
                        <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
                            <line x1="12" y1="4" x2="12" y2="20" />
                            <line x1="4" y1="12" x2="20" y2="12" />
                        </svg>
                    </div>
                    <div className="header-titles">
                        <div className="header-title-line">
                            <h1>Clinical Decision Support System</h1>
                            <span className="version-pill">v3.0</span>
                        </div>
                        <p className="header-subtitle">Evidence-grounded progressive clinical reasoning</p>
                    </div>
                </div>

                {/* Center / Mode Selector */}
                <div className="consultation-mode-toggle" role="group" aria-label="Consultation Mode">
                    <button
                        type="button"
                        className={`mode-btn ${consultationMode === "conversational" ? "mode-btn-active" : ""}`}
                        onClick={() => setConsultationMode("conversational")}
                        title="Fast interactive clinical dialogue with progressive questioning"
                    >
                        <span className="mode-dot" />
                        <span>Conversational Consultation</span>
                    </button>
                    <button
                        type="button"
                        className={`mode-btn ${consultationMode === "deep_diagnostic" ? "mode-btn-active" : ""}`}
                        onClick={() => setConsultationMode("deep_diagnostic")}
                        title="Full 5-Agent Diagnostic Fusion Pipeline (Clarifier + RAG + Web + Fusion + Reasoning)"
                    >
                        <span className="mode-dot" />
                        <span>5-Agent Deep Diagnostic</span>
                    </button>
                </div>

                {/* Actions & Utilities */}
                <div className="header-actions">
                    {/* Live System Status Trigger */}
                    <button
                        type="button"
                        className={`status-indicator-btn ${isHealthy ? "status-ready" : "status-warning"}`}
                        onClick={onOpenSystemStatus}
                        title="Click to view infrastructure & agent health"
                        aria-label="View system infrastructure and agent status"
                    >
                        <span className="status-dot" aria-hidden="true" />
                        <span className="status-label">{statusLabel}</span>
                    </button>

                    {/* Theme Toggle (Light / Dark / System) */}
                    <ThemeToggle />

                    {/* New Case Button */}
                    <button
                        type="button"
                        className="btn-new-case"
                        onClick={handleReset}
                        disabled={isLoading}
                        title="Clear conversation and begin a new patient case"
                        aria-label="Start new clinical consultation"
                    >
                        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
                            <path d="M3 12a9 9 0 0 1 9-9 9.75 9.75 0 0 1 6.74 2.74L21 8" />
                            <path d="M21 3v5h-5" />
                            <path d="M21 12a9 9 0 0 1-9 9 9.75 9.75 0 0 1-6.74-2.74L3 16" />
                            <path d="M8 16H3v5" />
                        </svg>
                        <span>New Consultation</span>
                    </button>
                </div>
            </div>
        </header>
    );
}

export default Header;
