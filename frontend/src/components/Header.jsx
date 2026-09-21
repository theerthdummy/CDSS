import ThemeToggle from "./ThemeToggle";
import { useChatState } from "../context/ChatContext";

function Header({ onOpenSystemStatus, onReset, onShowToast }) {
    const {
        systemHealth,
        isLoading,
        isSidebarOpen,
        setIsSidebarOpen,
        isTelemetryOpen,
        setIsTelemetryOpen,
        evidence,
    } = useChatState();

    const isHealthy = systemHealth?.overall === "healthy";
    const statusLabel = isHealthy ? "System Ready" : systemHealth?.overall === "degraded" ? "Degraded" : "Connecting...";

    const handleReset = () => {
        if (isLoading) return;
        onReset && onReset();
        onShowToast && onShowToast("Consultation reset. Context cleared.", "info");
    };

    return (
        <header className="chatgpt-top-header">
            <div className="top-header-left">
                {!isSidebarOpen && (
                    <button
                        type="button"
                        className="btn-expand-sidebar-top"
                        onClick={() => setIsSidebarOpen(true)}
                        title="Open Navigation Sidebar"
                        aria-label="Open Sidebar"
                    >
                        <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                            <rect x="3" y="3" width="18" height="18" rx="2" ry="2" />
                            <line x1="9" y1="3" x2="9" y2="21" />
                        </svg>
                    </button>
                )}

                <div className="header-scope-indicator">
                    <span className="scope-dot" />
                    <span className="scope-text">
                        Scope: <strong>Active Clinical Decision Support (CDSS)</strong>
                    </span>
                </div>
            </div>

            {/* Navigation Badges / Links (DocPilot Studio Style) */}
            <div className="top-header-center-links">
                <span className="studio-nav-item active-nav">Clinical Co-Pilot</span>
                <span className="studio-nav-item" onClick={() => onShowToast && onShowToast("Biomedical Vector RAG (BioBERT + Qdrant) Active", "info")}>Vector RAG</span>
                <span className="studio-nav-item" onClick={() => onShowToast && onShowToast("PubMed & Tavily Evidence Scanner Active", "info")}>PubMed Scanner</span>
                <span className="studio-nav-item" onClick={() => setIsTelemetryOpen(true)}>
                    📖 Patient File {evidence.length > 0 ? `(${evidence.length})` : ""}
                </span>
            </div>

            {/* Right Actions */}
            <div className="top-header-right-actions">
                <button
                    type="button"
                    className={`status-chip-btn ${isHealthy ? "status-ready" : "status-warning"}`}
                    onClick={onOpenSystemStatus}
                    title="View 5-Agent Microservice Health & LLM Status"
                >
                    <span className="chip-dot" />
                    <span>{statusLabel}</span>
                </button>

                <ThemeToggle />

                <button
                    type="button"
                    className="btn-clear-consultation"
                    onClick={handleReset}
                    disabled={isLoading}
                    title="Clear current case and start fresh consultation"
                >
                    ✕ Clear Case
                </button>
            </div>
        </header>
    );
}

export default Header;
