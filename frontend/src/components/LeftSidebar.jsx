import { useState } from "react";
import { useChatState } from "../context/ChatContext";

function LeftSidebar({ onOpenSystemStatus, onShowToast }) {
    const {
        recentSessions,
        deleteRecentSession,
        clearAllSessions,
        resetConsultation,
        sendMessage,
        isLoading,
        systemHealth,
        isSidebarOpen,
        setIsSidebarOpen,
        isTelemetryOpen,
        setIsTelemetryOpen,
    } = useChatState();

    const [searchTerm, setSearchTerm] = useState("");

    const filteredSessions = recentSessions.filter((session) =>
        session.title.toLowerCase().includes(searchTerm.toLowerCase()) ||
        (session.preview && session.preview.toLowerCase().includes(searchTerm.toLowerCase()))
    );

    const handleNewConsultation = () => {
        if (isLoading) return;
        resetConsultation();
        onShowToast && onShowToast("New clinical consultation started.", "info");
    };

    const handleSelectSession = (session) => {
        if (isLoading) return;
        resetConsultation();
        sendMessage(session.title);
    };

    const handleDelete = (e, id) => {
        e.stopPropagation();
        deleteRecentSession(id);
    };

    const isHealthy = systemHealth?.overall === "healthy";

    return (
        <aside className={`docpilot-left-sidebar ${isSidebarOpen ? "sidebar-open" : "sidebar-collapsed"}`}>
            {/* Top Brand Header */}
            <div className="sidebar-brand-row">
                <div className="sidebar-brand-group">
                    <div className="brand-logo-glow" aria-hidden="true">
                        <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
                            <path d="M22 12h-4l-3 9L9 3l-3 9H2" />
                        </svg>
                    </div>
                    <div className="brand-text-block">
                        <div className="brand-title-wrap">
                            <h2 className="brand-name">DocPilot</h2>
                            <span className="brand-badge-lab">CDSS</span>
                        </div>
                        <span className="brand-subtext">@clinical_copilot</span>
                    </div>
                </div>

                <button
                    type="button"
                    className="btn-sidebar-toggle"
                    onClick={() => setIsSidebarOpen(!isSidebarOpen)}
                    title={isSidebarOpen ? "Collapse Sidebar" : "Expand Sidebar"}
                    aria-label="Toggle Sidebar"
                >
                    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                        <rect x="3" y="3" width="18" height="18" rx="2" ry="2" />
                        <line x1="9" y1="3" x2="9" y2="21" />
                    </svg>
                </button>
            </div>

            {/* + New Conversation Primary Button */}
            <div className="sidebar-action-wrap">
                <button
                    type="button"
                    className="btn-new-conversation"
                    onClick={handleNewConsultation}
                    disabled={isLoading}
                >
                    <span className="plus-icon">+</span>
                    <span>New Conversation</span>
                </button>
            </div>

            {/* Upload Source / Case File Card */}
            <div className="sidebar-upload-card" onClick={() => onShowToast && onShowToast("Clinical document ingestion ready", "info")}>
                <div className="upload-icon-box">
                    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                        <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z" />
                        <polyline points="14 2 14 8 20 8" />
                        <line x1="12" y1="18" x2="12" y2="12" />
                        <line x1="9" y1="15" x2="15" y2="15" />
                    </svg>
                </div>
                <span className="upload-card-text">Upload source files</span>
            </div>

            {/* Recent Chats Section */}
            <div className="sidebar-recent-section">
                <div className="recent-header-row">
                    <span className="recent-label">RECENT CHATS</span>
                    {recentSessions.length > 0 && (
                        <button
                            type="button"
                            className="btn-clear-recent"
                            onClick={clearAllSessions}
                            title="Clear all recent chats"
                        >
                            Clear
                        </button>
                    )}
                </div>

                {/* Search Box */}
                <div className="recent-search-wrap">
                    <svg className="search-icon" width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                        <circle cx="11" cy="11" r="8" />
                        <line x1="21" y1="21" x2="16.65" y2="16.65" />
                    </svg>
                    <input
                        type="text"
                        placeholder="Search..."
                        value={searchTerm}
                        onChange={(e) => setSearchTerm(e.target.value)}
                        className="recent-search-input"
                    />
                </div>

                {/* Chat History List */}
                <div className="recent-sessions-list">
                    {filteredSessions.length === 0 ? (
                        <div className="empty-history-hint">
                            {searchTerm ? "No matching chats" : "No recent consultations"}
                        </div>
                    ) : (
                        filteredSessions.map((session) => (
                            <div
                                key={session.id}
                                className="recent-session-item"
                                onClick={() => handleSelectSession(session)}
                            >
                                <span className="session-item-title" title={session.title}>
                                    {session.title}
                                </span>
                                <button
                                    type="button"
                                    className="btn-delete-session"
                                    onClick={(e) => handleDelete(e, session.id)}
                                    title="Delete chat"
                                    aria-label="Delete chat"
                                >
                                    ✕
                                </button>
                            </div>
                        ))
                    )}
                </div>
            </div>

            {/* Bottom Footer / Telemetry & Health */}
            <div className="sidebar-footer">
                <button
                    type="button"
                    className="sidebar-telemetry-toggle-btn"
                    onClick={() => setIsTelemetryOpen(!isTelemetryOpen)}
                >
                    <span className="telemetry-icon">📊</span>
                    <span>Patient Telemetry</span>
                    <span className={`telemetry-pill-state ${isTelemetryOpen ? "active" : ""}`}>
                        {isTelemetryOpen ? "Open" : "View"}
                    </span>
                </button>

                <div className="sidebar-status-card" onClick={onOpenSystemStatus}>
                    <div className="status-indicator-dot-wrap">
                        <span className={`status-dot-ping ${isHealthy ? "status-ready" : "status-warning"}`} />
                    </div>
                    <div className="status-meta">
                        <span className="status-model-name">GPT-OSS 120B</span>
                        <span className="status-agent-count">
                            {isHealthy ? "5 Microservices Ready" : "System Degraded"}
                        </span>
                    </div>
                </div>
            </div>
        </aside>
    );
}

export default LeftSidebar;
