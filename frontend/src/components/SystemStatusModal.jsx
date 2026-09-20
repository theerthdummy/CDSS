import { useChatState } from "../context/ChatContext";

function SystemStatusModal({ isOpen, onClose }) {
    const { systemHealth, refreshSystemHealth, llmMetadata } = useChatState();

    if (!isOpen) return null;

    const overall = systemHealth?.overall || "unknown";
    const isHealthy = overall === "healthy";
    const agents = systemHealth?.agents || {};
    const openai = systemHealth?.openai_gpt || {};

    const agentDisplayNames = {
        agent1_clarifier: "Agent 1 — Clinical Clarifier",
        agent2_retrieval: "Agent 2 — Biomedical Vector RAG",
        agent3_scanner: "Agent 3 — Live Evidence Scanner",
        agent4_fusion: "Agent 4 — Clinical Data Fusion",
        agent5_reasoning: "Agent 5 — Clinical Decision Support",
    };

    return (
        <div className="modal-backdrop" onClick={onClose} role="presentation">
            <div
                className="modal-card"
                onClick={(e) => e.stopPropagation()}
                role="dialog"
                aria-modal="true"
                aria-labelledby="system-status-title"
            >
                <div className="modal-header">
                    <div className="modal-title-group">
                        <span className={`status-badge ${isHealthy ? "status-badge-healthy" : "status-badge-warning"}`}>
                            {isHealthy ? "Operational" : overall.toUpperCase()}
                        </span>
                        <h3 id="system-status-title">System Infrastructure & Agents</h3>
                    </div>
                    <button
                        type="button"
                        className="modal-close-btn"
                        onClick={onClose}
                        aria-label="Close dialog"
                    >
                        ✕
                    </button>
                </div>

                <div className="modal-body">
                    {/* Hosted Model Card */}
                    <div className="status-section">
                        <div className="status-section-title">Primary Clinical Intelligence Model</div>
                        <div className="model-info-grid">
                            <div className="info-row">
                                <span className="info-label">Hosted Provider:</span>
                                <span className="info-value font-mono">{openai.provider || llmMetadata?.provider || "OpenAI"}</span>
                            </div>
                            <div className="info-row">
                                <span className="info-label">Model Engine:</span>
                                <span className="info-value font-mono model-tag">{openai.model || llmMetadata?.model || "openai/gpt-oss-120b"}</span>
                            </div>
                            <div className="info-row">
                                <span className="info-label">API Gateway:</span>
                                <span className="info-value font-mono text-xs">{openai.base_url || "https://api.groq.com/openai/v1"}</span>
                            </div>
                            <div className="info-row">
                                <span className="info-label">SDK Architecture:</span>
                                <span className="info-value status-pill success">AsyncOpenAI (Non-blocking)</span>
                            </div>
                        </div>
                    </div>

                    {/* Downstream Agents Grid */}
                    <div className="status-section">
                        <div className="status-section-title">Multi-Agent Microservices</div>
                        <div className="agents-status-list">
                            {Object.entries(agentDisplayNames).map(([key, displayName]) => {
                                const agentInfo = agents[key];
                                const agentHealthy = agentInfo?.status === "healthy";
                                return (
                                    <div key={key} className="agent-status-row">
                                        <div className="agent-name-cell">
                                            <span className={`agent-dot ${agentHealthy ? "dot-online" : "dot-offline"}`} />
                                            <span>{displayName}</span>
                                        </div>
                                        <div className="agent-meta-cell">
                                            <span className="agent-url">{agentInfo?.url || "N/A"}</span>
                                            <span className={`agent-tag ${agentHealthy ? "tag-online" : "tag-offline"}`}>
                                                {agentHealthy ? "Online" : "Unreachable"}
                                            </span>
                                        </div>
                                    </div>
                                );
                            })}
                        </div>
                    </div>
                </div>

                <div className="modal-footer">
                    <button
                        type="button"
                        className="btn-refresh"
                        onClick={refreshSystemHealth}
                    >
                        Refresh Health Audit
                    </button>
                    <button
                        type="button"
                        className="btn-primary"
                        onClick={onClose}
                    >
                        Done
                    </button>
                </div>
            </div>
        </div>
    );
}

export default SystemStatusModal;
