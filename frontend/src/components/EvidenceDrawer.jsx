import { useChatState } from "../context/ChatContext";

function EvidenceDrawer({ isOpen, onClose }) {
    const { evidence, clinicalAssessment } = useChatState();

    if (!isOpen) return null;

    const ragItems = evidence.filter((item) => item.source === "Vector RAG" || item.document_type === "literature");
    const webItems = evidence.filter((item) => item.source === "Web Scanner" || item.source === "web");

    return (
        <div className="drawer-backdrop" onClick={onClose} role="presentation">
            <aside
                className="evidence-drawer"
                onClick={(e) => e.stopPropagation()}
                role="dialog"
                aria-modal="true"
                aria-labelledby="evidence-drawer-title"
            >
                <div className="drawer-header">
                    <div className="drawer-title-group">
                        <span className="drawer-icon" aria-hidden="true">
                            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                                <path d="M4 19.5A2.5 2.5 0 0 1 6.5 17H20" />
                                <path d="M6.5 2H20v20H6.5A2.5 2.5 0 0 1 4 19.5v-15A2.5 2.5 0 0 1 6.5 2z" />
                            </svg>
                        </span>
                        <div>
                            <h3 id="evidence-drawer-title">Clinical Evidence & Provenance</h3>
                            <p className="drawer-subtitle">Biomedical Vector RAG & Live Web Grounding Sources</p>
                        </div>
                    </div>
                    <button
                        type="button"
                        className="modal-close-btn"
                        onClick={onClose}
                        aria-label="Close evidence drawer"
                    >
                        ✕
                    </button>
                </div>

                <div className="drawer-body">
                    {/* Diagnostic Summary if available */}
                    {clinicalAssessment?.reasoning_summary && (
                        <div className="evidence-summary-card">
                            <span className="evidence-summary-title">Clinical Synthesis Grounding</span>
                            <p className="evidence-summary-text">{clinicalAssessment.reasoning_summary}</p>
                        </div>
                    )}

                    {/* Section 1: Biomedical Vector RAG */}
                    <div className="evidence-section">
                        <div className="evidence-section-header">
                            <div className="section-title-with-badge">
                                <h4>Biomedical Vector RAG (Qdrant)</h4>
                                <span className={`source-status-badge ${ragItems.length > 0 ? "status-success" : "status-neutral"}`}>
                                    {ragItems.length > 0 ? `${ragItems.length} verified` : "No direct matches"}
                                </span>
                            </div>
                            <span className="source-meta">Agent 2 · Vector Similarity Search</span>
                        </div>

                        {ragItems.length === 0 ? (
                            <div className="empty-evidence-box">
                                <span>No direct literature articles cited for this turn.</span>
                            </div>
                        ) : (
                            <div className="evidence-cards-list">
                                {ragItems.map((item, idx) => (
                                    <article key={`rag-${idx}`} className="evidence-card">
                                        <div className="evidence-card-header">
                                            <span className="evidence-title">{item.title || "Medical Evidence Article"}</span>
                                            {typeof item.score === "number" && (
                                                <span className="relevance-score" title="Vector similarity score">
                                                    Match: {(item.score * 100).toFixed(1)}%
                                                </span>
                                            )}
                                        </div>
                                        <p className="evidence-snippet">{item.snippet || item.text}</p>
                                        <div className="evidence-card-footer">
                                            <span className="provenance-tag">Source: {item.source}</span>
                                            {item.document_type && (
                                                <span className="doc-type-tag">{item.document_type}</span>
                                            )}
                                        </div>
                                    </article>
                                ))}
                            </div>
                        )}
                    </div>

                    {/* Section 2: Live Web Evidence Scanner */}
                    <div className="evidence-section">
                        <div className="evidence-section-header">
                            <div className="section-title-with-badge">
                                <h4>Live Web Clinical Evidence</h4>
                                <span className={`source-status-badge ${webItems.length > 0 ? "status-success" : "status-warning"}`}>
                                    {webItems.length > 0 ? `${webItems.length} citations` : "Unavailable / Timeout"}
                                </span>
                            </div>
                            <span className="source-meta">Agent 3 · Evidence Scanner</span>
                        </div>

                        {webItems.length === 0 ? (
                            <div className="empty-evidence-box warning-border">
                                <span className="empty-warning-title">⚠ Live Web Scanner Unavailable</span>
                                <p className="empty-warning-desc">
                                    Web retrieval timed out or is unreachable. Clinical reasoning safely proceeded with Biomedical Vector RAG literature grounding.
                                </p>
                            </div>
                        ) : (
                            <div className="evidence-cards-list">
                                {webItems.map((item, idx) => (
                                    <article key={`web-${idx}`} className="evidence-card">
                                        <div className="evidence-card-header">
                                            <span className="evidence-title">{item.title || "Clinical Web Source"}</span>
                                            {item.url && (
                                                <a
                                                    href={item.url}
                                                    target="_blank"
                                                    rel="noopener noreferrer"
                                                    className="evidence-link"
                                                    title="Open external source"
                                                >
                                                    View Source ↗
                                                </a>
                                            )}
                                        </div>
                                        <p className="evidence-snippet">{item.snippet || item.text}</p>
                                        <div className="evidence-card-footer">
                                            <span className="provenance-tag">Live Web Evidence</span>
                                        </div>
                                    </article>
                                ))}
                            </div>
                        )}
                    </div>
                </div>

                <div className="drawer-footer">
                    <span className="drawer-disclaimer">
                        Evidence provenance is preserved to support transparent clinician verification.
                    </span>
                </div>
            </aside>
        </div>
    );
}

export default EvidenceDrawer;
