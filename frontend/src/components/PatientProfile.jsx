function PatientProfile({ data }) {
    if (!data) {
        return null;
    }

    const { clinical_assessment, reasoning, decision_support, uncertainty, agent_metadata } = data;
    const recommendedActions = Array.isArray(decision_support?.recommended_actions)
        ? decision_support.recommended_actions
        : [];

    return (
        <section className="patient-profile-card" aria-label="CDSS Output">
            <div className="patient-profile-header">
                <div className="patient-profile-title">
                    <span className="profile-icon" aria-hidden="true">
                        <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                            <path d="M22 12h-4l-3 9L9 3l-3 9H2" />
                        </svg>
                    </span>
                    <span>AI Clinical Assessment</span>
                </div>
            </div>

            <div className="profile-item" style={{ padding: "16px", fontSize: "1.05em", lineHeight: "1.6" }}>
                <span className="profile-value">
                    {reasoning?.reasoning_summary || clinical_assessment?.primary_interpretation || "No interpretation generated."}
                </span>
            </div>

            {clinical_assessment?.primary_interpretation && (
                <div className="profile-item" style={{ padding: "0 16px 16px", lineHeight: "1.5" }}>
                    <div className="profile-label">Clinical Assessment</div>
                    <div className="profile-value">{clinical_assessment.primary_interpretation}</div>
                </div>
            )}

            {reasoning?.reasoning_summary && (
                <div className="profile-item" style={{ padding: "0 16px 16px", lineHeight: "1.5" }}>
                    <div className="profile-label">Reasoning Summary</div>
                    <div className="profile-value">{reasoning.reasoning_summary}</div>
                </div>
            )}

            {recommendedActions.length > 0 && (
                <div className="profile-item" style={{ padding: "0 16px 16px", lineHeight: "1.5" }}>
                    <div className="profile-label">Recommended Actions</div>
                    <div className="profile-value">{recommendedActions.slice(0, 3).join("; ")}</div>
                </div>
            )}

            <div className="profile-item" style={{ padding: "0 16px 16px", lineHeight: "1.5" }}>
                <div className="profile-label">Execution Source</div>
                <div className="profile-value">
                    {agent_metadata?.reasoning_mode === "deterministic_fallback"
                        ? "Agent 5 deterministic fallback"
                        : "Agent 5 clinical reasoning"}
                </div>
                {typeof uncertainty?.confidence_score === "number" && (
                    <div className="profile-value">Confidence: {(uncertainty.confidence_score * 100).toFixed(0)}%</div>
                )}
            </div>
        </section>
    );
}

export default PatientProfile;
