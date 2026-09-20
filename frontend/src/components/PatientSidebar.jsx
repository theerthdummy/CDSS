import { useChatState } from "../context/ChatContext";

function PatientSidebar({ onOpenEvidence }) {
    const { patientState, clinicalAssessment, evidence, validation } = useChatState();

    const confirmed = patientState.confirmed_symptoms || [];
    const denied = patientState.denied_symptoms || [];
    const unknown = patientState.unknown_symptoms || [];
    const measurements = patientState.measurements || {};
    const age = patientState.age;
    const gender = patientState.gender;
    const temp = measurements.temperature;

    const isUrgent = clinicalAssessment?.is_urgent;
    const priority = clinicalAssessment?.priority_level || (isUrgent ? "URGENT" : "ROUTINE");
    const redFlags = clinicalAssessment?.red_flags || [];
    const differentials = clinicalAssessment?.differential_considerations || patientState.suspected_hypotheses || [];

    const isTempElevated = temp && (temp.includes("100") || temp.includes("101") || temp.includes("102") || temp.includes("103") || temp.includes("104") || parseFloat(temp) >= 38.0);

    return (
        <aside className="patient-sidebar" aria-label="Clinical State & Patient Profile">
            {/* Urgent Red Flag Alert Banner */}
            {isUrgent && (
                <div className="urgent-banner" role="alert">
                    <div className="urgent-banner-header">
                        <span className="urgent-icon" aria-hidden="true">⚠</span>
                        <strong>Urgent Medical Assessment Indicated</strong>
                    </div>
                    {redFlags.length > 0 && (
                        <div className="urgent-flags-list">
                            {redFlags.map((flag, idx) => (
                                <span key={idx} className="urgent-flag-pill">• {flag}</span>
                            ))}
                        </div>
                    )}
                </div>
            )}

            {/* Vitals & Demographics Card */}
            <div className="sidebar-card">
                <div className="sidebar-card-header">
                    <span className="card-icon" aria-hidden="true">
                        <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                            <path d="M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2" />
                            <circle cx="12" cy="7" r="4" />
                        </svg>
                    </span>
                    <h3>Patient Demographics & Vitals</h3>
                </div>
                <div className="vitals-grid">
                    <div className="vital-item">
                        <span className="vital-label">Age</span>
                        <span className="vital-val">{age ? `${age} y.o` : "Unspecified"}</span>
                    </div>
                    <div className="vital-item">
                        <span className="vital-label">Gender</span>
                        <span className="vital-val capitalize">{gender || "Unspecified"}</span>
                    </div>
                    <div className="vital-item">
                        <span className="vital-label">Temperature</span>
                        <span className={`vital-val ${isTempElevated ? "vital-elevated" : ""}`}>
                            {temp || "Not reported"}
                            {isTempElevated && <span className="fever-badge">Fever</span>}
                        </span>
                    </div>
                    <div className="vital-item">
                        <span className="vital-label">Medical History</span>
                        <span className="vital-val text-xs">
                            {patientState.medical_history?.length > 0
                                ? patientState.medical_history.join(", ")
                                : patientState.medical_history_status === "NONE_REPORTED"
                                ? "None reported"
                                : "Unknown"}
                        </span>
                    </div>
                </div>
            </div>

            {/* Confirmed / Denied / Unknown Symptoms */}
            <div className="sidebar-card">
                <div className="sidebar-card-header">
                    <span className="card-icon" aria-hidden="true">
                        <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                            <path d="M9 11l3 3L22 4" />
                            <path d="M21 12v7a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h11" />
                        </svg>
                    </span>
                    <h3>Symptom Verification</h3>
                </div>

                {/* Confirmed */}
                <div className="symptom-group">
                    <div className="group-header">
                        <span className="group-title confirmed-title">Confirmed Present</span>
                        <span className="badge-count confirmed-count">{confirmed.length}</span>
                    </div>
                    {confirmed.length === 0 ? (
                        <span className="symptom-empty-hint">None reported yet</span>
                    ) : (
                        <div className="symptom-tags-wrap">
                            {confirmed.map((sym, idx) => (
                                <span key={idx} className="symptom-tag tag-confirmed">
                                    ✓ {sym}
                                </span>
                            ))}
                        </div>
                    )}
                </div>

                {/* Explicitly Denied */}
                <div className="symptom-group">
                    <div className="group-header">
                        <span className="group-title denied-title">Explicitly Denied</span>
                        <span className="badge-count denied-count">{denied.length}</span>
                    </div>
                    {denied.length === 0 ? (
                        <span className="symptom-empty-hint">No denials reported yet</span>
                    ) : (
                        <div className="symptom-tags-wrap">
                            {denied.map((sym, idx) => (
                                <span key={idx} className="symptom-tag tag-denied">
                                    ✕ {sym}
                                </span>
                            ))}
                        </div>
                    )}
                </div>

                {/* Unknown / Pending */}
                {unknown.length > 0 && (
                    <div className="symptom-group">
                        <div className="group-header">
                            <span className="group-title unknown-title">Pending Inquiry</span>
                            <span className="badge-count unknown-count">{unknown.length}</span>
                        </div>
                        <div className="symptom-tags-wrap">
                            {unknown.slice(0, 6).map((sym, idx) => (
                                <span key={idx} className="symptom-tag tag-unknown">
                                    ? {sym}
                                </span>
                            ))}
                        </div>
                    </div>
                )}
            </div>

            {/* Clinical Triage & Hypotheses */}
            <div className="sidebar-card">
                <div className="sidebar-card-header">
                    <span className="card-icon" aria-hidden="true">
                        <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                            <circle cx="12" cy="12" r="10" />
                            <line x1="12" y1="8" x2="12" y2="12" />
                            <line x1="12" y1="16" x2="12.01" y2="16" />
                        </svg>
                    </span>
                    <h3>Differential Hypotheses</h3>
                    <span className={`priority-pill priority-${priority.toLowerCase()}`}>
                        {priority}
                    </span>
                </div>

                {differentials.length === 0 ? (
                    <span className="symptom-empty-hint">Awaiting sufficient clinical findings</span>
                ) : (
                    <div className="differentials-list">
                        {differentials.map((diff, idx) => (
                            <div key={idx} className="differential-row">
                                <span className="diff-bullet">•</span>
                                <span className="diff-name">{diff}</span>
                                <span className="diff-hypothesis-tag">Hypothesis</span>
                            </div>
                        ))}
                    </div>
                )}
            </div>

            {/* Evidence Quick Panel Trigger */}
            <div className="sidebar-card evidence-card-cta">
                <div className="evidence-cta-header">
                    <div>
                        <h4>Retrieved Medical Evidence</h4>
                        <p>{evidence.length} literature & web sources grounded</p>
                    </div>
                    <button
                        type="button"
                        className="btn-open-evidence"
                        onClick={onOpenEvidence}
                        aria-label="View evidence provenance"
                    >
                        View Sources
                    </button>
                </div>
            </div>

            {/* Safety & Guardrail Verification Badge */}
            <div className="sidebar-card safety-card">
                <div className="safety-row">
                    <span className="safety-shield">🛡️</span>
                    <div>
                        <span className="safety-title">Anti-Hallucination Guardrail</span>
                        <p className="safety-desc">
                            {validation?.is_valid
                                ? "Response validated: No unconfirmed symptoms asserted."
                                : "Responses are strictly constrained to patient-reported facts."}
                        </p>
                    </div>
                </div>
            </div>
        </aside>
    );
}

export default PatientSidebar;
