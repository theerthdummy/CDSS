function LoadingSpinner({ stage }) {
    const displayStage = stage || "Analyzing clinical evidence & formulating response...";

    return (
        <div className="loading-card" role="status" aria-live="polite">
            <div className="loading-pulse-ring">
                <span className="pulse-dot" />
            </div>
            <div className="loading-text-group">
                <span className="loading-title">Clinical Intelligence Engine</span>
                <span className="loading-stage-text">{displayStage}</span>
            </div>
        </div>
    );
}

export default LoadingSpinner;
