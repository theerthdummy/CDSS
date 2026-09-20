import { useEffect } from "react";

function Toast({ message, type = "info", onClose, duration = 3000 }) {
    useEffect(() => {
        if (!message) return;
        const timer = setTimeout(() => {
            onClose && onClose();
        }, duration);
        return () => clearTimeout(timer);
    }, [message, duration, onClose]);

    if (!message) return null;

    return (
        <div className={`toast-container toast-${type}`} role="alert" aria-live="polite">
            <span className="toast-icon">
                {type === "success" && "✓"}
                {type === "error" && "⚠"}
                {type === "info" && "ℹ"}
            </span>
            <span className="toast-message">{message}</span>
            <button
                type="button"
                className="toast-close"
                onClick={onClose}
                aria-label="Dismiss notification"
            >
                ✕
            </button>
        </div>
    );
}

export default Toast;
