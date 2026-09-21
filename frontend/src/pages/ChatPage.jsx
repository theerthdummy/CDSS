import { useState } from "react";
import Header from "../components/Header";
import LeftSidebar from "../components/LeftSidebar";
import ChatWindow from "../components/ChatWindow";
import ChatInput from "../components/ChatInput";
import PatientSidebar from "../components/PatientSidebar";
import EvidenceDrawer from "../components/EvidenceDrawer";
import SystemStatusModal from "../components/SystemStatusModal";
import Toast from "../components/Toast";
import Footer from "../components/Footer";
import { useChatState } from "../context/ChatContext";

function ChatPage() {
    const {
        sendMessage,
        resetConsultation,
        isLoading,
        isTelemetryOpen,
        setIsTelemetryOpen,
    } = useChatState();

    const [inputText, setInputText] = useState("");
    const [isStatusModalOpen, setIsStatusModalOpen] = useState(false);
    const [isEvidenceDrawerOpen, setIsEvidenceDrawerOpen] = useState(false);
    const [toast, setToast] = useState(null);

    const showToast = (message, type = "info") => {
        setToast({ message, type });
    };

    const handleSend = (text) => {
        sendMessage(text);
        setInputText("");
    };

    const handleSelectPrompt = (promptText) => {
        setInputText(promptText);
    };

    return (
        <div className="docpilot-app-shell">
            {/* ChatGPT / DocPilot Left Navigation Sidebar */}
            <LeftSidebar
                onOpenSystemStatus={() => setIsStatusModalOpen(true)}
                onShowToast={showToast}
            />

            {/* Central Fluid Conversation Canvas */}
            <main className="docpilot-main-canvas" id="main-content">
                <Header
                    onOpenSystemStatus={() => setIsStatusModalOpen(true)}
                    onReset={resetConsultation}
                    onShowToast={showToast}
                />

                <div className="docpilot-conversation-container">
                    <ChatWindow
                        onSelectPrompt={handleSelectPrompt}
                        onShowToast={showToast}
                    />
                </div>

                <ChatInput
                    value={inputText}
                    onChange={setInputText}
                    onSend={handleSend}
                    disabled={isLoading}
                />

                <Footer />
            </main>

            {/* Slide-out Patient Demographics & Telemetry Drawer */}
            <div
                className={`telemetry-drawer-overlay ${isTelemetryOpen ? "open" : ""}`}
                onClick={() => setIsTelemetryOpen(false)}
                aria-hidden={!isTelemetryOpen}
            >
                <aside
                    className="telemetry-drawer-panel"
                    onClick={(e) => e.stopPropagation()}
                    aria-label="Clinical Telemetry & Patient Demographics"
                >
                    <div className="telemetry-drawer-header">
                        <div className="drawer-title-wrap">
                            <span className="drawer-icon">📊</span>
                            <div>
                                <h3 className="drawer-title">Patient Telemetry & State</h3>
                                <span className="drawer-subtitle">Authoritative Clinical State & Hypotheses</span>
                            </div>
                        </div>
                        <button
                            type="button"
                            className="btn-close-drawer"
                            onClick={() => setIsTelemetryOpen(false)}
                            aria-label="Close telemetry drawer"
                        >
                            ✕
                        </button>
                    </div>

                    <div className="telemetry-drawer-content">
                        <PatientSidebar
                            onOpenEvidence={() => setIsEvidenceDrawerOpen(true)}
                        />
                    </div>
                </aside>
            </div>

            {/* Overlays & Modals */}
            <EvidenceDrawer
                isOpen={isEvidenceDrawerOpen}
                onClose={() => setIsEvidenceDrawerOpen(false)}
            />

            <SystemStatusModal
                isOpen={isStatusModalOpen}
                onClose={() => setIsStatusModalOpen(false)}
            />

            {toast && (
                <Toast
                    message={toast.message}
                    type={toast.type}
                    onClose={() => setToast(null)}
                />
            )}
        </div>
    );
}

export default ChatPage;
