import { useState } from "react";
import Header from "../components/Header";
import ChatWindow from "../components/ChatWindow";
import ChatInput from "../components/ChatInput";
import PatientSidebar from "../components/PatientSidebar";
import EvidenceDrawer from "../components/EvidenceDrawer";
import SystemStatusModal from "../components/SystemStatusModal";
import Toast from "../components/Toast";
import Footer from "../components/Footer";
import { useChatState } from "../context/ChatContext";

function ChatPage() {
    const { sendMessage, resetConsultation, isLoading } = useChatState();
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
        <div className="app-layout">
            <Header
                onOpenSystemStatus={() => setIsStatusModalOpen(true)}
                onReset={resetConsultation}
                onShowToast={showToast}
            />

            <div className="main-content-grid">
                {/* Primary Column: Conversation Window and Clinical Input */}
                <main className="conversation-column" id="main-content">
                    <ChatWindow
                        onSelectPrompt={handleSelectPrompt}
                        onShowToast={showToast}
                    />
                    <ChatInput
                        value={inputText}
                        onChange={setInputText}
                        onSend={handleSend}
                        disabled={isLoading}
                    />
                    <Footer />
                </main>

                {/* Secondary Column: Authoritative Patient State & Triage Sidebar */}
                <aside className="sidebar-column">
                    <PatientSidebar
                        onOpenEvidence={() => setIsEvidenceDrawerOpen(true)}
                    />
                </aside>
            </div>

            {/* Overlays */}
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
