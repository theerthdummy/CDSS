import ChatPage from "./pages/ChatPage";
import { ThemeProvider } from "./context/ThemeContext";
import { ChatProvider } from "./context/ChatContext";
import "./styles/tokens.css";
import "./styles/chat.css";
import "./styles/components.css";

function App() {
    return (
        <ThemeProvider>
            <ChatProvider>
                <ChatPage />
            </ChatProvider>
        </ThemeProvider>
    );
}

export default App;
