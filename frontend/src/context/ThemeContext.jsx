/* eslint-disable react-refresh/only-export-components */
import { createContext, useContext, useEffect, useState } from "react";

const THEME_STORAGE_KEY = "cdss_theme_preference";

const ThemeContext = createContext({
    theme: "system",
    resolvedTheme: "light",
    setTheme: () => {},
    toggleTheme: () => {},
});

export function ThemeProvider({ children }) {
    const [theme, setThemeState] = useState(() => {
        try {
            return localStorage.getItem(THEME_STORAGE_KEY) || "system";
        } catch {
            return "system";
        }
    });

    const [resolvedTheme, setResolvedTheme] = useState("light");

    useEffect(() => {
        const mediaQuery = window.matchMedia("(prefers-color-scheme: dark)");

        function updateResolvedTheme() {
            let active = theme;
            if (theme === "system") {
                active = mediaQuery.matches ? "dark" : "light";
            }
            setResolvedTheme(active);
            document.documentElement.setAttribute("data-theme", active);
            document.documentElement.classList.toggle("dark", active === "dark");
        }

        updateResolvedTheme();

        const handleChange = () => {
            if (theme === "system") {
                updateResolvedTheme();
            }
        };

        mediaQuery.addEventListener("change", handleChange);
        return () => mediaQuery.removeEventListener("change", handleChange);
    }, [theme]);

    const setTheme = (newTheme) => {
        setThemeState(newTheme);
        try {
            localStorage.setItem(THEME_STORAGE_KEY, newTheme);
        } catch (e) {
            console.warn("Could not persist theme preference:", e);
        }
    };

    const toggleTheme = () => {
        setTheme(resolvedTheme === "dark" ? "light" : "dark");
    };

    return (
        <ThemeContext.Provider value={{ theme, resolvedTheme, setTheme, toggleTheme }}>
            {children}
        </ThemeContext.Provider>
    );
}

export function useTheme() {
    const context = useContext(ThemeContext);
    if (!context) {
        throw new Error("useTheme must be used within a ThemeProvider");
    }
    return context;
}
