/**
 * MAANG-Grade Markdown parser and renderer for conversational clinical messages.
 * 
 * Safely parses:
 * - Section headings (### or **Heading**)
 * - Urgent safety banners (### ⚠️ Important Medical Safety Advice or **URGENT ADVICE**)
 * - Bullet points (- or * or •)
 * - Numbered questions/lists (1. 2.)
 * - Inline bold (**text**), italic (*text*), and code (`text`)
 * - Cleans up escaped markdown (\*\* -> **)
 * - Strips accidental HTML tags (<br>, <table>, etc.)
 */
function parseInlineMarkdown(text) {
    if (!text) return "";

    // 1. Clean backslash escaping
    let clean = text
        .replace(/\\\*/g, "*")
        .replace(/\\_/g, "_")
        .replace(/\\#/g, "#")
        .replace(/\\</g, "<")
        .replace(/\\>/g, ">");

    // 2. Split by bold **text**, code `text`, and italic *text*
    const parts = clean.split(/(\*\*[^*]+?\*\*|`[^`]+?`|\*[^*]+?\*)/g);

    return parts.map((part, idx) => {
        if (part.startsWith("**") && part.endsWith("**")) {
            return (
                <strong key={idx} className="md-strong">
                    {part.slice(2, -2)}
                </strong>
            );
        }
        if (part.startsWith("`") && part.endsWith("`")) {
            return (
                <code key={idx} className="md-inline-code">
                    {part.slice(1, -1)}
                </code>
            );
        }
        if (part.startsWith("*") && part.endsWith("*") && !part.startsWith("**")) {
            return (
                <em key={idx} className="md-italic">
                    {part.slice(1, -1)}
                </em>
            );
        }
        return part;
    });
}

function MarkdownRenderer({ content }) {
    if (!content) return null;

    // Normalize newlines and clean table artifacts / HTML
    let cleaned = content
        .replace(/<br\s*\/?>/gi, "\n")
        .replace(/\\<br\s*\/?>/gi, "\n")
        .replace(/<\/?[a-z][a-z0-9]*[^<>]*>/gi, ""); // strip raw html tags

    const lines = cleaned.split("\n");
    const elements = [];

    let currentList = null; // { type: 'ul' | 'ol', items: [] }

    function flushList() {
        if (!currentList) return;
        const key = `list-${elements.length}`;
        if (currentList.type === "ul") {
            elements.push(
                <ul key={key} className="md-ul">
                    {currentList.items.map((item, idx) => (
                        <li key={idx} className="md-li">
                            <span className="md-bullet-marker" aria-hidden="true">•</span>
                            <span className="md-li-content">{parseInlineMarkdown(item)}</span>
                        </li>
                    ))}
                </ul>
            );
        } else {
            elements.push(
                <ol key={key} className="md-ol">
                    {currentList.items.map((item, idx) => (
                        <li key={idx} className="md-oli">
                            <span className="md-num-badge">{idx + 1}</span>
                            <span className="md-oli-content">{parseInlineMarkdown(item)}</span>
                        </li>
                    ))}
                </ol>
            );
        }
        currentList = null;
    }

    for (let i = 0; i < lines.length; i++) {
        const rawLine = lines[i];
        const trimmed = rawLine.trim();

        if (!trimmed) {
            flushList();
            continue;
        }

        // Table separator artifact e.g. |---|---| or header | Category | Considerations |
        if (/^\|?[\s\-:|]+\|?$/.test(trimmed)) {
            continue;
        }
        if (/^\|\s*(?:Category|Considerations|Details|Key|Value|Field|Information|Status|Description|Priority)\s*(?:\|.*)?\|?$/i.test(trimmed)) {
            continue;
        }

        // Table row e.g. | Key | Value | or | Key | Value -> convert to neat bullet
        if (trimmed.startsWith("|") || (trimmed.includes("|") && !trimmed.startsWith("-") && !trimmed.startsWith("*"))) {
            flushList();
            const cells = trimmed
                .replace(/^\||\|$/g, "")
                .split("|")
                .map((c) => c.trim())
                .filter(Boolean);
            if (cells.length === 2) {
                const header = cells[0].replace(/^\*+|\*+$/g, "");
                elements.push(
                    <p key={`tbl-${i}`} className="md-bullet-line">
                        <span className="md-bullet-dot">•</span>
                        <strong>{header}:</strong> {parseInlineMarkdown(cells[1])}
                    </p>
                );
                continue;
            } else if (cells.length > 2) {
                const header = cells[0].replace(/^\*+|\*+$/g, "");
                elements.push(
                    <p key={`tbl-${i}`} className="md-bullet-line">
                        <span className="md-bullet-dot">•</span>
                        <strong>{header}:</strong> {parseInlineMarkdown(cells.slice(1).join(" — "))}
                    </p>
                );
                continue;
            } else if (cells.length === 1 && trimmed.startsWith("|")) {
                const header = cells[0].replace(/^\*+|\*+$/g, "");
                elements.push(
                    <h4 key={`tbl-h-${i}`} className="md-heading">
                        {header}
                    </h4>
                );
                continue;
            }
        }

        // Urgent alert heading e.g. ### ⚠️ Important Medical Safety Advice or **URGENT ADVICE**
        if (
            trimmed.includes("⚠️") ||
            trimmed.toLowerCase().includes("urgent advice") ||
            trimmed.toLowerCase().includes("urgent medical safety")
        ) {
            flushList();
            const cleanHeading = trimmed.replace(/^#{1,4}\s*/, "").replace(/\*\*/g, "");
            elements.push(
                <div key={`alert-${i}`} className="md-urgent-callout-header">
                    <span className="md-urgent-icon">⚠️</span>
                    <span className="md-urgent-title">{cleanHeading}</span>
                </div>
            );
            continue;
        }

        // Heading: ### Heading or ## Heading
        const headingMatch = trimmed.match(/^#{1,4}\s+(.+)$/);
        if (headingMatch) {
            flushList();
            elements.push(
                <h4 key={`h-${i}`} className="md-heading">
                    {parseInlineMarkdown(headingMatch[1])}
                </h4>
            );
            continue;
        }

        // Standalone bold heading e.g. **Clinical Status & Known Facts:**
        if (/^\*\*[^*]+?\*\*:?$/.test(trimmed)) {
            flushList();
            elements.push(
                <h4 key={`bh-${i}`} className="md-heading">
                    {trimmed.replace(/\*\*/g, "").replace(/:$/, "")}
                </h4>
            );
            continue;
        }

        // Bullet list: - item or * item or • item
        const bulletMatch = trimmed.match(/^[-*•]\s+(.+)$/);
        if (bulletMatch) {
            if (!currentList || currentList.type !== "ul") {
                flushList();
                currentList = { type: "ul", items: [] };
            }
            currentList.items.push(bulletMatch[1]);
            continue;
        }

        // Numbered list: 1. question?
        const numberedMatch = trimmed.match(/^(\d+)[.)]\s+(.+)$/);
        if (numberedMatch) {
            if (!currentList || currentList.type !== "ol") {
                flushList();
                currentList = { type: "ol", items: [] };
            }
            currentList.items.push(numberedMatch[2]);
            continue;
        }

        // Regular paragraph
        flushList();
        elements.push(
            <p key={`p-${i}`} className="md-paragraph">
                {parseInlineMarkdown(trimmed)}
            </p>
        );
    }

    flushList();

    return <div className="formatted-markdown">{elements}</div>;
}

export default MarkdownRenderer;
