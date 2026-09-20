/**
 * Clean, lightweight Markdown parser and renderer for conversational clinical messages.
 * 
 * Safely parses:
 * - Section headings (### or **Heading**)
 * - Bullet points (- or *)
 * - Numbered questions/lists (1. 2.)
 * - Inline bold (**text**) and italic (*text*)
 * - Cleans up escaped markdown (\\*\\* -> **)
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

    // 2. Split by bold **text**
    const parts = clean.split(/(\*\*[^*]+?\*\*)/g);

    return parts.map((part, idx) => {
        if (part.startsWith("**") && part.endsWith("**")) {
            return (
                <strong key={idx} className="md-strong">
                    {part.slice(2, -2)}
                </strong>
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
                            {parseInlineMarkdown(item)}
                        </li>
                    ))}
                </ul>
            );
        } else {
            elements.push(
                <ol key={key} className="md-ol">
                    {currentList.items.map((item, idx) => (
                        <li key={idx} className="md-li">
                            {parseInlineMarkdown(item)}
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

        // Table separator artifact e.g. |---|---|
        if (/^\|?[\s\-:|]+\|?$/.test(trimmed)) {
            continue;
        }

        // Table row e.g. | Key | Value | -> convert to bullet
        if (trimmed.startsWith("|") && trimmed.endsWith("|")) {
            flushList();
            const cells = trimmed
                .slice(1, -1)
                .split("|")
                .map((c) => c.trim())
                .filter(Boolean);
            if (cells.length === 2) {
                elements.push(
                    <p key={`tbl-${i}`} className="md-bullet-line">
                        <span className="md-bullet-dot">•</span>
                        <strong>{cells[0]}:</strong> {parseInlineMarkdown(cells[1])}
                    </p>
                );
            } else if (cells.length > 0) {
                elements.push(
                    <p key={`tbl-${i}`} className="md-bullet-line">
                        <span className="md-bullet-dot">•</span>
                        {parseInlineMarkdown(cells.join(" — "))}
                    </p>
                );
            }
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

        // Standalone bold heading e.g. **What you've told me**
        if (/^\*\*[^*]+?\*\*:?$/.test(trimmed)) {
            flushList();
            elements.push(
                <h4 key={`bh-${i}`} className="md-heading">
                    {trimmed.replace(/\*\*/g, "").replace(/:$/, "")}
                </h4>
            );
            continue;
        }

        // Bullet list: - item or * item
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
