import re

test_text = """
### Stratified Differential Diagnosis (Risk Considerations)

| Category | Considerations | Key Distinguishing Features |
|---|---|---|
| Must-Not-Miss | • **Compartment syndrome** of the lateral foot (rare):<br>• Progressive pain with tense swelling → neurovascular compromise. | Sudden, localized bone pain, swelling, inability to bear weight. |
| Likely / Primary | • **Peroneus brevis tendonitis / tendinopathy**<br>• **Peroneal tendon subluxation/dislocation** | Tender, "rope-like" structure along lateral foot → tendon.<br>• Pain worsens with eversion, resisted plantarflexion, or walking on uneven surfaces. |
| Atypical / Secondary | • **Cuboid syndrome** (lateral midfoot pain)<br>• **Sural nerve entrapment** (neuropathic pain)<br>• **Gout or calcium pyrophosphate deposition** (acute inflammatory arthritis)<br>• Soft-tissue mass (e.g., ganglion, lipoma)**: | Less common, but can mimic tendon pain; consider if exam reveals nodular mass or neuropathic distribution. |

### Recommended Initial Diagnostic Workup
- **Weight-bearing plain radiography (3-view foot)**: Rule out acute 5th metatarsal fracture.
- **Diagnostic Ultrasound**: Rapid assessment of peroneus brevis tendon integrity.
"""

def sanitize_formatting_v2(text: str) -> str:
    if not text:
        return ""

    # 1. Un-escape backslash-escaped characters
    cleaned = text.replace(r"\*\*", "**")
    cleaned = cleaned.replace(r"\_", "_")
    cleaned = cleaned.replace(r"\#", "#")
    cleaned = cleaned.replace(r"\<", "<")
    cleaned = cleaned.replace(r"\>", ">")
    cleaned = cleaned.replace(r"\[", "[")
    cleaned = cleaned.replace(r"\]", "]")

    # 2. In lines with table pipes (|), replace <br> with a clean separator before line splitting
    table_lines = []
    for raw_line in cleaned.split("\n"):
        if "|" in raw_line:
            raw_line = re.sub(r"(?i)<br\s*/?>", " <SEP> ", raw_line)
            raw_line = re.sub(r"(?i)\\<br\s*/?>", " <SEP> ", raw_line)
        table_lines.append(raw_line)
    cleaned = "\n".join(table_lines)

    # 3. Replace remaining HTML line breaks with newlines
    cleaned = re.sub(r"(?i)<br\s*/?>", "\n", cleaned)
    cleaned = re.sub(r"(?i)\\<br\s*/?>", "\n", cleaned)

    # 4. Strip HTML tags
    cleaned = re.sub(r"(?i)</?(?:table|thead|tbody|tfoot|tr|th|td|p|div|span|b|strong|i|em)[^>]*>", " ", cleaned)

    # 5. Process lines & convert markdown tables
    lines = cleaned.split("\n")
    new_lines = []
    
    for line in lines:
        stripped = line.strip()
        if not stripped:
            new_lines.append("")
            continue

        # Ignore table separator lines like |---|---|
        if re.match(r"^\|?[\s\-:|]+\|?$", stripped):
            continue

        # Ignore table header rows e.g. | Category | Considerations | ...
        if re.match(r"^\|\s*(?:Category|Considerations|Details|Key|Value|Field|Information|Status|Description|Priority|Key Distinguishing Features)\s*(?:\|.*)?\|?$", stripped, re.IGNORECASE):
            continue

        # If table row like | Must-Not-Miss | • Compartment syndrome... | ... |
        if stripped.startswith("|") or (stripped.count("|") >= 2 and not stripped.startswith("-") and not stripped.startswith("*")):
            cells = [c.strip() for c in stripped.strip("|").split("|") if c.strip()]
            if len(cells) >= 2:
                category = re.sub(r"^\*+|\*+$", "", cells[0]).strip()
                new_lines.append(f"\n**{category}:**")
                
                cons_raw = cells[1]
                details_raw = cells[2] if len(cells) > 2 else ""
                
                items = [i.strip() for i in re.split(r"<SEP>|•|\n", cons_raw) if i.strip()]
                details_items = [d.strip() for d in re.split(r"<SEP>|•|\n", details_raw) if d.strip()]
                
                if items:
                    for idx, itm in enumerate(items):
                        clean_itm = re.sub(r"^[\s•\-\u2022]+", "", itm).strip()
                        # Fix broken bold tags if opening asterisks were missing
                        if clean_itm.endswith("**:") and not clean_itm.startswith("**"):
                            clean_itm = f"**{clean_itm}"
                        elif clean_itm.endswith("**") and not clean_itm.startswith("**"):
                            clean_itm = f"**{clean_itm}"
                        
                        det = details_items[idx] if idx < len(details_items) else (details_items[0] if len(details_items) == 1 and idx == 0 else "")
                        det_clean = re.sub(r"^[\s•\-\u2022]+", "", det).strip()
                        
                        if det_clean and not clean_itm.endswith(":"):
                            new_lines.append(f"- {clean_itm}: {det_clean}")
                        elif det_clean:
                            new_lines.append(f"- {clean_itm} {det_clean}")
                        else:
                            new_lines.append(f"- {clean_itm}")
                else:
                    new_lines.append(f"- {cells[1]}")
                continue
            elif len(cells) == 1:
                header_c = re.sub(r"^\*+|\*+$", "", cells[0]).strip()
                new_lines.append(f"\n**{header_c}:**")
                continue

        # Strip remaining orphan pipes at end/start of line
        cleaned_line = re.sub(r"\s*\|\s*$", "", line)
        cleaned_line = re.sub(r"^\s*\|\s*", "", cleaned_line)
        
        # Strip duplicate bullets at start of line: "• • text" -> "- text", "- • text" -> "- text"
        cleaned_line = re.sub(r"^(\s*[-*•]\s*)+", "- ", cleaned_line)
        
        # Strip inline redundant bullets after bold labels: "- **Label**: • Text" -> "- **Label**: Text"
        cleaned_line = re.sub(r"(\*\*[^*]+?\*\*:\s*)[-*•]\s+", r"\1", cleaned_line)
        
        # Fix broken bold tags if line has unmatched "**:"
        if re.search(r"^[-\s*•]*[^*]+?\*\*:", cleaned_line):
            cleaned_line = re.sub(r"^([-\s*•]*)([^*]+?\*\*:)", r"\1**\2", cleaned_line)

        new_lines.append(cleaned_line)

    cleaned = "\n".join(new_lines)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()

import sys
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

output = sanitize_formatting_v2(test_text)
print("--- SANITIZED RESULT ---")
print(output)
