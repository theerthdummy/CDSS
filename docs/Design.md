# CDSS — Design & UI Document

*(Note: The CDSS is primarily a headless backend microservice architecture. However, it includes a React frontend for Agent 1 and future-proofing UI rules).*

## 1. Visual Identity & Theme
* **Primary Color (Medical Trust):** Slate Blue (`#3b82f6` to `#1e40af`)
* **Secondary Color (Alerts):** Coral/Crimson (`#ef4444`) for critical clinical alerts or low confidence.
* **Success/Validation:** Emerald Green (`#10b981`) for high confidence (≥0.70) reasoning outputs.
* **Background:** Soft Gray (`#f8fafc`) for reduced eye strain during clinical data entry.
* **Text:** Dark Charcoal (`#0f172a`) for maximum readability.

## 2. Typography
* **Primary Font:** `Inter` or `Roboto` (Clean, legible, sans-serif, widely adopted in healthcare IT).
* **Headings:** Bold, high contrast.
* **Data points/Tables:** Monospace font (`JetBrains Mono` or `Fira Code`) for SNOMED CT codes and JSON structured outputs to ensure easy scanning.

## 3. UI/UX Principles
* **High Information Density:** Medical professionals need to see symptoms, conflicting evidence, and confidence scores simultaneously without excessive scrolling.
* **Traceability:** Every decision or finding presented to the user MUST have a clickable or visible attribution (e.g., `Source: PubMed 38192011`).
* **Clear State Representation:** 
  * "Gathering Evidence" (Spinner)
  * "Reasoning & Fusing" (Progress Bar)
  * "Refining (Feedback Loop Activated)" (Amber warning state indicating multi-pass evaluation).

## 4. Frontend Assets (Agent 1)
* Styled with Tailwind CSS.
* Core components: `PatientProfile.jsx`, `ChatInput.jsx`, `MessageBubble.jsx`.
* Uses standard accessible contrast ratios (WCAG AA compliant).
