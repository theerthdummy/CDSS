# Detailed Steps to Run the CDSS Multi-Agent Pipeline

This guide outlines exactly how to launch and test the entire 5-agent CDSS pipeline from scratch.

## Prerequisites
1. **Python 3.10 or 3.11** installed on your system.
2. **Node.js** installed (for the frontend).
3. **Docker Desktop** installed and running (for the Qdrant database).
4. Required environment variables (`GEMINI_API_KEY`, etc.) populated in the respective `.env` files in each service directory.

---

## Step 1: Start the Backend Services
We have created a master PowerShell script that automatically starts the Qdrant database and all 6 backend services (5 agents + 1 orchestrator) in separate terminal windows.

1. Open a PowerShell terminal.
2. Navigate to the root of the project:
   ```powershell
   cd c:\Users\Karth\Documents\CDSS
   ```
3. Execute the startup script:
   ```powershell
   .\start_all.ps1
   ```
   *(Note: If Windows gives you a Security Warning, press `R` to run once).*

4. **Verify Windows:** You should now see 6 new terminal windows open, each running a Uvicorn FastAPI server on a different port (8000, 8002, 8003, 8004, 8005, 9000). Leave these running.

---

## Step 2: Start the Frontend UI
The frontend is a React application that provides a chat interface to interact with the Orchestrator.

1. Open a **new** terminal window.
2. Navigate to the frontend directory:
   ```powershell
   cd c:\Users\Karth\Documents\CDSS\frontend
   ```
3. Start the Vite development server:
   ```powershell
   npm run dev
   ```
4. Look for the `Local: http://localhost:5173/` line in the terminal.

---

## Step 3: Run a Clinical Scenario
1. Open your web browser (Chrome, Edge, Safari) and navigate to **`http://localhost:5173/`**.
2. You should see the Clinical Decision Support System dashboard.
3. In the chat box at the bottom, enter a clinical case, for example:
   > *"55 year old male with severe chest pain radiating to left arm for 2 hours, sweating, shortness of breath. History of diabetes and hypertension."*
4. Press **Send**.
5. Wait for the pipeline to finish (it usually takes 15-30 seconds depending on API response times).
6. **Result:** The system will populate the "CDSS Analysis Results" card on the left with the Primary Interpretation, Differential Considerations, Safety Flags, and the overall Confidence Score.

---

## Testing via API (Optional)
If you prefer to bypass the frontend and test the orchestrator directly, you can use tools like Thunder Client, Postman, or `curl`.

**Check Health:**
```http
GET http://localhost:9000/health
```

**Run Pipeline:**
```http
POST http://localhost:9000/analyze
Content-Type: application/json

{
  "text": "55 year old male with chest pain for 2 hours, sweating, SOB. History of diabetes."
}
```
