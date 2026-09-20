# CDSS — Full System Startup Script
# Run this from the project root containing this script.
# Starts all 6 services + Qdrant in separate terminal windows.
# 
# Prerequisites:
#   - Docker Desktop running (for Qdrant)
#   - Python 3.10+ with pip
#   - Virtual environments created for each agent (or use system Python)
#
# Port Map:
#   6333 - Qdrant Vector DB (Docker)
#   8000 - Agent 1: Clinical Text Clarifier
#   8002 - Agent 2: Vector Retrieval (RAG)
#   8003 - Agent 3: Evidence Scanner
#   8004 - Agent 4: Data Fusion
#   8005 - Agent 5: Clinical Reasoning
#   9000 - Orchestrator

$ProjectRoot = $PSScriptRoot

function Start-ServiceWindow {
    param(
        [string]$Title,
        [string]$WorkingDirectory,
        [string]$StartCommand
    )

    Start-Process powershell -WorkingDirectory $WorkingDirectory -ArgumentList "-NoExit", "-Command", $StartCommand
}

Write-Host "===== CDSS Full System Startup =====" -ForegroundColor Cyan
Write-Host ""

# 0. Qdrant Docker
Write-Host "[0/6] Starting Qdrant Docker..." -ForegroundColor Yellow
docker start vectordb-cdss 2>$null
if ($LASTEXITCODE -ne 0) {
    Write-Host "  Qdrant container not found. Creating..." -ForegroundColor DarkYellow
    docker run -d --name vectordb-cdss -p 6333:6333 -p 6334:6334 qdrant/qdrant:latest
}
Start-Sleep -Seconds 3
Write-Host "  Qdrant: http://localhost:6333" -ForegroundColor Green

# 1. Agent 1: Clinical Text Clarifier (port 8000)
Write-Host "[1/6] Starting Agent 1 (Clinical Text Clarifier) on port 8000..." -ForegroundColor Yellow
Start-ServiceWindow -Title "Agent 1" -WorkingDirectory (Join-Path $ProjectRoot 'clinicalTextClassifier\backend') -StartCommand "if (Test-Path 'venv\Scripts\Activate.ps1') { .\venv\Scripts\Activate.ps1 }; Write-Host 'Agent 1 starting...'; uvicorn app.main:app --host 0.0.0.0 --port 8000"

# 2. Agent 2: Vector Retrieval (port 8002)
Write-Host "[2/6] Starting Agent 2 (Vector Retrieval) on port 8002..." -ForegroundColor Yellow
Start-ServiceWindow -Title "Agent 2" -WorkingDirectory (Join-Path $ProjectRoot 'vectorRetrieval') -StartCommand "if (Test-Path 'venv\Scripts\Activate.ps1') { .\venv\Scripts\Activate.ps1 }; Write-Host 'Agent 2 starting...'; uvicorn app:app --host 0.0.0.0 --port 8002"

# 3. Agent 3: Evidence Scanner (port 8003)
Write-Host "[3/6] Starting Agent 3 (Evidence Scanner) on port 8003..." -ForegroundColor Yellow
Start-ServiceWindow -Title "Agent 3" -WorkingDirectory (Join-Path $ProjectRoot 'evidenceScanner') -StartCommand "if (Test-Path 'venv\Scripts\Activate.ps1') { .\venv\Scripts\Activate.ps1 }; Write-Host 'Agent 3 starting...'; uvicorn app:app --host 0.0.0.0 --port 8003"

# 4. Agent 4: Data Fusion (port 8004)
Write-Host "[4/6] Starting Agent 4 (Data Fusion) on port 8004..." -ForegroundColor Yellow
Start-ServiceWindow -Title "Agent 4" -WorkingDirectory (Join-Path $ProjectRoot 'dataFusion') -StartCommand "if (Test-Path 'venv\Scripts\Activate.ps1') { .\venv\Scripts\Activate.ps1 }; Write-Host 'Agent 4 starting...'; uvicorn app.main:app --host 0.0.0.0 --port 8004"

# 5. Agent 5: Clinical Reasoning (port 8005)
Write-Host "[5/6] Starting Agent 5 (Clinical Reasoning) on port 8005..." -ForegroundColor Yellow
Start-ServiceWindow -Title "Agent 5" -WorkingDirectory (Join-Path $ProjectRoot 'clinicalReasoning') -StartCommand "if (Test-Path 'venv\Scripts\Activate.ps1') { .\venv\Scripts\Activate.ps1 }; Write-Host 'Agent 5 starting...'; uvicorn app.main:app --host 0.0.0.0 --port 8005"

# 6. Orchestrator (port 9000)
Write-Host "[6/6] Starting Orchestrator on port 9000..." -ForegroundColor Yellow
Start-ServiceWindow -Title "Orchestrator" -WorkingDirectory (Join-Path $ProjectRoot 'orchestrator') -StartCommand "if (Test-Path 'venv\Scripts\Activate.ps1') { .\venv\Scripts\Activate.ps1 }; Write-Host 'Orchestrator starting...'; uvicorn app:app --host 0.0.0.0 --port 9000"

Write-Host ""
Write-Host "===== All services launched! =====" -ForegroundColor Green
Write-Host ""
Write-Host "Wait ~15 seconds for all services to initialize, then test:" -ForegroundColor Cyan
Write-Host "  Health Check:   GET  http://localhost:9000/health" -ForegroundColor White
Write-Host "  Full Pipeline:  POST http://localhost:9000/analyze" -ForegroundColor White
Write-Host ""
Write-Host "Test payload:" -ForegroundColor Cyan
Write-Host '  {"text": "55 year old male with severe chest pain radiating to left arm for 2 hours, sweating, shortness of breath. History of diabetes and hypertension."}' -ForegroundColor White
