"""
FastAPI application for the Evidence Scanner (Agent 3).
Exposes the core EvidenceScannerAgent over HTTP.
"""

import logging
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import os

from src.scanner_agent import EvidenceScannerAgent, EvidenceRequest, ScannerResponse

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(name)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Evidence Scanner Service",
    description="Agent 3: Gathers live web evidence from PubMed and Tavily (guidelines).",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Instantiate the agent globally
agent = EvidenceScannerAgent()

@app.get("/")
def health_check():
    """Health check endpoint."""
    return {
        "service": "evidenceScanner",
        "status": "running"
    }

@app.post("/api/v1/scan", response_model=ScannerResponse)
def scan_evidence(request: EvidenceRequest):
    """
    Main endpoint for triggering the Evidence Scanner.
    Expects structured patient entities and runs dynamic searches against 
    PubMed (if no chronic conditions) and Tavily (for treatment guidelines).
    """
    logger.info(f"Received scan request for symptoms: {request.entities}")
    try:
        response = agent.run(request)
        logger.info(f"Scan complete. Retrieved {response.total_found} records.")
        return response
    except Exception as e:
        logger.error(f"Error during evidence scanning: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Evidence scanner failed: {str(e)}")
