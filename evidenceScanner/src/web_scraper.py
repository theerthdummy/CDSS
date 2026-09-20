"""
web_scraper.py - Tavily REST Client for Clinical Practice Guidelines & Treatment Protocols.
Handles acute presentations with chronic risk factors (e.g., CKD dosing, Diabetes complications).
"""

import os
import requests
from typing import Dict, Any, Optional, List
from dotenv import load_dotenv
from tenacity import retry, stop_after_attempt, wait_exponential

load_dotenv()

TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "")

EXCLUDED_DOMAINS = [
    "wikipedia.org", "youtube.com", "facebook.com", "twitter.com",
    "reddit.com", "quora.com", "amazon.com", "pinterest.com"
]


class WebScraper:
    def __init__(self, timeout: int = 10):
        self.timeout = timeout
        self.api_url = "https://api.tavily.com/search"

    @retry(
        stop=stop_after_attempt(2),
        wait=wait_exponential(multiplier=1, min=2, max=4),
        retry_error_callback=lambda retry_state: None
    )
    def crawl_and_extract(
        self,
        start_url: str = "",
        topic: str = "",
        chronic_risks: Optional[List[str]] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Executes Tavily search targeting treatment protocols, drug adjustments, and guidelines.
        Accepts topic and optional chronic_risks list.
        """
        if not TAVILY_API_KEY:
            print("[WebScraper Warning] TAVILY_API_KEY is not configured in .env.")
            return None

        # Build query incorporating chronic comorbidity risk factors if present
        if chronic_risks:
            risk_context = " | ".join(chronic_risks)
            search_query = (
                f"{topic} | Clinical Practice Guidelines & Safe Treatments. "
                f"Comorbidity Risk Measures: {risk_context}. "
                f"Address drug dosing adjustments, treatment contraindications, and infection management."
            )
        else:
            search_query = f"{topic} clinical practice guidelines management and treatment recommendations"

        payload = {
            "api_key": TAVILY_API_KEY,
            "query": search_query,
            "search_depth": "advanced",
            "include_answer": "advanced",
            "topic": "general",
            "max_results": 3,
            "exclude_domains": EXCLUDED_DOMAINS
        }

        try:
            resp = requests.post(self.api_url, json=payload, timeout=self.timeout)
            if resp.status_code != 200:
                err_msg = f"[WebScraper Error] Tavily returned status {resp.status_code}: {resp.text}"
                print(err_msg)
                raise requests.HTTPError(err_msg)

            data = resp.json()
            ai_answer = data.get("answer", "").strip()
            results = data.get("results", [])

            # Primary reference details
            primary_url = results[0].get("url", "https://tavily.com") if results else "https://tavily.com"
            primary_title = results[0].get("title", f"{topic} Guidelines") if results else "Tavily Clinical Guidelines"

            # Prefer the Tavily synthesized answer; fallback to top result snippet
            combined_summary = ai_answer
            if not combined_summary and results:
                combined_summary = results[0].get("content", "").strip()

            if not combined_summary:
                return None

            words = combined_summary.split()
            if len(words) > 350:
                combined_summary = " ".join(words[:350]) + "... [truncated]"

            return {
                "source": "Tavily Guidelines & Treatment Safety",
                "url": primary_url,
                "title": primary_title,
                "abstract": combined_summary,
                "extracted_text": combined_summary,
                "query_matched": search_query,
                "chronic_risk_adjusted": bool(chronic_risks),
                "success": True
            }
        except Exception as e:
            print(f"[WebScraper Error] Failed to execute Tavily query: {e}")
            raise e