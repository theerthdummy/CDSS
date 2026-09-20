"""
Ingestion module for ClinicalTrials.gov API v2.
Fetches study data and formats it into standardized document dicts.
"""

import time
import logging
import requests
import config

logger = logging.getLogger(__name__)

def ingest_clinical_trials(queries: list[str], max_per_query: int, delay: float) -> list[dict]:
    """
    Search and fetch studies from ClinicalTrials.gov.

    Args:
        queries: List of search query strings.
        max_per_query: Maximum number of studies to fetch per query.
        delay: Delay in seconds between API requests.

    Returns:
        List of standardized document dictionaries.
    """
    documents = []
    seen_ids = set()
    base_url = "https://clinicaltrials.gov/api/v2/studies"
    
    for query in queries:
        logger.info(f"Searching ClinicalTrials.gov for query: {query}")
        try:
            time.sleep(delay)
            params = {
                "query.cond": query,
                "pageSize": min(max_per_query, 1000)
            }
            response = requests.get(base_url, params=params)
            response.raise_for_status()
            data = response.json()
            
            studies = data.get("studies", [])
            count = 0
            
            for study in studies:
                if count >= max_per_query:
                    break
                    
                protocol = study.get("protocolSection", {})
                ident = protocol.get("identificationModule", {})
                nct_id = ident.get("nctId")
                
                if not nct_id or nct_id in seen_ids:
                    continue
                    
                seen_ids.add(nct_id)
                
                desc = protocol.get("descriptionModule", {})
                brief_title = ident.get("briefTitle", "")
                brief_summary = desc.get("briefSummary", "")
                detailed_desc = desc.get("detailedDescription", "")
                
                conditions_mod = protocol.get("conditionsModule", {})
                conditions = conditions_mod.get("conditions", [])
                
                interventions_mod = protocol.get("armsInterventionsModule", {})
                interventions = interventions_mod.get("interventions", [])
                intervention_names = [inv.get("name", "") for inv in interventions]
                
                eligibility_mod = protocol.get("eligibilityModule", {})
                eligibility = eligibility_mod.get("eligibilityCriteria", "")
                
                full_body_text = f"Brief Summary: {brief_summary}\n\nDetailed Description: {detailed_desc}\n\n"
                full_body_text += f"Eligibility Criteria: {eligibility}\n\n"
                full_body_text += f"Interventions: {', '.join(intervention_names)}"
                
                sections = [
                    {"section_title": "Brief Summary", "text": brief_summary},
                    {"section_title": "Detailed Description", "text": detailed_desc},
                    {"section_title": "Eligibility Criteria", "text": eligibility},
                    {"section_title": "Interventions", "text": ", ".join(intervention_names)}
                ]
                
                doc = {
                    'title': brief_title,
                    'abstract': brief_summary,
                    'publication_year': "",
                    'journal': "",
                    'keywords': conditions,
                    'sections': sections,
                    'full_body_text': full_body_text.strip(),
                    'source_id': nct_id,
                    'source_type': 'clinical-trial',
                    'evidence_level': 'Clinical Trial'
                }
                documents.append(doc)
                count += 1
                
        except Exception as e:
            logger.warning(f"Error processing query '{query}': {e}")
            continue

    return documents
