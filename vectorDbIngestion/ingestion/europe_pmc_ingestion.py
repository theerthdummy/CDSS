"""
Ingestion module for Europe PMC.
Uses the REST API to search and fetch full-text JATS XML,
then parses it to extract structured text.
"""

import time
import logging
import requests
import xml.etree.ElementTree as ET
import config

logger = logging.getLogger(__name__)

def ingest_europe_pmc(queries: list[str], max_per_query: int, delay: float, exclude_ids: set) -> list[dict]:
    """
    Search and fetch articles from Europe PMC.

    Args:
        queries: List of search query strings.
        max_per_query: Maximum number of articles to fetch per query.
        delay: Delay in seconds between API requests.
        exclude_ids: Set of PMC IDs to exclude (to avoid duplicates).

    Returns:
        List of standardized document dictionaries.
    """
    documents = []
    seen_ids = set(exclude_ids) if exclude_ids else set()
    search_url = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"
    
    for query in queries:
        logger.info(f"Searching Europe PMC for query: {query}")
        try:
            time.sleep(delay)
            params = {
                "query": f'{query} OPEN_ACCESS:y',
                "resultType": "core",
                "format": "json",
                "pageSize": min(max_per_query, 1000)
            }
            response = requests.get(search_url, params=params)
            response.raise_for_status()
            data = response.json()
            
            results = data.get("resultList", {}).get("result", [])
            count = 0
            
            for item in results:
                if count >= max_per_query:
                    break
                    
                pmcid = item.get("pmcid")
                if not pmcid:
                    continue
                    
                if pmcid in seen_ids:
                    continue
                    
                seen_ids.add(pmcid)
                logger.debug(f"Fetching Europe PMC article {pmcid}")
                
                # Fetch full XML
                time.sleep(delay)
                xml_url = f"https://www.ebi.ac.uk/europepmc/webservices/rest/{pmcid}/fullTextXML"
                xml_resp = requests.get(xml_url)
                
                if xml_resp.status_code != 200:
                    logger.warning(f"Failed to fetch XML for {pmcid}: {xml_resp.status_code}")
                    continue
                    
                # Parse XML
                try:
                    root = ET.fromstring(xml_resp.content)
                except ET.ParseError as e:
                    logger.warning(f"Failed to parse XML for {pmcid}: {e}")
                    continue
                
                title = ""
                abstract = ""
                pub_year = ""
                journal = ""
                keywords = []
                sections = []
                
                article_meta = root.find(".//article-meta")
                if article_meta is not None:
                    title_elem = article_meta.find(".//article-title")
                    if title_elem is not None:
                        title = "".join(title_elem.itertext()).strip()
                        
                    abstract_elem = article_meta.find(".//abstract")
                    if abstract_elem is not None:
                        abstract = "".join(abstract_elem.itertext()).strip()
                        
                    pub_date = article_meta.find(".//pub-date")
                    if pub_date is not None:
                        year_elem = pub_date.find("year")
                        if year_elem is not None and year_elem.text:
                            pub_year = year_elem.text.strip()
                            
                    for kwd in article_meta.findall(".//kwd"):
                        if kwd.text:
                            keywords.append("".join(kwd.itertext()).strip())
                            
                journal_meta = root.find(".//journal-meta")
                if journal_meta is not None:
                    journal_elem = journal_meta.find(".//journal-title")
                    if journal_elem is not None:
                        journal = "".join(journal_elem.itertext()).strip()
                        
                body = root.find(".//body")
                full_body_text = ""
                if body is not None:
                    for sec in body.findall(".//sec"):
                        sec_title = ""
                        title_elem = sec.find("title")
                        if title_elem is not None:
                            sec_title = "".join(title_elem.itertext()).strip()
                            
                        sec_text = ""
                        for p in sec.findall(".//p"):
                            sec_text += "".join(p.itertext()).strip() + "\n"
                            
                        if sec_text:
                            sections.append({"section_title": sec_title, "text": sec_text.strip()})
                            full_body_text += sec_text + "\n"
                            
                doc = {
                    'title': title,
                    'abstract': abstract,
                    'publication_year': pub_year,
                    'journal': journal,
                    'keywords': keywords,
                    'sections': sections,
                    'full_body_text': full_body_text.strip(),
                    'source_id': pmcid,
                    'source_type': 'medical-paper'
                }
                documents.append(doc)
                count += 1
                
        except Exception as e:
            logger.warning(f"Error processing query '{query}': {e}")
            continue

    return documents
