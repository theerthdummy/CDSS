import logging
import config
import nltk
from typing import List, Dict, Any
from tqdm import tqdm
from processing.text_cleaner import clean_text, is_valid_chunk

logger = logging.getLogger(__name__)

# Ensure punkt_tab is downloaded
try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    logger.info("Downloading NLTK punkt_tab data...")
    nltk.download('punkt_tab')

def recursive_character_split(text: str, chunk_size: int, chunk_overlap: int) -> List[str]:
    """Simple recursive character splitter. Splits text into chunks of `chunk_size`."""
    if len(text) <= chunk_size:
        return [text]
        
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        if end >= len(text):
            chunks.append(text[start:])
            break
            
        # Try to find a good breaking point (newline, period, space)
        break_point = end
        for sep in ['\n\n', '\n', '. ', ' ']:
            pos = text.rfind(sep, start, end)
            if pos != -1:
                break_point = pos + len(sep)
                break
                
        chunk = text[start:break_point].strip()
        if chunk:
            chunks.append(chunk)
            
        # Move start forward, accounting for overlap
        next_start = break_point - chunk_overlap
        if next_start <= start:  # Guard against infinite loop / going backwards
            start = break_point
        else:
            start = next_start
            
    return chunks

def chunk_structured_document(doc: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Chunks a structured document by section boundaries."""
    chunks_out = []
    
    sections = doc.get('sections', [])
    for section in sections:
        section_title = section.get('section_title', '')
        section_text = section.get('text', '')
        
        if not section_text:
            continue
            
        text_chunks = recursive_character_split(
            section_text, 
            config.CHUNK_SIZE, 
            config.CHUNK_OVERLAP
        )
        
        for text_chunk in text_chunks:
            cleaned = clean_text(text_chunk)
            if is_valid_chunk(cleaned):
                chunks_out.append({
                    'text': cleaned,
                    'source_id': doc.get('source_id', ''),
                    'source_type': doc.get('source_type', ''),
                    'publication_year': doc.get('publication_year'),
                    'disease_category': doc.get('disease_category', ''),
                    'evidence_level': doc.get('evidence_level', ''),
                    'section_title': section_title,
                    'journal': doc.get('journal', '')
                })
                
    return chunks_out

def chunk_unstructured_document(doc: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Chunks an unstructured document using sentence-level tokenization."""
    chunks_out = []
    text = doc.get('full_body_text', '') or doc.get('text', '')
    
    if not text:
        return chunks_out
        
    sentences = nltk.sent_tokenize(text)
    current_chunk_sentences = []
    current_length = 0
    
    for sentence in sentences:
        sentence_len = len(sentence)
        if current_length + sentence_len > config.CHUNK_SIZE and current_chunk_sentences:
            # Join and save current chunk
            chunk_text = ' '.join(current_chunk_sentences)
            cleaned = clean_text(chunk_text)
            if is_valid_chunk(cleaned):
                chunks_out.append({
                    'text': cleaned,
                    'source_id': doc.get('source_id', ''),
                    'source_type': doc.get('source_type', ''),
                    'publication_year': doc.get('publication_year'),
                    'disease_category': doc.get('disease_category', ''),
                    'evidence_level': doc.get('evidence_level', ''),
                    'section_title': '',
                    'journal': doc.get('journal', '')
                })
            # Reset
            current_chunk_sentences = [sentence]
            current_length = sentence_len
        else:
            current_chunk_sentences.append(sentence)
            current_length += sentence_len + 1 # +1 for space
            
    # Process remaining
    if current_chunk_sentences:
        chunk_text = ' '.join(current_chunk_sentences)
        cleaned = clean_text(chunk_text)
        if is_valid_chunk(cleaned):
            chunks_out.append({
                'text': cleaned,
                'source_id': doc.get('source_id', ''),
                'source_type': doc.get('source_type', ''),
                'publication_year': doc.get('publication_year'),
                'disease_category': doc.get('disease_category', ''),
                'evidence_level': doc.get('evidence_level', ''),
                'section_title': '',
                'journal': doc.get('journal', '')
            })
            
    return chunks_out

def chunk_documents(documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Processes all documents and returns a flat list of chunk dicts.
    """
    logger.info(f"Chunking {len(documents)} documents...")
    all_chunks = []
    
    for doc in tqdm(documents, desc="Chunking documents"):
        # Use structured chunking if the document has sections with actual text
        sections = doc.get('sections', [])
        has_sections = any(s.get('text', '').strip() for s in sections) if sections else False
        
        if has_sections:
            all_chunks.extend(chunk_structured_document(doc))
        else:
            all_chunks.extend(chunk_unstructured_document(doc))
            
    logger.info(f"Generated {len(all_chunks)} chunks.")
    return all_chunks
