import logging
import re
import unicodedata
import config

logger = logging.getLogger(__name__)

# Compile regex patterns for performance
URL_PATTERN = re.compile(r'https?://\S+|www\.\S+')
EMAIL_PATTERN = re.compile(r'\S+@\S+\.\S+')
# Matches [1], [1, 2], [1-3], etc.
REF_PATTERN = re.compile(r'\[\s*\d+(?:\s*,\s*\d+)*\s*(?:-\s*\d+)?\s*\]')
WHITESPACE_PATTERN = re.compile(r'\s+')
NEWLINE_PATTERN = re.compile(r'\n{3,}')
COPYRIGHT_PATTERN = re.compile(r'(?i)copyright\s+.*?\d{4}')
DISCLAIMER_PATTERN = re.compile(r'(?i)disclaimer.*?(?=\n|$)')
HEADER_ONLY_PATTERN = re.compile(r'^[\w\s:-]{1,50}$')

def clean_text(text: str) -> str:
    """
    Applies text normalization and regex-based noise removal to the given text.
    
    Args:
        text (str): The raw text to clean.
        
    Returns:
        str: The cleaned text.
    """
    if not text:
        return ""

    # Unicode normalization
    text = unicodedata.normalize('NFKC', text)
    
    # Noise removal
    text = URL_PATTERN.sub('', text)
    text = EMAIL_PATTERN.sub('', text)
    text = REF_PATTERN.sub('', text)
    text = COPYRIGHT_PATTERN.sub('', text)
    text = DISCLAIMER_PATTERN.sub('', text)
    
    # Normalize excessive newlines and whitespace
    text = NEWLINE_PATTERN.sub('\n\n', text)
    text = WHITESPACE_PATTERN.sub(' ', text)
    
    return text.strip()

def is_valid_chunk(text: str) -> bool:
    """
    Checks if a chunk meets the minimum quality requirements.
    
    Args:
        text (str): The text chunk to validate.
        
    Returns:
        bool: True if the chunk is valid, False otherwise.
    """
    if not text:
        return False
        
    # Minimum content length validation
    if len(text) < config.MIN_CHUNK_LENGTH:
        return False
        
    # Check if it consists only of labels/headers (heuristic: short strings with mostly words and colons)
    if HEADER_ONLY_PATTERN.match(text):
        return False
        
    return True
