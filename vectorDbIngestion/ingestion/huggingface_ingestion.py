"""
Ingestion module for HuggingFace medical datasets.
Streams data and formats it into standardized document dicts.
"""

import logging
from datasets import load_dataset
import config

logger = logging.getLogger(__name__)

def ingest_huggingface(max_samples: int) -> list[dict]:
    """
    Stream and fetch samples from HuggingFace datasets.

    Args:
        max_samples: Maximum number of samples to fetch per dataset.

    Returns:
        List of standardized document dictionaries.
    """
    documents = []
    
    for dataset_info in config.HUGGINGFACE_DATASETS:
        ds_name = dataset_info["name"]
        logger.info(f"Streaming HuggingFace dataset: {ds_name}")
        
        try:
            dataset = load_dataset(ds_name, split=dataset_info["split"], streaming=True)
            
            text_field = dataset_info.get("text_field")
            label_field = dataset_info.get("label_field")
            
            count = 0
            for sample in dataset:
                if count >= max_samples:
                    break
                    
                if ds_name == "gretelai/symptom_to_diagnosis":
                    symptoms = sample.get(text_field, "")
                    diagnosis = sample.get(label_field, "")
                    
                    title = f"Diagnosis: {diagnosis}"
                    text = f"Symptoms:\n{symptoms}\n\nDiagnosis:\n{diagnosis}"
                    
                    doc = {
                        'title': title,
                        'abstract': "",
                        'publication_year': "",
                        'journal': "",
                        'keywords': [diagnosis] if diagnosis else [],
                        'sections': [
                            {"section_title": "Symptoms", "text": symptoms},
                            {"section_title": "Diagnosis", "text": diagnosis}
                        ],
                        'full_body_text': text,
                        'source_id': f"hf_{ds_name.replace('/', '_')}_{count}",
                        'source_type': 'symptom-disease-dataset'
                    }
                    documents.append(doc)
                    
                elif ds_name == "QuyenAnhDE/Diseases_Symptoms":
                    title_field = sample.get("Disease", sample.get("disease", ""))
                    if not title_field:
                        for k, v in sample.items():
                            if 'disease' in k.lower():
                                title_field = v
                                break
                                
                    full_text = []
                    sections = []
                    for k, v in sample.items():
                        if v and isinstance(v, str):
                            sections.append({"section_title": k, "text": v})
                            full_text.append(f"{k}: {v}")
                            
                    text = "\n\n".join(full_text)
                    
                    doc = {
                        'title': title_field,
                        'abstract': "",
                        'publication_year': "",
                        'journal': "",
                        'keywords': [title_field] if title_field else [],
                        'sections': sections,
                        'full_body_text': text,
                        'source_id': f"hf_{ds_name.replace('/', '_')}_{count}",
                        'source_type': 'symptom-disease-dataset'
                    }
                    documents.append(doc)
                    
                count += 1
                
        except Exception as e:
            logger.warning(f"Error processing dataset '{ds_name}': {e}")
            continue
            
    return documents
