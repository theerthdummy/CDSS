import logging
import time
import config
from typing import List, Dict, Any
from tqdm import tqdm
from qdrant_client import QdrantClient, models

logger = logging.getLogger(__name__)

class QdrantManager:
    """Manager for Qdrant vector database operations."""
    
    def __init__(self):
        """Initializes the Qdrant client with Docker container URL and extended timeout."""
        logger.info(f"Initializing QdrantClient at {config.QDRANT_URL}")
        self.client = QdrantClient(
            url=config.QDRANT_URL,
            api_key=config.QDRANT_API_KEY if config.QDRANT_API_KEY else None,
            timeout=120,
        )
        self.collection_name = config.COLLECTION_NAME
        self._current_id = None
        
    def collection_exists(self) -> bool:
        """Checks if the configured collection exists."""
        return self.client.collection_exists(collection_name=self.collection_name)
        
    def create_collection(self):
        """Creates the collection with dense vector configuration (no sparse)."""
        logger.info(f"Creating collection '{self.collection_name}'...")
        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config={
                "dense": models.VectorParams(
                    size=config.DENSE_VECTOR_DIM,
                    distance=models.Distance.COSINE
                )
            },
        )
        logger.info("Collection created successfully.")
        
    def create_payload_indexes(self):
        """Creates payload indexes for optimized metadata filtering."""
        logger.info("Creating payload indexes...")
        
        for field in ["source_type", "disease_category", "evidence_level"]:
            self.client.create_payload_index(
                collection_name=self.collection_name,
                field_name=field,
                field_schema=models.PayloadSchemaType.KEYWORD
            )
            
        self.client.create_payload_index(
            collection_name=self.collection_name,
            field_name="publication_year",
            field_schema=models.PayloadSchemaType.INTEGER
        )
        logger.info("Payload indexes created.")

    def _get_next_id(self) -> int:
        """Get the next auto-increment ID, initializing from collection count if needed."""
        if self._current_id is None:
            if self.collection_exists():
                collection_info = self.client.get_collection(self.collection_name)
                self._current_id = collection_info.points_count + 1
            else:
                self._current_id = 1
        current = self._current_id
        self._current_id += 1
        return current
        
    def upsert_batch(self, embedded_chunks: List[Dict[str, Any]]):
        """
        Uploads embedded chunks to Qdrant in streaming batches with retry logic.
        Dense-only vectors (no sparse).
        """
        if not embedded_chunks:
            return
            
        total = len(embedded_chunks)
        batch_size = config.UPSERT_BATCH_SIZE
        logger.info(f"Upserting {total} points in batches of {batch_size}...")
        
        total_batches = (total + batch_size - 1) // batch_size
        
        for batch_idx in tqdm(range(total_batches), desc="Uploading to Qdrant"):
            start = batch_idx * batch_size
            end = min(start + batch_size, total)
            batch_chunks = embedded_chunks[start:end]
            
            points = []
            for chunk in batch_chunks:
                point_id = self._get_next_id()
                
                pub_year = chunk.get('publication_year')
                if pub_year:
                    try:
                        pub_year = int(pub_year)
                    except (ValueError, TypeError):
                        pub_year = None
                else:
                    pub_year = None

                payload = {
                    'text': chunk.get('text', ''),
                    'source_id': chunk.get('source_id', ''),
                    'source_type': chunk.get('source_type', ''),
                    'publication_year': pub_year,
                    'disease_category': chunk.get('disease_category', ''),
                    'evidence_level': chunk.get('evidence_level', 'Research Article'),
                    'section_title': chunk.get('section_title', ''),
                    'journal': chunk.get('journal', '')
                }
                
                point = models.PointStruct(
                    id=point_id,
                    vector={
                        "dense": chunk['dense_vector'],
                    },
                    payload=payload
                )
                points.append(point)
            
            max_retries = 3
            for attempt in range(1, max_retries + 1):
                try:
                    self.client.upsert(
                        collection_name=self.collection_name,
                        points=points,
                    )
                    break
                except Exception as e:
                    if attempt < max_retries:
                        wait_time = 2 ** attempt
                        logger.warning(
                            f"Batch {batch_idx+1}/{total_batches} failed (attempt {attempt}/{max_retries}): {e}. "
                            f"Retrying in {wait_time}s..."
                        )
                        time.sleep(wait_time)
                    else:
                        logger.error(
                            f"Batch {batch_idx+1}/{total_batches} failed after {max_retries} attempts: {e}. Skipping."
                        )
            
            del points
            
    def get_collection_info(self) -> Dict[str, Any]:
        """Returns information about the collection."""
        if not self.collection_exists():
            return {"status": "Collection does not exist"}
            
        info = self.client.get_collection(self.collection_name)
        return {
            "status": info.status,
            "points_count": info.points_count
        }
        
    def close(self):
        """Closes the client connection."""
        self.client.close()
