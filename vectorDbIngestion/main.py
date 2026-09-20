"""
Main orchestrator for the Medical RAG Vector Database pipeline.

Usage:
    python main.py --mode build       # Full pipeline: ingest -> process -> embed -> store
    python main.py --mode build-sample # Test pipeline with a small sample
    python main.py --mode query       # Launch interactive search CLI
    python main.py --mode stats       # Print database statistics
"""

import argparse
import gc
import logging
import sys
import time

import config


def setup_logging():
    """Configure logging for the entire application."""
    logging.basicConfig(
        level=getattr(logging, config.LOG_LEVEL),
        format=config.LOG_FORMAT,
    )


logger = logging.getLogger(__name__)


def process_and_store(documents: list, db, source_label: str) -> dict:
    """
    Process documents in memory-safe sub-batches: chunk -> embed -> store -> free.

    For 16GB RAM systems, processing all chunks at once can cause OOM when the
    embedding model holds ~3GB and we try to store 50,000+ dense vectors in RAM.
    This function splits chunks into sub-batches of PROCESS_SUB_BATCH_SIZE,
    embeds and stores each sub-batch, then frees the memory before the next.

    Args:
        documents: List of standardized document dicts from an ingestion source.
        db: An initialized QdrantManager instance.
        source_label: A human-readable label for logging.

    Returns:
        dict with keys: 'documents', 'chunks', 'embedded'
    """
    from processing.chunker import chunk_documents
    from embedding.embedder import generate_embeddings

    stats = {'documents': len(documents), 'chunks': 0, 'embedded': 0}

    if not documents:
        logger.warning("[%s] No documents to process. Skipping.", source_label)
        return stats

    # --- Chunking ---
    logger.info("[%s] Chunking %d documents...", source_label, len(documents))
    chunks = chunk_documents(documents)
    stats['chunks'] = len(chunks)
    logger.info("[%s] Generated %d chunks.", source_label, len(chunks))

    if not chunks:
        logger.warning("[%s] No valid chunks produced. Skipping.", source_label)
        return stats

    # --- Sub-batch embed + store (memory-safe) ---
    sub_batch_size = config.PROCESS_SUB_BATCH_SIZE
    total_embedded = 0

    for start in range(0, len(chunks), sub_batch_size):
        end = min(start + sub_batch_size, len(chunks))
        sub_chunks = chunks[start:end]

        logger.info(
            "[%s] Embedding sub-batch %d-%d / %d...",
            source_label, start + 1, end, len(chunks),
        )
        embedded = generate_embeddings(sub_chunks)
        total_embedded += len(embedded)

        logger.info("[%s] Storing %d vectors in Qdrant...", source_label, len(embedded))
        db.upsert_batch(embedded)

        # Free this sub-batch immediately
        del embedded, sub_chunks
        gc.collect()

    stats['embedded'] = total_embedded
    logger.info("[%s] Stored %d vectors total.", source_label, total_embedded)

    # Free all chunks
    del chunks
    gc.collect()

    return stats


def run_build(sample_mode: bool = False):
    """
    Execute the full pipeline using source-by-source batch processing.
    Each source is ingested, chunked, embedded, and stored before moving to the next.
    """
    from ingestion.pmc_ingestion import ingest_pmc
    from ingestion.europe_pmc_ingestion import ingest_europe_pmc
    from ingestion.clinical_trials_ingestion import ingest_clinical_trials
    from ingestion.huggingface_ingestion import ingest_huggingface
    from ingestion.preprint_ingestion import ingest_preprints
    from database.qdrant_manager import QdrantManager
    from database.docker_manager import ensure_container_running

    start_time = time.time()

    # Calculate per-query limits
    if sample_mode:
        queries = config.SEARCH_QUERIES[:3]
        pmc_per_query = 1
        epmc_per_query = 1
        ct_per_query = 1
        hf_max = 50
        preprint_per_query = 1
        logger.info("=== SAMPLE MODE: Using 3 queries, 1 article per query ===")
    else:
        queries = config.SEARCH_QUERIES
        total_queries = len(queries)
        pmc_per_query = max(1, config.PMC_MAX_ARTICLES // total_queries)
        epmc_per_query = max(1, config.EPMC_MAX_ARTICLES // total_queries)
        ct_per_query = max(1, config.CLINICAL_TRIALS_MAX // total_queries)
        hf_max = config.HUGGINGFACE_MAX_SAMPLES
        preprint_per_query = max(1, config.PREPRINT_MAX_ARTICLES // total_queries)

    # =========================================================================
    # INITIALIZE DOCKER CONTAINER & DATABASE
    # =========================================================================
    logger.info("=" * 70)
    logger.info("INITIALIZING DOCKER CONTAINER & QDRANT DATABASE")
    logger.info("=" * 70)

    ensure_container_running()

    db = QdrantManager()
    if not db.collection_exists():
        db.create_collection()
        logger.info("Created new collection '%s'.", config.COLLECTION_NAME)
    else:
        logger.info("Collection '%s' already exists. Appending data.", config.COLLECTION_NAME)

    total_stats = {'documents': 0, 'chunks': 0, 'embedded': 0}
    pmc_ids_collected = set()

    try:
        # =================================================================
        # BATCH 1: PubMed Central
        # =================================================================
        logger.info("=" * 70)
        logger.info("BATCH 1/5: PubMed Central (PMC)")
        logger.info("=" * 70)

        try:
            pmc_docs = ingest_pmc(
                queries=queries,
                max_per_query=pmc_per_query,
                delay=config.PMC_DELAY,
            )
            for doc in pmc_docs:
                if doc.get("source_id"):
                    pmc_ids_collected.add(doc["source_id"])
            logger.info("PMC: Ingested %d documents.", len(pmc_docs))

            stats = process_and_store(pmc_docs, db, "PubMed Central")
            for k in total_stats:
                total_stats[k] += stats[k]

            del pmc_docs
            gc.collect()
        except Exception as e:
            logger.error("PMC batch failed: %s", e)

        # =================================================================
        # BATCH 2: Europe PMC
        # =================================================================
        logger.info("=" * 70)
        logger.info("BATCH 2/5: Europe PMC")
        logger.info("=" * 70)

        try:
            epmc_docs = ingest_europe_pmc(
                queries=queries,
                max_per_query=epmc_per_query,
                delay=config.EPMC_DELAY,
                exclude_ids=pmc_ids_collected,
            )
            logger.info("Europe PMC: Ingested %d documents.", len(epmc_docs))

            stats = process_and_store(epmc_docs, db, "Europe PMC")
            for k in total_stats:
                total_stats[k] += stats[k]

            del epmc_docs
            gc.collect()
        except Exception as e:
            logger.error("Europe PMC batch failed: %s", e)

        # =================================================================
        # BATCH 3: ClinicalTrials.gov
        # =================================================================
        logger.info("=" * 70)
        logger.info("BATCH 3/5: ClinicalTrials.gov")
        logger.info("=" * 70)

        try:
            ct_docs = ingest_clinical_trials(
                queries=queries,
                max_per_query=ct_per_query,
                delay=config.CT_DELAY,
            )
            logger.info("ClinicalTrials: Ingested %d documents.", len(ct_docs))

            stats = process_and_store(ct_docs, db, "ClinicalTrials.gov")
            for k in total_stats:
                total_stats[k] += stats[k]

            del ct_docs
            gc.collect()
        except Exception as e:
            logger.error("ClinicalTrials batch failed: %s", e)

        # =================================================================
        # BATCH 4: HuggingFace Datasets
        # =================================================================
        logger.info("=" * 70)
        logger.info("BATCH 4/5: HuggingFace Datasets")
        logger.info("=" * 70)

        try:
            hf_docs = ingest_huggingface(max_samples=hf_max)
            logger.info("HuggingFace: Ingested %d documents.", len(hf_docs))

            stats = process_and_store(hf_docs, db, "HuggingFace")
            for k in total_stats:
                total_stats[k] += stats[k]

            del hf_docs
            gc.collect()
        except Exception as e:
            logger.error("HuggingFace batch failed: %s", e)

        # =================================================================
        # BATCH 5: Preprints (medRxiv / bioRxiv)
        # =================================================================
        logger.info("=" * 70)
        logger.info("BATCH 5/5: Preprints (medRxiv / bioRxiv)")
        logger.info("=" * 70)

        try:
            preprint_docs = ingest_preprints(
                queries=queries,
                max_per_query=preprint_per_query,
                delay=config.PREPRINT_DELAY,
            )
            logger.info("Preprints: Ingested %d documents.", len(preprint_docs))

            stats = process_and_store(preprint_docs, db, "Preprints")
            for k in total_stats:
                total_stats[k] += stats[k]

            del preprint_docs
            gc.collect()
        except Exception as e:
            logger.error("Preprint batch failed: %s", e)

        # =================================================================
        # CREATE PAYLOAD INDEXES
        # =================================================================
        db.create_payload_indexes()

        # =================================================================
        # FINAL SUMMARY
        # =================================================================
        info = db.get_collection_info()
        logger.info("Database stats: %s", info)

    finally:
        db.close()

    elapsed = time.time() - start_time
    logger.info("=" * 70)
    logger.info("PIPELINE COMPLETE")
    logger.info("=" * 70)
    logger.info("Documents ingested:  %d", total_stats['documents'])
    logger.info("Chunks created:      %d", total_stats['chunks'])
    logger.info("Embeddings stored:   %d", total_stats['embedded'])
    logger.info("Time elapsed:        %.1f seconds (%.1f minutes)", elapsed, elapsed / 60)
    logger.info("Database location:   %s", config.QDRANT_URL)
    logger.info("=" * 70)


def run_stats():
    """Print database statistics."""
    from database.qdrant_manager import QdrantManager

    db = QdrantManager()
    try:
        if not db.collection_exists():
            logger.info("No collection found at '%s'.", config.QDRANT_URL)
            return

        info = db.get_collection_info()
        print("\n" + "=" * 50)
        print("  MEDICAL RAG DATABASE - Statistics")
        print("=" * 50)
        for key, value in info.items():
            print(f"  {key}: {value}")
        print("=" * 50 + "\n")
    finally:
        db.close()


def run_query():
    """Launch the interactive search CLI."""
    from query.search import run_interactive_cli
    run_interactive_cli()


def main():
    parser = argparse.ArgumentParser(
        description="Medical RAG Vector Database Pipeline"
    )
    parser.add_argument(
        "--mode",
        choices=["build", "build-sample", "query", "stats"],
        required=True,
        help="Pipeline mode: 'build' (full), 'build-sample' (test), "
             "'query' (interactive search), 'stats' (database info)",
    )
    args = parser.parse_args()

    setup_logging()

    if args.mode in ("build", "build-sample"):
        if not config.NCBI_EMAIL:
            logger.error(
                "NCBI_EMAIL is not set. Please set it in .env before running the build."
            )
            sys.exit(1)

    if args.mode == "build":
        run_build(sample_mode=False)
    elif args.mode == "build-sample":
        run_build(sample_mode=True)
    elif args.mode == "query":
        run_query()
    elif args.mode == "stats":
        run_stats()


if __name__ == "__main__":
    main()
