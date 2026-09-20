# Biomedical-Vector-DB

A specialized, local Retrieval-Augmented Generation (RAG) system for the biomedical domain. It ingests, processes, and embeds medical literature from multiple sources, storing them in a Dockerized Qdrant vector database. It uses dense vector search with BioBERT embeddings, source-priority re-ranking, and local LLMs (via Ollama) to answer complex medical questions.

## Architecture & Workflow

```
[Data Sources]          [Processing]           [Storage]              [Retrieval]
PMC Articles   ──┐                                                   
Europe PMC     ──┤      Text Cleaning         Qdrant Docker         Dense Search
ClinicalTrials ──┼──►   Chunking (512c)  ──►  Container       ──►  (BioBERT Cosine)
HuggingFace    ──┤      BioBERT Embedding     "vectordb-cdss"      + Source Priority
Preprints      ──┘      (768-dim dense)                              Re-Ranking
```

1. **Ingestion**: Fetches data from PubMed Central (PMC), Europe PMC, ClinicalTrials.gov, HuggingFace datasets, and medRxiv/bioRxiv preprints.
2. **Processing**: Normalizes text, cleans noise, and chunks documents intelligently (respecting section boundaries in papers).
3. **Embedding**: Uses `pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb` via `sentence-transformers` to generate 768-dimensional dense vectors optimized for biomedical text.
4. **Storage**: Stores vectors and payloads in a Dockerized Qdrant container (`vectordb-cdss`).
5. **Retrieval**: Uses dense cosine similarity search with source-priority re-ranking and multi-level fallback thresholds.
6. **Generation**: Injects retrieved context into a local LLM (e.g., `qwen2.5:3b` via Ollama) to synthesize answers.

## Source Priority Re-Ranking

Not all medical sources are equally trustworthy. The system applies score multipliers after retrieval to prioritize higher-quality evidence:

| Source Type | Multiplier | Description |
|---|---|---|
| `clinical-trial` | 1.15x | Peer-reviewed clinical evidence |
| `medical-paper` | 1.10x | Peer-reviewed research papers |
| `symptom-disease-dataset` | 1.00x | Curated datasets (neutral baseline) |
| `preprint` | 0.70x | Not yet peer-reviewed (lowest priority) |

## Project Structure

```text
Biomedical-Vector-DB/
├── config.py                     # Central configuration (API keys, model params, queries)
├── main.py                       # Main entry point for building DB or running search
├── requirements.txt              # Python dependencies
├── steps_to_run.md               # Detailed setup guide from scratch
├── .env                          # API keys (gitignored)
├── .env.example                  # Template for environment variables
├── .gitignore                    # Git ignore rules
├── ingestion/
│   ├── pmc_ingestion.py          # PubMed Central Open Access XML fetcher
│   ├── europe_pmc_ingestion.py   # Europe PMC full-text fetcher
│   ├── clinical_trials_ingestion.py  # ClinicalTrials.gov API v2 fetcher
│   ├── huggingface_ingestion.py  # Symptom-Disease dataset streamer
│   └── preprint_ingestion.py     # medRxiv/bioRxiv preprint fetcher
├── processing/
│   ├── chunker.py                # Section-aware & sentence-level chunking
│   └── text_cleaner.py           # Regex and unicode normalization
├── embedding/
│   └── embedder.py               # BioBERT dense embedding generator
├── database/
│   ├── qdrant_manager.py         # Qdrant collection & indexing manager
│   └── docker_manager.py         # Docker container lifecycle manager
├── query/
│   └── search.py                 # Dense search engine with cosine similarity + re-ranking
└── test/
    ├── evaluate_pubmedqa.py      # PubMedQA yes/no/maybe benchmark
    ├── evaluate_bioasq.py        # BioASQ yes/no benchmark
    └── evaluate_medbullets.py    # MedBullets MCQ benchmark
```

## Prerequisites

1. **Python 3.11+**
2. **Docker Desktop**: Installed and running ([download here](https://www.docker.com/products/docker-desktop/)).
3. **Ollama**: Installed and running ([download here](https://ollama.com/)).
4. **Hardware**: 16GB RAM minimum. A dedicated GPU is recommended but not required.

## Quick Start

```bash
# 1. Clone and install
git clone <repository_url>
cd Biomedical-Vector-DB
pip install -r requirements.txt

# 2. Configure environment
copy .env.example .env     # Then edit .env with your NCBI credentials

# 3. Pull the LLM model
ollama pull qwen2.5:3b

# 4. Build the database (requires Docker Desktop running)
python main.py --mode build

# 5. Search the database
python main.py --mode query

# 6. Run benchmarks
python -m test.evaluate_pubmedqa
python -m test.evaluate_bioasq
python -m test.evaluate_medbullets
```

## Data Sources

| Source | Type | Articles | Description |
|---|---|---|---|
| PubMed Central (PMC) | `medical-paper` | ~2,000 | Peer-reviewed Open Access papers |
| Europe PMC | `medical-paper` | ~1,500 | Additional European Open Access research |
| ClinicalTrials.gov | `clinical-trial` | ~800 | Clinical study protocols and results |
| HuggingFace | `symptom-disease-dataset` | ~15,000 | Symptom-to-diagnosis mappings |
| medRxiv / bioRxiv | `preprint` | ~1,000 | Pre-publication medical research |

## Configuration

All parameters are centralized in `config.py`. Key settings:

| Parameter | Default | Description |
|---|---|---|
| `EMBEDDING_MODEL` | `pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb` | BioBERT sentence embedding model |
| `DENSE_VECTOR_DIM` | `768` | BioBERT output dimensions |
| `QDRANT_URL` | `http://localhost:6333` | Docker container endpoint |
| `CHUNK_SIZE` | `512` | Characters per text chunk |
| `EMBEDDING_BATCH_SIZE` | `32` | Chunks per embedding batch |

## Benchmarks

| Benchmark | Dataset | Task | Command |
|---|---|---|---|
| PubMedQA | `qiaojin/PubMedQA` | Yes/No/Maybe QA | `python -m test.evaluate_pubmedqa` |
| BioASQ | `jmhb/BioASQ` | Yes/No biomedical QA | `python -m test.evaluate_bioasq` |
| MedBullets | `LangAGI-Lab/medbullets` | Clinical MCQ (USMLE-style) | `python -m test.evaluate_medbullets` |

## License

This project is intended for educational and research purposes. Medical decisions should not be made based on the outputs of this system without consulting a qualified healthcare professional.
