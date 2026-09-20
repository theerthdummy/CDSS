# Steps to Run Biomedical-Vector-DB From Scratch

This guide walks you through setting up the Medical RAG Vector Database entirely from scratch on a fresh machine.

---

## Step 1: System Requirements

Ensure you have the following installed:

| Software | Version | Download |
|---|---|---|
| Python | 3.11+ | [python.org](https://www.python.org/downloads/) |
| Git | Latest | [git-scm.com](https://git-scm.com/downloads) |
| Docker Desktop | Latest | [docker.com](https://www.docker.com/products/docker-desktop/) |
| Ollama | Latest | [ollama.com](https://ollama.com/) |

**Hardware Requirements:**
- **RAM**: 16GB minimum (BioBERT loads ~400MB into memory — much lighter than BGE-M3)
- **Storage**: ~5GB free disk space (for Docker image, model weights, and vector data)
- **GPU**: Recommended but not required. A dedicated NVIDIA GPU speeds up embedding generation and LLM inference.

---

## Step 2: Get Required API Keys

### 1. NCBI / PubMed Central API Key (Recommended)

Without this key, you are limited to 3 API requests per second. With it, you get 10 requests per second.

1. Go to [NCBI Registration](https://www.ncbi.nlm.nih.gov/account/).
2. Create an account or log in.
3. Go to **Account Settings** (click your username in the top right).
4. Under **API Key Management**, click **Create an API Key**.
5. Copy the generated key string.

### 2. HuggingFace Account (Required for Datasets)

1. Create a free account at [HuggingFace](https://huggingface.co/).
2. Go to your **Settings > Access Tokens** and generate a token (read permission).
3. Authenticate your CLI:
   ```bash
   pip install -U "huggingface_hub[cli]"
   huggingface-cli login
   ```
   Paste your token when prompted.

---

## Step 3: Clone the Repository

```bash
git clone <your-github-repo-url>
cd Biomedical-Vector-DB
```

---

## Step 4: Create a Virtual Environment (Recommended)

```bash
python -m venv venv

# On Windows:
venv\Scripts\activate

# On Mac/Linux:
source venv/bin/activate
```

---

## Step 5: Install Python Dependencies

```bash
pip install -r requirements.txt
```

**For GPU support (CUDA):** Install the CUDA version of PyTorch first by following the instructions at [pytorch.org](https://pytorch.org/get-started/locally/), then run the command above.

---

## Step 6: Configure Environment Variables

1. Copy the example environment file:
   ```bash
   # Windows:
   copy .env.example .env

   # Mac/Linux:
   cp .env.example .env
   ```

2. Open `.env` in a text editor and fill in your credentials:
   ```env
   NCBI_EMAIL="your.real.email@example.com"
   NCBI_API_KEY="your_copied_api_key_here"
   ```

---

## Step 7: Start Docker Desktop

1. Open **Docker Desktop** from your Start Menu / Applications.
2. Wait until the Docker Engine status shows **"Running"** (green icon in the system tray).
3. Verify Docker is working by running:
   ```bash
   docker --version
   ```
   You should see something like `Docker version 27.x.x`.

> **Important:** Docker Desktop MUST be running before you execute the build command. The system will automatically create and manage a container named `vectordb-cdss`.

---

## Step 8: Setup Ollama Local LLM

1. Start Ollama (open the Ollama application, or run `ollama serve` in a separate terminal).
2. Pull the Qwen 2.5 model (~2GB download):
   ```bash
   ollama pull qwen2.5:3b
   ```

---

## Step 9: Build the Vector Database

This is the main step. It will:
- Start a fresh Docker container `vectordb-cdss` with Qdrant
- Download medical papers from PMC, Europe PMC, ClinicalTrials.gov
- Stream symptom datasets from HuggingFace
- Fetch preprints from medRxiv/bioRxiv
- Chunk all text into ~512-character segments
- Generate 768-dim dense embeddings using BioBERT
- Store everything in the Dockerized Qdrant database

### Full Build
```bash
python main.py --mode build
```

### Sample Build (Quick Test — 3 queries, 1 article each)
```bash
python main.py --mode build-sample
```

**Expected duration (full build):** 2-4 hours depending on internet speed and hardware.

**Expected output:** ~250,000+ chunks across 5 data sources.

> **Note:** If the script is interrupted, re-running `python main.py --mode build` will start fresh (the Docker container is cleared and recreated each time).

---

## Step 10: Use the System

### Interactive Search

```bash
python main.py --mode query
```

Type any medical question and the system will:
1. Encode your query using BioBERT into a 768-dim dense vector.
2. Search the Qdrant database using cosine similarity.
3. Apply source priority re-ranking (peer-reviewed > preprints).
4. Return the most relevant research passages.

**Example queries:**
```
What are the symptoms and treatment options for myocardial infarction?
Differentiate between Crohn's disease and ulcerative colitis
How is subarachnoid hemorrhage diagnosed?
```

### View Database Statistics

```bash
python main.py --mode stats
```

---

## Step 11 (Optional): Run Benchmarks

Make sure Ollama is running (`ollama serve`), then:

### PubMedQA (Yes/No/Maybe medical questions)
```bash
python -m test.evaluate_pubmedqa
```

### BioASQ (Yes/No biomedical questions)
```bash
python -m test.evaluate_bioasq
```

### MedBullets (Clinical MCQ — USMLE Step 2/3 style)
```bash
python -m test.evaluate_medbullets
```

Each benchmark tests 100 questions by default and outputs a final accuracy percentage.
By default, benchmarks use `MAX_SAMPLES=100` from `.env` or `config.py`; set `MAX_SAMPLES=None` to run the full dataset for final evaluation.

---

## Troubleshooting

| Error | Cause | Fix |
|---|---|---|
| `Cannot connect to Docker` | Docker Desktop is not running | Start Docker Desktop and wait for it to be ready |
| `Port 6333 already in use` | Another Qdrant instance is on that port | Stop the conflicting service or change `QDRANT_HOST_PORT` in `config.py` |
| `CUDA out of memory` | GPU VRAM is insufficient | Reduce `EMBEDDING_BATCH_SIZE` in `config.py` |
| `Connection refused on localhost:11434` | Ollama is not running | Start Ollama with `ollama serve` |
| `NCBI rate limit exceeded` | Too many API requests without an API key | Add your NCBI API key to `.env` |
| `No results found` for queries | Database is empty or Docker container was stopped | Re-run `python main.py --mode build` |
