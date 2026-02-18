# RFQ — RAG pipeline (LangChain + local LLM + Qdrant)

RAG (Retrieval-Augmented Generation) pipeline that uses:

- **Local LLM** — same setup as `local_model_run.py` (OpenAI-compatible server, e.g. llama.cpp at `localhost:12434`).
- **Qdrant Cloud** — same connection as `quadrant_service.py` for vector storage and search.
- **Open-source embeddings** — [sentence-transformers/all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) (384 dimensions).

Documents are loaded from a `docs/` folder, chunked, embedded with the open-source model, stored in a dedicated Qdrant collection, and retrieved at query time; the local LLM answers using that context.

---

## Prerequisites

1. **Local LLM server**  
   Running and reachable at the URL you set in `.env` (e.g. `http://localhost:12434/engines/v1`), with a model loaded (e.g. `ai/llama3.2:1B-F16`). See `local_model_run.py`.

2. **Qdrant Cloud**  
   A Qdrant instance and API key, same as used in `quadrant_service.py` (`QDRANT_CLOUD_HOST`, `QDRANT_CLOUD_API_KEY`).

3. **Python 3.10+** and a virtual environment (e.g. `.venv`).

---

## Setup

### 1. Create and activate venv

```bash
python -m venv .venv
.venv\Scripts\activate   # Windows
# source .venv/bin/activate   # Linux/macOS
```

### 2. Install dependencies

```bash
pip install python-dotenv openai
pip install "langchain>=0.2" langchain-community langchain-openai langchain-text-splitters
pip install qdrant-client sentence-transformers
pip install pypdf
```

Optional: use a `requirements.txt` (see below).

### 3. Environment variables

Create a `.env` in the project root (same as for `quadrant_service.py`, plus optional RAG overrides):

```env
# Qdrant (required for RAG; same as quadrant_service.py)
QDRANT_CLOUD_HOST=https://your-cluster.qdrant.io
QDRANT_CLOUD_API_KEY=your-api-key

# Local LLM (optional; defaults match local_model_run.py)
LOCAL_LLM_BASE_URL=http://localhost:12434/engines/v1
LOCAL_LLM_MODEL=ai/llama3.2:1B-F16
LOCAL_LLM_API_KEY=anything

# RAG (optional)
RAG_COLLECTION_NAME=rag_docs
RAG_DOCS_DIR=docs
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
```

---

## Usage

### Ingest documents into Qdrant

Put `.txt` and/or `.pdf` files in a folder (default: `docs/`). Then run:

```bash
python rag_langchain.py --ingest
```

Optional:

- `--docs-dir path/to/folder` — folder to read from (default: `docs`).
- `--collection name` — Qdrant collection name (default: `rag_docs`).

This creates the collection (if needed) with the correct embedding size (384 for the default model) and upserts chunk embeddings.

### Ask questions (interactive)

```bash
python rag_langchain.py
```

Then type questions; the app retrieves relevant chunks from Qdrant and calls the local LLM with that context. Type `exit` or `quit` to stop.

### Single query from CLI

```bash
python rag_langchain.py --query "What is the main topic of the documents?"
```

Optional: `--collection name`, `--k 4` (number of chunks to retrieve).

---

## How it fits with your existing code

| File | Role |
|------|------|
| `local_model_run.py` | Defines how to call the local LLM (base URL, model, messages). |
| `quadrant_service.py` | Defines Qdrant connection and helpers for the `test` collection (1536-dim, user-scoped). |
| `rag_langchain.py` | RAG pipeline: uses the **same** local LLM config and **same** Qdrant env vars, but a **separate** collection (`rag_docs`) and **open-source** 384-dim embeddings so document search is self-contained and free of OpenAI embedding API. |

The RAG collection is independent of the `test` collection used in `quadrant_service.py`; you can keep using both.

---

## Optional: `requirements.txt`

You can add a `requirements.txt` for the RAG stack, for example:

```text
python-dotenv
openai
langchain>=0.2
langchain-community
langchain-openai
langchain-text-splitters
qdrant-client
sentence-transformers
pypdf
```

Then: `pip install -r requirements.txt`.

---

## Troubleshooting

- **“Set QDRANT_CLOUD_HOST and QDRANT_CLOUD_API_KEY”**  
  Ensure `.env` has these set (same as for `quadrant_service.py`).

- **“No documents found”**  
  Use a folder that contains at least one `.txt` or `.pdf`, or set `--docs-dir` / `RAG_DOCS_DIR`.

- **Local LLM connection errors**  
  Confirm the server is running and `LOCAL_LLM_BASE_URL` and `LOCAL_LLM_MODEL` match `local_model_run.py`.

- **First run slow**  
  The first time you run, `sentence-transformers` may download the embedding model; later runs use the cache.
