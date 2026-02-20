# RFQ — RAG pipeline (LangChain + local LLM + Qdrant)

RAG (Retrieval-Augmented Generation) pipeline for **Tekrowe RFQ Feasibility Analysis**. Two variants:

| Script | Qdrant | Use case |
|--------|--------|----------|
| **`local_rag_langchain.py`** | Local Docker (`localhost:6333`) | Local development, no cloud API keys |
| **`rag_langchain.py`** | Qdrant Cloud | Production / shared vector DB |

Both use:

- **Local LLM** — OpenAI-compatible server (e.g. llama.cpp at `localhost:12434`), same as `local_model_run.py`.
- **Open-source embeddings** — [sentence-transformers/all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) (384 dimensions).
- **Tekrowe RFQ prompt** — `rag_prompt.py` defines the system prompt (RFQ Feasibility Analyst, vertical alignment, evidence from knowledge base). The local script imports this prompt; the cloud script can use the same or its own.

Documents are loaded from a folder (default `data/` for local), chunked, embedded, stored in Qdrant, and retrieved at query time; the LLM answers using that context and the structured output format from `rag_prompt.py`.

---

## Prerequisites

1. **Local LLM server**  
   Running at the URL in `.env` (e.g. `http://localhost:12434/engines/v1`) with a model loaded (e.g. `ai/llama3.2:1B-F16`). See `local_model_run.py`.

2. **Qdrant**
   - **Local:** Docker Qdrant on `localhost:6333` (see below).
   - **Cloud:** Only if using `rag_langchain.py` — set `QDRANT_CLOUD_HOST` and `QDRANT_CLOUD_API_KEY` in `.env`.

3. **Python 3.10+** and a virtual environment (e.g. `.venv`).

---

## Local setup (`local_rag_langchain.py`)

### 1. Start Qdrant with Docker

```bash
docker run -p 6333:6333 -p 6334:6334 -v $(pwd)/qdrant_storage:/qdrant/storage qdrant/qdrant
```

### 2. Create and activate venv

```bash
python -m venv .venv
.venv\Scripts\activate   # Windows
# source .venv/bin/activate   # Linux/macOS
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Environment variables

Create a `.env` in the project root. For **local only** you do **not** need Qdrant Cloud vars:

```env
# Local LLM (optional; defaults match local_model_run.py)
LOCAL_LLM_BASE_URL=http://localhost:12434/engines/v1
LOCAL_LLM_MODEL=ai/llama3.2:1B-F16
LOCAL_LLM_API_KEY=anything

# Local Qdrant (optional; default is http://localhost:6333)
QDRANT_LOCAL_URL=http://localhost:6333

# RAG (optional)
RAG_COLLECTION_NAME=rag_docs
RAG_DOCS_DIR=data
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
```

### 5. Add documents

Put `.txt`, `.pdf`, and/or `.docx` files in the `data/` folder (or the path set by `RAG_DOCS_DIR`). PPTX is disabled on Windows due to loader instability.

### 6. Ingest into Qdrant

```bash
python local_rag_langchain.py --ingest
```

Optional: `--docs-dir path/to/folder`, `--collection name`.

### 7. Run the RAG (interactive or single query)

```bash
# Interactive
python local_rag_langchain.py

# Single question
python local_rag_langchain.py --query "Summarize the RFQ and provide a feasibility recommendation."
```

Optional: `--collection name`, `--k 4` (chunks to retrieve).

The model uses the **Tekrowe RFQ Feasibility Analyst** prompt from `rag_prompt.py`: structured assessment (RFQ Summary, Vertical Alignment, Technical/Delivery Feasibility, Strategic Fit, Recommendation, Evidence from Knowledge Base).

---

## Cloud setup (`rag_langchain.py`)

Use this when you want to use **Qdrant Cloud** instead of local Docker.

1. In `.env`, set:
   ```env
   QDRANT_CLOUD_HOST=https://your-cluster.qdrant.io
   QDRANT_CLOUD_API_KEY=your-api-key
   ```
2. Install and run as above, but use the script name `rag_langchain.py`:
   ```bash
   python rag_langchain.py --ingest
   python rag_langchain.py
   ```
3. Default docs folder for the cloud script can be set via `RAG_DOCS_DIR` (e.g. `docs` or `data`).

---

## How it fits with your code

| File | Role |
|------|------|
| `local_model_run.py` | How to call the local LLM (base URL, model, messages). |
| `quadrant_service.py` | Qdrant Cloud connection and helpers for the `test` collection (e.g. user-scoped embeddings). |
| **`rag_prompt.py`** | Tekrowe RFQ Feasibility Analyst system prompt and `RAG_PROMPT` (LangChain `ChatPromptTemplate`). Used by `local_rag_langchain.py`; can be used by `rag_langchain.py` as well. |
| **`local_rag_langchain.py`** | Local RAG: local LLM + **local Qdrant (Docker)** + open-source embeddings. Imports prompt from `rag_prompt.py`. Docs from `data/` by default. |
| **`rag_langchain.py`** | Cloud RAG: local LLM + **Qdrant Cloud** + open-source embeddings. Same collection name and embedding model; separate deployment. |

The RAG collection (`rag_docs`) is independent of the `test` collection in `quadrant_service.py`.

---

## Troubleshooting

- **“No documents found”**  
  Add at least one `.txt`, `.pdf`, or `.docx` to `data/` (or `--docs-dir` / `RAG_DOCS_DIR`).

- **Local Qdrant connection errors**  
  Ensure the Qdrant container is running and `QDRANT_LOCAL_URL` is `http://localhost:6333` (or the correct host/port).

- **Cloud: “Set QDRANT_CLOUD_HOST and QDRANT_CLOUD_API_KEY”**  
  Required only for `rag_langchain.py`; set both in `.env`.

- **Local LLM connection errors**  
  Confirm the LLM server is running and `LOCAL_LLM_BASE_URL` / `LOCAL_LLM_MODEL` match `local_model_run.py`.

- **First run slow**  
  The first time, `sentence-transformers` may download the embedding model; later runs use the cache.

- **PPTX / segfault on Windows**  
  PPTX loading is disabled in the loader registry to avoid crashes; only .txt, .pdf, and .docx are ingested.
