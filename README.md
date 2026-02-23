# RFQ — RAG pipeline (LangChain + local LLM + Qdrant)

RAG (Retrieval-Augmented Generation) pipeline for **Tekrowe RFQ Feasibility Analysis**. Two variants:

| Script | Qdrant | Use case |
|--------|--------|----------|
| **`local_rag_langchain.py`** | Local Docker (`localhost:6333`) | Local development, no cloud API keys |
| **`rag_langchain.py`** | Qdrant Cloud | Production / shared vector DB |

Both use:

- **Local LLM** — OpenAI-compatible server (e.g. at `localhost:12434`). See **Prerequisites** below.
- **Open-source embeddings** — [sentence-transformers/all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) (384 dimensions).
- **Tekrowe RFQ prompt** — `rag_prompt.py` defines the system prompt (RFQ Feasibility Analyst, vertical alignment, evidence from knowledge base).

Documents are loaded from a folder (default `data/` for local), chunked, embedded, stored in Qdrant, and retrieved at query time; the LLM answers using that context and the structured output format from `rag_prompt.py`.

---

## Project structure

| File / folder | Purpose |
|---------------|---------|
| **`rfq_api.py`** | FastAPI app: `/ingestRFQs`, `/analyze`, `/analyzeFile`. Uses `local_rag_langchain` for RAG. |
| **`local_rag_langchain.py`** | RAG CLI + core: local Qdrant, embeddings, chain. Run with `--ingest` or `--query`. |
| **`rag_langchain.py`** | Same RAG logic but with Qdrant Cloud. |
| **`rag_prompt.py`** | Tekrowe RFQ Feasibility Analyst prompt and `RAG_PROMPT` template. |
| **`local_model_run.py`** | Minimal example: call the local LLM (no RAG). |
| **`quadrant_service.py`** | Qdrant Cloud helpers for a separate `test` collection (user-scoped embeddings). Not used by the RAG pipeline. |
| **`data/`** | Default folder for RFQ documents (`.txt`, `.pdf`, `.docx`) and `data/profiles/tekrowe_profiles.json`. |
| **`.env`** | Local config (LLM URL, Qdrant, RAG options). Not committed; see **Environment variables** below. |

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

Create a `.env` in the project root (it is gitignored). For **local only** you do **not** need Qdrant Cloud vars:

```env
# Local LLM (optional; defaults match local_model_run.py)
LOCAL_LLM_BASE_URL=http://localhost:12434/engines/v1
LOCAL_LLM_MODEL=ai/llama3.2:1B-F16
LOCAL_LLM_API_KEY=anything
# Optional: increase context window if your LLM server supports it (e.g. Ollama num_ctx). Default 0 = do not set.
# LOCAL_LLM_NUM_CTX=8192

# Local Qdrant (optional; default is http://localhost:6333)
QDRANT_LOCAL_URL=http://localhost:6333

# RAG (optional)
RAG_COLLECTION_NAME=rag_docs
RAG_DOCS_DIR=data
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
# Optional: max characters for the question sent to the LLM (avoids exceed_context_size_error). Default 6000.
# MAX_QUESTION_CHARS=6000
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

## FastAPI RFQ API (`rfq_api.py`)

In addition to the CLI script, you can expose the RFQ feasibility engine over HTTP using FastAPI.

### 1. Install dependencies

Already covered by:

```bash
pip install -r requirements.txt
```

(`requirements.txt` includes `fastapi` and `uvicorn[standard]`.)

### 2. Run the API server

From the project root:

```bash
uvicorn rfq_api:app --reload
```

By default this starts on `http://127.0.0.1:8000`.

### 3. CORS (external frontend)

The API has **CORS** enabled so an external frontend (e.g. on another origin) can call it. Default middleware allows all origins (`*`), all methods, and all headers. Restrict `allow_origins` in `rfq_api.py` for production if needed.

### 4. Endpoints

- **POST `/ingestRFQs`**  
  Trigger ingestion of RFQ documents into Qdrant.

  Request body:

  ```json
  {
    "docs_dir": "data",
    "collection": "rag_docs"
  }
  ```

  Both optional; defaults: `RAG_DOCS_DIR` / `data`, `RAG_COLLECTION_NAME` / `rag_docs`.

  Response: `{"message": "Ingestion completed", "docs_dir": "...", "collection": "..."}`.

- **POST `/analyze`**  
  Run RFQ feasibility analysis using the RAG pipeline.

  Request body:

  ```json
  {
    "question": "Summarize the RFQ and provide a feasibility recommendation.",
    "collection": "rag_docs",
    "k": 4
  }
  ```

  Response: `{"answer": "...", "references": [...], "raw_sources": [...]}`.

- **POST `/analyzeFile`**  
  Upload an RFQ file (.txt, .pdf, .docx); converted to text and analyzed.

  - Content type: `multipart/form-data`.
  - Form fields: `file` (required), `collection` (optional), `k` (optional).
  - Response shape same as `/analyze`.
  - Long documents are truncated to fit the LLM context (see `MAX_QUESTION_CHARS`); if you see context errors, increase `LOCAL_LLM_NUM_CTX` on the server or set `MAX_QUESTION_CHARS` in `.env`.

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

## Optional: other LLM backends

You can use any OpenAI-compatible server. For example, to serve a model with **vLLM** and point this project at it:

```bash
pip install vllm
vllm serve Qwen/Qwen2.5-1.5B-Instruct
```

Then set in `.env`: `LOCAL_LLM_BASE_URL=http://localhost:8000/v1` and `LOCAL_LLM_MODEL=Qwen/Qwen2.5-1.5B-Instruct`.

---

## Troubleshooting

- **“No documents found”**  
  Add at least one `.txt`, `.pdf`, or `.docx` to `data/` (or `--docs-dir` / `RAG_DOCS_DIR`).

- **Local Qdrant connection errors**  
  Ensure the Qdrant container is running and `QDRANT_LOCAL_URL` is `http://localhost:6333` (or the correct host/port).

- **Cloud: “Set QDRANT_CLOUD_HOST and QDRANT_CLOUD_API_KEY”**  
  Required only for `rag_langchain.py`; set both in `.env`.

- **Local LLM connection errors**  
  Confirm the LLM server is running and `LOCAL_LLM_BASE_URL` / `LOCAL_LLM_MODEL` in `.env` point to it. See `local_model_run.py` for a minimal test.

- **Context size exceeded (e.g. 4096 tokens)**  
  Long documents are truncated (see `MAX_QUESTION_CHARS`). To allow longer input, set `LOCAL_LLM_NUM_CTX` if your server supports it, or use a model with a larger context window.

- **Reset RAG collection (start fresh)**  
  With local Qdrant running, delete and recreate the collection:
  ```python
  from qdrant_client import QdrantClient
  client = QdrantClient(url="http://localhost:6333")
  client.delete_collection(collection_name="rag_docs")
  ```
  Then run `python local_rag_langchain.py --ingest` again.

- **First run slow**  
  The first time, `sentence-transformers` may download the embedding model; later runs use the cache.

- **PPTX / segfault on Windows**  
  PPTX loading is disabled in the loader registry; only .txt, .pdf, and .docx are ingested.

- **Clean build artifacts**  
  The repo uses a `.gitignore` for `__pycache__/`, `.venv/`, `.env`, etc. To remove cached bytecode:  
  `find . -type d -name __pycache__ -exec rm -rf {} +` (Linux/macOS) or delete `__pycache__` folders manually on Windows.
