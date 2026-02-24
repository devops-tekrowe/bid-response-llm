"""
RAG pipeline — LangChain + local LLM + local Qdrant (Docker).
Uses the Tekrowe RFQ Feasibility Analyst prompt from rag_prompt.py.
Supports: .txt  .pdf  .docx  (PPTX disabled on Windows due to loader segfaults)

Run order (first time):
    python local_rag_langchain.py --ingest   ← creates collection and loads docs
    python local_rag_langchain.py             ← interactive query mode
    python local_rag_langchain.py --query "..."  ← single question
"""

import os
import time
import uuid
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv

# ── LangChain ──────────────────────────────────────────────────────────────
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain_community.document_loaders import (
    DirectoryLoader, TextLoader, PyPDFLoader,
    Docx2txtLoader,
)
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_openai import ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ── Prompt (Tekrowe RFQ Feasibility Analyst) ───────────────────────────────
from rag_prompt import RAG_PROMPT

# ── Qdrant client (direct) ─────────────────────────────────────────────────
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct,
    Filter,
    FieldCondition,
    MatchValue,
)
try:
    from qdrant_client.models import FilterSelector
except (ImportError, AttributeError):
    try:
        from qdrant_client.http.models import FilterSelector
    except ImportError:
        from qdrant_client.http.models.models import FilterSelector

# ===========================================================================
#  COLORFUL PRINT HELPERS  (pure ANSI — zero extra dependencies)
# ===========================================================================

R   = "\033[0m"   # Reset
BLD = "\033[1m"   # Bold
DIM = "\033[2m"   # Dim

CYN = "\033[96m"  # Cyan    → START
GRN = "\033[92m"  # Green   → END / success
YLW = "\033[93m"  # Yellow  → warnings
RED = "\033[91m"  # Red     → errors
MAG = "\033[95m"  # Magenta → sub-steps
BLU = "\033[94m"  # Blue    → info
WHT = "\033[97m"  # White   → results


def _badge(color, text):
    return f"{BLD}{color}[{text:^5}]{R}"

def p_start(fn: str, **kw):
    kw_str = "  ".join(f"{k}={v!r}" for k, v in kw.items())
    print(f"\n{_badge(CYN, 'START')} {BLD}{fn}{R}  {DIM}{kw_str}{R}")

def p_end(fn: str, detail: str = ""):
    tail = f"  {DIM}{detail}{R}" if detail else ""
    print(f"{_badge(GRN, 'END')}{BLD} {fn}{R}{tail}")

def p_step(msg: str):
    print(f"{_badge(MAG, 'STEP')}  {msg}")

def p_info(msg: str):
    print(f"{_badge(BLU, 'INFO')}  {msg}")

def p_warn(msg: str):
    print(f"{_badge(YLW, 'WARN')}  {YLW}{msg}{R}")

def p_error(msg: str):
    print(f"{_badge(RED, 'ERR')}   {RED}{BLD}{msg}{R}")

def p_result(label: str, value):
    print(f"{_badge(WHT, 'RSLT')}  {BLD}{label}:{R} {DIM}{value}{R}")

def _t(t0: float) -> str:
    return f"{time.time() - t0:.2f}s"


# ===========================================================================
#  CONFIG
# ===========================================================================

load_dotenv()

BASE_URL   = os.getenv("LOCAL_LLM_BASE_URL", "http://localhost:12434/engines/v1")
MODEL_NAME = os.getenv("LOCAL_LLM_MODEL",    "ai/llama3.2:1B-F16")
API_KEY    = os.getenv("LOCAL_LLM_API_KEY",  "anything")

# Local Qdrant (Docker) only; cloud support lives in rag_langchain.py
QDRANT_LOCAL_URL   = os.getenv("QDRANT_LOCAL_URL", "http://localhost:6333")
RAG_COLLECTION_NAME = os.getenv("RAG_COLLECTION_NAME", "rag_docs")
PROFILES_COLLECTION_NAME = os.getenv("PROFILES_COLLECTION_NAME", "profiles_docs")
PROFILES_DOCS_DIR = os.getenv("PROFILES_DOCS_DIR", os.path.join("data", "profiles"))
EMBEDDING_MODEL     = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
DOCS_DIR            = os.getenv("RAG_DOCS_DIR", "data")

# all-MiniLM-L6-v2 always produces 384-dimensional vectors
EMBEDDING_DIM = 384

# How many chunks to upsert per HTTP request
UPSERT_BATCH_SIZE = 100


# ===========================================================================
#  PROFILES CONTEXT (TEAM CAPABILITIES)
# ===========================================================================
# Profile context is retrieved from the profiles collection at query time.
# When no profile chunks are returned, format_profiles_docs() returns a simple
# message; no fallback (e.g. JSON) is supported.
#
# ===========================================================================
#  QDRANT CLIENT
# ===========================================================================

def get_qdrant_client() -> QdrantClient:
    p_start("get_qdrant_client")
    t0 = time.time()

    # Local Docker default: http://localhost:6333
    p_step(f"Connecting (local) → {QDRANT_LOCAL_URL}")
    client = QdrantClient(url=QDRANT_LOCAL_URL)
    p_result("qdrant_url", QDRANT_LOCAL_URL)
    p_end("get_qdrant_client", detail=_t(t0))
    return client


# ===========================================================================
#  LLM
# ===========================================================================

def get_llm() -> ChatOpenAI:
    p_start("get_llm", base_url=BASE_URL, model=MODEL_NAME)
    t0 = time.time()

    llm = ChatOpenAI(
        base_url=BASE_URL,
        api_key=API_KEY,
        model=MODEL_NAME,
        temperature=0.2,
    )

    p_result("model", MODEL_NAME)
    p_end("get_llm", detail=_t(t0))
    return llm


# ===========================================================================
#  EMBEDDINGS
# ===========================================================================

def get_embeddings() -> HuggingFaceEmbeddings:
    p_start("get_embeddings", model=EMBEDDING_MODEL)
    t0 = time.time()

    emb = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

    p_result("embedding_model", EMBEDDING_MODEL)
    p_end("get_embeddings", detail=_t(t0))
    return emb


# ===========================================================================
#  DOCUMENT LOADING
# ===========================================================================

def load_documents(docs_dir: str = DOCS_DIR) -> list:
    p_start("load_documents", docs_dir=docs_dir)
    t0 = time.time()

    path = Path(docs_dir)
    if not path.exists():
        p_warn(f"Directory '{docs_dir}' not found.")
        p_end("load_documents", detail="0 docs")
        return []

    path_str = str(path.resolve())
    p_step(f"Scanning → {path_str}")

    # PPTX disabled: UnstructuredPowerPointLoader causes segfaults on Windows
    loader_registry = [
        ("**/*.txt",  TextLoader,      "txt"),
        ("**/*.pdf",  PyPDFLoader,     "pdf"),
        ("**/*.docx", Docx2txtLoader,  "docx"),
    ]

    all_docs = []
    for glob, loader_cls, ftype in loader_registry:
        p_step(f"Loading {ftype.upper():4s} files  (glob={glob}) ...")
        loader = DirectoryLoader(path_str, glob=glob, loader_cls=loader_cls)
        try:
            loaded = loader.load()
            for doc in loaded:
                src = doc.metadata.get("source", "unknown")
                doc.metadata.update({
                    "file_name": Path(src).name,
                    "file_type": ftype,
                    "file_path": src,
                })
            p_result(f"{ftype.upper()} docs", len(loaded))
            all_docs.extend(loaded)
        except Exception as exc:
            p_warn(f"{ftype.upper()} loader failed: {exc}")

    p_result("TOTAL docs loaded", len(all_docs))
    p_end("load_documents", detail=_t(t0))
    return all_docs


def load_single_document(file_path: str) -> list:
    """
    Load one file (.txt, .pdf, .docx) into a list of LangChain Documents.
    Same formats as rag_docs; used for ingesting a single profile file into the profiles collection.
    """
    p_start("load_single_document", file_path=file_path)
    t0 = time.time()
    path = Path(file_path)
    if not path.exists():
        p_warn(f"File not found: {file_path}")
        p_end("load_single_document", detail="0 docs")
        return []

    suffix = path.suffix.lower()
    if suffix == ".txt":
        loader = TextLoader(str(path), encoding="utf-8")
    elif suffix == ".pdf":
        loader = PyPDFLoader(str(path))
    elif suffix == ".docx":
        loader = Docx2txtLoader(str(path))
    else:
        p_warn(f"Unsupported file type: {suffix}")
        p_end("load_single_document", detail="0 docs")
        return []

    docs = loader.load()
    ftype = suffix.lstrip(".")
    for doc in docs:
        doc.metadata.update({
            "file_name": path.name,
            "file_type": ftype,
            "file_path": str(path),
        })
    p_result("docs loaded", len(docs))
    p_end("load_single_document", detail=_t(t0))
    return docs


# ===========================================================================
#  TEXT SPLITTING
# ===========================================================================

def get_text_splitter(chunk_size: int = 1000, chunk_overlap: int = 200):
    p_start("get_text_splitter", chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
    )
    p_end("get_text_splitter")
    return splitter


# ===========================================================================
#  COLLECTION MANAGEMENT
# ===========================================================================

def ensure_collection(client: QdrantClient, collection_name: str, vector_size: int):
    """
    Create the Qdrant collection if it doesn't exist.
    Safe to call multiple times — skips creation if already present.

    WHY NOT Qdrant.from_documents()?
    ─────────────────────────────────
    LangChain's from_documents() internally calls the deprecated
    client.recreate_collection() and passes an 'init_from' kwarg.
    Newer qdrant-client versions reject that kwarg with:
        AssertionError: Unknown arguments: ['init_from']
    Managing the collection ourselves with create_collection() sidesteps
    both the deprecated method and the unknown-kwarg crash.
    """
    p_start("ensure_collection", collection=collection_name, vector_size=vector_size)
    t0 = time.time()

    existing = [c.name for c in client.get_collections().collections]
    p_result("existing collections", existing)

    if collection_name in existing:
        p_warn(f"Collection '{collection_name}' already exists — skipping creation.")
        p_info("To rebuild from scratch: delete the collection in Qdrant Cloud UI, then re-ingest.")
    else:
        p_step(f"Creating '{collection_name}' ({vector_size}D, cosine distance) ...")
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(
                size=vector_size,
                distance=Distance.COSINE,
            ),
        )
        p_result("created", collection_name)

    p_end("ensure_collection", detail=_t(t0))


# ===========================================================================
#  DOCUMENT ID CHECKING
# ===========================================================================

def doc_id_exists(client: QdrantClient, collection_name: str, doc_id: str) -> bool:
    """
    Check if a document with the given doc_id already exists in Qdrant.
    
    We search for any points that have metadata.doc_id matching the given doc_id.
    If any points are found, the document already exists.
    """
    p_start("doc_id_exists", doc_id=doc_id)
    t0 = time.time()
    
    try:
        # Use scroll with a filter on metadata.doc_id; we don't need vectors here
        scroll_result, _ = client.scroll(
            collection_name=collection_name,
            scroll_filter=Filter(
                must=[
                    FieldCondition(
                        key="metadata.doc_id",
                        match=MatchValue(value=doc_id),
                    )
                ]
            ),
            limit=1,  # only need to know if at least one exists
        )

        exists = len(scroll_result) > 0
        p_result("exists", exists)
        p_end("doc_id_exists", detail=_t(t0))
        return exists
    except Exception as exc:
        # If collection doesn't exist or query fails, assume doc doesn't exist
        p_warn(f"Error checking doc_id existence: {exc}")
        p_end("doc_id_exists", detail="error (assuming not exists)")
        return False


def delete_points_by_doc_id(client: QdrantClient, collection_name: str, doc_id: str) -> None:
    """
    Delete all points whose payload metadata.doc_id equals doc_id.
    Used to replace a profile file on re-ingest (delete then upsert).
    """
    p_start("delete_points_by_doc_id", collection=collection_name, doc_id=doc_id)
    t0 = time.time()
    try:
        client.delete(
            collection_name=collection_name,
            points_selector=FilterSelector(
                filter=Filter(
                    must=[
                        FieldCondition(
                            key="metadata.doc_id",
                            match=MatchValue(value=doc_id),
                        )
                    ]
                )
            ),
        )
        p_result("deleted", "points matching doc_id")
    except Exception as exc:
        p_warn(f"Delete by doc_id failed (collection may be missing): {exc}")
    p_end("delete_points_by_doc_id", detail=_t(t0))


def delete_document_by_filename(collection_name: str, filename: str) -> None:
    """
    Convenience wrapper to delete all points for a given filename (doc_id) in a collection.
    Used by the API when deleting documents based on their original file name.
    """
    client = get_qdrant_client()
    delete_points_by_doc_id(client, collection_name, doc_id=filename)


# ===========================================================================
#  UPSERT
# ===========================================================================

def upsert_chunks(client: QdrantClient, collection_name: str, chunks: list, embeddings, doc_id: str, file_name: str):
    """
    Embed each chunk and store it in Qdrant as a PointStruct.

    HOW METADATA IS STORED
    ──────────────────────
    Every point stored in Qdrant has exactly three fields:

        id      → a UUID string we generate (must be unique per point)

        vector  → list of 384 floats produced by the embedding model
                  This is the "address" of the chunk in semantic space.
                  Qdrant uses this to find the nearest neighbours when you query.

        payload → a plain Python dict — this is the metadata store.
                  We put TWO things in here:

                  "page_content" : the raw text of the chunk
                      ↳ LangChain needs this to reconstruct a Document object
                        when it gets results back from Qdrant.

                  "metadata"     : a nested dict with everything we know about
                                   the source file:
                                       doc_id      → unique document ID (file name)
                                       file_name   → "Medical_book.pdf"
                                       uploaded_at → "2026-02-20" (ISO date)
                                       file_type   → "pdf"
                                       file_path   → full Windows path
                                       page        → page number (from PyPDFLoader)
                                       total_pages, producer, creator, etc.
                                   When retrieved, this shows up in
                                   Document.metadata so format_docs() can print
                                   "[Chunk 1 | Medical_book.pdf | page=3]"

    WHY BATCH?
    ──────────
    3428 chunks × 384 floats = ~5 MB of raw vectors.
    Sending in one shot can hit Qdrant Cloud's HTTP body limit or just be
    very slow. Batches of 100 keep each request tiny and show progress.
    """
    p_start("upsert_chunks", collection=collection_name, total=len(chunks))
    t0 = time.time()

    total     = len(chunks)
    n_batches = (total + UPSERT_BATCH_SIZE - 1) // UPSERT_BATCH_SIZE
    inserted  = 0

    for batch_idx in range(n_batches):
        start = batch_idx * UPSERT_BATCH_SIZE
        end   = min(start + UPSERT_BATCH_SIZE, total)
        batch = chunks[start:end]

        p_step(f"Batch {batch_idx + 1}/{n_batches}  (chunks {start+1}–{end}) ...")

        # 1. Get the raw text strings for this batch
        texts = [chunk.page_content for chunk in batch]

        # 2. Embed them — returns a list-of-lists, each inner list is 384 floats
        vectors = embeddings.embed_documents(texts)

        # 3. Pack into PointStructs
        # Add doc_id, file_name, and uploaded_at to metadata for deduplication
        uploaded_at = datetime.now().strftime("%Y-%m-%d")
        points = [
            PointStruct(
                id=str(uuid.uuid4()),          # random unique ID
                vector=vector,                 # 384-float embedding
                payload={
                    "page_content": chunk.page_content,
                    "metadata": {
                        **chunk.metadata,      # file_name, file_type, page, etc.
                        "doc_id": doc_id,      # unique document ID (file name)
                        "file_name": file_name,
                        "uploaded_at": uploaded_at,
                    },
                },
            )
            for chunk, vector in zip(batch, vectors)
        ]

        # 4. Send to Qdrant
        client.upsert(collection_name=collection_name, points=points)
        inserted += len(points)
        p_result("uploaded so far", f"{inserted}/{total}")

    p_result("total upserted", inserted)
    p_end("upsert_chunks", detail=_t(t0))


# ===========================================================================
#  INGEST  (full pipeline)
# ===========================================================================

def ingest_documents(docs_dir: str = DOCS_DIR, collection_name: str = RAG_COLLECTION_NAME):
    p_start("ingest_documents", docs_dir=docs_dir, collection=collection_name)
    t0 = time.time()

    p_step("Step 1/4 — Loading documents ...")
    docs = load_documents(docs_dir)
    if not docs:
        p_warn(f"No documents found in '{docs_dir}'.")
        p_end("ingest_documents", detail="aborted")
        return

    p_step(f"Step 2/4 — Splitting {len(docs)} document(s) into chunks ...")
    chunks = get_text_splitter().split_documents(docs)
    p_result("chunks created", len(chunks))
    if chunks:
        p_info(f"Sample metadata → {chunks[0].metadata}")

    p_step("Step 3/4 — Initialising embeddings and ensuring collection ...")
    embeddings = get_embeddings()
    client     = get_qdrant_client()
    ensure_collection(client, collection_name, vector_size=EMBEDDING_DIM)

    p_step("Step 4/4 — Checking for existing documents and upserting new chunks ...")
    
    # Group chunks by file_name (doc_id)
    chunks_by_doc = defaultdict(list)
    for chunk in chunks:
        file_name = chunk.metadata.get("file_name", "unknown")
        chunks_by_doc[file_name].append(chunk)
    
    p_result("unique documents", len(chunks_by_doc))
    
    # Process each document
    total_inserted = 0
    total_skipped = 0
    for file_name, doc_chunks in chunks_by_doc.items():
        doc_id = file_name  # doc_id is the file name
        
        p_step(f"Checking document: {doc_id} ({len(doc_chunks)} chunks) ...")
        
        if doc_id_exists(client, collection_name, doc_id):
            p_warn(f"Document '{doc_id}' already exists — skipping ingestion")
            total_skipped += len(doc_chunks)
        else:
            p_info(f"Document '{doc_id}' not found — inserting {len(doc_chunks)} chunks")
            upsert_chunks(client, collection_name, doc_chunks, embeddings, doc_id, file_name)
            total_inserted += len(doc_chunks)
    
    p_result("chunks inserted", total_inserted)
    p_result("chunks skipped", total_skipped)
    p_result("DONE — collection", collection_name)
    p_end("ingest_documents", detail=_t(t0))


# ===========================================================================
#  PROFILES INGEST  (single file → profiles collection)
# ===========================================================================


def ingest_profile_file(
    file_path: str,
    collection_name: str = PROFILES_COLLECTION_NAME,
) -> None:
    """
    Ingest a single profile file (e.g. PDF/DOCX/TXT from OneDrive) into the profiles collection.
    Replaces any existing points for the same doc_id (filename) so re-ingesting updates the profile.
    """
    p_start("ingest_profile_file", file_path=file_path, collection=collection_name)
    t0 = time.time()

    path = Path(file_path)
    doc_id = path.name
    suffix = path.suffix.lower()
    if suffix not in (".txt", ".pdf", ".docx"):
        raise ValueError(f"Unsupported profile file type '{suffix}'. Use .txt, .pdf, or .docx.")

    p_step("Loading single document ...")
    docs = load_single_document(str(path))
    if not docs:
        p_warn("No content loaded from file.")
        p_end("ingest_profile_file", detail="aborted")
        return

    p_step("Splitting into chunks ...")
    chunks = get_text_splitter().split_documents(docs)
    p_result("chunks", len(chunks))

    p_step("Ensuring collection and removing previous version of this doc ...")
    embeddings = get_embeddings()
    client = get_qdrant_client()
    ensure_collection(client, collection_name, vector_size=EMBEDDING_DIM)
    delete_points_by_doc_id(client, collection_name, doc_id)

    p_step("Upserting chunks ...")
    upsert_chunks(client, collection_name, chunks, embeddings, doc_id=doc_id, file_name=doc_id)

    p_result("DONE — collection", collection_name)
    p_end("ingest_profile_file", detail=_t(t0))


# ===========================================================================
#  VECTOR STORE  (query path)
# ===========================================================================

def get_vector_store(embeddings, collection_name: str) -> QdrantVectorStore:
    """
    Create QdrantVectorStore wrapper for querying.

    langchain-qdrant 1.1.0 requires a QdrantClient instance (not url/api_key).
    It calls client.query_points() internally — compatible with qdrant-client >=1.15.
    """
    p_start("get_vector_store", collection=collection_name)
    t0 = time.time()

    client = get_qdrant_client()

    vectorstore = QdrantVectorStore(
        client=client,
        collection_name=collection_name,
        embedding=embeddings,
    )

    p_result("collection", collection_name)
    p_end("get_vector_store", detail=_t(t0))
    return vectorstore


# ===========================================================================
#  FORMAT RETRIEVED DOCS
# ===========================================================================

def format_docs(docs: list) -> str:
    p_step(f"format_docs — {len(docs)} chunk(s) retrieved")
    parts = []
    for i, doc in enumerate(docs):
        m      = doc.metadata
        source = m.get("file_name", m.get("source", "unknown"))
        ftype  = m.get("file_type", "?")
        page   = m.get("page", "")
        header = (
            f"[Chunk {i+1} | {source} | type={ftype}"
            + (f" | page={page}]" if page != "" else "]")
        )
        parts.append(f"{header}\n{doc.page_content}")
    return "\n\n".join(parts)


def format_references(docs: list) -> str:
    """Format retrieved docs' metadata for printing as references."""
    if not docs:
        return "  (no sources retrieved)"
    lines = []
    for i, doc in enumerate(docs, 1):
        m = doc.metadata or {}
        file_name = m.get("file_name", m.get("source", "unknown"))
        if isinstance(file_name, str) and os.path.sep in file_name:
            file_name = os.path.basename(file_name)
        file_type = m.get("file_type", "?")
        page = m.get("page", "")
        page_str = f", page {page}" if page != "" else ""
        lines.append(f"  {i}. {file_name} ({file_type}{page_str})")
    return "\n".join(lines)


def format_profiles_docs(docs: list) -> str:
    """
    Format retrieved profile chunks for the RAG prompt's {profiles} placeholder.
    When no profile docs are retrieved, returns a simple message; no fallback is supported.
    """
    if not docs:
        return "No team profile data available."
    return "\n\n".join(doc.page_content for doc in docs)


# RAG_PROMPT is imported from rag_prompt.py (Tekrowe RFQ Feasibility Analyst).


# ===========================================================================
#  RAG CHAIN
# ===========================================================================

def build_rag_chain(
    collection_name: str = RAG_COLLECTION_NAME,
    profiles_collection_name: str = PROFILES_COLLECTION_NAME,
    k: int = 4,
    k_profiles: int = 10,
):
    """
    Build the RAG chain: retrieve from both RFQ docs and profiles collection,
    then run the LLM with context + profiles + question.
    Profiles are ingested via /ingestProfileFromOneDrive (PDF/DOCX/TXT from OneDrive links).
    """
    p_start(
        "build_rag_chain",
        collection=collection_name,
        profiles_collection=profiles_collection_name,
        k=k,
        k_profiles=k_profiles,
    )
    t0 = time.time()

    p_step("Initialising embeddings ...")
    embeddings = get_embeddings()

    p_step("Wrapping vector stores (RFQ docs + profiles) ...")
    vectorstore = get_vector_store(embeddings, collection_name)
    profiles_store = get_vector_store(embeddings, profiles_collection_name)

    p_step(f"Building retrievers (RFQ k={k}, profiles k={k_profiles}) ...")
    retriever = vectorstore.as_retriever(search_kwargs={"k": k})
    profiles_retriever = profiles_store.as_retriever(search_kwargs={"k": k_profiles})

    p_step("Initialising LLM ...")
    llm = get_llm()

    p_step("Assembling LCEL chain (answer + references) ...")

    def _log_retrieval(label: str, retr, question: str):
        p_step(f"Querying {label} retriever ...")
        docs = retr.invoke(question)
        p_step(f"{label} retriever returned {len(docs)} chunk(s)")
        return docs

    llm_chain = RAG_PROMPT | llm | StrOutputParser()

    def _log_llm_call(context: str, question: str, profiles: str):
        p_step("Calling LLM for RFQ feasibility analysis ...")
        answer = llm_chain.invoke(
            {
                "context": context,
                "question": question,
                "profiles": profiles,
            }
        )
        p_step("LLM inference completed.")
        return answer

    chain = (
        RunnableLambda(lambda q: {"question": q} if isinstance(q, str) else q)
        | RunnablePassthrough.assign(
            sources=lambda x: _log_retrieval("RFQ docs", retriever, x["question"]),
            profile_docs=lambda x: _log_retrieval("profiles", profiles_retriever, x["question"]),
        )
        | RunnablePassthrough.assign(
            context=lambda x: format_docs(x["sources"]),
            profiles=lambda x: format_profiles_docs(x["profile_docs"]),
        )
        | RunnablePassthrough.assign(
            answer=lambda x: _log_llm_call(
                context=x["context"],
                question=x["question"],
                profiles=x["profiles"],
            ),
        )
        | RunnableLambda(lambda x: {"answer": x["answer"], "sources": x["sources"]})
    )

    p_end("build_rag_chain", detail=_t(t0))
    return chain


# ===========================================================================
#  CLI
# ===========================================================================

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="RAG pipeline — LangChain + local LLM + Qdrant Cloud",
        epilog=(
            "FIRST TIME:\n"
            "  1. Add docs to data/ (.txt .pdf .docx .pptx)\n"
            "  2. python local_rag_langchain.py --ingest\n"
            "  3. python local_rag_langchain.py\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--ingest",     action="store_true",         help="Ingest docs into Qdrant (run first)")
    parser.add_argument("--docs-dir",   default=DOCS_DIR,            help="Folder with docs (default: data/)")
    parser.add_argument("--query",      type=str,                    help="Single question then exit")
    parser.add_argument("--collection", default=RAG_COLLECTION_NAME, help="Qdrant collection name")
    parser.add_argument("--k",          type=int, default=4,         help="Chunks to retrieve per query")
    args = parser.parse_args()

    if args.ingest:
        p_info("Mode: INGEST")
        ingest_documents(docs_dir=args.docs_dir, collection_name=args.collection)
        p_info("Done. Now run:  python local_rag_langchain.py")
        return

    p_info("Mode: QUERY")
    p_info("Haven't ingested yet? Run:  python local_rag_langchain.py --ingest")

    try:
        chain = build_rag_chain(collection_name=args.collection, k=args.k)
    except Exception as exc:
        p_error(f"Failed to build chain: {exc}")
        return

    def print_result(result):
        if isinstance(result, dict) and "answer" in result and "sources" in result:
            print(f"\n{BLD}{GRN}Answer:{R}\n{result['answer']}")
            print(f"\n{BLD}{CYN}References:{R}\n{format_references(result['sources'])}\n")
        else:
            print(f"\n{BLD}{GRN}Answer:{R}\n{result}\n")

    if args.query:
        t0 = time.time()
        print_result(chain.invoke(args.query))
        p_info(f"Done in {_t(t0)}")
        return

    print(f"\n{BLD}{CYN}RAG ready.{R}  Ask a question or type 'exit'.\n")
    while True:
        try:
            q = input(f"{BLD}{BLU}You:{R} ").strip()
        except (EOFError, KeyboardInterrupt):
            p_warn("Interrupted.")
            break
        if not q or q.lower() in ("exit", "quit", "q"):
            p_info("Goodbye.")
            break
        t0 = time.time()
        p_step(f"Querying: '{q}'")
        try:
            print_result(chain.invoke(q))
        except Exception as exc:
            p_error(f"Chain error: {exc}")
        p_info(f"Answered in {_t(t0)}")


if __name__ == "__main__":
    main()