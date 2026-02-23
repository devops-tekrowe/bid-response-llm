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
import json
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
from qdrant_client.models import Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchValue

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
EMBEDDING_MODEL     = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
DOCS_DIR            = os.getenv("RAG_DOCS_DIR", "data")

# all-MiniLM-L6-v2 always produces 384-dimensional vectors
EMBEDDING_DIM = 384

# How many chunks to upsert per HTTP request
UPSERT_BATCH_SIZE = 100


# ===========================================================================
#  PROFILES CONTEXT (TEAM CAPABILITIES)
# ===========================================================================


def get_profiles_context(profiles_path: str = os.path.join("data", "profiles", "tekrowe_profiles.json")) -> str:
    """
    Load Tekrowe team profiles and convert them into a concise, readable
    text block that can be injected into the system prompt.

    If the file is missing or invalid, return a short fallback string so
    the prompt still works.
    """
    p_start("get_profiles_context", profiles_path=profiles_path)
    t0 = time.time()

    try:
        full_path = Path(profiles_path)
        if not full_path.exists():
            p_warn(f"Profiles file not found at '{profiles_path}'.")
            p_end("get_profiles_context", detail=_t(t0))
            return "No explicit team profile data is available."

        with full_path.open("r", encoding="utf-8") as f:
            data = json.load(f)

        if not isinstance(data, list):
            p_warn("Profiles JSON is not a list; using generic description.")
            p_end("get_profiles_context", detail=_t(t0))
            return "Structured team profile data is present but not in the expected list format."

        lines = []
        for profile in data:
            name = profile.get("name", "Unknown")
            designation = profile.get("designation", "")
            band = profile.get("band", "")
            summary = profile.get("summary", "")
            key_focus = profile.get("key_focus_areas", []) or []
            tech = profile.get("technologies", []) or []
            projects = profile.get("major_projects", []) or []

            header_parts = [name]
            if designation:
                header_parts.append(designation)
            if band:
                header_parts.append(f"Band {band}")
            header = " — ".join(header_parts)

            lines.append(f"- {header}")
            if summary:
                lines.append(f"  Summary: {summary}")
            if key_focus:
                lines.append(f"  Focus areas: {', '.join(key_focus)}")
            if tech:
                lines.append(f"  Technologies: {', '.join(tech)}")
            if projects:
                proj_summaries = []
                for p in projects:
                    pname = p.get("name", "")
                    domain = p.get("domain", "")
                    if pname and domain:
                        proj_summaries.append(f"{pname} ({domain})")
                    elif pname:
                        proj_summaries.append(pname)
                if proj_summaries:
                    lines.append(f"  Major projects: {', '.join(proj_summaries)}")
            lines.append("")  # blank line between profiles

        result = "\n".join(lines).strip() or "Team profiles file loaded but contained no usable entries."
        p_end("get_profiles_context", detail=_t(t0))
        return result

    except Exception as exc:
        p_warn(f"Failed to load profiles: {exc}")
        p_end("get_profiles_context", detail=_t(t0))
        return "Profiles file could not be loaded due to an error."


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


# RAG_PROMPT is imported from rag_prompt.py (Tekrowe RFQ Feasibility Analyst).


# ===========================================================================
#  RAG CHAIN
# ===========================================================================

def build_rag_chain(collection_name: str = RAG_COLLECTION_NAME, k: int = 4):
    p_start("build_rag_chain", collection=collection_name, k=k)
    t0 = time.time()

    p_step("Initialising embeddings ...")
    embeddings = get_embeddings()

    p_step("Wrapping vector store ...")
    vectorstore = get_vector_store(embeddings, collection_name)

    p_step(f"Building retriever (top k={k}) ...")
    retriever = vectorstore.as_retriever(search_kwargs={"k": k})

    p_step("Initialising LLM ...")
    llm = get_llm()

    p_step("Assembling LCEL chain (answer + references) ...")
    profiles_text = get_profiles_context()
    # Return both the answer and the retrieved docs so we can print metadata references.
    chain = (
        RunnableLambda(lambda q: {"question": q} if isinstance(q, str) else q)
        # Step 1: retrieve docs
        | RunnablePassthrough.assign(
            sources=lambda x: retriever.invoke(x["question"]),
        )
        # Step 2: build context text from retrieved docs
        | RunnablePassthrough.assign(
            context=lambda x: format_docs(x["sources"]),
        )
        # Step 3: run the LLM
        | RunnablePassthrough.assign(
            answer=lambda x: (RAG_PROMPT | llm | StrOutputParser()).invoke(
                {
                    "context": x["context"],
                    "question": x["question"],
                    "profiles": profiles_text,
                }
            ),
        )
        # Output only what the CLI needs
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