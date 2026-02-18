"""
RAG pipeline — LangChain + local LLM + Qdrant (local OR cloud)
Supports: .txt  .pdf  .docx  .pptx

LOCAL QDRANT (recommended — no cloud needed):
    docker run -p 6333:6333 -v $(pwd)/qdrant_storage:/qdrant/storage qdrant/qdrant
    Set in .env:  QDRANT_CLOUD_HOST=http://localhost:6333
                  QDRANT_CLOUD_API_KEY=    (leave blank)

Run order:
    python rag_langchain.py --ingest          ← first time
    python rag_langchain.py                   ← interactive query
    python rag_langchain.py --query "..."     ← single question

Install:
    pip install langchain-qdrant langchain-huggingface qdrant-client
"""

import os
import time
import uuid
from pathlib import Path
from dotenv import load_dotenv

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import Qdrant
from langchain_community.document_loaders import (
    DirectoryLoader, TextLoader, PyPDFLoader,
    Docx2txtLoader, UnstructuredPowerPointLoader,
)
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_openai import ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

# ===========================================================================
#  COLORFUL PRINT HELPERS
# ===========================================================================

R   = "\033[0m"
BLD = "\033[1m"
DIM = "\033[2m"
CYN = "\033[96m"
GRN = "\033[92m"
YLW = "\033[93m"
RED = "\033[91m"
MAG = "\033[95m"
BLU = "\033[94m"
WHT = "\033[97m"


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

QDRANT_URL          = os.getenv("QDRANT_CLOUD_HOST", "http://localhost:6333")
QDRANT_API_KEY      = os.getenv("QDRANT_CLOUD_API_KEY", "")      # blank = local, no auth needed
RAG_COLLECTION_NAME = os.getenv("RAG_COLLECTION_NAME", "rag_docs")
EMBEDDING_MODEL     = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
DOCS_DIR            = os.getenv("RAG_DOCS_DIR", "data")

EMBEDDING_DIM     = 384
UPSERT_BATCH_SIZE = 100


# ===========================================================================
#  QDRANT CLIENT  (works for both local and cloud)
# ===========================================================================

def get_qdrant_client() -> QdrantClient:
    p_start("get_qdrant_client")
    t0 = time.time()
    p_step(f"Connecting → {QDRANT_URL}")

    # If API key is blank (local), don't pass it — local Qdrant rejects auth headers
    if QDRANT_API_KEY:
        client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
    else:
        client = QdrantClient(url=QDRANT_URL)

    mode = "LOCAL" if "localhost" in QDRANT_URL else "CLOUD"
    p_result("mode", mode)
    p_result("url", QDRANT_URL)
    p_end("get_qdrant_client", detail=_t(t0))
    return client


# ===========================================================================
#  LLM
# ===========================================================================

def get_llm() -> ChatOpenAI:
    p_start("get_llm", model=MODEL_NAME)
    t0 = time.time()
    llm = ChatOpenAI(base_url=BASE_URL, api_key=API_KEY, model=MODEL_NAME, temperature=0.2)
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
    p_result("model", EMBEDDING_MODEL)
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
        return []

    path_str = str(path.resolve())
    p_step(f"Scanning → {path_str}")

    loader_registry = [
        ("**/*.txt",  TextLoader,                   "txt"),
        ("**/*.pdf",  PyPDFLoader,                  "pdf"),
        ("**/*.docx", Docx2txtLoader,               "docx"),
        ("**/*.pptx", UnstructuredPowerPointLoader, "pptx"),
    ]

    all_docs = []
    for glob, loader_cls, ftype in loader_registry:
        p_step(f"Loading {ftype.upper():4s}  (glob={glob}) ...")
        try:
            loaded = DirectoryLoader(path_str, glob=glob, loader_cls=loader_cls).load()
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

    p_result("TOTAL docs", len(all_docs))
    p_end("load_documents", detail=_t(t0))
    return all_docs


# ===========================================================================
#  TEXT SPLITTING
# ===========================================================================

def get_text_splitter(chunk_size: int = 1000, chunk_overlap: int = 200):
    p_start("get_text_splitter", chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap, length_function=len,
    )
    p_end("get_text_splitter")
    return splitter


# ===========================================================================
#  COLLECTION MANAGEMENT
# ===========================================================================

def ensure_collection(client: QdrantClient, collection_name: str, vector_size: int):
    p_start("ensure_collection", collection=collection_name, dims=vector_size)
    t0 = time.time()

    existing = [c.name for c in client.get_collections().collections]
    if collection_name in existing:
        p_warn(f"'{collection_name}' already exists — skipping. Delete it to rebuild.")
    else:
        p_step(f"Creating '{collection_name}' ({vector_size}D cosine) ...")
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
        )
        p_result("created", collection_name)

    p_end("ensure_collection", detail=_t(t0))


# ===========================================================================
#  UPSERT
# ===========================================================================

def upsert_chunks(client: QdrantClient, collection_name: str, chunks: list, embeddings):
    p_start("upsert_chunks", total=len(chunks))
    t0 = time.time()

    total     = len(chunks)
    n_batches = (total + UPSERT_BATCH_SIZE - 1) // UPSERT_BATCH_SIZE
    inserted  = 0

    for i in range(n_batches):
        start = i * UPSERT_BATCH_SIZE
        end   = min(start + UPSERT_BATCH_SIZE, total)
        batch = chunks[start:end]
        vecs  = embeddings.embed_documents([c.page_content for c in batch])

        points = [
            PointStruct(
                id=str(uuid.uuid4()),
                vector=vec,
                payload={"page_content": c.page_content, "metadata": c.metadata},
            )
            for c, vec in zip(batch, vecs)
        ]

        p_step(f"Batch {i+1}/{n_batches}  chunks {start+1}–{end} ...")
        client.upsert(collection_name=collection_name, points=points)
        inserted += len(points)
        p_result("uploaded", f"{inserted}/{total}")

    p_result("total upserted", inserted)
    p_end("upsert_chunks", detail=_t(t0))


# ===========================================================================
#  INGEST
# ===========================================================================

def ingest_documents(docs_dir: str = DOCS_DIR, collection_name: str = RAG_COLLECTION_NAME):
    p_start("ingest_documents", docs_dir=docs_dir, collection=collection_name)
    t0 = time.time()

    p_step("Step 1/4 — Loading ...")
    docs = load_documents(docs_dir)
    if not docs:
        p_warn("No docs found. Aborting.")
        return

    p_step(f"Step 2/4 — Splitting {len(docs)} doc(s) ...")
    chunks = get_text_splitter().split_documents(docs)
    p_result("chunks", len(chunks))
    if chunks:
        p_info(f"Sample metadata → {chunks[0].metadata}")

    p_step("Step 3/4 — Setup embeddings + collection ...")
    embeddings = get_embeddings()
    client     = get_qdrant_client()
    ensure_collection(client, collection_name, vector_size=EMBEDDING_DIM)

    p_step("Step 4/4 — Upserting ...")
    upsert_chunks(client, collection_name, chunks, embeddings)

    p_result("DONE", collection_name)
    p_end("ingest_documents", detail=_t(t0))


# ===========================================================================
#  VECTOR STORE
# ===========================================================================

def get_vector_store(client: QdrantClient, embeddings, collection_name: str) -> Qdrant:
    p_start("get_vector_store", collection=collection_name)
    t0 = time.time()
    vs = Qdrant(client=client, collection_name=collection_name, embeddings=embeddings)
    p_result("collection", collection_name)
    p_end("get_vector_store", detail=_t(t0))
    return vs


# ===========================================================================
#  METADATA DISPLAY
#
#  This is what prints the sources panel you see in the response.
#  We separate it from format_docs so we can show it AFTER the answer.
# ===========================================================================

def _render_sources(docs: list) -> str:
    """
    Build a colored sources block from retrieved docs.
    Shown after every answer so you always know exactly where the answer came from.
    """
    lines = [f"\n{BLD}{CYN}{'─' * 60}{R}"]
    lines.append(f"{BLD}{CYN}  SOURCES  ({len(docs)} chunk(s) retrieved){R}")
    lines.append(f"{BLD}{CYN}{'─' * 60}{R}")

    for i, doc in enumerate(docs):
        m        = doc.metadata
        fname    = m.get("file_name",   m.get("source", "unknown"))
        ftype    = m.get("file_type",   "?")
        page     = m.get("page",        None)
        total_pg = m.get("total_pages", None)
        fpath    = m.get("file_path",   "")

        lines.append(f"\n  {BLD}{WHT}Chunk {i+1}{R}")
        lines.append(f"  {BLD}File     :{R} {GRN}{fname}{R}")
        lines.append(f"  {BLD}Type     :{R} {ftype.upper()}")
        if page is not None:
            pg_str = f"{page}" + (f" / {total_pg}" if total_pg else "")
            lines.append(f"  {BLD}Page     :{R} {pg_str}")
        if fpath:
            lines.append(f"  {BLD}Path     :{R} {DIM}{fpath}{R}")

        # Show a short preview of the chunk text
        preview = doc.page_content.replace("\n", " ").strip()[:120]
        lines.append(f"  {BLD}Preview  :{R} {DIM}{preview}…{R}")

    lines.append(f"\n{BLD}{CYN}{'─' * 60}{R}\n")
    return "\n".join(lines)


# ===========================================================================
#  FORMAT DOCS  (context for the prompt — plain text, no colors)
# ===========================================================================

def format_docs(docs: list) -> str:
    """Plain-text context string that goes into the LLM prompt."""
    p_step(f"format_docs — {len(docs)} chunk(s)")
    parts = []
    for i, doc in enumerate(docs):
        m      = doc.metadata
        source = m.get("file_name", m.get("source", "unknown"))
        page   = m.get("page", "")
        header = f"[Source {i+1}: {source}" + (f" | page {page}]" if page != "" else "]")
        parts.append(f"{header}\n{doc.page_content}")
    return "\n\n".join(parts)


# ===========================================================================
#  PROMPT
# ===========================================================================

RAG_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a helpful AI assistant. "
        "Answer using ONLY the provided context. "
        "If the answer is not in the context, say 'I don't have that information.' "
        "Always mention which source document your answer comes from.",
    ),
    ("human", "Context:\n{context}\n\nQuestion: {question}\n\nAnswer:"),
])


# ===========================================================================
#  RAG CHAIN  (with metadata shown after answer)
# ===========================================================================

def build_rag_chain(collection_name: str = RAG_COLLECTION_NAME, k: int = 4):
    p_start("build_rag_chain", collection=collection_name, k=k)
    t0 = time.time()

    p_step("Initialising embeddings ...")
    embeddings = get_embeddings()

    p_step("Connecting to Qdrant ...")
    client = get_qdrant_client()

    p_step("Wrapping vector store ...")
    vs = get_vector_store(client, embeddings, collection_name)

    p_step(f"Retriever (k={k}) ...")
    retriever = vs.as_retriever(search_kwargs={"k": k})

    p_step("LLM ...")
    llm = get_llm()

    # ── Chain that returns BOTH the answer AND the source docs ────────────────
    #
    # We run retrieval once, then send the docs to two places in parallel:
    #   1. format_docs  → plain text context  → LLM  → answer string
    #   2. kept as-is   → printed as sources panel after the answer
    #
    # RunnablePassthrough.assign() lets us attach the raw docs to the chain
    # state without running retrieval twice.

    p_step("Assembling chain ...")

    # Step 1: retrieve docs and attach them to the chain state
    retrieve_step = RunnablePassthrough.assign(
        docs=lambda x: retriever.invoke(x["question"])
    )

    # Step 2: build context string from the retrieved docs
    context_step = RunnablePassthrough.assign(
        context=lambda x: format_docs(x["docs"])
    )

    # Step 3: run the prompt + LLM + parse
    answer_chain = RAG_PROMPT | llm | StrOutputParser()

    # Step 4: combine — produces {"answer": "...", "docs": [...]}
    full_chain = (
        retrieve_step
        | context_step
        | RunnablePassthrough.assign(answer=answer_chain)
    )

    p_end("build_rag_chain", detail=_t(t0))
    return full_chain


def run_query(chain, question: str) -> str:
    """
    Run the chain and print:
        1. The LLM answer
        2. The sources / metadata panel
    Returns the answer string.
    """
    p_step(f"Querying: '{question}'")
    result = chain.invoke({"question": question})

    answer = result["answer"]
    docs   = result["docs"]

    print(f"\n{BLD}{GRN}Answer:{R}\n{answer}")
    print(_render_sources(docs))
    return answer


# ===========================================================================
#  CLI
# ===========================================================================

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="RAG pipeline — LangChain + local LLM + Qdrant",
        epilog=(
            "LOCAL QDRANT SETUP:\n"
            "  docker run -p 6333:6333 -v $(pwd)/qdrant_storage:/qdrant/storage qdrant/qdrant\n\n"
            "FIRST TIME:\n"
            "  1. Add docs to data/ (.txt .pdf .docx .pptx)\n"
            "  2. python rag_langchain.py --ingest\n"
            "  3. python rag_langchain.py\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--ingest",     action="store_true",         help="Ingest docs into Qdrant")
    parser.add_argument("--docs-dir",   default=DOCS_DIR,            help="Docs folder (default: data/)")
    parser.add_argument("--query",      type=str,                    help="Single question then exit")
    parser.add_argument("--collection", default=RAG_COLLECTION_NAME, help="Qdrant collection name")
    parser.add_argument("--k",          type=int, default=4,         help="Chunks to retrieve")
    args = parser.parse_args()

    if args.ingest:
        p_info("Mode: INGEST")
        ingest_documents(docs_dir=args.docs_dir, collection_name=args.collection)
        p_info("Done. Now run:  python rag_langchain.py")
        return

    p_info("Mode: QUERY")
    p_info(f"Qdrant: {QDRANT_URL}")

    try:
        chain = build_rag_chain(collection_name=args.collection, k=args.k)
    except Exception as exc:
        p_error(f"Failed to build chain: {exc}")
        return

    if args.query:
        t0 = time.time()
        run_query(chain, args.query)
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
        try:
            run_query(chain, q)
        except Exception as exc:
            p_error(f"Chain error: {exc}")
        p_info(f"Answered in {_t(t0)}")


if __name__ == "__main__":
    main()