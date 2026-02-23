from typing import Optional, List, Dict, Any
import os
import tempfile
from pathlib import Path
from urllib.parse import urlparse
from uuid import uuid4

import requests
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from langchain_community.document_loaders import TextLoader, PyPDFLoader, Docx2txtLoader

from local_rag_langchain import (
    ingest_documents,
    build_rag_chain,
    DOCS_DIR,
    RAG_COLLECTION_NAME,
)

load_dotenv()

# Max characters for the "question" sent to the LLM (system + profiles + context also use tokens).
# ~4 chars ≈ 1 token; 4096 context − reserve for system/profiles/context/response ≈ 2500 → ~1500 tokens for question ≈ 6000 chars.
MAX_QUESTION_CHARS = int(os.getenv("MAX_QUESTION_CHARS", "6000"))

app = FastAPI(
    title="Tekrowe RFQ Feasibility API",
    description=(
        "FastAPI wrapper around local_rag_langchain.py.\n"
        "- /ingestRFQs: ingest RFQ documents into Qdrant\n"
        "- /analyze: run RFQ feasibility analysis using RAG\n"
        "- /analyzeFile: upload an RFQ file and analyze it"
    ),
    version="2.0.0",
)

# CORS for external frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class IngestRequest(BaseModel):
    docs_dir: Optional[str] = None
    collection: Optional[str] = None


class IngestResponse(BaseModel):
    message: str
    docs_dir: str
    collection: str


class OneDriveIngestRequest(BaseModel):
    file_url: str
    collection: Optional[str] = None


class AnalyzeRequest(BaseModel):
    question: str
    collection: Optional[str] = None
    k: Optional[int] = None


class AnalyzeResponse(BaseModel):
    answer: str
    references: List[str]
    raw_sources: Optional[List[Dict[str, Any]]] = None


def _extract_text_from_file(temp_path: Path, suffix: str) -> str:
    """Load a single uploaded file (.txt, .pdf, .docx) and return its text."""
    suffix = suffix.lower()
    if suffix == ".txt":
        loader = TextLoader(str(temp_path), encoding="utf-8")
    elif suffix == ".pdf":
        loader = PyPDFLoader(str(temp_path))
    elif suffix in (".docx",):
        loader = Docx2txtLoader(str(temp_path))
    else:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{suffix}'. Only .txt, .pdf, .docx are supported.",
        )

    docs = loader.load()
    return "\n\n".join(d.page_content for d in docs)


@app.post("/ingestRFQs", response_model=IngestResponse)
def ingest_rfqs(body: IngestRequest) -> IngestResponse:
    """
    Trigger ingestion of RFQ documents into Qdrant.
    - docs_dir: folder containing RFQ files (.txt, .pdf, .docx). Defaults to RAG_DOCS_DIR / 'data'.
    - collection: Qdrant collection name. Defaults to RAG_COLLECTION_NAME / 'rag_docs'.
    """
    docs_dir = body.docs_dir or DOCS_DIR
    collection = body.collection or RAG_COLLECTION_NAME

    try:
        ingest_documents(docs_dir=docs_dir, collection_name=collection)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {exc}") from exc

    return IngestResponse(
        message="Ingestion completed",
        docs_dir=docs_dir,
        collection=collection,
    )


@app.post("/ingestOneDrive", response_model=IngestResponse)
def ingest_onedrive(body: OneDriveIngestRequest) -> IngestResponse:
    """
    Download a pre-authenticated OneDrive file URL into DOCS_DIR and ingest it into Qdrant.
    The file must be one of: .txt, .pdf, .docx.
    """
    if not body.file_url or not body.file_url.strip():
        raise HTTPException(status_code=400, detail="file_url must not be empty")

    try:
        resp = requests.get(body.file_url, stream=True, timeout=30)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Failed to download OneDrive file: {exc}") from exc

    if resp.status_code >= 400:
        raise HTTPException(
            status_code=400,
            detail=f"Failed to download OneDrive file: HTTP {resp.status_code}",
        )

    filename: Optional[str] = None
    content_disposition = resp.headers.get("content-disposition") or resp.headers.get("Content-Disposition")
    if content_disposition and "filename=" in content_disposition:
        # Very lightweight parsing; handles: attachment; filename="foo.pdf"
        parts = content_disposition.split("filename=")
        if len(parts) > 1:
            filename = parts[1].strip().strip('";')

    if not filename:
        parsed = urlparse(body.file_url)
        filename = os.path.basename(parsed.path) or f"onedrive_{uuid4().hex}"

    # Try to infer extension from filename or content-type
    _, ext = os.path.splitext(filename)
    ext = ext.lower()

    if not ext:
        content_type = (
            resp.headers.get("content-type")
            or resp.headers.get("Content-Type")
            or ""
        ).split(";")[0].strip().lower()

        if content_type == "application/pdf":
            ext = ".pdf"
        elif content_type in {
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            "application/msword",
        }:
            ext = ".docx"
        elif content_type.startswith("text/"):
            ext = ".txt"

    if not ext:
        # Fallback for OneDrive/SharePoint Word docs that hide the extension
        ext = ".docx"

    if not filename.lower().endswith(ext):
        filename = f"{filename}{ext}"

    if ext not in {".txt", ".pdf", ".docx"}:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported OneDrive file type '{ext}'. Only .txt, .pdf, .docx are supported.",
        )

    docs_dir = DOCS_DIR
    collection = body.collection or RAG_COLLECTION_NAME

    try:
        target_dir = Path(docs_dir)
        target_dir.mkdir(parents=True, exist_ok=True)
        target_path = target_dir / filename

        with target_path.open("wb") as f:
            for chunk in resp.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)

        ingest_documents(docs_dir=str(target_dir), collection_name=collection)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"OneDrive ingestion failed: {exc}") from exc

    return IngestResponse(
        message="OneDrive ingestion completed",
        docs_dir=docs_dir,
        collection=collection,
    )


def _truncate_question_for_context(question: str, max_chars: int = MAX_QUESTION_CHARS) -> str:
    """Truncate question so total prompt fits within the LLM context window (e.g. 4096 tokens)."""
    if not question or len(question) <= max_chars:
        return question
    truncated = question[:max_chars].rstrip()
    return (
        truncated
        + f"\n\n[Document truncated for context limit — full text was {len(question)} characters; first {len(truncated)} shown.]"
    )


def _analyze_rfq_internal(collection: str, k: int, question: str) -> AnalyzeResponse:
    """Shared logic for /analyze and /analyzeFile: build chain, invoke, format response."""
    question = _truncate_question_for_context(question)
    try:
        chain = build_rag_chain(collection_name=collection, k=k)
        result = chain.invoke(question)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {exc}") from exc

    if isinstance(result, dict) and "answer" in result and "sources" in result:
        answer = result["answer"]
        sources = result["sources"] or []
        references = []
        raw_sources = []
        for i, doc in enumerate(sources, start=1):
            m = getattr(doc, "metadata", {}) or {}
            file_name = m.get("file_name") or m.get("source") or "unknown"
            file_type = m.get("file_type", "?")
            page = m.get("page", None)
            page_str = f", page {page}" if page not in (None, "") else ""
            references.append(f"{i}. {file_name} ({file_type}{page_str})")
            raw_sources.append(
                {"metadata": m, "page_content": getattr(doc, "page_content", "")},
            )
        return AnalyzeResponse(answer=answer, references=references, raw_sources=raw_sources)
    return AnalyzeResponse(answer=str(result), references=[], raw_sources=None)


@app.post("/analyze", response_model=AnalyzeResponse)
def analyze_rfq(body: AnalyzeRequest) -> AnalyzeResponse:
    """
    Run RFQ feasibility analysis using the RAG pipeline.
    - question: the RFQ or question text to analyze
    - collection: optional override for Qdrant collection (default: rag_docs)
    - k: optional top-k chunks to retrieve (default: 4)
    """
    if not body.question.strip():
        raise HTTPException(status_code=400, detail="question must not be empty")

    collection = body.collection or RAG_COLLECTION_NAME
    k = body.k or 4
    return _analyze_rfq_internal(collection=collection, k=k, question=body.question)


@app.post("/analyzeFile", response_model=AnalyzeResponse)
async def analyze_rfq_file(
    file: UploadFile = File(...),
    collection: Optional[str] = Form(None),
    k: Optional[int] = Form(None),
) -> AnalyzeResponse:
    """
    Upload an RFQ file (.txt, .pdf, .docx), convert to text, and run feasibility analysis.
    """
    filename = file.filename or ""
    _, ext = os.path.splitext(filename)
    if not ext:
        raise HTTPException(status_code=400, detail="Uploaded file must have an extension.")

    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = Path(tmp.name)

        try:
            text = _extract_text_from_file(tmp_path, ext)
        finally:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass

    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to read uploaded file: {exc}") from exc

    collection = collection or RAG_COLLECTION_NAME
    return _analyze_rfq_internal(collection=collection, k=k or 4, question=text)
