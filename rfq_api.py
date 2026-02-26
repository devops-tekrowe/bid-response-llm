from typing import Optional, List, Dict, Any
import os
import tempfile
from pathlib import Path
from urllib.parse import urlparse
from uuid import uuid4
import logging
import traceback

import requests
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from langchain_community.document_loaders import TextLoader, PyPDFLoader, Docx2txtLoader

from local_rag_langchain import (
    ingest_documents,
    ingest_profile_file,
    delete_document_by_filename,
    build_rag_chain,
    DOCS_DIR,
    RAG_COLLECTION_NAME,
    PROFILES_COLLECTION_NAME,
    PROFILES_DOCS_DIR,
)

load_dotenv()

logger = logging.getLogger(__name__)

# Max characters for the "question" sent to the LLM (system + profiles + context also use tokens).
# ~4 chars ≈ 1 token; 4096 context − reserve for system/profiles/context/response ≈ 2500 → ~1500 tokens for question ≈ 6000 chars.
MAX_QUESTION_CHARS = int(os.getenv("MAX_QUESTION_CHARS", "6000"))

app = FastAPI(
    title="Tekrowe RFQ Feasibility API",
    description=(
        "FastAPI wrapper around local_rag_langchain.py.\n"
        "- /ingestRFQs: ingest RFQ documents from a folder into Qdrant\n"
        "- /ingestOneDrive: ingest RFQ files from an array of pre-authenticated OneDrive URLs\n"
        "- /ingestProfileFromOneDrive: ingest profile files (PDF/DOCX/TXT) from an array of OneDrive URLs into the profiles collection\n"
        "- /analyze: run RFQ feasibility analysis using RAG\n"
        "- /analyzeFile: analyze an RFQ file from a pre-authenticated OneDrive URL"
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
    file_urls: List[str]
    collection: Optional[str] = None


class IngestProfileFromOneDriveRequest(BaseModel):
    file_urls: List[str]
    collection: Optional[str] = None


class AnalyzeRequest(BaseModel):
    question: str
    collection: Optional[str] = None
    k: Optional[int] = None


class AnalyzeFileRequest(BaseModel):
    file_urls: List[str]
    collection: Optional[str] = None
    k: Optional[int] = None


class DeleteByUrlRequest(BaseModel):
    file_urls: List[str]
    collection: Optional[str] = None


class AnalyzeResponse(BaseModel):
    answer: str
    references: List[str]
    raw_sources: Optional[List[Dict[str, Any]]] = None


def _download_onedrive_file_to_dir(
    file_url: str,
    target_dir: Path,
    default_name_prefix: str = "file",
) -> Path:
    """
    Download a file from a pre-authenticated OneDrive (or similar) URL into target_dir.
    Infers filename from Content-Disposition or URL path, and extension from filename or Content-Type.
    Returns the path to the saved file. Raises HTTPException on failure.
    """
    if not file_url or not file_url.strip():
        raise HTTPException(status_code=400, detail="file_url must not be empty")
    try:
        resp = requests.get(file_url, stream=True, timeout=30)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Failed to download file: {exc}") from exc
    if resp.status_code >= 400:
        raise HTTPException(
            status_code=400,
            detail=f"Failed to download file: HTTP {resp.status_code}",
        )

    filename: Optional[str] = None
    content_disposition = resp.headers.get("content-disposition") or resp.headers.get("Content-Disposition")
    if content_disposition and "filename=" in content_disposition:
        parts = content_disposition.split("filename=")
        if len(parts) > 1:
            filename = parts[1].strip().strip('";')

    if not filename:
        parsed = urlparse(file_url)
        filename = os.path.basename(parsed.path) or f"{default_name_prefix}_{uuid4().hex}"

    _, ext = os.path.splitext(filename)
    ext = ext.lower()

    if not ext:
        content_type = (
            resp.headers.get("content-type") or resp.headers.get("Content-Type") or ""
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
        else:
            ext = ".docx"

    if not filename.lower().endswith(ext):
        filename = f"{filename}{ext}"

    if ext not in {".txt", ".pdf", ".docx"}:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{ext}'. Only .txt, .pdf, .docx are supported.",
        )

    target_dir.mkdir(parents=True, exist_ok=True)
    target_path = target_dir / filename
    with target_path.open("wb") as f:
        for chunk in resp.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)
    return target_path


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


@app.post("/ingestProfileFromOneDrive", response_model=IngestResponse)
def ingest_profile_from_onedrive(body: IngestProfileFromOneDriveRequest) -> IngestResponse:
    """
    Download pre-authenticated OneDrive profile files (PDF, DOCX, or TXT) from an array of URLs
    and ingest them into the profiles Qdrant collection. Re-ingesting the same filename replaces its chunks.
    Downloaded files are deleted after successful ingestion.
    """
    if not body.file_urls:
        raise HTTPException(status_code=400, detail="file_urls must not be empty")

    collection = body.collection or PROFILES_COLLECTION_NAME
    target_dir = Path(PROFILES_DOCS_DIR)
    n = 0
    downloaded_paths: List[Path] = []
    success = False
    try:
        for file_url in body.file_urls:
            target_path = _download_onedrive_file_to_dir(file_url, target_dir, default_name_prefix="profile")
            downloaded_paths.append(target_path)
            ingest_profile_file(file_path=str(target_path), collection_name=collection)
            n += 1
        success = True
    except HTTPException:
        raise
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Profile ingestion failed: {exc}") from exc
    finally:
        if success:
            for path in downloaded_paths:
                try:
                    os.unlink(path)
                except OSError:
                    pass

    return IngestResponse(
        message=f"Profile ingestion from OneDrive completed for {n} file(s).",
        docs_dir=PROFILES_DOCS_DIR,
        collection=collection,
    )


@app.post("/ingestOneDrive", response_model=IngestResponse)
def ingest_onedrive(body: OneDriveIngestRequest) -> IngestResponse:
    """
    Download pre-authenticated OneDrive RFQ files from an array of URLs into DOCS_DIR,
    then ingest them into Qdrant. Supported formats: .txt, .pdf, .docx.
    Downloaded files are deleted after successful ingestion.
    """
    if not body.file_urls:
        raise HTTPException(status_code=400, detail="file_urls must not be empty")

    docs_dir = DOCS_DIR
    collection = body.collection or RAG_COLLECTION_NAME
    target_dir = Path(docs_dir)
    n = 0
    downloaded_paths: List[Path] = []
    success = False
    try:
        for file_url in body.file_urls:
            path = _download_onedrive_file_to_dir(file_url, target_dir, default_name_prefix="onedrive")
            downloaded_paths.append(path)
            n += 1
        ingest_documents(docs_dir=docs_dir, collection_name=collection)
        success = True
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"OneDrive ingestion failed: {exc}") from exc
    finally:
        if success:
            for path in downloaded_paths:
                try:
                    os.unlink(path)
                except OSError:
                    pass

    return IngestResponse(
        message=f"OneDrive ingestion completed for {n} file(s).",
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
        # Log full traceback to the terminal before returning a 500,
        # so you see the real underlying error instead of just "500".
        logger.exception("Analysis failed in _analyze_rfq_internal")
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
def analyze_rfq_file(body: AnalyzeFileRequest) -> AnalyzeResponse:
    """
    Download one or more RFQ files from pre-authenticated OneDrive URLs, extract text,
    and run a single feasibility analysis over the combined content.
    Supported formats: .txt, .pdf, .docx.
    """
    if not body.file_urls:
        raise HTTPException(status_code=400, detail="file_urls must not be empty")

    texts: List[str] = []
    with tempfile.TemporaryDirectory() as tmpdir:
        target_dir = Path(tmpdir)
        for file_url in body.file_urls:
            try:
                target_path = _download_onedrive_file_to_dir(
                    file_url, target_dir, default_name_prefix="analyze"
                )
            except HTTPException:
                raise
            suffix = target_path.suffix
            try:
                text = _extract_text_from_file(target_path, suffix)
            except HTTPException:
                raise
            except Exception as exc:
                raise HTTPException(status_code=500, detail=f"Failed to extract text from file: {exc}") from exc
            texts.append(text)

    combined_text = "\n\n-----\n\n".join(texts)
    collection = body.collection or RAG_COLLECTION_NAME
    k = body.k or 4
    return _analyze_rfq_internal(collection=collection, k=k, question=combined_text)


@app.post("/deleteByOneDriveUrl", response_model=IngestResponse)
def delete_by_onedrive_url(body: DeleteByUrlRequest) -> IngestResponse:
    """
    Delete embeddings for one or more documents from a Qdrant collection, using
    their pre-authenticated OneDrive URLs to derive the original filenames/doc_ids.

    - file_urls: array of OneDrive URLs that were previously ingested
    - collection: target collection (e.g. "rag_docs" or "profiles_docs").
                  Defaults to RAG_COLLECTION_NAME ("rag_docs").
    """
    if not body.file_urls:
        raise HTTPException(status_code=400, detail="file_urls must not be empty")

    collection = body.collection or RAG_COLLECTION_NAME

    # Derive the filenames (doc_ids) exactly the same way as ingestion, but
    # use a temporary directory and discard the downloaded files.
    filenames: List[str] = []
    with tempfile.TemporaryDirectory() as tmpdir:
        target_dir = Path(tmpdir)
        for url in body.file_urls:
            path = _download_onedrive_file_to_dir(url, target_dir, default_name_prefix="delete")
            filenames.append(path.name)

    # Delete embeddings for each derived filename in the target collection.
    for name in filenames:
        delete_document_by_filename(collection_name=collection, filename=name)

    return IngestResponse(
        message=f"Deleted documents from collection '{collection}' for {len(filenames)} file URL(s).",
        docs_dir="",
        collection=collection,
    )
