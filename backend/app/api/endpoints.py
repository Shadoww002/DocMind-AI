import uuid
import json
import asyncio
import logging
from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from pydantic import BaseModel, Field, field_validator

from backend.app.core.config import DomainConfig
from backend.app.services.parser import DocumentParser
from backend.app.services.vector_db import VectorStoreManager
from backend.app.pipelines.summarizer import DomainSummarizer
from backend.app.pipelines.extractor import DomainExtractor
from backend.app.pipelines.rag_engine import DomainRAGEngine

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1")

# ── Singletons — models loaded once at startup, not per request ───────────────
_vdb_manager = VectorStoreManager()
_summarizer  = DomainSummarizer()
_extractor   = DomainExtractor()
_rag_engine  = DomainRAGEngine()


# ── Request model ─────────────────────────────────────────────────────────────

class QueryRequest(BaseModel):
    document_id: str  = Field(..., description="Document ID returned by /process.")
    domain:      str  = Field(..., description="Domain used during processing.")
    question:    str  = Field(..., min_length=3, description="User question.")

    @field_validator("domain")
    @classmethod
    def domain_must_be_valid(cls, v: str) -> str:
        v = v.lower().strip()
        if not DomainConfig.is_valid_domain(v):
            raise ValueError(f"Unknown domain '{v}'. Valid: {DomainConfig.domain_names()}")
        return v

    @field_validator("question")
    @classmethod
    def question_not_empty(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("Question cannot be blank.")
        return v


# ── /process ──────────────────────────────────────────────────────────────────

@router.post("/process")
async def process_document(
    file:   UploadFile = File(...),
    domain: str        = Form(...),
    params: str        = Form("{}"),
):
    """
    Full ingestion pipeline:
      parse PDF → chunk → store in ChromaDB → summarize → extract entities

    WHY asyncio.to_thread():
      FastAPI runs on an asyncio event loop. Calling CPU-bound blocking functions
      (summarizer, extractor) directly inside an async def route freezes the entire
      event loop — no other request can be processed until the inference finishes.
      asyncio.to_thread() runs the blocking call in a separate thread from the
      default ThreadPoolExecutor, freeing the event loop immediately.
      On CPU with flan-t5-base, inference takes 5–20s — this matters.
    """
    normalized_domain = domain.lower().strip()
    if not DomainConfig.is_valid_domain(normalized_domain):
        raise HTTPException(
            status_code=400,
            detail=f"Domain '{domain}' is not supported. Choose from: {DomainConfig.domain_names()}",
        )

    try:
        tuning_params = json.loads(params)
        if not isinstance(tuning_params, dict):
            raise ValueError("params must be a JSON object.")
    except (json.JSONDecodeError, ValueError) as exc:
        raise HTTPException(status_code=422, detail=f"Invalid params JSON: {exc}")

    doc_id = f"doc_{uuid.uuid4().hex[:10]}"
    logger.info(f"[API] Processing '{file.filename}' | domain={normalized_domain} | doc_id={doc_id}")

    try:
        chunk_size    = DomainConfig.get_domain_property(normalized_domain, "chunk_size", 1000)
        chunk_overlap = DomainConfig.get_domain_property(normalized_domain, "chunk_overlap", 150)
        parser        = DocumentParser(chunk_size=chunk_size, overlap=chunk_overlap)

        # parse_pdf is already async (awaits file.read())
        pages_data = await parser.parse_pdf(file)
        if not pages_data:
            raise HTTPException(status_code=422, detail="PDF has no extractable text.")

        # create_chunks is CPU-bound but fast (<5ms) — fine to call directly
        chunks = parser.create_chunks(pages_data)

        if len(chunks) < 2:
            logger.warning(
                f"[API] Only {len(chunks)} chunk(s) from '{file.filename}'. "
                "Document may be image-based or very short."
            )

        # store_document encodes embeddings — CPU-bound, run in thread
        await asyncio.to_thread(_vdb_manager.store_document, doc_id, normalized_domain, chunks)

        raw_text      = " ".join(p["text"] for p in pages_data)
        string_chunks = [c["text"] for c in chunks]

        # Both are CPU-bound blocking calls — run concurrently in separate threads.
        # asyncio.gather() starts both threads simultaneously; total wall time =
        # max(summarize_time, extract_time) instead of their sum.
        summary_out, extracted_features = await asyncio.gather(
            asyncio.to_thread(_summarizer.summarize, string_chunks, normalized_domain, tuning_params),
            asyncio.to_thread(_extractor.extract_features, raw_text, normalized_domain, tuning_params),
        )

        _summarizer.clear_result_cache()

        return {
            "document_id":      doc_id,
            "domain":           normalized_domain,
            "filename":         file.filename,
            "page_count":       len(pages_data),
            "chunk_count":      len(chunks),
            "short_summary":    summary_out["short_summary"],
            "detailed_summary": summary_out["detailed_summary"],
            "extracted_data":   extracted_features,
            "processing_meta":  summary_out.get("meta", {}),
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[API] Pipeline failure on {doc_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Processing failed: {e}")


# ── /query ────────────────────────────────────────────────────────────────────

@router.post("/query")
async def query_document(request: QueryRequest):
    """
    RAG Q&A against the document's ChromaDB vector collection.

    vector similarity search + QA inference are both CPU-bound —
    both run in threads so the event loop stays free.
    """
    top_k = DomainConfig.get_domain_property(request.domain, "top_k_retrieval", 3)

    try:
        # Similarity search — CPU-bound (embedding encode + HNSW lookup)
        context = await asyncio.to_thread(
            _vdb_manager.query_similarity,
            request.question,
            request.document_id,
            request.domain,
            top_k,
        )

        if not context:
            return {
                "answer":     "No relevant content found for this question in the document.",
                "confidence": 0.0,
                "citations":  [],
            }

        # QA inference — CPU-bound (roberta-base-squad2 forward pass)
        answer = await asyncio.to_thread(
            _rag_engine.generate_answer,
            request.question,
            context,
            request.domain,
        )

        # Surface expanded query to frontend when query expansion was applied
        expanded = _rag_engine._expand_query(request.question, request.domain)
        if expanded and expanded.lower() != request.question.lower():
            answer["expanded_query"] = expanded

        return answer

    except Exception as e:
        logger.error(f"[API] Query failed for doc '{request.document_id}': {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Query pipeline failed: {e}")


# ── /document DELETE ──────────────────────────────────────────────────────────

class DeleteRequest(BaseModel):
    document_id: str = Field(..., description="Document ID to delete.")
    domain:      str = Field(..., description="Domain the document belongs to.")

    @field_validator("domain")
    @classmethod
    def domain_must_be_valid(cls, v: str) -> str:
        v = v.lower().strip()
        if v not in DomainConfig.DOMAINS:
            raise ValueError(f"Unknown domain '{v}'.")
        return v


@router.delete("/document")
async def delete_document(request: DeleteRequest):
    """
    FIX 6: Deletes all ChromaDB vectors for a document.
    Called by Streamlit reset_app() when the user clicks New Document.
    Keeps storage lean across multiple sessions.
    """
    try:
        await asyncio.to_thread(
            _vdb_manager.delete_document, request.document_id, request.domain
        )
        return {"status": "deleted", "document_id": request.document_id}
    except Exception as e:
        logger.error(f"[API] Delete failed for doc '{request.document_id}': {e}")
        raise HTTPException(status_code=500, detail=f"Delete failed: {e}")