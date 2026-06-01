import uuid
import json
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

# ── Singleton pipeline workers ───────────────────────────────────────────────
# IMPROVEMENT: instantiated once at module load; avoids reloading model weights
# on every request (critical on CPU where model load takes 3-10 seconds).
_vdb_manager = VectorStoreManager()
_summarizer  = DomainSummarizer()
_extractor   = DomainExtractor()
_rag_engine  = DomainRAGEngine()


# ── Request models ────────────────────────────────────────────────────────────

class QueryRequest(BaseModel):
    document_id: str = Field(..., description="Document ID returned by /process.")
    domain:      str = Field(..., description="Domain used during processing.")
    question:    str = Field(..., min_length=3, description="User question.")

    # IMPROVEMENT: validate domain on the model layer, not just in the route
    @field_validator("domain")
    @classmethod
    def domain_must_be_valid(cls, v: str) -> str:
        v = v.lower().strip()
        if v not in DomainConfig.DOMAINS:
            raise ValueError(f"Unknown domain '{v}'. Valid: {list(DomainConfig.DOMAINS)}")
        return v

    @field_validator("question")
    @classmethod
    def question_not_empty(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("Question cannot be blank.")
        return v


# ── /process ─────────────────────────────────────────────────────────────────

@router.post("/process")
async def process_document(
    file:   UploadFile = File(...),
    domain: str        = Form(...),
    params: str        = Form("{}"),
):
    """
    Full ingestion pipeline:
      parse PDF → chunk → store in ChromaDB → summarize → extract entities
    Returns summary, detailed breakdown, and extracted domain fields.
    """
    normalized_domain = domain.lower().strip()
    if normalized_domain not in DomainConfig.DOMAINS:
        raise HTTPException(
            status_code=400,
            detail=f"Domain '{domain}' is not supported. Choose from: {list(DomainConfig.DOMAINS.keys())}",
        )

    # IMPROVEMENT: params parse with clear user-facing error
    try:
        tuning_params = json.loads(params)
        if not isinstance(tuning_params, dict):
            raise ValueError("params must be a JSON object.")
    except (json.JSONDecodeError, ValueError) as exc:
        raise HTTPException(status_code=422, detail=f"Invalid params JSON: {exc}")

    doc_id = f"doc_{uuid.uuid4().hex[:10]}"
    logger.info(f"[API] Processing '{file.filename}' | domain={normalized_domain} | doc_id={doc_id}")

    try:
        # 1. Domain-tuned parser
        chunk_size    = DomainConfig.get_domain_property(normalized_domain, "chunk_size", 1000)
        chunk_overlap = DomainConfig.get_domain_property(normalized_domain, "chunk_overlap", 150)
        parser        = DocumentParser(chunk_size=chunk_size, overlap=chunk_overlap)

        # 2. Parse
        pages_data = await parser.parse_pdf(file)
        if not pages_data:
            raise HTTPException(status_code=422, detail="PDF has no extractable text.")

        chunks = parser.create_chunks(pages_data)

        # IMPROVEMENT: warn if very few chunks — likely a scanned/image PDF
        if len(chunks) < 2:
            logger.warning(f"[API] Only {len(chunks)} chunk(s) from '{file.filename}'. "
                           "Document may be image-based or very short.")

        # 3. Vector store ingestion
        _vdb_manager.store_document(doc_id, normalized_domain, chunks)

        # 4. Downstream pipelines
        raw_text      = " ".join(p["text"] for p in pages_data)
        string_chunks = [c["text"] for c in chunks]

        summary_out        = _summarizer.summarize(string_chunks, normalized_domain, tuning_params)
        extracted_features = _extractor.extract_features(raw_text, normalized_domain, tuning_params)

        # IMPROVEMENT: clear summarizer inference cache between documents to keep RAM stable
        _summarizer.clear_result_cache()

        return {
            "document_id":      doc_id,
            "domain":           normalized_domain,
            "filename":         file.filename,
            "page_count":       len(pages_data),          # new — useful for UI display
            "chunk_count":      len(chunks),               # new — helps debug short docs
            "short_summary":    summary_out["short_summary"],
            "detailed_summary": summary_out["detailed_summary"],
            "extracted_data":   extracted_features,
            "processing_meta":  summary_out.get("meta", {}),  # model, device, elapsed_seconds
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
    Returns answer text plus page-level citations.
    """
    top_k = DomainConfig.get_domain_property(request.domain, "top_k_retrieval", 3)

    try:
        context = _vdb_manager.query_similarity(
            query=request.question,
            document_id=request.document_id,
            domain=request.domain,
            n_results=top_k,
        )

        # IMPROVEMENT: if no context returned, give a clear answer rather than
        # letting the RAG engine hallucinate with empty context.
        if not context:
            return {
                "answer": "No relevant content found for this question in the document.",
                "citations": [],
            }

        return _rag_engine.generate_answer(request.question, context, request.domain)

    except Exception as e:
        logger.error(f"[API] Query failed for doc '{request.document_id}': {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Query pipeline failed: {e}")