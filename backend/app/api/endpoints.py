import uuid
import json
import logging
from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from pydantic import BaseModel, Field

# Core module and configuration dependencies
from backend.app.core.config import DomainConfig
from backend.app.services.parser import DocumentParser
from backend.app.services.vector_db import VectorStoreManager
from backend.app.pipelines.summarizer import DomainSummarizer
from backend.app.pipelines.extractor import DomainExtractor
from backend.app.pipelines.rag_engine import DomainRAGEngine

# Setup pipeline tracking logger
logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1")

# Initialize long-lived application dependency workers
vdb_manager = VectorStoreManager()
summarizer = DomainSummarizer()
extractor = DomainExtractor()
rag_engine = DomainRAGEngine()


class QueryRequest(BaseModel):
    document_id: str = Field(..., description="Unique hash string tracking target document.")
    domain: str = Field(..., description="Target execution domain profile name.")
    question: str = Field(..., description="The user's contextual question.")


@router.post("/process")
async def process_document(
    file: UploadFile = File(...), 
    domain: str = Form(...),
    params: str = Form("{}")  # <-- NEW: Captures the 5-parameter JSON string from the UI
):
    """
    Unified context extraction pipeline: dynamically chunks, indexes, summarizes, 
    and extracts metadata features based on custom domain matrices and UI tuning.
    """
    normalized_domain = domain.lower().strip()
    if normalized_domain not in DomainConfig.DOMAINS:
        raise HTTPException(
            status_code=400, 
            detail=f"Requested execution domain '{domain}' is unsupported by the underlying model matrix."
        )
        
    # Safely parse the UI tuning parameters
    try:
        tuning_params = json.loads(params)
    except json.JSONDecodeError:
        logger.warning("Failed to decode UI parameters. Defaulting to standard configurations.")
        tuning_params = {}

    doc_id = f"doc_{uuid.uuid4().hex[:8]}"
    logger.info(f"Initiating dynamic context routing pipeline for file: [{file.filename}] -> Assigned ID: {doc_id}")
    
    try:
        # 1. Fetch domain-specific tuning overrides from our configuration matrix
        chunk_size = DomainConfig.get_domain_property(normalized_domain, "chunk_size", 1000)
        chunk_overlap = DomainConfig.get_domain_property(normalized_domain, "chunk_overlap", 150)
        
        # Instantiate a dynamic parser instance tuned specifically for this domain's density layout
        dynamic_parser = DocumentParser(chunk_size=chunk_size, overlap=chunk_overlap)
        
        # 2. Stage: Text and Metadata Structural Extraction
        pages_data = await dynamic_parser.parse_pdf(file)
        if not pages_data:
            raise HTTPException(status_code=422, detail="PDF contains no extractable or readable text layers.")
            
        chunks = dynamic_parser.create_chunks(pages_data)
        
        # 3. Stage: Idempotent Vector DB Index Ingestion
        vdb_manager.store_document(doc_id, normalized_domain, chunks)
        
        # Prepare content string sequences for downstream transformer blocks
        raw_text_stream = " ".join([p["text"] for p in pages_data])
        string_chunks = [c["text"] for c in chunks]
        
        # 4. Stage: Unified Downstream Feature & Intelligence Mining
        # Passes the tuning parameters directly to the ML models
        summary_out = summarizer.summarize(string_chunks, normalized_domain, tuning_params)
        extracted_features = extractor.extract_features(raw_text_stream, normalized_domain, tuning_params)
        
        return {
            "document_id": doc_id,
            "domain": normalized_domain,
            "filename": file.filename,
            "short_summary": summary_out["short_summary"],
            "detailed_summary": summary_out["detailed_summary"],
            "extracted_data": extracted_features
        }
        
    except HTTPException as http_exc:
        raise http_exc
    except Exception as e:
        logger.error(f"Fatal processing pipeline interruption on doc {doc_id}: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Pipeline processing failed during automated context ingestion: {str(e)}"
        )


@router.post("/query")
async def query_document(request: QueryRequest):
    """
    Processes open-ended natural language queries against localized vector fields 
    using isolated metadata scope maps.
    """
    normalized_domain = request.domain.lower().strip()
    if normalized_domain not in DomainConfig.DOMAINS:
        raise HTTPException(status_code=400, detail="Invalid execution domain scope requested.")
        
    # Dynamically look up how many context nodes this domain wants to fetch
    top_k = DomainConfig.get_domain_property(normalized_domain, "top_k_retrieval", 3)
    
    try:
        # Retrieve context blocks bounded by the domain's unique top_k configuration parameter
        context = vdb_manager.query_similarity(
            query=request.question,
            document_id=request.document_id,
            domain=normalized_domain,
            n_results=top_k
        )
        
        # Run contextual generative question answering
        answer_out = rag_engine.generate_answer(request.question, context, normalized_domain)
        return answer_out
        
    except Exception as e:
        logger.error(f"RAG query generation failure on index key {request.document_id}: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Context collection fault or downstream generation pipeline exception: {str(e)}"
        )