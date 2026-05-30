import os
# Force telemetry off at the earliest possible import step
os.environ["ANONYMIZED_TELEMETRY"] = "False"

import logging
import torch
from typing import List, Dict, Any
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from backend.app.core.config import DomainConfig

logger = logging.getLogger(__name__)

class VectorStoreManager:
    def __init__(self):
        """Initializes ChromaDB persistent client and maps the embedding engine to local hardware accelerators."""
        os.makedirs(DomainConfig.VECTOR_DB_DIR, exist_ok=True)
        
        # Initialize standard persistent storage instance
        self.chroma_client = chromadb.PersistentClient(
            path=DomainConfig.VECTOR_DB_DIR,
            settings=Settings(anonymized_telemetry=False)
        )
        
        # Get or create collection
        self.collection = self.chroma_client.get_or_create_collection(name="domain_intelligence")
        
        # Hardware-accelerate the embedding model generation tier
        self.device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
        self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2", device=self.device)
        logger.info(f"Embedding framework mapped successfully to hardware target: [{self.device.upper()}]")

    def store_document(self, document_id: str, domain: str, chunks: List[Dict[str, Any]]) -> None:
        """
        Ingests document chunks into persistent storage using an idempotent upsert sequence.
        Batches execution to handle high-density files efficiently.
        """
        if not chunks:
            logger.warning(f"Aborted ingestion sequence: No text chunks provided for document ID: {document_id}")
            return

        ids = [f"{document_id}_ch_{idx}" for idx in range(len(chunks))]
        documents = [c["text"] for c in chunks]
        
        # Hardware-accelerated matrix batch encoding
        embeddings = self.embedding_model.encode(
            documents, 
            batch_size=32, 
            show_progress_bar=False,
            convert_to_numpy=True
        ).tolist()
        
        metadatas = []
        for c in chunks:
            meta = c["metadata"].copy()
            # Ensure critical structural metadata items are cast as simple supported scalars
            meta.update({
                "doc_id": str(document_id), 
                "domain": str(domain)
            })
            metadatas.append(meta)

        try:
            # .upsert replaces existing matching records rather than crashing with an ID collision error
            self.collection.upsert(
                ids=ids,
                embeddings=embeddings,
                documents=documents,
                metadatas=metadatas
            )
            logger.info(f"Successfully processed and indexed {len(chunks)} chunks for Document ID: {document_id}")
        except Exception as e:
            logger.error(f"Vector Database Insertion Failure on document tracking key {document_id}: {str(e)}")
            raise e

    def query_similarity(self, query: str, document_id: str, domain: str, n_results: int = 3) -> List[Dict[str, Any]]:
        """
        Queries the localized vector space using metadata containment filters.
        Safely isolated against empty matching arrays.
        """
        if not query.strip():
            return []

        # Encode search string using identical local hardware target mapping
        query_vector = self.embedding_model.encode([query], convert_to_numpy=True).tolist()
        
        try:
            results = self.collection.query(
                query_embeddings=query_vector,
                n_results=n_results,
                # Safe implicit logical AND operator mapping supported natively by ChromaDB core engines
                where={
                    "$and": [
                        {"doc_id": {"$eq": str(document_id)}},
                        {"domain": {"$eq": str(domain)}}
                    ]
                }
            )
        except Exception as e:
            logger.error(f"Vector spatial indexing lookup exception encountered: {str(e)}")
            return []
        
        output = []
        
        # Hard defensive parsing validation to safeguard against internal extraction index crashes
        if results and "documents" in results and results["documents"] and len(results["documents"][0]) > 0:
            documents_list = results["documents"][0]
            metadatas_list = results["metadatas"][0] if results.get("metadatas") else [{}] * len(documents_list)
            
            for doc, meta in zip(documents_list, metadatas_list):
                if doc and meta:
                    output.append({
                        "text": doc, 
                        "page": meta.get("page", "Unknown")
                    })
                    
        return output