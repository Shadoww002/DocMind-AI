import os
import logging
import torch
from typing import List, Dict, Any, Optional

# Telemetry must be off before chromadb is imported
os.environ["ANONYMIZED_TELEMETRY"] = "False"

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from backend.app.core.config import DomainConfig

logger = logging.getLogger(__name__)

# IMPROVEMENT: one collection per domain instead of one shared collection.
# In v1 all three domains wrote to "domain_intelligence" — a medical chunk
# and a legal chunk with the same text could match each other's queries.
# Per-domain collections make the $and filter unnecessary and improve
# retrieval precision (especially important on CPU with small top_k).
_COLLECTION_NAMES: Dict[str, str] = {
    "medical": "docmind_medical",
    "legal":   "docmind_legal",
    "resume":  "docmind_resume",
}

# IMPROVEMENT: batch size tuned for CPU RAM — 32 is fine for all-MiniLM-L6-v2
# (384-dim embeddings, ~90MB model). Increase to 64 if you have >= 8GB RAM.
_EMBED_BATCH_SIZE = 32


class VectorStoreManager:
    """
    ChromaDB-backed vector store with per-domain isolated collections.

    Improvements over v1:
    - Per-domain collections eliminate cross-domain retrieval contamination
    - Embedding model loaded once; device selection logic extracted to helper
    - store_document validates chunk content before encoding (skips empty strings)
    - query_similarity uses simple doc_id filter (no $and needed with isolated collections)
    - _get_collection() is a safe accessor — raises clearly if domain is unknown
    - distance scores included in query output for RAG engine ranking
    - Explicit encode() call uses normalize_embeddings=True for cosine similarity accuracy
    """

    def __init__(self):
        os.makedirs(DomainConfig.VECTOR_DB_DIR, exist_ok=True)

        self.chroma_client = chromadb.PersistentClient(
            path=DomainConfig.VECTOR_DB_DIR,
            settings=Settings(anonymized_telemetry=False),
        )

        # IMPROVEMENT: pre-create all three collections at startup so
        # the first store_document call doesn't pay the creation overhead.
        self._collections: Dict[str, Any] = {}
        for domain, cname in _COLLECTION_NAMES.items():
            self._collections[domain] = self.chroma_client.get_or_create_collection(
                name=cname,
                # IMPROVEMENT: store cosine distance metadata so ChromaDB uses
                # cosine similarity for ranking (correct for sentence-transformers).
                metadata={"hnsw:space": "cosine"},
            )
        logger.info(f"[VectorStore] Collections ready: {list(_COLLECTION_NAMES.values())}")

        self.device = self._pick_device()
        # all-MiniLM-L6-v2 is ~90MB — fast on CPU, good quality for retrieval
        self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2", device=self.device)
        logger.info(f"[VectorStore] Embedding model loaded on {self.device.upper()}")

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _pick_device() -> str:
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    def _get_collection(self, domain: str):
        """
        Returns the ChromaDB collection for a domain.
        Raises ValueError for unknown domains instead of returning None silently.
        """
        col = self._collections.get(domain.lower())
        if col is None:
            raise ValueError(
                f"[VectorStore] Unknown domain '{domain}'. "
                f"Valid: {list(_COLLECTION_NAMES.keys())}"
            )
        return col

    def _encode(self, texts: List[str]) -> List[List[float]]:
        """
        Encodes a list of strings to embeddings.
        normalize_embeddings=True is required for cosine similarity to be correct.
        """
        return self.embedding_model.encode(
            texts,
            batch_size=_EMBED_BATCH_SIZE,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,   # IMPROVEMENT: v1 omitted this
        ).tolist()

    # ── Store ─────────────────────────────────────────────────────────────────

    def store_document(
        self,
        document_id: str,
        domain: str,
        chunks: List[Dict[str, Any]],
    ) -> None:
        """
        Upserts all chunks for a document into the domain's isolated collection.
        """
        if not chunks:
            logger.warning(f"[VectorStore] No chunks provided for doc_id='{document_id}'.")
            return

        collection = self._get_collection(domain)

        # IMPROVEMENT: filter out empty or whitespace-only chunk texts before
        # encoding — SentenceTransformer silently produces zero-vectors for them
        # which pollute similarity results.
        valid: List[Dict[str, Any]] = [
            c for c in chunks if c.get("text", "").strip()
        ]
        skipped = len(chunks) - len(valid)
        if skipped:
            logger.warning(f"[VectorStore] Skipped {skipped} empty chunks for doc_id='{document_id}'.")

        if not valid:
            logger.error(f"[VectorStore] All chunks were empty for doc_id='{document_id}'. Aborting store.")
            return

        ids       = [f"{document_id}_ch_{i}" for i in range(len(valid))]
        documents = [c["text"] for c in valid]

        # IMPROVEMENT: metadata values must all be scalars for ChromaDB.
        # Cast page to int explicitly — pypdf sometimes returns strings.
        metadatas = []
        for c in valid:
            raw_meta = c.get("metadata", {})
            page_val = raw_meta.get("page", 0)
            metadatas.append({
                "doc_id": str(document_id),
                "domain": str(domain),
                "page":   int(page_val) if str(page_val).isdigit() else 0,
            })

        embeddings = self._encode(documents)

        try:
            collection.upsert(
                ids=ids,
                embeddings=embeddings,
                documents=documents,
                metadatas=metadatas,
            )
            logger.info(
                f"[VectorStore] Stored {len(valid)} chunks | "
                f"domain={domain} | doc_id={document_id}"
            )
        except Exception as e:
            logger.error(f"[VectorStore] Upsert failed for doc_id='{document_id}': {e}")
            raise

    # ── Query ─────────────────────────────────────────────────────────────────

    def query_similarity(
        self,
        query: str,
        document_id: str,
        domain: str,
        n_results: int = 3,
    ) -> List[Dict[str, Any]]:
        """
        Returns the top-n most similar chunks for a query, scoped to one document.

        IMPROVEMENT: uses a simple single-field doc_id filter instead of $and —
        the domain is already isolated by the per-domain collection, so the $and
        was redundant and slowed down small collections on CPU.
        IMPROVEMENT: includes the cosine distance score in output so the RAG
        engine can skip low-confidence chunks (score > 0.8 means weak match).
        """
        query = query.strip()
        if not query:
            return []

        collection = self._get_collection(domain)

        # Clamp n_results to the actual number of stored vectors to avoid
        # ChromaDB raising "n_results > collection size" on small documents
        try:
            collection_size = collection.count()
        except Exception:
            collection_size = n_results

        safe_n = min(n_results, max(1, collection_size))

        query_vector = self._encode([query])

        try:
            results = collection.query(
                query_embeddings=query_vector,
                n_results=safe_n,
                # IMPROVEMENT: simple single-field filter — domain already
                # isolated by the collection; only scope by document ID.
                where={"doc_id": {"$eq": str(document_id)}},
                include=["documents", "metadatas", "distances"],
            )
        except Exception as e:
            logger.error(f"[VectorStore] Query failed for doc_id='{document_id}': {e}")
            return []

        output: List[Dict[str, Any]] = []

        docs      = (results.get("documents")  or [[]])[0]
        metas     = (results.get("metadatas")  or [[]])[0]
        distances = (results.get("distances")  or [[]])[0]

        for doc, meta, dist in zip(docs, metas, distances):
            if not doc:
                continue
            output.append({
                "text":     doc,
                "page":     meta.get("page", 0),
                "score":    round(float(dist), 4),  # cosine distance; lower = more similar
            })

        logger.debug(
            f"[VectorStore] Query returned {len(output)} chunks | "
            f"doc_id={document_id} | domain={domain}"
        )
        return output

    # ── Maintenance ───────────────────────────────────────────────────────────

    def delete_document(self, document_id: str, domain: str) -> None:
        """
        IMPROVEMENT: new method — deletes all chunks for a document from its
        collection. Call this from endpoints.py if you add a DELETE /document
        route, or from Streamlit's 'New Document' button to keep storage lean.
        """
        collection = self._get_collection(domain)
        try:
            collection.delete(where={"doc_id": {"$eq": str(document_id)}})
            logger.info(f"[VectorStore] Deleted chunks for doc_id='{document_id}' from '{domain}' collection.")
        except Exception as e:
            logger.warning(f"[VectorStore] Delete failed for doc_id='{document_id}': {e}")

    def collection_stats(self) -> Dict[str, int]:
        """
        IMPROVEMENT: new method — returns chunk counts per domain collection.
        Useful for the /health endpoint and Streamlit sidebar diagnostics.
        """
        stats = {}
        for domain, col in self._collections.items():
            try:
                stats[domain] = col.count()
            except Exception:
                stats[domain] = -1
        return stats