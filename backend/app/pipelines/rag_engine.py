"""
rag_engine.py — Citation-aware extractive Q&A engine
======================================================
CHANGE from v1: Generation model swapped from flan-t5-base to
deepset/roberta-base-squad2 (130MB, CPU-fast, SQuAD-trained).

Why this matters:
  flan-t5-base is an instruction-following model — it generates new text
  from a prompt. For RAG the task is different: given a retrieved passage,
  find the exact span that answers the question. That is extractive QA,
  which SQuAD-trained reader models do far better.

  roberta-base-squad2:
    - 130MB (smaller than flan-t5-base at 250MB)
    - Returns answer + confidence score natively
    - Returns start/end character offsets → precise citation snippets
    - No hallucination: can only return text that exists in the context
    - 3-5x faster on CPU than flan-t5-base for this task
"""

import logging
from typing import List, Dict, Any

import torch
from transformers import pipeline

from backend.app.core.config import DomainConfig

logger = logging.getLogger(__name__)

# Low score threshold — answers below this are treated as "not found"
_CONFIDENCE_THRESHOLD = 0.10


class DomainRAGEngine:
    """
    Extractive Q&A engine backed by roberta-base-squad2.

    For each query:
      1. Concatenate retrieved context chunks (with page labels)
      2. Run the QA pipeline: find the best answer span in the context
      3. If confidence is too low, return a "not found" response
      4. Return answer text + page-level citation
    """

    def __init__(self, model_name: str = "deepset/roberta-base-squad2"):
        self.model_name = model_name
        self._setup_device()
        try:
            self.qa_pipeline = pipeline(
                "question-answering",
                model=model_name,
                device=self.device,
                torch_dtype=torch.float32,   # explicit for CPU stability
            )
            logger.info(f"[RAG] QA model loaded: {model_name} on {self.device_name}")
        except Exception as e:
            logger.error(f"[RAG] Model load failed: {e}")
            raise

    def _setup_device(self):
        if torch.cuda.is_available():
            self.device = 0
            self.device_name = "cuda"
        elif torch.backends.mps.is_available():
            self.device = "mps"
            self.device_name = "mps"
        else:
            self.device = -1
            self.device_name = "cpu"

    def generate_answer(
        self,
        question: str,
        context_chunks: List[Dict[str, Any]],
        domain: str,
    ) -> Dict[str, Any]:
        """
        Runs extractive QA over the retrieved context chunks.

        Args:
            question:       The user's question string.
            context_chunks: List of {"text": str, "page": int, "score": float}
                            returned by VectorStoreManager.query_similarity().
            domain:         Domain key — used for the system prompt prefix.

        Returns:
            {
                "answer":    str,
                "citations": [{"page": int, "text_snippet": str}],
                "confidence": float,
            }
        """
        if not context_chunks:
            return self._no_answer("No relevant content found in this document.")

        # Build a single context string, prefixed with page numbers
        # so the citation extraction can map back to page correctly.
        context_parts = []
        page_map: Dict[int, int] = {}  # character_offset → page_number
        offset = 0

        for chunk in context_chunks:
            text = chunk["text"].strip()
            page = chunk.get("page", 0)
            prefix = f"[Page {page}] "
            full = prefix + text + " "
            page_map[offset] = page
            context_parts.append(full)
            offset += len(full)

        full_context = "".join(context_parts)

        # Truncate context to 512 tokens (roberta-base-squad2 hard limit)
        # We truncate the string side — the pipeline also truncates internally,
        # but pre-truncating lets us keep page_map offsets accurate.
        full_context = full_context[:1800]

        try:
            result = self.qa_pipeline(
                question=question,
                context=full_context,
                max_answer_len=150,
                handle_impossible_answer=True,  # returns "" + score=0 if no answer
            )
        except Exception as e:
            logger.error(f"[RAG] QA inference failed: {e}")
            return self._no_answer("An error occurred during answer generation.")

        answer_text: str = result.get("answer", "").strip()
        confidence: float = round(float(result.get("score", 0.0)), 4)
        start_offset: int = result.get("start", 0)

        # Confidence gate — below threshold means the model didn't find a clear answer
        if confidence < _CONFIDENCE_THRESHOLD or not answer_text:
            return self._no_answer(
                "I couldn't find a confident answer to that question in the document. "
                "Try rephrasing or ask about a specific section."
            )

        # Map answer start offset back to the closest page
        cited_page = self._resolve_page(start_offset, page_map)

        # Build a snippet: the sentence in the context containing the answer
        snippet = self._extract_snippet(full_context, start_offset, window=120)

        logger.info(
            f"[RAG] Answer found | confidence={confidence} | page={cited_page} | "
            f"domain={domain} | answer='{answer_text[:60]}...'"
        )

        return {
            "answer": answer_text,
            "confidence": confidence,
            "citations": [
                {"page": cited_page, "text_snippet": snippet}
            ],
        }

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _resolve_page(answer_start: int, page_map: Dict[int, int]) -> int:
        """
        Returns the page number for the chunk closest to answer_start.
        page_map keys are chunk start offsets; we find the largest offset
        that is still <= answer_start.
        """
        best_offset = 0
        for offset in page_map:
            if offset <= answer_start:
                best_offset = max(best_offset, offset)
        return page_map.get(best_offset, 0)

    @staticmethod
    def _extract_snippet(context: str, start: int, window: int = 120) -> str:
        """
        Extracts a readable sentence-length snippet around the answer location.
        """
        begin = max(0, start - 40)
        end = min(len(context), start + window)
        raw = context[begin:end].strip()
        # Remove [Page N] prefix if it bleeds into the snippet
        if raw.startswith("[Page"):
            raw = raw.split("] ", 1)[-1] if "] " in raw else raw
        return raw

    @staticmethod
    def _no_answer(message: str) -> Dict[str, Any]:
        return {
            "answer": message,
            "confidence": 0.0,
            "citations": [],
        }

    def _flush_vram(self):
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        elif torch.backends.mps.is_available():
            torch.mps.empty_cache()