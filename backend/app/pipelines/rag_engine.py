"""
rag_engine.py — Citation-aware extractive Q&A engine
deepset/roberta-base-squad2 — 130MB, CPU-fast, SQuAD-trained

Key improvement in this version:
- Query expansion: vague/short queries are rewritten into proper questions
  before being sent to the QA model. This dramatically improves recall.
- Domain-aware expansion: medical "history" → "What is the patient's medical history?"
- Gibberish/irrelevant queries return a helpful message with example questions
- Confidence threshold raised slightly to reduce low-quality answers
"""

import re
import logging
from typing import List, Dict, Any, Optional

import torch
from transformers import pipeline

from backend.app.core.config import DomainConfig

logger = logging.getLogger(__name__)

_CONFIDENCE_THRESHOLD = 0.15  # raised from 0.10 — filters weak answers better

# ── Query expansion maps ──────────────────────────────────────────────────────
# Maps short/vague keywords to proper questions per domain.
# roberta-base-squad2 works much better with complete question sentences.

_QUERY_EXPANSION = {
    "medical": {
        "patient":          "Who is the patient and what are their details?",
        "doctor":           "Who is the attending physician or doctor?",
        "diagnosis":        "What is the patient's diagnosis or medical condition?",
        "diagnoses":        "What are the patient's diagnoses and conditions?",
        "disease":          "What disease or condition does the patient have?",
        "diseases":         "What diseases or conditions does the patient have?",
        "medication":       "What medications are prescribed to the patient?",
        "medications":      "What medications and dosages are prescribed?",
        "medicine":         "What medicines are prescribed with dosages?",
        "treatment":        "What is the treatment plan for the patient?",
        "history":          "What is the patient's past medical history?",
        "lab":              "What are the patient's lab test results?",
        "labs":             "What are the laboratory values and test results?",
        "vitals":           "What are the patient's vital signs?",
        "procedure":        "What medical procedures were performed?",
        "surgery":          "What surgical procedure was performed?",
        "date":             "What are the important dates mentioned in the document?",
        "dates":            "What dates are mentioned in the medical record?",
        "year":             "What year or dates are mentioned in this document?",
        "age":              "How old is the patient?",
        "name":             "What is the patient's name?",
        "complaint":        "What is the chief complaint of the patient?",
        "summary":          "What is the overall clinical summary of this case?",
        "medical history":  "What is the complete past medical history of the patient?",
        "medical details":  "What are the key medical details in this document?",
    },
    "legal": {
        "parties":      "Who are the parties involved in this agreement?",
        "party":        "Who are the parties to this legal document?",
        "date":         "What are the execution or signing dates?",
        "dates":        "What important dates are mentioned in this document?",
        "year":         "What year was this document executed?",
        "amount":       "What are the monetary amounts mentioned?",
        "money":        "What is the consideration or monetary amount?",
        "rent":         "What is the monthly rent amount?",
        "penalty":      "What is the penalty clause in this agreement?",
        "duration":     "What is the duration or term of this agreement?",
        "obligations":  "What are the key obligations of each party?",
        "clauses":      "What are the important clauses in this agreement?",
        "termination":  "What are the termination conditions?",
        "jurisdiction": "What is the jurisdiction for dispute resolution?",
        "law":          "Which Indian laws or acts apply to this document?",
        "risk":         "What are the risks or red flags in this agreement?",
        "summary":      "What is a summary of this legal agreement?",
        "signees":      "Who are the signatories to this document?",
        "deposit":      "What is the security deposit amount?",
        "notice":       "What is the notice period mentioned?",
    },
    "resume": {
        "name":           "What is the candidate's full name?",
        "candidate":      "Who is the candidate and what is their background?",
        "skills":         "What are the candidate's technical skills?",
        "education":      "What is the candidate's educational background?",
        "degree":         "What degree does the candidate hold?",
        "university":     "Which university did the candidate attend?",
        "experience":     "What work experience does the candidate have?",
        "company":        "Which companies has the candidate worked at?",
        "job":            "What job titles or roles has the candidate held?",
        "achievement":    "What are the candidate's notable achievements?",
        "certification":  "What certifications does the candidate hold?",
        "cgpa":           "What is the candidate's CGPA or academic score?",
        "project":        "What projects has the candidate worked on?",
        "summary":        "What is a professional summary of this candidate?",
        "intern":         "What internship experience does the candidate have?",
        "location":       "Where is the candidate located?",
        "email":          "What is the candidate's email or contact information?",
        "year":           "What years are mentioned in the candidate's timeline?",
        "dates":          "What dates are mentioned in the resume?",
    },
}

# Generic expansions for all domains
_GENERIC_EXPANSION = {
    "summary":  "What is the overall summary of this document?",
    "date":     "What dates are mentioned in this document?",
    "dates":    "What are the important dates in this document?",
    "year":     "What year is mentioned in this document?",
    "name":     "What names are mentioned in this document?",
    "details":  "What are the key details in this document?",
    "info":     "What is the main information in this document?",
    "content":  "What is the content of this document?",
}

# Minimum word count to be a valid question (below this → try expansion)
_MIN_QUESTION_WORDS = 4


class DomainRAGEngine:
    """
    Extractive Q&A engine backed by deepset/roberta-base-squad2.
    Includes query expansion for vague/short queries.
    """

    def __init__(self, model_name: str = "deepset/roberta-base-squad2"):
        self.model_name = model_name
        self._setup_device()
        try:
            self.qa_pipeline = pipeline(
                "question-answering",
                model=model_name,
                device=self.device,
                torch_dtype=torch.float32,
            )
            logger.info(f"[RAG] Model loaded: {model_name} on {self.device_name}")
        except Exception as e:
            logger.error(f"[RAG] Model load failed: {e}")
            raise

    def _setup_device(self):
        if torch.cuda.is_available():
            self.device, self.device_name = 0, "cuda"
        elif torch.backends.mps.is_available():
            self.device, self.device_name = "mps", "mps"
        else:
            self.device, self.device_name = -1, "cpu"

    # ── Query expansion ───────────────────────────────────────────────────────

    def _expand_query(self, question: str, domain: str) -> Optional[str]:
        """
        Expands short/vague queries into proper questions.
        Returns expanded question or None if no expansion found.
        """
        q = question.strip().lower().rstrip("?")

        # Already a proper question — no expansion needed
        if len(q.split()) >= _MIN_QUESTION_WORDS:
            return None

        # Try domain-specific expansion first
        domain_map = _QUERY_EXPANSION.get(domain, {})
        if q in domain_map:
            return domain_map[q]

        # Try generic expansion
        if q in _GENERIC_EXPANSION:
            return _GENERIC_EXPANSION[q]

        # Try partial match — "medical hist" → "medical history"
        for key, expanded in domain_map.items():
            if q in key or key in q:
                return expanded

        return None  # no expansion found

    def _is_gibberish(self, question: str) -> bool:
        """
        Returns True if the question appears to be gibberish or nonsensical.
        Checks: very short with no vowels, all consonants, random chars.
        """
        q = question.strip().lower()
        if len(q) < 3:
            return True
        # Check vowel ratio — real words have vowels
        vowels = sum(1 for c in q if c in "aeiou")
        if len(q) > 3 and vowels / len(q) < 0.1:
            return True
        # Check if it's all non-alphabetic
        if not any(c.isalpha() for c in q):
            return True
        return False

    # ── Main entry point ──────────────────────────────────────────────────────

    def generate_answer(
        self,
        question: str,
        context_chunks: List[Dict[str, Any]],
        domain: str,
    ) -> Dict[str, Any]:
        """
        Runs extractive QA over retrieved context chunks.
        Expands vague queries before sending to the model.
        """
        if not context_chunks:
            return self._no_answer("No relevant content found in this document.")

        # Gibberish check
        if self._is_gibberish(question):
            return self._no_answer(
                "That doesn't look like a valid question. "
                + self._example_questions(domain)
            )

        # Query expansion for short/vague queries
        expanded = self._expand_query(question, domain)
        effective_question = expanded or question

        if expanded:
            logger.info(f"[RAG] Query expanded: '{question}' → '{expanded}'")

        # Build context string with page labels
        context_parts = []
        page_map: Dict[int, int] = {}
        offset = 0

        for chunk in context_chunks:
            text   = chunk["text"].strip()
            page   = chunk.get("page", 0)
            prefix = f"[Page {page}] "
            full   = prefix + text + " "
            page_map[offset] = page
            context_parts.append(full)
            offset += len(full)

        full_context = "".join(context_parts)[:1800]

        try:
            result = self.qa_pipeline(
                question=effective_question,
                context=full_context,
                max_answer_len=150,
                handle_impossible_answer=True,
            )
        except Exception as e:
            logger.error(f"[RAG] QA inference failed: {e}")
            return self._no_answer("An error occurred during answer generation.")

        answer_text: str   = result.get("answer", "").strip()
        confidence: float  = round(float(result.get("score", 0.0)), 4)
        start_offset: int  = result.get("start", 0)

        if confidence < _CONFIDENCE_THRESHOLD or not answer_text:
            # Give a helpful domain-specific hint instead of generic message
            hint = self._hint_for_query(question, domain)
            return self._no_answer(hint)

        cited_page = self._resolve_page(start_offset, page_map)
        snippet    = self._extract_snippet(full_context, start_offset, window=130)

        logger.info(
            f"[RAG] Answer | confidence={confidence} | page={cited_page} | "
            f"domain={domain} | q='{question[:50]}' | a='{answer_text[:50]}'"
        )

        return {
            "answer":     answer_text,
            "confidence": confidence,
            "citations":  [{"page": cited_page, "text_snippet": snippet}],
        }

    # ── Hint & example helpers ────────────────────────────────────────────────

    def _hint_for_query(self, question: str, domain: str) -> str:
        """
        Returns a helpful message when no answer is found.
        Suggests rephrasing based on domain context.
        """
        q_lower = question.lower()

        if domain == "medical":
            if any(w in q_lower for w in ["patient", "who", "name"]):
                return "Patient name not clearly identified. Try: 'What is the patient's age?' or 'Who is the attending physician?'"
            if any(w in q_lower for w in ["year", "date", "when"]):
                return "Date not found as a direct answer. Try: 'What is the admission date?' or 'When was the patient seen?'"
            return (
                "Answer not found. Try specific questions like: "
                "'What medications are prescribed?' or 'What is the blood pressure reading?'"
            )
        elif domain == "legal":
            if any(w in q_lower for w in ["year", "date", "when"]):
                return "Date not found directly. Try: 'When was this agreement signed?' or 'What is the commencement date?'"
            return (
                "Answer not found. Try: "
                "'What is the monthly rent?' or 'Who are the parties to this agreement?'"
            )
        elif domain == "resume":
            if any(w in q_lower for w in ["year", "date", "when"]):
                return "Timeline not found directly. Try: 'When did the candidate graduate?' or 'What are the employment dates?'"
            return (
                "Answer not found. Try: "
                "'What degree does the candidate hold?' or 'Which companies has the candidate worked at?'"
            )
        return (
            "I couldn't find a confident answer. "
            "Try rephrasing as a specific factual question."
        )

    def _example_questions(self, domain: str) -> str:
        examples = {
            "medical": "Try: 'What is the patient's diagnosis?' or 'What medications are prescribed?'",
            "legal":   "Try: 'Who are the parties to this agreement?' or 'What is the monthly rent?'",
            "resume":  "Try: 'What degree does the candidate hold?' or 'What are the candidate's skills?'",
        }
        return examples.get(domain, "Try asking a specific factual question about the document.")

    # ── Standard helpers ──────────────────────────────────────────────────────

    @staticmethod
    def _resolve_page(answer_start: int, page_map: Dict[int, int]) -> int:
        best_offset = 0
        for offset in page_map:
            if offset <= answer_start:
                best_offset = max(best_offset, offset)
        return page_map.get(best_offset, 0)

    @staticmethod
    def _extract_snippet(context: str, start: int, window: int = 130) -> str:
        begin = max(0, start - 40)
        end   = min(len(context), start + window)
        raw   = context[begin:end].strip()
        if raw.startswith("[Page"):
            raw = raw.split("] ", 1)[-1] if "] " in raw else raw
        return raw

    @staticmethod
    def _no_answer(message: str) -> Dict[str, Any]:
        return {"answer": message, "confidence": 0.0, "citations": []}

    def _flush_vram(self):
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        elif torch.backends.mps.is_available():
            torch.mps.empty_cache()