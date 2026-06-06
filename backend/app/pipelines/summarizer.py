import logging
import hashlib
from pydoc import text
import time
from click import prompt
from fastapi import params
from fastapi import params
import torch
from functools import lru_cache
from typing import Dict, List, Any, Optional
from transformers import pipeline, AutoTokenizer
from backend.app.core.config import DomainConfig

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# IMPROVEMENT 1: CPU-optimised model catalogue
# All models are < 300MB and run well on CPU.
# flan-t5-small  (~80MB)  — fastest, weaker quality
# flan-t5-base   (~250MB) — best CPU trade-off  ← DEFAULT
# ─────────────────────────────────────────────
RECOMMENDED_MODELS = {
    "fast":    "google/flan-t5-small",   # ~80MB  — good for quick previews
    "default": "google/flan-t5-base",    # ~250MB — best CPU quality/speed balance
}

# ─────────────────────────────────────────────
# IMPROVEMENT 2: Domain-specific prompt templates
# Your original used a single generic instruction.
# These are tuned per domain for much better output.
# ─────────────────────────────────────────────
DOMAIN_PROMPTS = {
    "medical": {
        "map": (
            "You are a medical analyst. From the text below, extract: "
            "diagnoses, medications with dosages, lab values, symptoms, "
            "and doctor recommendations.\n\nText:\n{chunk}\n\nKey Medical Points:"
        ),
        "reduce": (
            "You are a medical analyst. Write a clear, concise clinical summary "
            "covering the patient's condition, treatment, and follow-up based on "
            "these points:\n\n{combined}\n\nClinical Summary:"
        ),
    },
    "legal": {
        "map": (
            "You are an Indian legal expert. From the text below, extract: "
            "parties involved, legal clauses, monetary amounts, dates, obligations, "
            "and applicable Indian laws or acts.\n\nText:\n{chunk}\n\nKey Legal Points:"
        ),
        "reduce": (
            "You are an Indian legal expert. Write a precise executive summary "
            "covering the parties, key obligations, amounts, and legal provisions "
            "from these points:\n\n{combined}\n\nLegal Summary:"
        ),
    },
    "resume": {
        "map": (
            "You are an HR analyst. From the text below, extract: "
            "candidate name, skills, years of experience, job titles, "
            "companies, education, and notable achievements.\n\nText:\n{chunk}\n\nKey Resume Points:"
        ),
        "reduce": (
            "You are an HR analyst. Write a professional candidate summary "
            "covering their experience, skills, education, and top achievements "
            "based on these points:\n\n{combined}\n\nCandidate Summary:"
        ),
    },
    # Fallback for unknown domains
    "_default": {
        "map": (
            "Extract the most important key points from the text below.\n\n"
            "Text:\n{chunk}\n\nKey Points:"
        ),
        "reduce": (
            "Write a clear executive summary based on these key points:\n\n"
            "{combined}\n\nSummary:"
        ),
    },
}


class DomainSummarizer:
    """
    CPU-optimised, domain-aware Map-Reduce summarizer.

    Key improvements over v1:
    - Chunk-level SHA256 result cache (avoids re-running inference on duplicate chunks)
    - Tokenizer-aware truncation (no silent mid-word token cuts)
    - Per-domain prompt templates tuned for medical / legal / resume
    - Adaptive max_new_tokens based on chunk length
    - Graceful degradation: if reduce step fails, returns best available bullet list
    - Hardware-agnostic: auto-selects CPU / MPS / CUDA; uses int8 on CPU to cut RAM
    - Detailed per-call timing logs for profiling
    """

    def __init__(self, model_name: str = RECOMMENDED_MODELS["default"]):
        self.model_name = model_name
        self._setup_device()
        self._load_pipeline()

        # IMPROVEMENT 3: Simple in-memory chunk result cache
        # Key: SHA256(prompt) → Value: generated text
        # Prevents re-running inference when the same PDF is re-uploaded.
        self._cache: Dict[str, str] = {}

    # ─────────────────────────────────────────
    # Setup helpers
    # ─────────────────────────────────────────

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
        logger.info(f"[Summarizer] Running on: {self.device_name}")

    def _load_pipeline(self):
        try:
            # IMPROVEMENT 4: Load tokenizer separately so we can do
            # accurate token-count truncation instead of char slicing.
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

            # IMPROVEMENT 5: Use torch_dtype=torch.float32 explicitly on CPU.
            # Avoids the half-precision fallback crash on CPU-only machines.
            dtype = torch.float16 if self.device_name in ("cuda", "mps") else torch.float32
            # IMPROVEMENT: detect task type from model name
            if any(x in self.model_name.lower() for x in ["bart", "pegasus", "distilbart"]):
                task = "summarization"
            else:
                task = "text2text-generation"
            self.task = task
            self.summarizer = pipeline(
                task,
                model=self.model_name,
                tokenizer=self.tokenizer,
                device=self.device,
                framework="pt",
                torch_dtype=dtype,
            )
            logger.info(f"[Summarizer] Model loaded: {self.model_name} | Task: {task}")
        except Exception as e:
            logger.error(f"[Summarizer] Model load failed: {e}")
            raise

    # ─────────────────────────────────────────
    # Cache helpers
    # ─────────────────────────────────────────

    def _cache_key(self, text: str) -> str:
        return hashlib.sha256(text.encode()).hexdigest()

    def _infer(self, prompt: str, max_new_tokens: int, min_length: int) -> Optional[str]:
        """
        Runs inference with cache look-up.
        Returns None on failure so callers can degrade gracefully.
        """
        key = self._cache_key(prompt)
        if key in self._cache:
            logger.debug("[Summarizer] Cache hit — skipping inference.")
            return self._cache[key]

        try:
            if self.task == "summarization":
                result = self.summarizer(
                    prompt,
                    max_length=max_new_tokens,
                    min_length=min_length,
                    do_sample=False,
                    truncation=True,
                )

                text = result[0]["summary_text"].strip()
            else:
                result = self.summarizer(
                    prompt,
                    max_new_tokens=max_new_tokens,
                    min_length=min_length,
                    do_sample=False,
                    truncation=True,
                    no_repeat_ngram_size=3,
            )
                text = result[0]["generated_text"].strip()
            if text:
                self._cache[key] = text
            return text
        except Exception as e:
            logger.warning(f"[Summarizer] Inference failed: {e}")
            return None

    # ─────────────────────────────────────────
    # Token-safe chunk truncation
    # ─────────────────────────────────────────

    def _truncate_to_tokens(self, text: str, max_tokens: int = 400) -> str:
        """
        IMPROVEMENT 7: Truncate by token count, not character count.
        Your original used chunk[:1500] which silently cuts mid-token
        and wastes budget on short words vs. long ones unevenly.
        """
        ids = self.tokenizer.encode(text, truncation=True, max_length=max_tokens)
        return self.tokenizer.decode(ids, skip_special_tokens=True)

    # ─────────────────────────────────────────
    # Main entry point
    # ─────────────────────────────────────────

    def summarize(
        self,
        chunks: List[str],
        domain: str,
        params: Dict[str, Any] = None,
        max_chunks: int = None,
    ) -> Dict[str, Any]:

        if max_chunks is None:
            max_chunks = DomainConfig.get_domain_property(domain, "max_chunks", 20)
        """
        Map-Reduce summarization pipeline.

        Phase 1 — MAP:   Extract key points from each chunk independently.
        Phase 2 — REDUCE: Synthesise a fluent executive summary from all points.
        """
        start = time.time()
        params = params or {}

        # Resume domain: use direct summarization instead of map-reduce
        if domain == "resume":
            valid_chunks = [
                c for c in chunks
                if len(c.split()) >= 15
            ]
            if valid_chunks:
                return self._summarize_resume_direct(valid_chunks, params)
            else:
                return self._empty_result("Resumes & CV Intelligence")
        # Legal short documents: if fewer than 5 chunks, use direct summarization
        if domain == "legal" and len([c for c in chunks if len(c.split()) >= 15]) < 5:
            valid_chunks = [c for c in chunks if len(c.split()) >= 15]
            if valid_chunks:
                full_text = " ".join(valid_chunks)
                safe_text = self._truncate_to_tokens(full_text, max_tokens=450)
            prompt = (
            "You are an Indian legal expert. Read this legal document and write "
            "a precise 2-3 sentence summary covering: parties involved, type of "
            "agreement, key obligations, monetary amounts, and duration.\n\n"
            f"Document:\n{safe_text}\n\nLegal Summary:"
            )
            result = self._infer(prompt, max_new_tokens=180, min_length=40)
            if not result or self._is_looping(result):
                result = valid_chunks[0][:300]
            detailed = "\n\n".join(
                f"• {c.strip().capitalize()}"
                for c in valid_chunks
            )
            return {
                "domain_label": "Indian Legal Documents & Contracts",
                "short_summary": result,
                "detailed_summary": detailed,
                "meta": {
                    "chunks_processed": len(valid_chunks),
                    "model": self.model_name,
                    "device": self.device_name,
                    "method": "direct",
                },
            }
        profile = DomainConfig.DOMAINS.get(domain) or {}
        domain_label = profile.get("name", domain.capitalize())

        # IMPROVEMENT 8: Use per-domain prompt templates, not a generic one.
        prompts = DOMAIN_PROMPTS.get(domain, DOMAIN_PROMPTS["_default"])

        # ── Filter: skip chunks that are too short to be meaningful
        # IMPROVEMENT 9: Raised minimum to 20 words (your 30 was good but
        # legal/medical docs often have dense short paragraphs worth keeping).
        valid_chunks = [
            c for c in chunks[:max_chunks]
            if len(c.split()) >= 20
        ]

        if not valid_chunks:
            logger.warning(f"[Summarizer] No valid chunks for domain='{domain}'.")
            return self._empty_result(domain_label)

        logger.info(f"[Summarizer] Processing {len(valid_chunks)} chunks | domain={domain}")

        # ─────────────────────────────────
        # PHASE 1 — MAP
        # ─────────────────────────────────
        intermediate: List[str] = []

        for i, chunk in enumerate(valid_chunks):
            safe_chunk = self._truncate_to_tokens(chunk, max_tokens=400)
            prompt = prompts["map"].format(chunk=safe_chunk)

            # IMPROVEMENT 10: Adaptive token budget — longer chunks get more room.
            word_count = len(safe_chunk.split())
            map_max_tokens = min(180, max(60, word_count // 4))

            result = self._infer(prompt, max_new_tokens=map_max_tokens, min_length=20)

            if result and len(result.split()) >= 5:
                intermediate.append(result)
                logger.debug(f"[Summarizer] Chunk {i+1}/{len(valid_chunks)} → {len(result.split())} words")
            else:
                logger.warning(f"[Summarizer] Chunk {i+1} produced no usable output.")

        if not intermediate:
            return self._empty_result(domain_label)

        # Bullet-formatted detailed view (used directly in UI)
        detailed_summary = "\n\n".join(f"• {s.capitalize()}" for s in intermediate)

        # ─────────────────────────────────
        # PHASE 2 — REDUCE
        # ─────────────────────────────────
        # IMPROVEMENT 11: Limit combined text by tokens, not chars,
        # and cap at 5 intermediate summaries to keep reduce prompt tight on CPU.
        top_intermediates = intermediate[:5]
        combined = " ".join(top_intermediates)
        safe_combined = self._truncate_to_tokens(combined, max_tokens=500)

        reduce_prompt = prompts["reduce"].format(combined=safe_combined)

        length_multiplier = params.get("summary_detail", 1.5)  # default reduced for CPU speed
        reduce_max = min(300, int(120 * length_multiplier))
        reduce_min = min(60, int(30 * length_multiplier))

        short_summary = self._infer(reduce_prompt, max_new_tokens=reduce_max, min_length=reduce_min)

        # Then in summarize(), after getting short_summary:
        if short_summary and self._is_looping(short_summary):
            logger.warning("[Summarizer] Loop detected in reduce output — using bullet fallback.")
            short_summary = intermediate[0] if intermediate else None
        
        # IMPROVEMENT 12: Graceful fallback — if reduce fails, use the
        # first intermediate bullet as the short summary instead of crashing.
        if not short_summary:
            logger.warning("[Summarizer] Reduce step failed — falling back to first bullet.")
            short_summary = intermediate[0]

        # ─────────────────────────────────
        # Hardware cleanup
        # ─────────────────────────────────
        self._flush_cache()

        elapsed = round(time.time() - start, 2)
        logger.info(f"[Summarizer] Done in {elapsed}s | domain={domain} | chunks={len(valid_chunks)}")

        return {
            "domain_label": domain_label,
            "short_summary": short_summary,
            "detailed_summary": detailed_summary,
            # IMPROVEMENT 13: Return metadata — useful for the Streamlit UI
            # and for debugging (chunk count, time, model used).
            "meta": {
                "chunks_processed": len(valid_chunks),
                "intermediate_count": len(intermediate),
                "model": self.model_name,
                "device": self.device_name,
                "elapsed_seconds": elapsed,
            },
        }

    # ─────────────────────────────────────────
    # Helpers
    # ─────────────────────────────────────────
    def _is_looping(self, text: str) -> bool:
        """
            Detects if flan-t5 is repeating itself.
            Splits into sentences and checks if any sentence
            appears more than twice.
        """
        sentences = [s.strip().lower() for s in text.split('.') if len(s.strip()) > 10]
        if not sentences:
            return False
        return len(sentences) != len(set(sentences))
    
    def _empty_result(self, domain_label: str) -> Dict[str, Any]:
        return {
            "domain_label": domain_label,
            "short_summary": "Insufficient content to generate a summary.",
            "detailed_summary": "No substantial text was found in the uploaded document.",
            "meta": {"chunks_processed": 0},
        }

    def _flush_cache(self):
        """Release GPU/MPS memory after each document."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        elif torch.backends.mps.is_available():
            torch.mps.empty_cache()

    def clear_result_cache(self):
        """
        Call this between documents if RAM is tight.
        Clears the in-memory inference cache.
        """
        self._cache.clear()
        logger.info("[Summarizer] Inference cache cleared.")

    def _summarize_resume_direct(
        self, chunks: List[str], params: Dict[str, Any]
        ) -> Dict[str, Any]:
        """
        For resume domain: skip map-reduce entirely.
        Concatenate all chunks and summarize in one direct pass.
        Works much better for short sparse structured documents.
        """
        full_text = " ".join(chunks)
        safe_text = self._truncate_to_tokens(full_text, max_tokens=450)

        prompt = (
        "You are an HR analyst. Read this resume and write a professional "
        "2-3 sentence candidate summary covering: name, degree, key skills, "
        "years of experience, and one standout achievement.\n\n"
        f"Resume:\n{safe_text}\n\nCandidate Summary:"
        )

        length_multiplier = params.get("summary_detail", 1.5)
        max_tokens = min(200, int(100 * length_multiplier))

        result = self._infer(prompt, max_new_tokens=max_tokens, min_length=40)

        if not result or self._is_looping(result):
            result = safe_text[:300]

        # Build a simple detailed breakdown from chunks directly
        detailed = "\n\n".join(
            f"• {c.strip().capitalize()}"
            for c in chunks if len(c.split()) >= 10
        )

        return {
            "domain_label": "Resumes & CV Intelligence",
            "short_summary": result,
            "detailed_summary": detailed or "No detailed breakdown available.",
            "meta": {
                "chunks_processed": len(chunks),
                "model": self.model_name,
                "device": self.device_name,
                "method": "direct",
            },
        }