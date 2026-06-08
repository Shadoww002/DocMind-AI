import logging
import hashlib
import time
import torch
from typing import Dict, List, Any, Optional
from transformers import pipeline, AutoTokenizer
from backend.app.core.config import DomainConfig

logger = logging.getLogger(__name__)

# ── Model ─────────────────────────────────────────────────────────────────────
# sshleifer/distilbart-cnn-12-6 (~306MB)
# Distilled BART trained on CNN/DailyMail news summarization.
# CRITICAL: distilbart is NOT an instruction-following model.
# Do NOT pass instruction prompts — it summarizes them as text.
# Feed ONLY clean raw text and it produces coherent prose summaries.
DEFAULT_MODEL = "sshleifer/distilbart-cnn-12-6"

# Echo prefixes distilbart sometimes starts output with — strip these
ECHO_PREFIXES = [
    "this article", "this document", "this report", "this resume",
    "the following", "in this article", "the article", "the document",
    "according to", "in summary", "summary:",
]


class DomainSummarizer:
    """
    Domain-aware summarizer using sshleifer/distilbart-cnn-12-6.

    Key design principle: distilbart is a news summarizer, NOT an instruction
    model. Feed it clean raw text only — no prompts, no instructions.
    It will produce coherent extractive-abstractive summaries natively.

    Strategy:
    - Resume + short legal (<5 chunks): direct single-pass on full text
    - Medical + long legal:             map-reduce
      MAP:    distilbart summarizes each chunk independently → bullet points
      REDUCE: distilbart summarizes the combined bullets → short executive summary
    """

    def __init__(self, model_name: str = DEFAULT_MODEL):
        self.model_name = model_name
        self._setup_device()
        self._load_pipeline()
        self._cache: Dict[str, str] = {}

    # ── Setup ─────────────────────────────────────────────────────────────────

    def _setup_device(self):
        if torch.cuda.is_available():
            self.device, self.device_name = 0, "cuda"
        elif torch.backends.mps.is_available():
            self.device, self.device_name = "mps", "mps"
        else:
            self.device, self.device_name = -1, "cpu"
        logger.info(f"[Summarizer] Device: {self.device_name}")

    def _load_pipeline(self):
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            dtype = torch.float16 if self.device_name in ("cuda", "mps") else torch.float32

            if any(x in self.model_name.lower() for x in ["bart", "pegasus", "distilbart"]):
                self.task = "summarization"
            else:
                self.task = "text2text-generation"

            self.summarizer = pipeline(
                self.task,
                model=self.model_name,
                tokenizer=self.tokenizer,
                device=self.device,
                framework="pt",
                torch_dtype=dtype,
            )
            logger.info(f"[Summarizer] Loaded: {self.model_name} | Task: {self.task}")
        except Exception as e:
            logger.error(f"[Summarizer] Load failed: {e}")
            raise

    # ── Inference ─────────────────────────────────────────────────────────────

    def _infer(self, text: str, max_length: int = 180, min_length: int = 30) -> Optional[str]:
        """
        Runs distilbart on raw text. No prompt prefix — just clean text.
        SHA256 cache to avoid re-running on duplicate chunks.
        """
        # Clean the text before inference
        text = text.strip()
        if not text:
            return None

        key = self._cache_key(text)
        if key in self._cache:
            logger.debug("[Summarizer] Cache hit.")
            return self._cache[key]
        try:
            if self.task == "summarization":
                result = self.summarizer(
                    text,
                    max_length=max_length,
                    min_length=min_length,
                    do_sample=False,
                    truncation=True,
                )
                out = result[0]["summary_text"].strip()
            else:
                result = self.summarizer(
                    text,
                    max_new_tokens=max_length,
                    min_length=min_length,
                    do_sample=False,
                    truncation=True,
                    no_repeat_ngram_size=3,
                )
                out = result[0]["generated_text"].strip()

            out = self._strip_echo(out)
            if out:
                self._cache[key] = out
            return out
        except Exception as e:
            logger.warning(f"[Summarizer] Inference failed: {e}")
            return None

    # ── Main entry point ──────────────────────────────────────────────────────

    def summarize(
        self,
        chunks: List[str],
        domain: str,
        params: Dict[str, Any] = None,
        max_chunks: int = None,
    ) -> Dict[str, Any]:
        """
        Routes to correct strategy:
          resume          → direct single-pass
          legal < 5 chunks → direct single-pass
          medical + long legal → map-reduce
        """
        start  = time.time()
        params = params or {}

        if max_chunks is None:
            max_chunks = DomainConfig.get_domain_property(domain, "max_chunks", 20)

        # ── Resume: always direct ─────────────────────────────────────────────
        if domain == "resume":
            valid = [c for c in chunks if len(c.split()) >= 15]
            return (
                self._summarize_direct(valid, domain, params)
                if valid else self._empty_result("Resumes & CV Intelligence")
            )

        # ── Legal short doc: direct ───────────────────────────────────────────
        valid_legal = [c for c in chunks if len(c.split()) >= 15]
        if domain == "legal" and len(valid_legal) < 5:
            return (
                self._summarize_direct(valid_legal, domain, params)
                if valid_legal else self._empty_result("Indian Legal Documents & Contracts")
            )

        # ── Map-reduce for medical + long legal ───────────────────────────────
        profile      = DomainConfig.DOMAINS.get(domain) or {}
        domain_label = profile.get("name", domain.capitalize())

        valid_chunks = [c for c in chunks[:max_chunks] if len(c.split()) >= 20]
        if not valid_chunks:
            return self._empty_result(domain_label)

        logger.info(f"[Summarizer] Map-reduce | {len(valid_chunks)} chunks | domain={domain}")

        # MAP — distilbart summarizes each chunk independently (NO PROMPT)
        intermediate: List[str] = []
        for i, chunk in enumerate(valid_chunks):
            safe = self._truncate_to_tokens(chunk, max_tokens=350)
            word_count = len(safe.split())
            map_max = min(160, max(50, word_count // 3))
            result = self._infer(safe, max_length=map_max, min_length=15)
            if result and len(result.split()) >= 5:
                intermediate.append(result)
            else:
                logger.warning(f"[Summarizer] Chunk {i+1} no output.")

        if not intermediate:
            return self._empty_result(domain_label)

        detailed = self._format_detailed(intermediate)

        # REDUCE — summarize the combined bullets (NO PROMPT)
        combined      = " ".join(intermediate[:5])
        safe_combined = self._truncate_to_tokens(combined, max_tokens=400)

        length_multiplier = params.get("summary_detail", 1.5)
        reduce_max = min(200, int(100 * length_multiplier))
        reduce_min = min(50,  int(25  * length_multiplier))

        short_summary = self._infer(safe_combined, max_length=reduce_max, min_length=reduce_min)

        if not short_summary or self._is_looping(short_summary):
            logger.warning("[Summarizer] Reduce poor output — using first bullet.")
            short_summary = intermediate[0]

        self._flush_vram()
        elapsed = round(time.time() - start, 2)
        logger.info(f"[Summarizer] Done {elapsed}s | domain={domain} | chunks={len(valid_chunks)}")

        return {
            "domain_label":     domain_label,
            "short_summary":    short_summary,
            "detailed_summary": detailed,
            "meta": {
                "chunks_processed":   len(valid_chunks),
                "intermediate_count": len(intermediate),
                "model":              self.model_name,
                "device":             self.device_name,
                "elapsed_seconds":    elapsed,
                "method":             "map-reduce",
            },
        }

    # ── Direct single-pass ────────────────────────────────────────────────────

    def _summarize_direct(
        self,
        chunks: List[str],
        domain: str,
        params: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        For resume and short legal.
        Feeds raw text to distilbart — no instruction prefix.
        distilbart summarizes naturally; produces readable prose.
        """
        start        = time.time()
        profile      = DomainConfig.DOMAINS.get(domain) or {}
        domain_label = profile.get("name", domain.capitalize())

        # Per-chunk detailed bullets (NO PROMPT — pure text)
        intermediate: List[str] = []
        for chunk in chunks:
            if len(chunk.split()) < 10:
                continue
            safe   = self._truncate_to_tokens(chunk, max_tokens=350)
            result = self._infer(safe, max_length=120, min_length=15)
            if result:
                intermediate.append(result)

        detailed = self._format_detailed(intermediate) if intermediate else f"• {chunks[0][:300]}"

        # Short summary — full document in one pass (NO PROMPT)
        full_text = " ".join(chunks)
        safe_text = self._truncate_to_tokens(full_text, max_tokens=450)

        length_multiplier = params.get("summary_detail", 1.5)
        max_len = min(200, int(100 * length_multiplier))

        short_summary = self._infer(safe_text, max_length=max_len, min_length=40)

        if not short_summary or self._is_looping(short_summary):
            short_summary = intermediate[0] if intermediate else safe_text[:300]

        elapsed = round(time.time() - start, 2)
        logger.info(f"[Summarizer] Direct {elapsed}s | domain={domain}")

        return {
            "domain_label":     domain_label,
            "short_summary":    short_summary,
            "detailed_summary": detailed,
            "meta": {
                "chunks_processed": len(chunks),
                "model":            self.model_name,
                "device":           self.device_name,
                "elapsed_seconds":  elapsed,
                "method":           "direct",
            },
        }

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _format_detailed(self, items: List[str]) -> str:
        """Clean readable bullet list — deduped, capitalised, echo-stripped."""
        seen   = set()
        result = []
        for item in items:
            clean = self._strip_echo(item.strip())
            if not clean or len(clean.split()) < 4:
                continue
            key = clean.lower()[:60]
            if key in seen:
                continue
            seen.add(key)
            clean = clean[0].upper() + clean[1:]
            result.append(f"• {clean}")
        return "\n\n".join(result) if result else "• No detailed breakdown available."

    def _cache_key(self, text: str) -> str:
        return hashlib.sha256(text.encode()).hexdigest()

    def _truncate_to_tokens(self, text: str, max_tokens: int = 400) -> str:
        ids = self.tokenizer.encode(text, truncation=True, max_length=max_tokens)
        return self.tokenizer.decode(ids, skip_special_tokens=True)

    def _is_looping(self, text: str) -> bool:
        sentences = [s.strip().lower() for s in text.split('.') if len(s.strip()) > 10]
        if not sentences:
            return False
        return len(sentences) != len(set(sentences))

    def _strip_echo(self, text: str) -> str:
        """Strips common echo prefixes distilbart starts output with."""
        lower = text.lower()
        for echo in ECHO_PREFIXES:
            if lower.startswith(echo):
                text = text[len(echo):].lstrip(":., ").strip()
                break
        return text[0].upper() + text[1:] if text else text

    def _empty_result(self, domain_label: str) -> Dict[str, Any]:
        return {
            "domain_label":     domain_label,
            "short_summary":    "Insufficient content to generate a summary.",
            "detailed_summary": "• No substantial text was found in the uploaded document.",
            "meta":             {"chunks_processed": 0},
        }

    def _flush_vram(self):
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        elif torch.backends.mps.is_available():
            torch.mps.empty_cache()

    def clear_result_cache(self):
        self._cache.clear()
        logger.info("[Summarizer] Cache cleared.")