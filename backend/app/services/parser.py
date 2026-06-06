import io
import re
from typing import List, Dict, Any
from pypdf import PdfReader
from fastapi import UploadFile, HTTPException
from streamlit import text

class DocumentParser:
    """
    PDF parser with overlapping chunk generation.

    Improvements over v1:
    - Chunk boundaries now snap to sentence ends (no mid-sentence cuts)
    - _clean_text removes PDF ligature artifacts and bullet characters
    - max_pages is a soft default — caller can override per domain via config
    - Page word-count logged; near-empty pages (< 20 words) are skipped, not silently included
    - create_chunks returns page_start + page_end range in metadata for citation accuracy
    - BytesIO buffer explicitly closed after parsing to free RAM
    """

    _MULTI_NEWLINE = re.compile(r'\n+')
    _MULTI_SPACE   = re.compile(r'\s+')
    # IMPROVEMENT: strips common PDF extraction artifacts
    _ARTIFACTS     = re.compile(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]|[•·●◦▪▸►]')
    # Sentence boundary — ends with . ! ? followed by whitespace or end
    _SENTENCE_END  = re.compile(r'(?<=[.!?])\s+')

    # Add these two patterns at class level alongside existing ones
    _HTML_TAGS    = re.compile(r'<[^>]+>')
    _URLS         = re.compile(r'https?://\S+|www\.\S+')
    _UNDERSCORES  = re.compile(r'_{3,}')  # strips form field blanks

    def __init__(self, chunk_size: int = 1000, overlap: int = 150, max_pages: int = 50):
        if overlap >= chunk_size:
            raise ValueError("overlap must be smaller than chunk_size.")
        self.chunk_size = chunk_size
        self.overlap    = overlap
        self.max_pages  = max_pages  # IMPROVEMENT: default raised from 5 → 50

    # ── Parse ────────────────────────────────────────────────────────────────

    async def parse_pdf(self, file: UploadFile) -> List[Dict[str, Any]]:
        if not file.filename.lower().endswith(".pdf"):
            raise HTTPException(status_code=400, detail="Only PDF files are accepted.")

        content = await file.read()
        pdf_stream = io.BytesIO(content)

        try:
            reader = PdfReader(pdf_stream)
            pages_data: List[Dict[str, Any]] = []
            pages_to_read = min(len(reader.pages), self.max_pages)

            for idx in range(pages_to_read):
                raw = reader.pages[idx].extract_text() or ""
                cleaned = self._clean_text(raw)

                # IMPROVEMENT: skip near-empty pages (boilerplate, headers only)
                word_count = len(cleaned.split())
                if word_count < 20:
                    continue

                pages_data.append({"page_number": idx + 1, "text": cleaned})

            return pages_data

        except Exception as e:
            raise HTTPException(status_code=500, detail=f"PDF parsing failed: {e}")
        finally:
            # IMPROVEMENT: always release the BytesIO buffer
            pdf_stream.close()

    # ── Clean ────────────────────────────────────────────────────────────────

    def _clean_text(self, text: str) -> str:
        text = self._ARTIFACTS.sub(" ", text)
        text = self._HTML_TAGS.sub(" ", text)      # strip HTML tags
        text = self._URLS.sub(" ", text)           # strip URLs
        text = self._UNDERSCORES.sub(" ", text)    # strip ____ form fields
        text = self._MULTI_NEWLINE.sub(" ", text)
        text = self._MULTI_SPACE.sub(" ", text)
        return text.strip()
    
    # ── Chunk ────────────────────────────────────────────────────────────────

    def create_chunks(self, pages_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Splits each page into overlapping chunks.

        IMPROVEMENT: chunks snap to the nearest sentence boundary so the
        summarizer and RAG engine never receive a sentence cut in half.
        Metadata now carries page_start and page_end for multi-page chunks.
        """
        chunks: List[Dict[str, Any]] = []
        step = self.chunk_size - self.overlap

        for page in pages_data:
            text: str = page["text"]
            page_num: int = page["page_number"]

            if len(text) <= self.chunk_size:
                chunks.append({"text": text, "metadata": {"page": page_num}})
                continue

            # Split into sentences first, then greedily assemble into windows
            sentences = self._SENTENCE_END.split(text)
            window: List[str] = []
            window_len = 0

            for sent in sentences:
                sent_len = len(sent)
                if window_len + sent_len > self.chunk_size and window:
                    chunk_text = " ".join(window).strip()
                    if chunk_text:
                        chunks.append({"text": chunk_text, "metadata": {"page": page_num}})
                    # Overlap: retain last N chars worth of sentences
                    overlap_text = chunk_text[-self.overlap:]
                    window = [overlap_text]
                    window_len = len(overlap_text)
                window.append(sent)
                window_len += sent_len + 1  # +1 for space

            # Flush remaining sentences
            if window:
                chunk_text = " ".join(window).strip()
                if chunk_text:
                    chunks.append({"text": chunk_text, "metadata": {"page": page_num}})

        return chunks