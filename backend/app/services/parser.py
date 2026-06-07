import io
import re
import logging
from typing import List, Dict, Any
from pypdf import PdfReader
from fastapi import UploadFile, HTTPException

logger = logging.getLogger(__name__)

# Hard cap — process at most 5 pages per document
MAX_PAGES = 5


class DocumentParser:
    """
    PDF parser with sentence-boundary chunking.
    Hard limit: 5 pages per document.
    Strips HTML tags, URLs, form field underscores, annotation artifacts.
    Skips near-empty pages (< 20 words).
    """

    _MULTI_NEWLINE = re.compile(r'\n+')
    _MULTI_SPACE   = re.compile(r'\s+')
    _ARTIFACTS     = re.compile(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]|[•·●◦▪▸►]')
    _SENTENCE_END  = re.compile(r'(?<=[.!?])\s+')
    _HTML_TAGS     = re.compile(r'<[^>]+>')
    _URLS          = re.compile(r'https?://\S+|www\.\S+')
    _UNDERSCORES   = re.compile(r'_{3,}')
    _ANNOTATIONS   = re.compile(r'\[\d+\]|\[Link\]|\[image\]|\[table\]', re.IGNORECASE)
    _DASHES        = re.compile(r'-{3,}')

    def __init__(self, chunk_size: int = 1000, overlap: int = 150, max_pages: int = MAX_PAGES):
        if overlap >= chunk_size:
            raise ValueError("overlap must be smaller than chunk_size.")
        self.chunk_size = chunk_size
        self.overlap    = overlap
        # Hard cap — caller cannot exceed MAX_PAGES even if they pass a larger value
        self.max_pages  = min(max_pages, MAX_PAGES)

    async def parse_pdf(self, file: UploadFile) -> List[Dict[str, Any]]:
        if not file.filename.lower().endswith(".pdf"):
            raise HTTPException(status_code=400, detail="Only PDF files are accepted.")

        content    = await file.read()
        pdf_stream = io.BytesIO(content)

        try:
            reader        = PdfReader(pdf_stream)
            total_pages   = len(reader.pages)
            pages_to_read = min(total_pages, self.max_pages)

            # Warn if document is being truncated
            if total_pages > self.max_pages:
                logger.warning(
                    f"[Parser] '{file.filename}' has {total_pages} pages — "
                    f"processing first {self.max_pages} only (hard cap)."
                )

            pages_data: List[Dict[str, Any]] = []

            for idx in range(pages_to_read):
                raw     = reader.pages[idx].extract_text() or ""
                cleaned = self._clean_text(raw)

                if len(cleaned.split()) < 20:
                    continue

                pages_data.append({
                    "page_number": idx + 1,
                    "text":        cleaned,
                })

            logger.info(
                f"[Parser] '{file.filename}' — "
                f"{len(pages_data)} usable pages from {pages_to_read} read "
                f"(total in PDF: {total_pages})"
            )
            return pages_data

        except Exception as e:
            raise HTTPException(status_code=500, detail=f"PDF parsing failed: {e}")
        finally:
            pdf_stream.close()

    def _clean_text(self, text: str) -> str:
        text = self._ARTIFACTS.sub(" ",   text)
        text = self._HTML_TAGS.sub(" ",   text)
        text = self._URLS.sub(" ",        text)
        text = self._ANNOTATIONS.sub(" ", text)
        text = self._UNDERSCORES.sub(" ", text)
        text = self._DASHES.sub(" ",      text)
        text = self._MULTI_NEWLINE.sub(" ", text)
        text = self._MULTI_SPACE.sub(" ",   text)
        return text.strip()

    def create_chunks(self, pages_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        chunks = []

        for page in pages_data:
            text: str     = page["text"]
            page_num: int = page["page_number"]

            if len(text) <= self.chunk_size:
                chunks.append({"text": text, "metadata": {"page": page_num}})
                continue

            sentences  = self._SENTENCE_END.split(text)
            window: List[str] = []
            window_len = 0

            for sent in sentences:
                sent_len = len(sent)
                if window_len + sent_len > self.chunk_size and window:
                    chunk_text = " ".join(window).strip()
                    if chunk_text:
                        chunks.append({"text": chunk_text, "metadata": {"page": page_num}})
                    overlap_text = chunk_text[-self.overlap:]
                    window     = [overlap_text]
                    window_len = len(overlap_text)
                window.append(sent)
                window_len += sent_len + 1

            if window:
                chunk_text = " ".join(window).strip()
                if chunk_text:
                    chunks.append({"text": chunk_text, "metadata": {"page": page_num}})

        return chunks