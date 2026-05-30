import io
import re
from typing import List, Dict, Any, Generator
from pypdf import PdfReader
from fastapi import UploadFile, HTTPException

class DocumentParser:
    # Pre-compile regex engines once at the class level to save CPU cycles on every loop iteration
    _MULTIPLE_NEWLINES_PATTERN = re.compile(r'\n+')
    _MULTIPLE_SPACES_PATTERN = re.compile(r'\s+')

    def __init__(self, chunk_size: int = 1000, overlap: int = 150, max_pages: int = 5):
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.max_pages = max_pages
        
        # Guard rail optimization: ensuring safe chunking boundary windows
        if self.overlap >= self.chunk_size:
            raise ValueError("Overlap size must be strictly smaller than the chunk size.")

    async def parse_pdf(self, file: UploadFile) -> List[Dict[str, Any]]:
        """
        Parses an uploaded PDF file safely up to a strict page threshold.
        Optimized for memory overhead and high processing throughput.
        """
        if not file.filename.lower().endswith('.pdf'):
            raise HTTPException(
                status_code=400, 
                detail="Invalid file format. Only standard PDF formats are accepted by the extraction engine."
            )
        
        try:
            # Read streaming bytes efficiently into an in-memory buffer
            content = await file.read()
            pdf_stream = io.BytesIO(content)
            reader = PdfReader(pdf_stream)
            
            pages_data: List[Dict[str, Any]] = []
            
            # Enforce strict early exit directly within the extraction loop
            total_pages = len(reader.pages)
            pages_to_extract = min(total_pages, self.max_pages)
            
            for page_idx in range(pages_to_extract):
                page = reader.pages[page_idx]
                raw_text = page.extract_text()
                
                if raw_text:
                    cleaned_text = self._clean_text(raw_text)
                    # Skip pages that end up completely empty after text normalization
                    if cleaned_text:
                        pages_data.append({
                            "page_number": page_idx + 1,
                            "text": cleaned_text
                        })
                        
            return pages_data
            
        except Exception as e:
            # Shield internal stack traces while providing context for monitoring systems
            raise HTTPException(
                status_code=500, 
                detail=f"Pipeline Processing Interruption during parsing engine execution: {str(e)}"
            )

    def _clean_text(self, text: str) -> str:
        """Sanitizes layout string fragments into clean sequential semantic tokens."""
        # Highly efficient substitution using our pre-compiled engines
        text = self._MULTIPLE_NEWLINES_PATTERN.sub(' ', text)
        text = self._MULTIPLE_SPACES_PATTERN.sub(' ', text)
        return text.strip()

    def create_chunks(self, pages_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Generates overlapping vector token chunks using a memory-conscious sequence slice window.
        Preserves original metadata tracking footprints cleanly.
        """
        processed_chunks: List[Dict[str, Any]] = []
        step_size = self.chunk_size - self.overlap

        for page in pages_data:
            text: str = page["text"]
            page_num: int = page["page_number"]
            text_len = len(text)
            
            # Fallback for pages with very short strings to ensure they are captured
            if text_len <= self.chunk_size:
                processed_chunks.append({
                    "text": text,
                    "metadata": {"page": page_num}
                })
                continue

            # Optimized rolling matrix slice window loop
            for start in range(0, text_len - self.overlap, step_size):
                end = start + self.chunk_size
                chunk_text = text[start:end]
                
                processed_chunks.append({
                    "text": chunk_text,
                    "metadata": {"page": page_num}
                })
                
                # Performance early break to eliminate unnecessary trailing empty slices
                if end >= text_len:
                    break
                    
        return processed_chunks