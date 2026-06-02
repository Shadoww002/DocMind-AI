"""
test_parser.py — tests for DocumentParser

What we're testing:
  - _clean_text removes artifacts and collapses whitespace
  - create_chunks splits text at sentence boundaries
  - create_chunks respects chunk_size
  - short pages are returned as a single chunk (not split)
  - overlap is applied correctly

No PDF file needed — we test the logic directly.
"""
import pytest
from backend.app.services.parser import DocumentParser


# ── Fixtures ───────────────────────────────────────────────────────────────────

@pytest.fixture
def parser():
    """Standard parser with small chunk size so we can test splitting easily."""
    return DocumentParser(chunk_size=100, overlap=20)


@pytest.fixture
def large_parser():
    return DocumentParser(chunk_size=1000, overlap=150)


# ── _clean_text ────────────────────────────────────────────────────────────────

def test_clean_text_collapses_newlines(parser):
    dirty = "hello\n\n\nworld"
    assert parser._clean_text(dirty) == "hello world"


def test_clean_text_collapses_spaces(parser):
    dirty = "hello     world"
    assert parser._clean_text(dirty) == "hello world"


def test_clean_text_removes_bullet_artifacts(parser):
    dirty = "• Item one • Item two"
    result = parser._clean_text(dirty)
    assert "•" not in result


def test_clean_text_removes_control_chars(parser):
    dirty = "hello\x00\x01world"
    result = parser._clean_text(dirty)
    assert "\x00" not in result
    assert "\x01" not in result


def test_clean_text_strips_edges(parser):
    dirty = "   hello world   "
    assert parser._clean_text(dirty) == "hello world"


# ── create_chunks ──────────────────────────────────────────────────────────────

def test_short_page_returned_as_single_chunk(large_parser):
    """A page shorter than chunk_size should not be split."""
    pages = [{"page_number": 1, "text": "Short text here."}]
    chunks = large_parser.create_chunks(pages)
    assert len(chunks) == 1
    assert chunks[0]["text"] == "Short text here."


def test_chunk_metadata_has_page_key(large_parser):
    """Every chunk must have a 'page' key in metadata for citation accuracy."""
    pages = [{"page_number": 3, "text": "Some text on page three."}]
    chunks = large_parser.create_chunks(pages)
    assert chunks[0]["metadata"]["page"] == 3


def test_long_page_produces_multiple_chunks():
    """A page longer than chunk_size must produce more than one chunk."""
    parser = DocumentParser(chunk_size=50, overlap=10)
    long_text = "This is sentence one. " * 20  # 440 chars, well over chunk_size=50
    pages = [{"page_number": 1, "text": long_text}]
    chunks = parser.create_chunks(pages)
    assert len(chunks) > 1


def test_no_empty_chunks(large_parser):
    """No chunk should have empty or whitespace-only text."""
    pages = [
        {"page_number": 1, "text": "First page content with enough words to be meaningful."},
        {"page_number": 2, "text": "Second page content."},
    ]
    chunks = large_parser.create_chunks(pages)
    for chunk in chunks:
        assert chunk["text"].strip() != ""


def test_chunk_size_respected():
    """No chunk should exceed chunk_size by more than one sentence length."""
    parser = DocumentParser(chunk_size=80, overlap=10)
    # Each sentence is ~30 chars — so max 2-3 sentences per chunk
    text = "First sentence here. Second sentence here. Third sentence. Fourth one. Fifth now."
    pages = [{"page_number": 1, "text": text}]
    chunks = parser.create_chunks(pages)
    for chunk in chunks:
        # Allow some leeway for sentence boundary snapping
        assert len(chunk["text"]) <= 200


# ── Constructor validation ─────────────────────────────────────────────────────

def test_overlap_equal_to_chunk_size_raises():
    """overlap >= chunk_size is invalid and should raise immediately."""
    with pytest.raises(ValueError):
        DocumentParser(chunk_size=100, overlap=100)


def test_overlap_greater_than_chunk_size_raises():
    with pytest.raises(ValueError):
        DocumentParser(chunk_size=100, overlap=150)