"""
test_rag_engine.py — tests for DomainRAGEngine

What we're testing:
  - Empty context returns _no_answer response
  - Context is assembled correctly with page labels
  - Low confidence triggers fallback
  - _resolve_page maps offsets to correct pages
  - _extract_snippet returns readable text around the answer
  - Response always has answer, confidence, citations keys

NER pipeline is mocked — no model download needed.
"""
import pytest
from unittest.mock import MagicMock, patch


@pytest.fixture
def rag_engine():
    """DomainRAGEngine with QA pipeline mocked out."""
    with patch("backend.app.pipelines.rag_engine.pipeline") as mock_pipeline:
        mock_pipeline.return_value = MagicMock()
        from backend.app.pipelines.rag_engine import DomainRAGEngine
        engine = DomainRAGEngine.__new__(DomainRAGEngine)
        engine.device = -1
        engine.device_name = "cpu"
        engine.model_name = "deepset/roberta-base-squad2"
        engine.qa_pipeline = MagicMock()
        return engine


@pytest.fixture
def sample_context():
    return [
        {"text": "The patient was prescribed Metformin 1000mg twice daily.", "page": 1, "score": 0.12},
        {"text": "Blood pressure recorded at 158/96 mmHg.", "page": 2, "score": 0.18},
    ]


# ── Empty context ─────────────────────────────────────────────────────────────

def test_empty_context_returns_no_answer(rag_engine):
    result = rag_engine.generate_answer("What is the diagnosis?", [], "medical")
    assert result["answer"] != ""
    assert result["confidence"] == 0.0
    assert result["citations"] == []


# ── Response shape ────────────────────────────────────────────────────────────

def test_response_always_has_required_keys(rag_engine, sample_context):
    rag_engine.qa_pipeline.return_value = {
        "answer": "Metformin 1000mg",
        "score": 0.85,
        "start": 20,
        "end": 34,
    }
    result = rag_engine.generate_answer("What medication?", sample_context, "medical")
    assert "answer" in result
    assert "confidence" in result
    assert "citations" in result


def test_high_confidence_answer_returned(rag_engine, sample_context):
    rag_engine.qa_pipeline.return_value = {
        "answer": "Metformin 1000mg",
        "score": 0.85,
        "start": 20,
        "end": 34,
    }
    result = rag_engine.generate_answer("What medication?", sample_context, "medical")
    assert result["answer"] == "Metformin 1000mg"
    assert result["confidence"] == 0.85


def test_low_confidence_triggers_fallback(rag_engine, sample_context):
    """Score below _CONFIDENCE_THRESHOLD (0.10) should return fallback message."""
    rag_engine.qa_pipeline.return_value = {
        "answer": "something",
        "score": 0.05,   # below threshold
        "start": 0,
        "end": 5,
    }
    result = rag_engine.generate_answer("What?", sample_context, "medical")
    assert result["confidence"] == 0.0
    assert result["citations"] == []


def test_empty_answer_triggers_fallback(rag_engine, sample_context):
    rag_engine.qa_pipeline.return_value = {
        "answer": "",
        "score": 0.50,
        "start": 0,
        "end": 0,
    }
    result = rag_engine.generate_answer("What?", sample_context, "medical")
    assert result["confidence"] == 0.0


# ── _resolve_page ─────────────────────────────────────────────────────────────

def test_resolve_page_exact_offset(rag_engine):
    page_map = {0: 1, 100: 2, 200: 3}
    assert rag_engine._resolve_page(0, page_map) == 1
    assert rag_engine._resolve_page(100, page_map) == 2
    assert rag_engine._resolve_page(200, page_map) == 3


def test_resolve_page_between_offsets(rag_engine):
    """Answer at offset 150 is inside chunk starting at 100 (page 2)."""
    page_map = {0: 1, 100: 2, 200: 3}
    assert rag_engine._resolve_page(150, page_map) == 2


def test_resolve_page_empty_map(rag_engine):
    assert rag_engine._resolve_page(50, {}) == 0


# ── _extract_snippet ──────────────────────────────────────────────────────────

def test_extract_snippet_returns_string(rag_engine):
    context = "The patient has Type 2 Diabetes. HbA1c is 8.4%. Metformin prescribed."
    result = rag_engine._extract_snippet(context, start=10, window=50)
    assert isinstance(result, str)
    assert len(result) > 0


def test_extract_snippet_strips_page_prefix(rag_engine):
    context = "[Page 1] The patient has diabetes."
    result = rag_engine._extract_snippet(context, start=0, window=50)
    assert not result.startswith("[Page")


# ── _no_answer helper ─────────────────────────────────────────────────────────

def test_no_answer_shape(rag_engine):
    result = rag_engine._no_answer("Test message")
    assert result == {"answer": "Test message", "confidence": 0.0, "citations": []}


# ── Inference exception ───────────────────────────────────────────────────────

def test_inference_exception_returns_no_answer(rag_engine, sample_context):
    rag_engine.qa_pipeline.side_effect = RuntimeError("CUDA out of memory")
    result = rag_engine.generate_answer("What?", sample_context, "medical")
    assert result["confidence"] == 0.0
    assert result["citations"] == []