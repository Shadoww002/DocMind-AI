"""
test_extractor.py — tests for DomainExtractor

What we're testing:
  - Regex patterns find known entities in sample text
  - Scored output has correct shape {"value": str, "score": float}
  - Scores are within valid range (0.0 to 1.0)
  - High-frequency entities score higher than single-occurrence entities
  - anonymize_pii replaces names with redaction string
  - Unknown domain raises ValueError
  - _assess_legal_risk returns correct risk level strings

IMPORTANT: We mock the NER pipeline so these tests run without
downloading any model. They test YOUR logic, not HuggingFace's model.
"""
import pytest
from unittest.mock import MagicMock, patch


# ── Mock NER pipeline before importing extractor ──────────────────────────────
# This patches transformers.pipeline so no model is downloaded during tests.

@pytest.fixture
def extractor():
    """Returns a DomainExtractor with NER pipeline mocked out."""
    with patch("backend.app.pipelines.extractor.pipeline") as mock_pipeline:
        # NER returns empty list by default — tests that need NER results
        # will override this per-test
        mock_pipeline.return_value = MagicMock(return_value=[])
        from backend.app.pipelines.extractor import DomainExtractor
        ext = DomainExtractor.__new__(DomainExtractor)
        ext.device = -1
        ext.device_name = "cpu"
        ext.ner_pipeline = MagicMock(return_value=[])
        return ext


# ── Scored output shape ────────────────────────────────────────────────────────

def test_scored_matches_returns_list(extractor, medical_text):
    result = extractor._scored_matches(
        extractor._MEDICAL_PATTERNS["Diagnoses"], medical_text
    )
    assert isinstance(result, list)


def test_scored_matches_items_have_value_and_score(extractor, medical_text):
    result = extractor._scored_matches(
        extractor._MEDICAL_PATTERNS["Diagnoses"], medical_text
    )
    for item in result:
        assert "value" in item
        assert "score" in item


def test_scored_matches_score_in_valid_range(extractor, medical_text):
    result = extractor._scored_matches(
        extractor._MEDICAL_PATTERNS["Diagnoses"], medical_text
    )
    for item in result:
        assert 0.0 <= item["score"] <= 1.0


def test_high_frequency_entity_scores_higher(extractor):
    """
    'Metformin' appears 3 times → should score 0.99
    'Amlodipine' appears 1 time → should score 0.75
    """
    text = "Metformin 1000mg prescribed. Metformin dose adjusted. Metformin continued. Amlodipine 5mg added."
    result = extractor._scored_matches(extractor._MEDICAL_PATTERNS["Medications"], text)
    scores = {item["value"].lower(): item["score"] for item in result}

    # Metformin appears 3x — should be 0.99
    metformin_score = next((v for k, v in scores.items() if "metformin" in k), None)
    assert metformin_score == 0.99

    # Amlodipine appears 1x — should be 0.75
    amlodipine_score = next((v for k, v in scores.items() if "amlodipine" in k), None)
    assert amlodipine_score == 0.75


# ── Medical extraction ────────────────────────────────────────────────────────

def test_medical_finds_diagnoses(extractor, medical_text):
    result = extractor._scored_matches(
        extractor._MEDICAL_PATTERNS["Diagnoses"], medical_text
    )
    values = [item["value"].lower() for item in result]
    # medical_text contains "Hypertension" and "Diabetes"
    assert any("hypertension" in v or "diabetes" in v for v in values)


def test_medical_finds_medications(extractor, medical_text):
    result = extractor._scored_matches(
        extractor._MEDICAL_PATTERNS["Medications"], medical_text
    )
    values = [item["value"].lower() for item in result]
    assert any("metformin" in v for v in values)


def test_medical_finds_lab_values(extractor, medical_text):
    result = extractor._scored_matches(
        extractor._MEDICAL_PATTERNS["Lab Values"], medical_text
    )
    # medical_text contains "HbA1c 8.4%" and "Fasting glucose 184 mg/dl"
    assert len(result) > 0


def test_anonymize_pii_redacts_patient_name(extractor, medical_text):
    result = extractor._extract_medical(
        medical_text,
        people=["John Doe"],
        dates=[],
        anonymize=True,
        params={}
    )
    assert "REDACTED" in result["Patient & Doctors Involved"]
    assert "John Doe" not in result["Patient & Doctors Involved"]


def test_no_anonymize_shows_name(extractor, medical_text):
    result = extractor._extract_medical(
        medical_text,
        people=["John Doe"],
        dates=[],
        anonymize=False,
        params={}
    )
    assert "John Doe" in result["Patient & Doctors Involved"]


# ── Legal extraction ──────────────────────────────────────────────────────────

def test_legal_finds_clauses(extractor, legal_text):
    result = extractor._scored_matches(
        extractor._LEGAL_PATTERNS["Clauses"], legal_text
    )
    values = [item["value"].lower() for item in result]
    assert any("indemnity" in v or "arbitration" in v or "termination" in v for v in values)


def test_legal_finds_indian_acts(extractor, legal_text):
    result = extractor._scored_matches(
        extractor._LEGAL_PATTERNS["Indian Acts"], legal_text
    )
    assert len(result) > 0


def test_legal_risk_high(extractor, legal_text):
    """legal_text has both termination and penalty — should be HIGH RISK at sensitivity 7."""
    result = extractor._assess_legal_risk(legal_text, risk_level=7)
    assert "HIGH RISK" in result


def test_legal_risk_low(extractor):
    result = extractor._assess_legal_risk("Standard contract with no special clauses.", risk_level=7)
    assert "LOW RISK" in result


def test_money_pattern_finds_rupee_amounts(extractor, legal_text):
    result = extractor._find_money(legal_text)
    assert len(result) > 0
    # Should find Rs. 85,00,000 and Rs. 4,25,000
    assert any("85" in m for m in result)


# ── Resume extraction ─────────────────────────────────────────────────────────

def test_resume_finds_skills(extractor, resume_text):
    result = extractor._scored_matches(
        extractor._RESUME_PATTERNS["Skills"], resume_text
    )
    values = [item["value"].lower() for item in result]
    assert any("python" in v or "docker" in v or "aws" in v for v in values)


def test_resume_finds_education(extractor, resume_text):
    result = extractor._scored_matches(
        extractor._RESUME_PATTERNS["Education"], resume_text
    )
    values = [item["value"].lower() for item in result]
    assert any("iit" in v or "b.tech" in v or "btech" in v for v in values)


def test_target_role_strong_match(extractor, resume_text):
    result = extractor._extract_resume(
        resume_text, orgs=[], dates=[], anonymize=False,
        params={"target_role": "Python Engineer"}
    )
    assert "match" in result["Target Role Alignment"].lower()


def test_target_role_no_target(extractor, resume_text):
    result = extractor._extract_resume(
        resume_text, orgs=[], dates=[], anonymize=False,
        params={"target_role": ""}
    )
    assert "No target role" in result["Target Role Alignment"]


# ── Unknown domain ────────────────────────────────────────────────────────────

def test_unknown_domain_raises(extractor, medical_text):
    with pytest.raises(ValueError, match="Unknown domain"):
        extractor.extract_features(medical_text, domain="finance")


# ── _as_scored_list ───────────────────────────────────────────────────────────

def test_as_scored_list_wraps_strings(extractor):
    items = ["Hypertension", "Diabetes"]
    result = extractor._as_scored_list(items)
    assert result == [
        {"value": "Hypertension", "score": 0.75},
        {"value": "Diabetes",     "score": 0.75},
    ]