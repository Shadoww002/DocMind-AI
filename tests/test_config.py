"""
test_config.py — tests for DomainConfig

What we're testing:
  - All three domains exist
  - Every domain has the required keys
  - Field lists match what the extractor actually returns
  - get_domain_property returns correct values and defaults
  - is_valid_domain works correctly
  - domain_names returns all three
  - Unknown domain does not silently fall back to 'legal'

No models, no files needed — pure config logic.
"""
import pytest
from backend.app.core.config import DomainConfig


REQUIRED_DOMAIN_KEYS = [
    "name", "ner_model", "llm_model", "fields",
    "chunk_size", "chunk_overlap", "max_chunks",
    "top_k_retrieval", "system_prompt",
]

EXPECTED_DOMAINS = ["medical", "legal", "resume"]


# ── Domain existence ──────────────────────────────────────────────────────────

def test_all_three_domains_exist():
    for domain in EXPECTED_DOMAINS:
        assert domain in DomainConfig.DOMAINS


def test_no_extra_domains():
    assert set(DomainConfig.domain_names()) == set(EXPECTED_DOMAINS)


# ── Required keys ─────────────────────────────────────────────────────────────

@pytest.mark.parametrize("domain", EXPECTED_DOMAINS)
def test_domain_has_all_required_keys(domain):
    profile = DomainConfig.DOMAINS[domain]
    for key in REQUIRED_DOMAIN_KEYS:
        assert key in profile, f"Domain '{domain}' missing key '{key}'"


# ── Field list non-empty ───────────────────────────────────────────────────────

@pytest.mark.parametrize("domain", EXPECTED_DOMAINS)
def test_domain_fields_not_empty(domain):
    fields = DomainConfig.all_fields(domain)
    assert len(fields) > 0


# ── Medical fields match extractor keys ──────────────────────────────────────

def test_medical_fields_match_extractor():
    expected = [
        "Patient & Doctors Involved", "Important Dates", "Diagnoses & Conditions",
        "Lab Values & Vitals", "Medications & Dosages", "Medical History", "Procedures & Treatments"
    ]
    assert DomainConfig.all_fields("medical") == expected


def test_legal_fields_match_extractor():
    expected = [
        "Parties & Signees", "Execution Dates", "Monetary Amounts", "Financial Liabilities",
        "Indemnity & Clauses", "Applicable Indian Laws", "Courts & Jurisdiction",
        "Obligations", "Risks & Red Flags"
    ]
    assert DomainConfig.all_fields("legal") == expected


def test_resume_fields_order_correct():
    """Certifications must come before Experience Summary — matches extractor output order."""
    fields = DomainConfig.all_fields("resume")
    cert_idx = fields.index("Certifications")
    exp_idx  = fields.index("Experience Summary")
    assert cert_idx < exp_idx


# ── Hyperparameters are sensible ──────────────────────────────────────────────

@pytest.mark.parametrize("domain", EXPECTED_DOMAINS)
def test_chunk_overlap_less_than_chunk_size(domain):
    size    = DomainConfig.get_domain_property(domain, "chunk_size")
    overlap = DomainConfig.get_domain_property(domain, "chunk_overlap")
    assert overlap < size, f"{domain}: overlap ({overlap}) must be < chunk_size ({size})"


@pytest.mark.parametrize("domain", EXPECTED_DOMAINS)
def test_top_k_retrieval_is_positive(domain):
    top_k = DomainConfig.get_domain_property(domain, "top_k_retrieval")
    assert top_k >= 1


# ── get_domain_property ───────────────────────────────────────────────────────

def test_get_domain_property_returns_correct_value():
    result = DomainConfig.get_domain_property("medical", "chunk_size")
    assert result == 800


def test_get_domain_property_returns_default_for_missing_key():
    result = DomainConfig.get_domain_property("medical", "nonexistent_key", default=42)
    assert result == 42


def test_get_domain_property_unknown_domain_returns_default():
    """Must NOT fall back to legal — must return the default."""
    result = DomainConfig.get_domain_property("finance", "chunk_size", default=999)
    assert result == 999


def test_get_domain_property_unknown_domain_not_legal():
    """Regression test: unknown domain must not silently return legal config."""
    legal_chunk_size = DomainConfig.get_domain_property("legal", "chunk_size")
    unknown_result   = DomainConfig.get_domain_property("finance", "chunk_size", default=0)
    assert unknown_result != legal_chunk_size


# ── is_valid_domain ───────────────────────────────────────────────────────────

@pytest.mark.parametrize("domain", EXPECTED_DOMAINS)
def test_valid_domains_return_true(domain):
    assert DomainConfig.is_valid_domain(domain) is True


def test_invalid_domain_returns_false():
    assert DomainConfig.is_valid_domain("finance") is False


def test_is_valid_domain_case_insensitive():
    assert DomainConfig.is_valid_domain("MEDICAL") is True
    assert DomainConfig.is_valid_domain("Legal") is True


# ── get_system_prompt ─────────────────────────────────────────────────────────

@pytest.mark.parametrize("domain", EXPECTED_DOMAINS)
def test_system_prompt_not_empty(domain):
    prompt = DomainConfig.get_system_prompt(domain)
    assert isinstance(prompt, str)
    assert len(prompt) > 20


def test_unknown_domain_gets_fallback_prompt():
    prompt = DomainConfig.get_system_prompt("finance")
    assert isinstance(prompt, str)
    assert len(prompt) > 0