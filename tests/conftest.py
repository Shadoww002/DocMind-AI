"""
conftest.py — shared fixtures for all tests
pytest loads this automatically before any test file runs.
"""
import pytest


# ── Sample text fixtures ───────────────────────────────────────────────────────
# These are plain strings that mimic what your parser would extract from a PDF.
# No actual PDF needed — we test the logic, not the PDF reading.

@pytest.fixture
def medical_text():
    return (
        "Patient John Doe, 45 years old, diagnosed with Type 2 Diabetes Mellitus. "
        "HbA1c recorded at 8.4%. Prescribed Metformin 1000mg twice daily. "
        "Blood pressure 158/96 mmHg — Stage II Hypertension. "
        "History of Hypertension. Fasting glucose 184 mg/dl. "
        "ECG shows normal sinus rhythm. Follow-up in 4 weeks. "
        "Metformin 1000mg to continue. Amlodipine 5mg added. "
        "Past medical history of cardiac issues. Previously diagnosed with anaemia."
    )

@pytest.fixture
def legal_text():
    return (
        "This Sale Deed is executed under the Registration Act 1908. "
        "Party A (Vendor) agrees to transfer property to Party B (Purchaser) "
        "for a consideration of Rs. 85,00,000. Stamp duty of Rs. 4,25,000 paid. "
        "The agreement includes an indemnity clause and arbitration provision. "
        "Jurisdiction: Mumbai High Court. Section 54 of Transfer of Property Act applies. "
        "Termination clause included. Penalty of Rs. 5,00,000 for breach. "
        "The property is free from all encumbrances and mortgage."
    )

@pytest.fixture
def resume_text():
    return (
        "Priya Sharma — Senior Software Engineer. "
        "Skills: Python, FastAPI, Docker, AWS, PostgreSQL, Machine Learning, LangChain. "
        "B.Tech Computer Science, IIT Bombay, 2019. CGPA 8.7. "
        "Experience: 5 years at TechCorp as Backend Engineer. "
        "AWS Certified Solutions Architect. "
        "Reduced API latency by 60% using Redis caching. "
        "Docker, Kubernetes, Git, Linux used daily."
    )

@pytest.fixture
def sample_chunks():
    """Mimics what DocumentParser.create_chunks() returns."""
    return [
        {"text": "Patient has Type 2 Diabetes Mellitus with HbA1c 8.4%.", "metadata": {"page": 1}},
        {"text": "Prescribed Metformin 1000mg twice daily. Blood pressure 158/96 mmHg.", "metadata": {"page": 1}},
        {"text": "Follow-up scheduled in 4 weeks. ECG normal sinus rhythm.", "metadata": {"page": 2}},
    ]