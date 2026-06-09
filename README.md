<div align="center">

# 🧠 DocMind AI

### Multi-Domain Document Intelligence Platform

**Medical Records · Indian Legal Contracts · Professional Resumes**

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat&logo=python&logoColor=white)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.111-009688?style=flat&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.35-FF4B4B?style=flat&logo=streamlit&logoColor=white)](https://streamlit.io)
[![ChromaDB](https://img.shields.io/badge/ChromaDB-0.5-orange?style=flat)](https://trychroma.com)
[![License](https://img.shields.io/badge/License-MIT-green?style=flat)](LICENSE)

---



https://github.com/user-attachments/assets/3f0f33a7-12a9-4000-85bc-dd9face29b4b



---

</div>

## Overview

DocMind AI is a fully offline, CPU-compatible document intelligence system that processes PDFs across three specialist domains. Upload a document, select a domain, and get a structured executive summary, entity extraction with confidence scores, and a ChromaDB-backed RAG chat interface — all running locally without any API keys or cloud services.

Built with a production-aware architecture: isolated vector collections per domain, map-reduce summarization pipeline, extractive Q&A with real confidence scores, and async FastAPI endpoints.

---

## Features

| Feature | Description |
|---|---|
| 🏥 Medical Analysis | Diagnoses, medications, lab values, procedures, patient history |
| ⚖️ Indian Legal Analysis | Parties, clauses, monetary amounts, applicable acts, risk flags |
| 📄 Resume Analysis | Skills matrix, education, experience, target role alignment |
| 📝 Domain Summaries | Map-reduce pipeline with domain-specific prompts |
| 🏷️ Confidence Scores | Entity chips with frequency-based confidence badges |
| 💬 RAG Chat | ChromaDB vector search + roberta-base-squad2 extractive Q&A |
| 🔒 PII Controls | HIPAA masking, blind screening, compliance redaction per domain |
| ⬇️ Export | Full session export as structured JSON |

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        Streamlit Frontend                        │
│          Upload · Parameters · Summary · Chat · Export           │
└──────────────────────────┬──────────────────────────────────────┘
                           │ HTTP (FastAPI)
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│                      FastAPI Backend                             │
│                                                                  │
│  POST /api/v1/process          POST /api/v1/query               │
│  DELETE /api/v1/document       GET  /health                     │
└────┬──────────────┬──────────────────┬───────────────┬──────────┘
     │              │                  │               │
     ▼              ▼                  ▼               ▼
┌─────────┐  ┌───────────┐  ┌──────────────┐  ┌────────────┐
│  Parser  │  │Summarizer │  │  Extractor   │  │ RAG Engine │
│          │  │           │  │              │  │            │
│ pypdf    │  │flan-t5    │  │bert-base-NER │  │roberta     │
│ sentence │  │map-reduce │  │+ regex       │  │-squad2     │
│ chunking │  │pipeline   │  │patterns      │  │extractive  │
└────┬─────┘  └─────┬─────┘  └──────┬───────┘  └─────┬──────┘
     │              │               │                 │
     └──────────────┴───────────────┘                 │
                    │                                  │
                    ▼                                  ▼
         ┌──────────────────┐              ┌───────────────────┐
         │    ChromaDB      │◄─────────────│  VectorStore      │
         │                  │              │  all-MiniLM-L6-v2 │
         │ docmind_medical  │              │  cosine similarity│
         │ docmind_legal    │              └───────────────────┘
         │ docmind_resume   │
         └──────────────────┘
```

### Pipeline Flow

```
PDF Upload
    │
    ▼
DocumentParser          — sentence-boundary chunks, artifact cleaning
    │
    ├──► VectorStoreManager   — encode with all-MiniLM-L6-v2, upsert to ChromaDB
    │
    ├──► DomainSummarizer     — map (extract key points) → reduce (fluent summary)
    │                           flan-t5-base, domain-specific prompts
    │
    └──► DomainExtractor      — NER (bert-base-NER) + regex patterns
                                returns scored entity lists per field

Query
    │
    ▼
VectorStoreManager      — cosine similarity search, scoped by doc_id
    │
    ▼
DomainRAGEngine         — deepset/roberta-base-squad2 extractive QA
                          returns answer + confidence + page citation
```

---

## Project Structure

```
ai_document_intelligence/
├── backend/
│   └── app/
│       ├── main.py                  # FastAPI app, lifespan, CORS, middleware
│       ├── api/
│       │   └── endpoints.py         # /process, /query, /document routes
│       ├── core/
│       │   └── config.py            # Domain profiles, prompts, field schemas
│       ├── services/
│       │   ├── parser.py            # PDF parser, sentence-boundary chunker
│       │   └── vector_db.py         # ChromaDB interface, per-domain collections
│       └── pipelines/
│           ├── summarizer.py        # Map-reduce summarization (flan-t5-base)
│           ├── extractor.py         # NER + regex entity extraction with scores
│           └── rag_engine.py        # Extractive Q&A (roberta-base-squad2)
├── frontend/
│   └── app.py                       # Streamlit UI
├── tests/
│   ├── conftest.py                  # Shared fixtures
│   ├── test_parser.py               # 11 tests
│   ├── test_extractor.py            # 20 tests
│   ├── test_rag_engine.py           # 14 tests
│   └── test_config.py               # 20 tests
├── data/
│   └── chroma_db/                   # ChromaDB storage (gitignored)
├── .env.example
├── requirements.txt
├── pytest.ini
└── README.md
```

---

## Models Used

| Model | Size | Task |
|---|---|---|
| `google/flan-t5-base` | ~250MB | Map-reduce summarization |
| `dslim/bert-base-NER` | ~400MB | Named entity recognition |
| `deepset/roberta-base-squad2` | ~130MB | Extractive Q&A (RAG) |
| `all-MiniLM-L6-v2` | ~90MB | Sentence embeddings (ChromaDB) |

All models run on CPU. Total RAM required: ~870MB for models + ~300MB overhead.

---

## Getting Started

### Prerequisites

- Python 3.10+
- 4GB RAM minimum (8GB recommended)
- No GPU required

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/Shadoww002/DocMind-AI.git
cd docmind-ai

# 2. Create and activate virtual environment
python -m venv venv

# Windows
venv\Scripts\activate
# macOS / Linux
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Set up environment variables
cp .env.example .env
# Edit .env if you want to change the ChromaDB storage path
```

### Running the App

Open **two terminals**:

**Terminal 1 — Backend**
```bash
uvicorn backend.app.main:app --reload --host 0.0.0.0 --port 8000
```

**Terminal 2 — Frontend**
```bash
streamlit run frontend/app.py
```

Then open **http://localhost:8501** in your browser.

### Running Tests

```bash
pytest
```

Expected output:
```
tests/test_config.py      ......................   20 passed
tests/test_extractor.py   ....................     20 passed
tests/test_parser.py      ...........             11 passed
tests/test_rag_engine.py  ..............           14 passed
65 passed in ~4s
```

---

## Domain Capabilities

### 🏥 Medical Records

| Extracted Field | Example Output |
|---|---|
| Diagnoses & Conditions | Type 2 DM `99%`, Hypertension `90%` |
| Lab Values & Vitals | HbA1c 8.4%, BP 158/96 mmHg |
| Medications & Dosages | Metformin 1000mg `99%`, Amlodipine 5mg `75%` |
| Procedures & Treatments | ECG `90%`, Ultrasound `75%` |
| Medical History | History of cardiac issues |

### ⚖️ Indian Legal Documents

| Extracted Field | Example Output |
|---|---|
| Applicable Indian Laws | Registration Act 1908 `99%`, IPC `90%` |
| Monetary Amounts | ₹85,00,000, Rs. 4,25,000 stamp duty |
| Courts & Jurisdiction | Mumbai High Court `90%` |
| Risks & Red Flags | HIGH RISK: termination + penalty clauses detected |

### 📄 Resumes & CVs

| Extracted Field | Example Output |
|---|---|
| Skills Matrix | Python `99%`, Docker `90%`, AWS `90%` |
| Education | IIT Bombay `99%`, B.Tech `99%` |
| Target Role Alignment | Strong match (85%) for: ML Engineer |
| Certifications | AWS Certified `75%` |

---

## Configuration

All domain parameters are configurable in `backend/app/core/config.py` or via environment variables:

```env
# .env
VECTOR_DB_DIR=./data/chroma_db
ENV=development
ALLOWED_ORIGINS=http://localhost:8501

# Override models per domain (optional)
MEDICAL_LLM_MODEL=google/flan-t5-small   # use small for faster CPU
LEGAL_LLM_MODEL=google/flan-t5-base
RESUME_LLM_MODEL=google/flan-t5-base
```

---

## API Reference

### `POST /api/v1/process`
Upload and process a PDF document.

```bash
curl -X POST http://localhost:8000/api/v1/process \
  -F "file=@document.pdf" \
  -F "domain=medical" \
  -F 'params={"extract_dosages": true, "anonymize_pii": false}'
```

### `POST /api/v1/query`
Ask a question about a processed document.

```bash
curl -X POST http://localhost:8000/api/v1/query \
  -H "Content-Type: application/json" \
  -d '{"document_id": "doc_abc123", "domain": "medical", "question": "What medications were prescribed?"}'
```

### `GET /health`
Check backend status and collection stats.

```bash
curl http://localhost:8000/health
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| Backend API | FastAPI + Uvicorn |
| Frontend | Streamlit |
| Vector Database | ChromaDB (persistent, per-domain collections) |
| Embeddings | sentence-transformers (all-MiniLM-L6-v2) |
| Summarization | HuggingFace Transformers (flan-t5-base) |
| NER | HuggingFace Transformers (bert-base-NER) |
| Q&A | HuggingFace Transformers (roberta-base-squad2) |
| PDF Parsing | pypdf |
| Testing | pytest |

---

## License

MIT License — see [LICENSE](LICENSE) for details.

---

<div align="center">

Built with 🧠 by [Sanjay](https://github.com/Shadoww002)

</div>
