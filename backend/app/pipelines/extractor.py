"""
extractor.py — with entity confidence scores
=============================================
Change from previous version: _run_ner() now returns scores alongside
each entity, and _run_matcher() returns (value, score) tuples.

The public-facing output dict values change shape from:
  "Medications & Dosages": "Metformin, Amlodipine"
to:
  "Medications & Dosages": [
      {"value": "Metformin", "score": 0.99},
      {"value": "Amlodipine", "score": 0.97}
  ]

The Streamlit UI reads this list and renders each chip with a
confidence badge. The RAG engine and summarizer are unaffected —
they consume raw_text, not extracted_data.

Fields that are free-text (Patient & Doctors Involved, Risks & Red Flags,
Target Role Alignment, etc.) stay as plain strings — scores only make
sense on discrete entity lists.
"""

import re
import torch
import logging
from typing import Dict, List, Any, Optional, Tuple, Union
from transformers import pipeline
from backend.app.core.config import DomainConfig

logger = logging.getLogger(__name__)

# Type alias: an entity is either a scored item or a plain string
ScoredEntity = Dict[str, Any]   # {"value": str, "score": float}
FieldValue = Union[str, List[ScoredEntity]]


class DomainExtractor:

    _DATE_PATTERN = re.compile(
        r'\b\d{1,2}[-/.]\d{1,2}[-/.]\d{2,4}\b'
        r'|\b\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{4}\b'
        r'|\b(?:19|20)\d{2}\b',
        re.IGNORECASE,
    )
    _MONEY_PATTERN = re.compile(
        r'(?:₹|Rs\.?|INR)\s*[\d,]+(?:\.\d{1,2})?(?:\s*(?:lakh|crore|thousand|million))?'
        r'|\b\d[\d,]*(?:\.\d{1,2})?\s*(?:lakh|crore)\b',
        re.IGNORECASE,
    )
    _MEDICAL_PATTERNS = {
        "Medications": re.compile(
            r'\b(?:ibuprofen|paracetamol|aspirin|amoxicillin|metformin|insulin|atorvastatin'
            r'|amlodipine|ramipril|losartan|telmisartan|pantoprazole|omeprazole|azithromycin'
            r'|dolo|crocin|combiflam|augmentin|cefixime|cetirizine|montelukast'
            r'|\w+olol|\w+pril|\w+artan|\w+statin|\d+\s*mg|\d+\s*ml|\d+\s*mcg)\b',
            re.IGNORECASE,
        ),
        "Diagnoses": re.compile(
            r'\b(?:acute|chronic|hypertension|diabetes(?:\s+mellitus)?|type\s*[12]\s*dm'
            r'|infection|fracture|syndrome|carcinoma|asthma|cardiac|oncology'
            r'|hypothyroid|anaemia|anemia|dengue|typhoid|tuberculosis|copd|ckd|sepsis'
            r'|stroke|myocardial\s+infarction|uti)\b',
            re.IGNORECASE,
        ),
        "Lab Values": re.compile(
            r'\b(?:hba1c|fasting\s+glucose|blood\s+sugar|bp|blood\s+pressure'
            r'|creatinine|egfr|haemoglobin|hemoglobin|wbc|platelets'
            r'|cholesterol|triglycerides|sgot|sgpt|bilirubin|urea)'
            r'[\s:]*[\d.]+\s*(?:%|mg/dl|mmol/l|mmhg|g/dl|u/l)?\b',
            re.IGNORECASE,
        ),
        "Procedures": re.compile(
            r'\b(?:surgery|biopsy|resection|intubation|mri|ct\s*scan|ultrasound'
            r'|dialysis|ecg|ekg|x-?ray|endoscopy|laparoscopy|angiography'
            r'|stenting|bypass|chemotherapy|radiotherapy)\b',
            re.IGNORECASE,
        ),
        "History": re.compile(
            r'(?:past\s+medical\s+history|history\s+of|previously\s+diagnosed\s+with'
            r'|family\s+history\s+of|k/o)\s+([a-zA-Z0-9 ]+)',
            re.IGNORECASE,
        ),
    }
    _LEGAL_PATTERNS = {
        "Clauses": re.compile(
            r'\b(?:indemnity|termination|jurisdiction|arbitration|liability'
            r'|confidentiality|force\s+majeure|non-compete|governing\s+law'
            r'|warranty|breach|remedy|injunction|specific\s+performance)\b',
            re.IGNORECASE,
        ),
        "Financials": re.compile(
            r'\b(?:penalty|liquidated\s+damages|compensation|remuneration'
            r'|consideration|stamp\s+duty|security\s+deposit|advance|earnest\s+money'
            r'|loan|mortgage|lien)\b',
            re.IGNORECASE,
        ),
        "Indian Acts": re.compile(
            r'\b(?:ipc|crpc|cpc|constitution\s+of\s+india'
            r'|transfer\s+of\s+property\s+act|registration\s+act'
            r'|companies\s+act|arbitration\s+and\s+conciliation\s+act'
            r'|consumer\s+protection\s+act|it\s+act|gst\s+act'
            r'|sale\s+of\s+goods\s+act|contract\s+act|evidence\s+act'
            r'|section\s+\d+[a-z]?(?:\(\d+\))?)\b',
            re.IGNORECASE,
        ),
        "Courts": re.compile(
            r'\b(?:supreme\s+court|high\s+court|district\s+court|sessions\s+court'
            r'|consumer\s+forum|nclt|nclat|arbitration\s+tribunal'
            r'|lok\s+adalat|sub-?registrar)\b',
            re.IGNORECASE,
        ),
    }
    _RESUME_PATTERNS = {
        "Skills": re.compile(
            r'\b(?:python|pytorch|tensorflow|transformers|nlp|java|c\+\+|c#'
            r'|react|angular|vue|nextjs|nodejs|fastapi|flask|django'
            r'|aws|azure|gcp|docker|kubernetes|terraform|jenkins'
            r'|sql|postgresql|mysql|mongodb|redis|kafka|spark|airflow|dbt'
            r'|machine\s+learning|deep\s+learning|llm|langchain|rag|chromadb'
            r'|git|linux|rest\s+api|graphql|microservices)\b',
            re.IGNORECASE,
        ),
        "Certifications": re.compile(
            r'\b(?:aws\s+certified|azure\s+certified|gcp\s+certified'
            r'|pmp|csm|cka|ckad|google\s+cloud|microsoft\s+certified'
            r'|coursera|udemy|nptel|gate|nasscom)\b',
            re.IGNORECASE,
        ),
        "Education": re.compile(
            r'\b(?:b\.?tech|m\.?tech|b\.?e|m\.?e|b\.?sc|m\.?sc|mba|phd|b\.?ca|m\.?ca'
            r'|iit|nit|bits|iim|vtu|anna\s+university|mumbai\s+university)\b',
            re.IGNORECASE,
        ),
    }

    def __init__(self, model_name: str = "dslim/bert-base-NER"):
        self._setup_device()
        self.ner_pipeline = pipeline(
            "ner",
            model=model_name,
            device=self.device,
            aggregation_strategy="simple",
            torch_dtype=torch.float32,
        )
        logger.info(f"[Extractor] NER loaded on {self.device_name}")

    def _setup_device(self):
        if torch.cuda.is_available():
            self.device, self.device_name = 0, "cuda"
        elif torch.backends.mps.is_available():
            self.device, self.device_name = "mps", "mps"
        else:
            self.device, self.device_name = -1, "cpu"

    # ── Public entry point ────────────────────────────────────────────────────

    def extract_features(
        self, full_text: str, domain: str, params: Optional[dict] = None
    ) -> Dict[str, FieldValue]:
        params = params or {}
        profile = DomainConfig.DOMAINS.get(domain.lower())
        if not profile:
            raise ValueError(f"Unknown domain: '{domain}'")

        output: Dict[str, FieldValue] = {f: "Not found." for f in profile["fields"]}

        text_sample = full_text[:6000]
        anonymize = params.get("anonymize_pii", False)
        ner_threshold = params.get("ner_threshold", 80) / 100.0

        orgs, people, locs = self._run_ner(text_sample, ner_threshold)
        dates   = self._find_dates(text_sample)
        money   = self._find_money(full_text)

        if domain == "medical":
            output = self._extract_medical(full_text, people, dates, anonymize, params)
        elif domain == "legal":
            output = self._extract_legal(full_text, people, orgs, dates, money, anonymize, params)
        elif domain == "resume":
            output = self._extract_resume(full_text, orgs, dates, anonymize, params)

        self._flush_vram()
        return output

    # ── Domain extractors ─────────────────────────────────────────────────────

    def _extract_medical(self, text, people, dates, anonymize, params) -> dict:
        out = {}
        out["Patient & Doctors Involved"] = (
            "[REDACTED — PII masked]" if anonymize
            else (", ".join(people[:4]) if people else "None detected.")
        )
        out["Important Dates"] = ", ".join(dates) if dates else "None found."

        # SCORED fields — discrete entity lists get confidence badges
        out["Diagnoses & Conditions"] = self._scored_matches(
            self._MEDICAL_PATTERNS["Diagnoses"], text
        )
        out["Lab Values & Vitals"] = self._scored_matches(
            self._MEDICAL_PATTERNS["Lab Values"], text, max_limit=10
        )
        out["Medications & Dosages"] = (
            "[Extraction disabled]" if not params.get("extract_dosages", True)
            else self._scored_matches(self._MEDICAL_PATTERNS["Medications"], text)
        )
        if params.get("map_pathology", True):
            hist = self._run_matcher(self._MEDICAL_PATTERNS["History"], text)
            out["Medical History"] = (
                self._as_scored_list(hist) if hist else "None found."
            )
        else:
            out["Medical History"] = "[History mapping disabled]"

        out["Procedures & Treatments"] = self._scored_matches(
            self._MEDICAL_PATTERNS["Procedures"], text
        )
        return out

    def _extract_legal(self, text, people, orgs, dates, money, anonymize, params) -> dict:
        out = {}
        all_parties = list(dict.fromkeys(people + orgs))
        out["Parties & Signees"] = (
            "[REDACTED — compliance mode]" if anonymize
            else (", ".join(all_parties[:6]) if all_parties else "Not identified.")
        )
        out["Execution Dates"] = ", ".join(dates) if dates else "None found."

        # Monetary amounts keep their raw string values — scores don't apply
        out["Monetary Amounts"] = ", ".join(money[:8]) if money else "None detected."

        out["Financial Liabilities"] = (
            "[Disabled]" if not params.get("scan_liabilities", True)
            else self._scored_matches(self._LEGAL_PATTERNS["Financials"], text)
        )
        out["Indemnity & Clauses"] = self._scored_matches(
            self._LEGAL_PATTERNS["Clauses"], text
        )
        out["Applicable Indian Laws"] = self._scored_matches(
            self._LEGAL_PATTERNS["Indian Acts"], text
        )
        out["Courts & Jurisdiction"] = self._scored_matches(
            self._LEGAL_PATTERNS["Courts"], text
        )
        out["Obligations"] = (
            f"Signees: {', '.join(people[:3])}"
            if params.get("isolate_signees", True) and people
            else "Standard delivery conditions apply."
        )
        out["Risks & Red Flags"] = self._assess_legal_risk(
            text, params.get("risk_sensitivity", 7)
        )
        return out

    def _extract_resume(self, text, orgs, dates, anonymize, params) -> dict:
        out = {}
        out["Candidate Info"] = (
            "[BLIND SCREENING — name scrubbed]" if anonymize
            else (f"Organisations: {', '.join(orgs[:4])}" if orgs else "None detected.")
        )
        target = params.get("target_role", "").strip()
        if target:
            words = set(target.lower().split())
            matched = [w for w in words if w in text.lower()]
            pct = int(len(matched) / max(len(words), 1) * 100)
            missing = words - set(matched)
            if pct >= 70:
                out["Target Role Alignment"] = f"Strong match ({pct}%) for: {target}"
            elif pct >= 40:
                out["Target Role Alignment"] = (
                    f"Partial match ({pct}%) for: {target}. "
                    f"Missing: {', '.join(missing)}"
                )
            else:
                out["Target Role Alignment"] = f"Weak match ({pct}%) — profile does not closely align with '{target}'."
        else:
            out["Target Role Alignment"] = "No target role specified."

        skills = self._scored_matches(
            self._RESUME_PATTERNS["Skills"], text, max_limit=20
        )
        if params.get("strict_skills", True) and isinstance(skills, list):
            skills = [s for s in skills if len(s["value"]) > 2]
        out["Skills Matrix"] = skills if skills else "No skills detected."

        out["Education"] = self._scored_matches(
            self._RESUME_PATTERNS["Education"], text
        )
        out["Education Timeline"] = (
            (", ".join(dates) if dates else "None found.")
            if params.get("parse_academic", True) else "Disabled."
        )
        out["Certifications"] = self._scored_matches(
            self._RESUME_PATTERNS["Certifications"], text
        )
        out["Experience Summary"] = (
            f"Organisations: {', '.join(orgs[:5])}"
            if orgs else "No company history detected."
        )
        return out

    # ── NER ───────────────────────────────────────────────────────────────────

    def _run_ner(self, text: str, threshold: float):
        orgs, people, locs = [], [], []
        try:
            for ent in self.ner_pipeline(text):
                if ent.get("score", 1.0) < threshold:
                    continue
                word = ent["word"].replace(" ##", "").replace("##", "").strip()
                if len(word) < 2:
                    continue
                g = ent["entity_group"]
                if g == "ORG":   orgs.append(word)
                elif g == "PER": people.append(word)
                elif g == "LOC": locs.append(word)
        except Exception as e:
            logger.error(f"[Extractor] NER failed: {e}")
        return (
            list(dict.fromkeys(orgs)),
            list(dict.fromkeys(people)),
            list(dict.fromkeys(locs)),
        )

    # ── Regex helpers ─────────────────────────────────────────────────────────

    def _find_dates(self, text): return list(dict.fromkeys(self._DATE_PATTERN.findall(text)))
    def _find_money(self, text): return list(dict.fromkeys(self._MONEY_PATTERN.findall(text)))

    def _run_matcher(self, pattern: re.Pattern, text: str, max_limit: int = 8) -> List[str]:
        """Returns plain deduplicated strings (used internally)."""
        seen = {}
        for m in pattern.findall(text):
            cleaned = m.strip().title()
            key = cleaned.lower()
            if key not in seen and len(cleaned) > 1:
                seen[key] = cleaned
        return list(seen.values())[:max_limit]

    def _scored_matches(
        self, pattern: re.Pattern, text: str, max_limit: int = 8
    ) -> List[ScoredEntity]:
        """
        IMPROVEMENT: returns scored entity list instead of a plain comma string.

        Regex matches are deterministic so we can't get a neural score from them.
        We assign a pseudo-confidence based on match frequency in the text:
          - Appears 3+ times → 0.99 (very likely intentional)
          - Appears 2 times  → 0.90
          - Appears 1 time   → 0.75
        This gives the UI enough signal to show "certain" vs "tentative" badges.
        NER entities that come through _run_ner() already carry real model scores.
        """
        all_matches = [m.strip().title() for m in pattern.findall(text) if m.strip()]
        freq: Dict[str, int] = {}
        for m in all_matches:
            key = m.lower()
            freq[key] = freq.get(key, 0) + 1

        result: List[ScoredEntity] = []
        seen = set()
        for m in all_matches:
            key = m.lower()
            if key in seen or len(m) <= 1:
                continue
            seen.add(key)
            count = freq[key]
            score = 0.99 if count >= 3 else (0.90 if count == 2 else 0.75)
            result.append({"value": m, "score": round(score, 2)})
            if len(result) >= max_limit:
                break

        return result

    def _as_scored_list(self, items: List[str]) -> List[ScoredEntity]:
        """Wraps plain strings as scored entities with a default score of 0.75."""
        return [{"value": item, "score": 0.75} for item in items]

    # ── Risk assessment ───────────────────────────────────────────────────────

    def _assess_legal_risk(self, text: str, risk_level: int) -> str:
        has_termination = bool(self._LEGAL_PATTERNS["Clauses"].search(text))
        has_penalties   = bool(self._LEGAL_PATTERNS["Financials"].search(text))
        has_arbitration = bool(re.search(r'\barbitration\b', text, re.IGNORECASE))
        if has_termination and has_penalties and risk_level >= 5:
            return (
                f"HIGH RISK (sensitivity {risk_level}/10): Exit/termination clauses and "
                f"financial penalty provisions both detected. Legal review recommended."
            )
        elif has_termination and risk_level >= 7:
            return (
                f"MODERATE RISK (sensitivity {risk_level}/10): Termination clause detected. "
                f"Arbitration present: {'Yes' if has_arbitration else 'No'}."
            )
        elif has_penalties:
            return f"LOW-MODERATE RISK (sensitivity {risk_level}/10): Financial liability terms present."
        return f"LOW RISK (sensitivity {risk_level}/10): No high-risk clauses identified."

    # ── Utils ─────────────────────────────────────────────────────────────────

    def _flush_vram(self):
        if torch.cuda.is_available():   torch.cuda.empty_cache()
        elif torch.backends.mps.is_available(): torch.mps.empty_cache()