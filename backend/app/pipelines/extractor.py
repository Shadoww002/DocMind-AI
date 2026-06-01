import re
import torch
import logging
from typing import Dict, List, Any, Optional
from transformers import pipeline
from backend.app.core.config import DomainConfig

logger = logging.getLogger(__name__)


class DomainExtractor:
    """
    Domain-aware entity extractor combining transformer NER with
    pre-compiled regex pattern matrices.

    Improvements over v1:
    - Regex expanded for Indian legal acts, rupee amounts, IPC sections, court names
    - Medical: added lab values (HbA1c, BP, eGFR), generic drug suffixes
    - Resume: added more modern skills (FastAPI, LangChain, Airflow, etc.)
    - NER text window raised from 4000 → 6000 chars (bert-base handles it fine)
    - _run_matcher deduplication is now case-insensitive (avoids "Python"/"python" duplicates)
    - Risk flag logic is deterministic text, not emoji strings (safer for HTML rendering)
    - All output values are plain strings, never left as the placeholder sentinel
    - _flush_vram() extracted as a reusable helper
    - Model loaded with torch.float32 explicitly to avoid CPU half-precision crash
    """

    # ── Date patterns (DD-MM-YYYY, D Month YYYY, YYYY) ──────────────────────
    _DATE_PATTERN = re.compile(
        r'\b\d{1,2}[-/.]\d{1,2}[-/.]\d{2,4}\b'
        r'|\b\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{4}\b'
        r'|\b(?:19|20)\d{2}\b',
        re.IGNORECASE,
    )

    # ── Rupee / monetary amounts (₹, Rs., INR) ──────────────────────────────
    _MONEY_PATTERN = re.compile(
        r'(?:₹|Rs\.?|INR)\s*[\d,]+(?:\.\d{1,2})?(?:\s*(?:lakh|crore|thousand|million))?'
        r'|\b\d[\d,]*(?:\.\d{1,2})?\s*(?:lakh|crore)\b',
        re.IGNORECASE,
    )

    # ── Medical ─────────────────────────────────────────────────────────────
    _MEDICAL_PATTERNS = {
        # IMPROVEMENT: added common Indian drug names + generic suffixes (-olol, -pril, -artan, -statin)
        "Medications": re.compile(
            r'\b(?:ibuprofen|paracetamol|aspirin|amoxicillin|metformin|insulin|atorvastatin'
            r'|amlodipine|ramipril|losartan|telmisartan|pantoprazole|omeprazole|azithromycin'
            r'|dolo|crocin|combiflam|pan-?d|augmentin|cefixime|cetirizine|montelukast'
            r'|\w+olol|\w+pril|\w+artan|\w+statin'
            r'|\d+\s*mg|\d+\s*ml|\d+\s*mcg|\d+\s*units)\b',
            re.IGNORECASE,
        ),
        "Diagnoses": re.compile(
            r'\b(?:acute|chronic|hypertension|diabetes(?:\s+mellitus)?|type\s*[12]\s*dm'
            r'|infection|fracture|pain|syndrome|carcinoma|asthma|cardiac|oncology'
            r'|hypothyroid|anaemia|anemia|dengue|typhoid|malaria|tuberculosis|tb'
            r'|copd|ckd|arf|sepsis|stroke|mi|myocardial\s+infarction|uti)\b',
            re.IGNORECASE,
        ),
        # IMPROVEMENT: lab values with numeric capture (HbA1c 8.4%, BP 130/80)
        "Lab Values": re.compile(
            r'\b(?:hba1c|fasting\s+glucose|blood\s+sugar|bp|blood\s+pressure'
            r'|creatinine|egfr|haemoglobin|hemoglobin|wbc|rbc|platelets'
            r'|cholesterol|triglycerides|sgot|sgpt|bilirubin|urea|sodium|potassium)'
            r'[\s:]*[\d.]+\s*(?:%|mg/dl|mmol/l|mmhg|g/dl|u/l|meq/l)?\b',
            re.IGNORECASE,
        ),
        "Procedures": re.compile(
            r'\b(?:surgery|biopsy|resection|intubation|mri|ct\s*scan|ultrasound'
            r'|dialysis|graft|sutures|ecg|ekg|x-?ray|endoscopy|laparoscopy'
            r'|angiography|stenting|bypass|chemotherapy|radiotherapy|ivf)\b',
            re.IGNORECASE,
        ),
        "History": re.compile(
            r'(?:past\s+medical\s+history|history\s+of|prior\s+to'
            r'|previously\s+diagnosed\s+with|family\s+history\s+of|k/o|k\.o\.)\s+([a-zA-Z0-9 ]+)',
            re.IGNORECASE,
        ),
    }

    # ── Indian Legal ─────────────────────────────────────────────────────────
    _LEGAL_PATTERNS = {
        "Clauses": re.compile(
            r'\b(?:indemnity|termination|jurisdiction|arbitration|liability'
            r'|confidentiality|force\s+majeure|non-compete|governing\s+law'
            r'|warranty|representation|covenant|limitation\s+of\s+liability'
            r'|dispute\s+resolution|breach|remedy|injunction|specific\s+performance)\b',
            re.IGNORECASE,
        ),
        "Financials": re.compile(
            r'\b(?:penalty|liquidated\s+damages|compensation|remuneration'
            r'|consideration|invoice|disbursement|stipend|payment|stamp\s+duty'
            r'|security\s+deposit|advance|earnest\s+money|loan|mortgage|lien)\b',
            re.IGNORECASE,
        ),
        # IMPROVEMENT: Indian-specific acts, IPC sections, court names
        "Indian Acts": re.compile(
            r'\b(?:ipc|crpc|cpc|constitution\s+of\s+india'
            r'|transfer\s+of\s+property\s+act|registration\s+act'
            r'|companies\s+act|arbitration\s+and\s+conciliation\s+act'
            r'|consumer\s+protection\s+act|it\s+act|gst\s+act'
            r'|sale\s+of\s+goods\s+act|contract\s+act|evidence\s+act'
            r'|section\s+\d+[a-z]?(?:\(\d+\))?(?:\s+of\s+the\s+\w[\w\s]+act)?)\b',
            re.IGNORECASE,
        ),
        # IMPROVEMENT: capture court/tribunal names
        "Courts & Jurisdiction": re.compile(
            r'\b(?:supreme\s+court|high\s+court|district\s+court|sessions\s+court'
            r'|consumer\s+forum|nclt|nclat|ncdrc|arbitration\s+tribunal'
            r'|lok\s+adalat|sub-?registrar|registrar)\b',
            re.IGNORECASE,
        ),
    }

    # ── Resume ───────────────────────────────────────────────────────────────
    _RESUME_PATTERNS = {
        # IMPROVEMENT: added modern stack (FastAPI, LangChain, dbt, Airflow, etc.)
        "Skills": re.compile(
            r'\b(?:python|pytorch|tensorflow|transformers|nlp|java|c\+\+|c#'
            r'|react|angular|vue|nextjs|nodejs|node\.js|fastapi|flask|django'
            r'|aws|azure|gcp|docker|kubernetes|terraform|ci/cd|jenkins|github\s+actions'
            r'|sql|postgresql|mysql|mongodb|redis|elasticsearch|kafka|spark|hadoop|airflow|dbt'
            r'|machine\s+learning|deep\s+learning|llm|langchain|rag|vector\s+db|chromadb'
            r'|git|linux|bash|rest\s+api|graphql|microservices|agile|scrum)\b',
            re.IGNORECASE,
        ),
        "Certifications": re.compile(
            r'\b(?:aws\s+certified|azure\s+certified|gcp\s+certified'
            r'|pmp|csm|cka|ckad|google\s+cloud|microsoft\s+certified'
            r'|coursera|udemy|nptel|gate|infosys\s+certified|nasscom)\b',
            re.IGNORECASE,
        ),
        # IMPROVEMENT: captures degree names
        "Education": re.compile(
            r'\b(?:b\.?tech|m\.?tech|b\.?e|m\.?e|b\.?sc|m\.?sc|mba|phd|b\.?ca|m\.?ca'
            r'|iit|nit|bits|iim|vtu|anna\s+university|mumbai\s+university|du|delhi\s+university)\b',
            re.IGNORECASE,
        ),
    }

    def __init__(self, model_name: str = "dslim/bert-base-NER"):
        self._setup_device()
        try:
            # IMPROVEMENT: explicit float32 prevents CPU half-precision crash
            self.ner_pipeline = pipeline(
                "ner",
                model=model_name,
                device=self.device,
                aggregation_strategy="simple",
                torch_dtype=torch.float32,
            )
            logger.info(f"[Extractor] NER model loaded on {self.device_name}")
        except Exception as e:
            logger.error(f"[Extractor] NER model load failed: {e}")
            raise

    def _setup_device(self):
        if torch.cuda.is_available():
            self.device = 0
            self.device_name = "cuda"
        elif torch.backends.mps.is_available():
            self.device = "mps"
            self.device_name = "mps"
        else:
            self.device = -1
            self.device_name = "cpu"

    # ── Public entry point ───────────────────────────────────────────────────

    def extract_features(
        self, full_text: str, domain: str, params: Optional[dict] = None
    ) -> Dict[str, Any]:
        params = params or {}

        profile = DomainConfig.DOMAINS.get(domain.lower())
        if not profile:
            raise ValueError(f"Unknown domain: '{domain}'")

        output_data = {field: "Not found." for field in profile["fields"]}

        # IMPROVEMENT: raised window to 6000 chars; bert-base handles this fine
        text_sample = full_text[:6000]
        anonymize = params.get("anonymize_pii", False)
        # IMPROVEMENT: default threshold lowered to 0.80 — 0.85 missed too many entities on CPU
        ner_threshold = params.get("ner_threshold", 80) / 100.0

        orgs, people, locs = self._run_ner(text_sample, ner_threshold)
        detected_dates = self._find_dates(text_sample)
        detected_money = self._find_money(full_text)

        if domain == "medical":
            output_data = self._extract_medical(full_text, text_sample, people, detected_dates, anonymize, params)
        elif domain == "legal":
            output_data = self._extract_legal(full_text, text_sample, people, orgs, locs, detected_dates, detected_money, anonymize, params)
        elif domain == "resume":
            output_data = self._extract_resume(full_text, orgs, detected_dates, anonymize, params)

        self._flush_vram()
        return output_data

    # ── Domain extractors ────────────────────────────────────────────────────

    def _extract_medical(self, full_text, text_sample, people, dates, anonymize, params) -> dict:
        out = {}

        out["Patient & Doctors Involved"] = (
            "[REDACTED — PII masked]"
            if anonymize
            else (", ".join(people[:4]) if people else "No named individuals detected.")
        )
        out["Important Dates"] = ", ".join(dates) if dates else "None found."
        out["Diagnoses & Conditions"] = (
            self._join(self._run_matcher(self._MEDICAL_PATTERNS["Diagnoses"], full_text))
            or "None identified."
        )

        # Lab values — new field
        lab_hits = self._run_matcher(self._MEDICAL_PATTERNS["Lab Values"], full_text, max_limit=10)
        out["Lab Values & Vitals"] = self._join(lab_hits) or "None extracted."

        out["Medications & Dosages"] = (
            "[Extraction disabled]"
            if not params.get("extract_dosages", True)
            else (self._join(self._run_matcher(self._MEDICAL_PATTERNS["Medications"], full_text)) or "None found.")
        )

        if params.get("map_pathology", True):
            hist = self._run_matcher(self._MEDICAL_PATTERNS["History"], full_text)
            out["Medical History"] = ("Indicators: " + self._join(hist)) if hist else "No history indicators found."
        else:
            out["Medical History"] = "[History mapping disabled]"

        out["Procedures & Treatments"] = (
            self._join(self._run_matcher(self._MEDICAL_PATTERNS["Procedures"], full_text))
            or "None documented."
        )
        return out

    def _extract_legal(self, full_text, text_sample, people, orgs, locs, dates, money, anonymize, params) -> dict:
        out = {}

        all_parties = list(dict.fromkeys(people + orgs))  # preserves order, deduplicates
        out["Parties & Signees"] = (
            "[REDACTED — compliance mode]"
            if anonymize
            else (", ".join(all_parties[:6]) if all_parties else "Parties not identified.")
        )

        out["Execution Dates"] = ", ".join(dates) if dates else "No dates found."

        # IMPROVEMENT: monetary amounts extracted by regex, not NER
        out["Monetary Amounts"] = self._join(money[:8]) if money else "None detected."

        out["Financial Liabilities"] = (
            "[Disabled]"
            if not params.get("scan_liabilities", True)
            else (self._join(self._run_matcher(self._LEGAL_PATTERNS["Financials"], full_text)) or "None found.")
        )

        out["Indemnity & Clauses"] = (
            self._join(self._run_matcher(self._LEGAL_PATTERNS["Clauses"], full_text))
            or "No standard clauses identified."
        )

        # IMPROVEMENT: Indian acts — new dedicated field
        out["Applicable Indian Laws"] = (
            self._join(self._run_matcher(self._LEGAL_PATTERNS["Indian Acts"], full_text))
            or "No specific acts referenced."
        )

        out["Courts & Jurisdiction"] = (
            self._join(self._run_matcher(self._LEGAL_PATTERNS["Courts & Jurisdiction"], full_text))
            or "No court or jurisdiction mentioned."
        )

        if params.get("isolate_signees", True) and people:
            out["Obligations"] = f"Signees identified: {', '.join(people[:3])}"
        else:
            out["Obligations"] = "Standard delivery conditions apply."

        # IMPROVEMENT: risk flag is clean text — no emoji in data layer
        out["Risks & Red Flags"] = self._assess_legal_risk(full_text, params.get("risk_sensitivity", 7))

        return out

    def _extract_resume(self, full_text, orgs, dates, anonymize, params) -> dict:
        out = {}

        out["Candidate Info"] = (
            "[BLIND SCREENING — name and location scrubbed]"
            if anonymize
            else (f"Institutional connections found: {', '.join(orgs[:4])}" if orgs else "No organisations detected.")
        )

        target = params.get("target_role", "").strip()
        if target:
            # IMPROVEMENT: keyword overlap score instead of simple substring match
            target_words = set(target.lower().split())
            text_lower = full_text.lower()
            matched = [w for w in target_words if w in text_lower]
            overlap = len(matched) / max(len(target_words), 1)
            if overlap >= 0.7:
                out["Target Role Alignment"] = f"Strong match ({int(overlap*100)}%) for role: {target}"
            elif overlap >= 0.4:
                out["Target Role Alignment"] = f"Partial match ({int(overlap*100)}%) for role: {target}. Missing: {', '.join(set(target_words)-set(matched))}"
            else:
                out["Target Role Alignment"] = f"Weak match ({int(overlap*100)}%) — profile does not closely align with '{target}'."
        else:
            out["Target Role Alignment"] = "No target role specified."

        skills = self._run_matcher(self._RESUME_PATTERNS["Skills"], full_text, max_limit=20)
        if params.get("strict_skills", True):
            skills = [s for s in skills if len(s) > 2]
        out["Skills Matrix"] = self._join(skills) if skills else "No technical skills detected."

        # IMPROVEMENT: dedicated education field
        edu_hits = self._run_matcher(self._RESUME_PATTERNS["Education"], full_text)
        out["Education"] = self._join(edu_hits) if edu_hits else "No education details found."

        out["Education Timeline"] = (
            (", ".join(dates) if dates else "No dates found.")
            if params.get("parse_academic", True)
            else "Academic parsing disabled."
        )
        out["Certifications"] = (
            self._join(self._run_matcher(self._RESUME_PATTERNS["Certifications"], full_text))
            or "None listed."
        )
        out["Experience Summary"] = (
            f"Organisations: {', '.join(orgs[:5])}"
            if orgs
            else "No company history detected."
        )
        return out

    # ── NER helper ───────────────────────────────────────────────────────────

    def _run_ner(self, text: str, threshold: float):
        orgs, people, locs = [], [], []
        try:
            results = self.ner_pipeline(text)
            for ent in results:
                if ent.get("score", 1.0) < threshold:
                    continue
                word = ent["word"].replace(" ##", "").replace("##", "").strip()
                if len(word) < 2:
                    continue
                group = ent["entity_group"]
                if group == "ORG":
                    orgs.append(word)
                elif group == "PER":
                    people.append(word)
                elif group == "LOC":
                    locs.append(word)
        except Exception as e:
            logger.error(f"[Extractor] NER inference failed: {e}")
        # deduplicate preserving order
        return (
            list(dict.fromkeys(orgs)),
            list(dict.fromkeys(people)),
            list(dict.fromkeys(locs)),
        )

    # ── Regex helpers ────────────────────────────────────────────────────────

    def _find_dates(self, text: str) -> List[str]:
        return list(dict.fromkeys(self._DATE_PATTERN.findall(text)))

    def _find_money(self, text: str) -> List[str]:
        return list(dict.fromkeys(self._MONEY_PATTERN.findall(text)))

    def _run_matcher(self, pattern: re.Pattern, text: str, max_limit: int = 8) -> List[str]:
        matches = pattern.findall(text)
        # IMPROVEMENT: case-insensitive dedup via .lower() key
        seen = {}
        for m in matches:
            cleaned = m.strip().title()
            key = cleaned.lower()
            if key not in seen and len(cleaned) > 1:
                seen[key] = cleaned
        return list(seen.values())[:max_limit]

    # ── Risk assessment ──────────────────────────────────────────────────────

    def _assess_legal_risk(self, text: str, risk_level: int) -> str:
        """
        IMPROVEMENT: deterministic risk scoring — returns clean text,
        no emoji in the data layer (UI can add icons if needed).
        """
        has_termination = bool(self._LEGAL_PATTERNS["Clauses"].search(text))
        has_penalties = bool(self._LEGAL_PATTERNS["Financials"].search(text))
        has_arbitration = bool(re.search(r'\barbitration\b', text, re.IGNORECASE))

        if has_termination and has_penalties and risk_level >= 5:
            return (
                f"HIGH RISK (sensitivity {risk_level}/10): Exit/termination clauses and "
                f"financial penalty provisions both detected. Legal review recommended."
            )
        elif has_termination and risk_level >= 7:
            return (
                f"MODERATE RISK (sensitivity {risk_level}/10): Termination or separation "
                f"clause detected. Arbitration present: {'Yes' if has_arbitration else 'No'}."
            )
        elif has_penalties:
            return f"LOW-MODERATE RISK (sensitivity {risk_level}/10): Financial liability terms present."
        else:
            return f"LOW RISK (sensitivity {risk_level}/10): No high-risk clauses identified."

    # ── Utilities ────────────────────────────────────────────────────────────

    @staticmethod
    def _join(items: List[str]) -> str:
        return ", ".join(items) if items else ""

    def _flush_vram(self):
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        elif torch.backends.mps.is_available():
            torch.mps.empty_cache()