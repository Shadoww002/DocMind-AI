import re
import torch
import logging
from typing import Dict, List, Any, Optional, Union
from transformers import pipeline
from backend.app.core.config import DomainConfig

logger = logging.getLogger(__name__)

ScoredEntity = Dict[str, Any]
FieldValue   = Union[str, List[ScoredEntity]]


class DomainExtractor:

    # ── Date & money patterns ──────────────────────────────────────────────────
    _DATE_PATTERN = re.compile(
        r'\b\d{1,2}[-/.]\d{1,2}[-/.]\d{2,4}\b'
        r'|\b\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{4}\b'
        r'|\b(?:19|20)\d{2}\b',
        re.IGNORECASE,
    )
    _MONEY_PATTERN = re.compile(
        r'(?:₹|Rs\.?|INR)\s*[\d,]+(?:\.\d{1,2})?(?:\s*(?:lakh|crore|thousand|million))?'
        r'|\b\d[\d,]*(?:\.\d{1,2})?\s*(?:lakh|crore|thousand)\b'
        r'|\b(?:one|two|three|four|five|six|seven|eight|nine|ten)\s+(?:lakh|crore)\b',
        re.IGNORECASE,
    )

    # ── Medical patterns ───────────────────────────────────────────────────────
    _MEDICAL_PATTERNS = {
        "Medications": re.compile(
            r'\b(?:'
            r'dolo|crocin|combiflam|pan-?d|augmentin|cefixime|cetirizine|montelukast'
            r'|azithromycin|amoxicillin|clavam|taxim|zifi|omez|pantop|razo|nexpro'
            r'|metformin|glucophage|glycomet|galvus|januvia|trajenta'
            r'|amlodipine|amlovas|stamlo|atenolol|tenormin|metoprolol|betaloc'
            r'|ramipril|hopace|cardace|enalapril|lisinopril|losartan|telmisartan'
            r'|atorvastatin|lipitor|rosuvas|rosuvastatin|crestor'
            r'|aspirin|ecosprin|clopidogrel|deplatt|warfarin|acitrom'
            r'|paracetamol|ibuprofen|diclofenac|voveran|aceclofenac|zerodol'
            r'|insulin|lantus|tresiba|novomix|humalog|novorapid'
            r'|pantoprazole|omeprazole|rabeprazole|esomeprazole|nexium'
            r'|levothyroxine|thyroxine|eltroxin|thyronorm'
            r'|prednisolone|dexamethasone|methylprednisolone|betamethasone'
            r'|amoxycillin|doxycycline|ciprofloxacin|levofloxacin|norfloxacin'
            r'|\w+olol|\w+pril|\w+artan|\w+statin|\w+zole|\w+floxacin|\w+mycin'
            r'|\d+\s*mg|\d+\s*ml|\d+\s*mcg|\d+\s*iu|\d+\s*units'
            r')\b',
            re.IGNORECASE,
        ),
        "Diagnoses": re.compile(
            r'\b(?:'
            r'acute|chronic|hypertension|htn|diabetes(?:\s+mellitus)?|type\s*[12]\s*dm|t2dm|t1dm'
            r'|infection|fracture|syndrome|carcinoma|cancer|tumour|tumor|malignancy'
            r'|asthma|copd|bronchitis|pneumonia|tuberculosis|tb|covid'
            r'|cardiac|heart\s+failure|chf|ischemia|angina|myocardial\s+infarction|mi'
            r'|ckd|arf|renal\s+failure|nephropathy|nephrotic'
            r'|hypothyroid|hyperthyroid|thyroiditis|goitre'
            r'|anaemia|anemia|thrombocytopenia|leukemia|lymphoma'
            r'|dengue|malaria|typhoid|jaundice|hepatitis|cirrhosis'
            r'|sepsis|uti|cellulitis|abscess'
            r'|stroke|tia|epilepsy|seizure|migraine|parkinsonism'
            r'|depression|anxiety|schizophrenia|bipolar'
            r'|arthritis|osteoporosis|gout|spondylosis'
            r'|appendicitis|cholecystitis|pancreatitis|ibs'
            r'|glaucoma|cataract|retinopathy|maculopathy'
            r')\b',
            re.IGNORECASE,
        ),
        "Lab Values": re.compile(
            r'\b(?:hba1c|a1c|fasting\s+(?:blood\s+)?(?:glucose|sugar)|ppbs|rbs'
            r'|blood\s+pressure|bp|systolic|diastolic'
            r'|creatinine|egfr|bun|urea|uric\s+acid'
            r'|haemoglobin|hemoglobin|hb|hgb|wbc|rbc|platelets|tlc|dlc'
            r'|total\s+cholesterol|ldl|hdl|triglycerides|vldl'
            r'|sgot|ast|sgpt|alt|alp|bilirubin|albumin|protein'
            r'|sodium|potassium|chloride|bicarbonate|calcium|phosphorus|magnesium'
            r'|tsh|t3|t4|ft3|ft4|psa|ca-?125|cea'
            r'|inr|pt|aptt|esr|crp|procalcitonin'
            r')[\s:=]*[\d.]+\s*(?:%|mg/dl|mmol/l|mmhg|g/dl|u/l|meq/l|ng/ml|iu/ml|miu/l)?\b',
            re.IGNORECASE,
        ),
        "Procedures": re.compile(
            r'\b(?:surgery|operation|biopsy|resection|intubation|ventilation'
            r'|mri|ct\s*scan|ctscan|pet\s*scan|ultrasound|usg|echocardiogram|echo'
            r'|ecg|ekg|holter|stress\s*test|tmt'
            r'|x-?ray|chest\s*x-?ray|mammogram|dexa'
            r'|endoscopy|colonoscopy|bronchoscopy|cystoscopy|laparoscopy'
            r'|angiography|angioplasty|stenting|bypass|cabg|catheterisation'
            r'|dialysis|haemo?dialysis|peritoneal\s+dialysis'
            r'|chemotherapy|radiotherapy|radiation|immunotherapy'
            r'|ivf|iui|embryo\s+transfer'
            r'|appendicectomy|cholecystectomy|hysterectomy|prostatectomy'
            r')\b',
            re.IGNORECASE,
        ),
        "History": re.compile(
            r'(?:past\s+(?:medical\s+)?history|h/o|h\.o\.|history\s+of'
            r'|previously\s+diagnosed\s+with|known\s+case\s+of|k/o|k\.o\.'
            r'|family\s+history\s+of|fh\s+of'
            r'|surgical\s+history|personal\s+history)\s*[:\-]?\s*([a-zA-Z0-9 ,]+)',
            re.IGNORECASE,
        ),
    }

    # ── Legal patterns ─────────────────────────────────────────────────────────
    _LEGAL_PATTERNS = {
        "Clauses": re.compile(
            r'\b(?:indemnity|termination|jurisdiction|arbitration|liability'
            r'|confidentiality|force\s+majeure|non-compete|non-?disclosure|nda'
            r'|governing\s+law|warranty|representation|covenant'
            r'|limitation\s+of\s+liability|dispute\s+resolution'
            r'|breach|remedy|injunction|specific\s+performance'
            r'|lock-?in|penalty|forfeiture|subletting|assignment'
            r'|renewal|escalation|notice\s+period|eviction'
            r')\b',
            re.IGNORECASE,
        ),
        "Financials": re.compile(
            r'\b(?:penalty|liquidated\s+damages|compensation|remuneration'
            r'|consideration|stamp\s+duty|security\s+deposit|advance'
            r'|earnest\s+money|token\s+money|loan|mortgage|lien'
            r'|rent|monthly\s+rent|maintenance\s+charges|society\s+charges'
            r'|registration\s+fees|brokerage|commission'
            r')\b',
            re.IGNORECASE,
        ),
        "Indian Acts": re.compile(
            r'\b(?:ipc|crpc|cpc|constitution\s+of\s+india'
            r'|transfer\s+of\s+property\s+act|registration\s+act'
            r'|companies\s+act|arbitration\s+and\s+conciliation\s+act'
            r'|consumer\s+protection\s+act|it\s+act|information\s+technology\s+act'
            r'|gst\s+act|goods\s+and\s+services\s+tax'
            r'|sale\s+of\s+goods\s+act|contract\s+act|evidence\s+act'
            r'|rent\s+control\s+act|tenancy\s+act'
            r'|limitation\s+act|negotiable\s+instruments\s+act|ni\s+act'
            r'|stamp\s+act|indian\s+stamp\s+act'
            r'|foreign\s+exchange\s+management\s+act|fema'
            r'|prevention\s+of\s+money\s+laundering\s+act|pmla'
            r'|real\s+estate\s+(?:regulation\s+and\s+development\s+)?act|rera'
            r'|section\s+\d+[a-z]?(?:\(\d+\))?(?:\s+of\s+the\s+[\w\s]+act)?'
            r')\b',
            re.IGNORECASE,
        ),
        "Courts": re.compile(
            r'\b(?:supreme\s+court|high\s+court|district\s+court|sessions\s+court'
            r'|civil\s+court|magistrate\s+court|family\s+court'
            r'|consumer\s+forum|consumer\s+court|ncdrc|scdrc|dcdrc'
            r'|nclt|nclat|drt|debt\s+recovery\s+tribunal'
            r'|arbitration\s+tribunal|sole\s+arbitrator'
            r'|lok\s+adalat|legal\s+services\s+authority'
            r'|sub-?registrar|registrar\s+of\s+(?:deeds|companies)'
            r')\b',
            re.IGNORECASE,
        ),
    }

    # ── Resume patterns ────────────────────────────────────────────────────────
    _RESUME_PATTERNS = {
        "Skills": re.compile(
            r'\b(?:python|pytorch|tensorflow|keras|transformers|nlp|spacy|nltk'
            r'|java|c\+\+|c#|go|golang|rust|scala|r\b|matlab'
            r'|react|angular|vue|nextjs|nodejs|node\.js|typescript|javascript'
            r'|fastapi|flask|django|spring|express|laravel'
            r'|aws|azure|gcp|google\s+cloud|oracle\s+cloud'
            r'|docker|kubernetes|terraform|ansible|jenkins|github\s+actions|ci/?cd'
            r'|sql|postgresql|mysql|mongodb|redis|elasticsearch|cassandra|dynamodb'
            r'|kafka|spark|hadoop|airflow|dbt|databricks|snowflake'
            r'|machine\s+learning|deep\s+learning|computer\s+vision|nlp'
            r'|llm|langchain|rag|vector\s+(?:database|db|store)|chromadb|faiss|pinecone'
            r'|hugging\s*face|openai|gemini|claude|llama'
            r'|git|linux|bash|rest\s*api|graphql|microservices|agile|scrum'
            r'|power\s+bi|tableau|excel|pandas|numpy|matplotlib|seaborn|plotly'
            r'|streamlit|gradio|flask|fastapi'
            r')\b',
            re.IGNORECASE,
        ),
        "Certifications": re.compile(
            r'\b(?:aws\s+certified|azure\s+certified|gcp\s+certified|google\s+certified'
            r'|microsoft\s+certified|oracle\s+certified|oci\s+certified'
            r'|pmp|csm|cka|ckad|cks|rhce|rhcsa'
            r'|coursera|udemy|nptel|edx|pluralsight'
            r'|gate|cat|gmat|gre'
            r'|infosys\s+certified|tcs\s+certified|nasscom'
            r'|cisco|ccna|ccnp|comptia'
            r')\b',
            re.IGNORECASE,
        ),
        "Education": re.compile(
            r'\b(?:b\.?\s*tech|m\.?\s*tech|b\.?\s*e\.?|m\.?\s*e\.?'
            r'|b\.?\s*sc|m\.?\s*sc|b\.?\s*ca|m\.?\s*ca|bca|mca'
            r'|mba|phd|ph\.d|m\.?\s*phil|pgdm|pgd'
            r'|iit\s*\w*|nit\s*\w*|bits\s*\w*|iim\s*\w*|iisc'
            r'|vtu|anna\s+university|mumbai\s+university|delhi\s+university|du'
            r'|pune\s+university|osmania\s+university|madras\s+university'
            r'|10th|12th|ssc|hsc|cbse|icse|state\s+board'
            r')\b',
            re.IGNORECASE,
        ),
        # NEW: capture years of experience explicitly
        "Experience Years": re.compile(
            r'\b(\d+(?:\.\d+)?)\s*\+?\s*years?\s*(?:of\s+)?(?:experience|exp|work)'
            r'|\b(?:fresher|entry.?level|junior|senior|lead|principal|staff)\b',
            re.IGNORECASE,
        ),
    }

    def __init__(self, model_name: str = "dslim/bert-base-NER"):
        self._setup_device()
        try:
            self.ner_pipeline = pipeline(
                "ner",
                model=model_name,
                device=self.device,
                aggregation_strategy="simple",
                torch_dtype=torch.float32,
            )
            logger.info(f"[Extractor] NER loaded on {self.device_name}")
        except Exception as e:
            logger.error(f"[Extractor] NER load failed: {e}")
            raise

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
        params  = params or {}
        profile = DomainConfig.DOMAINS.get(domain.lower())
        if not profile:
            raise ValueError(f"Unknown domain: '{domain}'")

        output: Dict[str, FieldValue] = {f: "Not found." for f in profile["fields"]}

        text_sample   = full_text[:6000]
        anonymize     = params.get("anonymize_pii", False)
        ner_threshold = params.get("ner_threshold", 80) / 100.0

        orgs, people, locs = self._run_ner(text_sample, ner_threshold)
        dates  = self._find_dates(text_sample)
        money  = self._find_money(full_text)

        if domain == "medical":
            output = self._extract_medical(full_text, people, dates, anonymize, params)
        elif domain == "legal":
            output = self._extract_legal(full_text, people, orgs, locs, dates, money, anonymize, params)
        elif domain == "resume":
            output = self._extract_resume(full_text, orgs, people, dates, anonymize, params)

        self._flush_vram()
        return output

    # ── Medical ───────────────────────────────────────────────────────────────

    def _extract_medical(self, text, people, dates, anonymize, params) -> dict:
        out = {}

        # Patient & Doctors — NER people with fallback message
        if anonymize:
            out["Patient & Doctors Involved"] = "[REDACTED — PII masked per HIPAA]"
        elif people:
            out["Patient & Doctors Involved"] = ", ".join(people[:5])
        else:
            out["Patient & Doctors Involved"] = "No named individuals detected in document."

        # Important Dates — formatted as readable list
        out["Important Dates"] = (
            " · ".join(dates[:8]) if dates
            else "No specific dates found."
        )

        # Diagnoses — scored chips
        diagnoses = self._scored_matches(self._MEDICAL_PATTERNS["Diagnoses"], text)
        out["Diagnoses & Conditions"] = (
            diagnoses if diagnoses
            else "No diagnoses detected. Ensure document contains clinical text."
        )

        # Lab Values — scored chips with numeric values preserved
        labs = self._scored_matches(self._MEDICAL_PATTERNS["Lab Values"], text, max_limit=12)
        out["Lab Values & Vitals"] = (
            labs if labs
            else "No lab values or vitals detected."
        )

        # Medications — scored chips or disabled
        if not params.get("extract_dosages", True):
            out["Medications & Dosages"] = "Medication extraction disabled by user."
        else:
            meds = self._scored_matches(self._MEDICAL_PATTERNS["Medications"], text, max_limit=12)
            out["Medications & Dosages"] = (
                meds if meds
                else "No medications detected."
            )

        # Medical History
        if params.get("map_pathology", True):
            hist = self._run_matcher(self._MEDICAL_PATTERNS["History"], text)
            out["Medical History"] = (
                self._as_scored_list(hist) if hist
                else "No past medical history indicators found."
            )
        else:
            out["Medical History"] = "History mapping disabled by user."

        # Procedures
        procs = self._scored_matches(self._MEDICAL_PATTERNS["Procedures"], text)
        out["Procedures & Treatments"] = (
            procs if procs
            else "No procedures or treatments documented."
        )
        return out

    # ── Legal ─────────────────────────────────────────────────────────────────

    def _extract_legal(self, text, people, orgs, locs, dates, money, anonymize, params) -> dict:
        out = {}

        # Parties — people + orgs combined, deduplicated
        all_parties = list(dict.fromkeys(people + orgs))
        if anonymize:
            out["Parties & Signees"] = "[REDACTED — compliance mode active]"
        elif all_parties:
            out["Parties & Signees"] = ", ".join(all_parties[:6])
        else:
            out["Parties & Signees"] = "Parties not identified via NER. Check if names are present."

        # Dates — readable separator
        out["Execution Dates"] = (
            " · ".join(dates[:6]) if dates
            else "No execution or signing dates found."
        )

        # Monetary amounts — raw values with currency symbols
        if money:
            # Deduplicate and limit
            seen_money = list(dict.fromkeys(money[:8]))
            out["Monetary Amounts"] = " · ".join(seen_money)
        else:
            out["Monetary Amounts"] = "No monetary amounts detected. Check if amounts use ₹, Rs., or INR."

        # Financial liabilities
        if not params.get("scan_liabilities", True):
            out["Financial Liabilities"] = "Financial liability scan disabled by user."
        else:
            fins = self._scored_matches(self._LEGAL_PATTERNS["Financials"], text)
            out["Financial Liabilities"] = (
                fins if fins
                else "No financial liability terms found."
            )

        # Clauses
        clauses = self._scored_matches(self._LEGAL_PATTERNS["Clauses"], text, max_limit=10)
        out["Indemnity & Clauses"] = (
            clauses if clauses
            else "No standard legal clauses detected."
        )

        # Indian Acts
        acts = self._scored_matches(self._LEGAL_PATTERNS["Indian Acts"], text, max_limit=10)
        out["Applicable Indian Laws"] = (
            acts if acts
            else "No specific Indian acts or sections referenced."
        )

        # Courts & Jurisdiction
        courts = self._scored_matches(self._LEGAL_PATTERNS["Courts"], text)
        # Also check locations for jurisdiction
        jurisdiction_hint = f" · Locations mentioned: {', '.join(locs[:3])}" if locs and not courts else ""
        out["Courts & Jurisdiction"] = (
            courts if courts
            else f"No court or tribunal mentioned.{jurisdiction_hint}"
        )

        # Obligations
        if params.get("isolate_signees", True) and people:
            out["Obligations"] = f"Primary signatories identified: {', '.join(people[:3])}"
        else:
            out["Obligations"] = "Signee isolation disabled or no named parties found."

        # Risk assessment
        out["Risks & Red Flags"] = self._assess_legal_risk(
            text, params.get("risk_sensitivity", 7)
        )
        return out

    # ── Resume ────────────────────────────────────────────────────────────────

    def _extract_resume(self, text, orgs, people, dates, anonymize, params) -> dict:
        out = {}

        # Candidate Info — show name (from NER people) + orgs
        if anonymize:
            out["Candidate Info"] = "[BLIND SCREENING — name and details scrubbed]"
        else:
            name_hint = f"Name detected: {people[0]}" if people else ""
            org_hint  = f"Organisations: {', '.join(orgs[:4])}" if orgs else "No organisations detected."
            out["Candidate Info"] = f"{name_hint}{'  ·  ' if name_hint else ''}{org_hint}"

        # Target role alignment
        target = params.get("target_role", "").strip()
        if target:
            target_words = set(target.lower().split())
            text_lower   = text.lower()
            matched      = [w for w in target_words if w in text_lower]
            pct          = int(len(matched) / max(len(target_words), 1) * 100)
            missing      = target_words - set(matched)
            if pct >= 70:
                out["Target Role Alignment"] = (
                    f"Strong match ({pct}%) — profile aligns well with '{target}'"
                )
            elif pct >= 40:
                out["Target Role Alignment"] = (
                    f"Partial match ({pct}%) for '{target}'. "
                    f"Missing keywords: {', '.join(missing)}"
                )
            else:
                out["Target Role Alignment"] = (
                    f"Weak match ({pct}%) — profile does not align with '{target}'. "
                    f"Missing: {', '.join(missing)}"
                )
        else:
            out["Target Role Alignment"] = "No target role specified. Enter a role in parameters for alignment scoring."

        # Skills — scored chips, expanded pattern
        skills = self._scored_matches(self._RESUME_PATTERNS["Skills"], text, max_limit=25)
        if params.get("strict_skills", True) and isinstance(skills, list):
            skills = [s for s in skills if len(s["value"]) > 2]
        out["Skills Matrix"] = skills if skills else "No technical skills detected via keyword matching."

        # Education
        edu = self._scored_matches(self._RESUME_PATTERNS["Education"], text, max_limit=10)
        out["Education"] = edu if edu else "No education qualifications detected."

        # Experience years — new field
        exp_matches = self._run_matcher(self._RESUME_PATTERNS["Experience Years"], text, max_limit=5)
        if exp_matches:
            out["Education Timeline"] = (
                f"Experience indicators: {', '.join(exp_matches[:3])}"
                + (f"  ·  Dates: {', '.join(dates[:4])}" if dates else "")
            ) if params.get("parse_academic", True) else "Disabled."
        else:
            out["Education Timeline"] = (
                (", ".join(dates[:6]) if dates else "No timeline dates found.")
                if params.get("parse_academic", True) else "Disabled."
            )

        # Certifications
        certs = self._scored_matches(self._RESUME_PATTERNS["Certifications"], text, max_limit=8)
        out["Certifications"] = certs if certs else "No certifications detected."

        # Experience summary — orgs + any experience year mentions
        if orgs:
            out["Experience Summary"] = f"Organisations mentioned: {', '.join(orgs[:5])}"
        else:
            out["Experience Summary"] = "No organisation names detected by NER. Ensure company names are clearly stated."

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
                if   g == "ORG": orgs.append(word)
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
        seen = {}
        for m in pattern.findall(text):
            if isinstance(m, tuple):
                m = " ".join(x for x in m if x).strip()
            cleaned = m.strip()
            key = cleaned.lower()
            if key not in seen and len(cleaned) > 1:
                seen[key] = cleaned
        return list(seen.values())[:max_limit]

    def _scored_matches(
        self, pattern: re.Pattern, text: str, max_limit: int = 8
    ) -> List[ScoredEntity]:
        all_matches = [m.strip() for m in pattern.findall(text) if isinstance(m, str) and m.strip()]
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
            display = re.sub(r'\s+', ' ', m.strip())
            display = display[0].upper() + display[1:] if display else display
            count   = freq[key]
            score   = 0.99 if count >= 3 else (0.90 if count == 2 else 0.75)
            result.append({"value": display, "score": round(score, 2)})
            if len(result) >= max_limit:
                break
        return result

    def _as_scored_list(self, items: List[str]) -> List[ScoredEntity]:
        return [{"value": item, "score": 0.75} for item in items]

    def _assess_legal_risk(self, text: str, risk_level: int) -> str:
        has_termination = bool(self._LEGAL_PATTERNS["Clauses"].search(text))
        has_penalties   = bool(self._LEGAL_PATTERNS["Financials"].search(text))
        has_arbitration = bool(re.search(r'\barbitration\b', text, re.IGNORECASE))
        high_risk_clauses = ["eviction", "forfeiture", "penalty", "liquidated damages"]
        has_high = any(c in text.lower() for c in high_risk_clauses)

        if has_termination and has_penalties and risk_level >= 5:
            return (
                f"HIGH RISK (sensitivity {risk_level}/10): Termination/exit clauses and "
                f"financial penalty provisions both detected. "
                f"{'Arbitration clause present.' if has_arbitration else ''} "
                f"Legal review strongly recommended."
            )
        elif has_termination and risk_level >= 7:
            return (
                f"MODERATE RISK (sensitivity {risk_level}/10): Termination clause detected. "
                f"Arbitration present: {'Yes' if has_arbitration else 'No'}. "
                f"Review exit conditions carefully."
            )
        elif has_penalties or has_high:
            return (
                f"LOW-MODERATE RISK (sensitivity {risk_level}/10): "
                f"Financial liability or penalty terms present. Review monetary obligations."
            )
        return (
            f"LOW RISK (sensitivity {risk_level}/10): "
            f"No high-risk clauses identified in this document."
        )

    def _flush_vram(self):
        if torch.cuda.is_available():   torch.cuda.empty_cache()
        elif torch.backends.mps.is_available(): torch.mps.empty_cache()
        