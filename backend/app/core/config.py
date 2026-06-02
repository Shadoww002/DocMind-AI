import os
import logging
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)


class DomainConfig:
    """
    Central configuration registry for all domain profiles.

    Improvements over v1:
    - BASE_DIR uses __file__ so it is stable regardless of where uvicorn is launched from
    - VECTOR_DB_DIR validated at class load time — fails fast with a clear message
    - Domain field schemas updated to match the improved extractor (Lab Values & Vitals,
      Monetary Amounts, Applicable Indian Laws, Courts & Jurisdiction, Education)
    - system_prompts tightened: explicit instruction not to hallucinate
    - get_domain_property fallback changed from "legal" to None — silently falling back
      to legal config when an unknown domain is passed hides bugs
    - New helpers: domain_names(), all_fields(), is_valid_domain(), get_system_prompt()
    - All env-var model overrides preserved
    """

    # ── Paths ─────────────────────────────────────────────────────────────────
    # IMPROVEMENT: anchor to this file's location, not os.getcwd()
    # getcwd() changes depending on where you run `uvicorn`, making the DB
    # path unpredictable. __file__ is always the same absolute location.
    BASE_DIR      = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    VECTOR_DB_DIR = os.getenv("VECTOR_DB_DIR", os.path.join(BASE_DIR, "data", "chroma_db"))

    # ── Domain profiles ───────────────────────────────────────────────────────
    DOMAINS: Dict[str, Dict[str, Any]] = {

        "medical": {
            "name":      "Medical Records & Clinical Reports",
            "ner_model": os.getenv("MEDICAL_NER_MODEL", "dslim/bert-base-NER"),
            "llm_model": os.getenv("MEDICAL_LLM_MODEL", "google/flan-t5-base"),

            # IMPROVEMENT: fields updated to match improved extractor output keys
            "fields": [
                "Patient & Doctors Involved",
                "Important Dates",
                "Diagnoses & Conditions",
                "Lab Values & Vitals",       # new — HbA1c, BP, creatinine, etc.
                "Medications & Dosages",
                "Medical History",
                "Procedures & Treatments",
            ],

            # Smaller chunks for dense clinical text — vitals/lab lines are short
            "chunk_size":      800,
            "chunk_overlap":   150,
            "max_chunks":      12,
            "top_k_retrieval": 3,

            "system_prompt": (
                "You are a clinical informatics assistant with expertise in Indian medical records. "
                "Analyze the provided text and extract only what is explicitly stated. "
                "Focus on diagnoses, medications with dosages, lab values, and timelines. "
                "Do not infer, extrapolate, or generate information absent from the source text."
            ),
        },

        "legal": {
            "name":      "Indian Legal Documents & Contracts",
            "ner_model": os.getenv("LEGAL_NER_MODEL", "dslim/bert-base-NER"),
            "llm_model": os.getenv("LEGAL_LLM_MODEL", "google/flan-t5-base"),

            # IMPROVEMENT: added Monetary Amounts, Applicable Indian Laws, Courts & Jurisdiction
            # to match the three new fields added in the improved extractor
            "fields": [
                "Parties & Signees",
                "Execution Dates",
                "Monetary Amounts",          # new — ₹ values, stamp duty, consideration
                "Financial Liabilities",
                "Indemnity & Clauses",
                "Applicable Indian Laws",    # new — IPC, CrPC, Registration Act, etc.
                "Courts & Jurisdiction",     # new — Supreme Court, High Court, NCLT, etc.
                "Obligations",
                "Risks & Red Flags",
            ],

            # High overlap to prevent legal clauses from being split across chunk boundaries
            "chunk_size":      600,
            "chunk_overlap":   200,
            "max_chunks":      15,
            "top_k_retrieval": 4,

            "system_prompt": (
                "You are a legal counsel specialising in Indian law, including the Indian Contract Act 1872, "
                "Transfer of Property Act, Registration Act 1908, and related statutes. "
                "Analyze the provided legal text strictly. Identify parties, financial obligations, "
                "indemnity clauses, applicable Indian acts, and jurisdiction. "
                "Base all output exclusively on the supplied text — do not infer unstated obligations."
            ),
        },

        "resume": {
            "name":      "Resumes & CV Intelligence",
            "ner_model": os.getenv("RESUME_NER_MODEL", "dslim/bert-base-NER"),
            "llm_model": os.getenv("RESUME_LLM_MODEL", "google/flan-t5-base"),

            # IMPROVEMENT: added Education field to match extractor; reordered for UI display
            "fields": [
                "Candidate Info",
                "Target Role Alignment",
                "Skills Matrix",
                "Education",             # new — degree names, IIT/NIT/BITS detection
                "Education Timeline",
                "Certifications",
                "Experience Summary",
            ],

            # Larger chunks to keep complete job entries together
            "chunk_size":      1200,
            "chunk_overlap":   100,
            "max_chunks":      8,
            "top_k_retrieval": 2,

            "system_prompt": (
                "You are a senior technical recruiter evaluating a candidate's profile. "
                "Extract concrete skills, institutional backgrounds, job titles, companies, "
                "years of experience, education details, and certifications. "
                "Base your output solely on the factual content of the provided resume text."
            ),
        },
    }

    # ── Class-level validation ────────────────────────────────────────────────

    @classmethod
    def validate(cls) -> bool:
        """
        Called by main.py during startup. Returns True if config is valid.
        Checks that VECTOR_DB_DIR exists and is writable.
        """
        try:
            os.makedirs(cls.VECTOR_DB_DIR, exist_ok=True)
            probe = os.path.join(cls.VECTOR_DB_DIR, ".config_probe")
            with open(probe, "w") as f:
                f.write("ok")
            os.remove(probe)
            return True
        except OSError as e:
            logger.critical(f"[Config] VECTOR_DB_DIR is not writable: {cls.VECTOR_DB_DIR} — {e}")
            return False

    # ── Access helpers ────────────────────────────────────────────────────────

    @classmethod
    def get_domain_property(cls, domain: str, key: str, default: Any = None) -> Any:
        """
        Safe property accessor with explicit None fallback.

        IMPROVEMENT: v1 silently fell back to the 'legal' profile for unknown
        domains, masking bugs where wrong domain strings were passed. Now returns
        the default value and logs a warning so the caller knows the domain was invalid.
        """
        profile = cls.DOMAINS.get(domain.lower())
        if profile is None:
            logger.warning(
                f"[Config] get_domain_property called with unknown domain '{domain}'. "
                f"Returning default for key '{key}'."
            )
            return default
        return profile.get(key, default)

    @classmethod
    def is_valid_domain(cls, domain: str) -> bool:
        """Single-call validation used in endpoints and Pydantic validators."""
        return domain.lower() in cls.DOMAINS

    @classmethod
    def domain_names(cls) -> List[str]:
        """Returns valid domain keys: ['medical', 'legal', 'resume']."""
        return list(cls.DOMAINS.keys())

    @classmethod
    def all_fields(cls, domain: str) -> List[str]:
        """
        Returns the field list for a domain, or [] for unknown domains.
        Used by the extractor to initialise the output dict safely.
        """
        return cls.get_domain_property(domain, "fields", default=[])

    @classmethod
    def get_system_prompt(cls, domain: str) -> str:
        """Convenience accessor for the domain system prompt."""
        return cls.get_domain_property(
            domain,
            "system_prompt",
            default="You are an expert analyst. Extract key information from the provided text.",
        )