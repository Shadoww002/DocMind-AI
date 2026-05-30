import os
from typing import Dict, Any

class DomainConfig:
    # Safe fallback mapping for localized data persistence directories
    BASE_DIR = os.getcwd()
    VECTOR_DB_DIR = os.getenv("VECTOR_DB_DIR", os.path.join(BASE_DIR, "data", "chroma_db"))
    
    DOMAINS: Dict[str, Dict[str, Any]] = {
        "medical": {
            "name": "Medical Records & Clinical Trials",
            "ner_model": os.getenv("MEDICAL_NER_MODEL", "dslim/bert-base-NER"),
            "llm_model": os.getenv("MEDICAL_LLM_MODEL", "google/flan-t5-base"),
            
            # Expanded feature extraction schema matching the new UI/Extractor
            "fields": [
                "Patient & Doctors Involved", 
                "Important Dates", 
                "Diagnoses & Conditions", 
                "Medications & Dosages", 
                "Medical History",
                "Procedures & Treatments"
            ],
            
            # Hyperparameter Tuning Matrix tailored for Clinical Text
            "chunk_size": 800,       # Smaller chunk window due to dense clinical vitals/metrics
            "chunk_overlap": 150,
            "max_chunks": 12,        # Guardrail cap to avoid exceeding model token constraints
            "top_k_retrieval": 3,    # Number of semantic vectors to surface during RAG lookup
            
            "system_prompt": (
                "You are an expert clinical informatics assistant. Analyze the provided medical context "
                "and extract clinical insights accurately. Focus heavily on dosages, diagnoses, and medical timelines. "
                "Do not infer or extrapolate any information not explicitly stated in the source text blocks."
            )
        },
        "legal": {
            "name": "Indian Legal Documents & Contracts",
            "ner_model": os.getenv("LEGAL_NER_MODEL", "dslim/bert-base-NER"),
            "llm_model": os.getenv("LEGAL_LLM_MODEL", "google/flan-t5-base"),
            
            # Expanded feature extraction schema matching the new UI/Extractor
            "fields": [
                "Parties & Signees", 
                "Execution Dates", 
                "Financial Liabilities", 
                "Indemnity & Clauses", 
                "Obligations",
                "Risks & Red Flags"
            ],
            
            # Hyperparameter Tuning Matrix tailored for Jurisdictional Documents
            "chunk_size": 600,       # Highly granular slicing to ensure legal clauses are not split mid-sentence
            "chunk_overlap": 200,    # Large overlap cushion to capture cross-references across boundaries
            "max_chunks": 15,
            "top_k_retrieval": 4,    # Surfacing higher context volume for comprehensive review
            
            "system_prompt": (
                "You are an expert legal counsel specialized in Indian Law, including the Indian Contract Act. "
                "Analyze the legal text strictly. Highlight jurisdictional limits, financial liabilities, indemnity clauses, "
                "and critical execution dates. Maintain absolute factual compliance based exclusively on the given text."
            )
        },
        "resume": {
            "name": "Resumes & CV Intelligence",
            "ner_model": os.getenv("RESUME_NER_MODEL", "dslim/bert-base-NER"),
            "llm_model": os.getenv("RESUME_LLM_MODEL", "google/flan-t5-base"),
            
            # Expanded feature extraction schema matching the new UI/Extractor
            "fields": [
                "Candidate Info", 
                "Target Role Alignment", 
                "Skills Matrix", 
                "Experience Summary", 
                "Certifications",
                "Education Timeline"
            ],
            
            # Hyperparameter Tuning Matrix tailored for Professional Profiles
            "chunk_size": 1200,      # Resumes are sparse; larger text spans keep whole job entries intact
            "chunk_overlap": 100,
            "max_chunks": 8,         # Low ceiling since resumes are brief documents
            "top_k_retrieval": 2,
            
            "system_prompt": (
                "You are an advanced technical talent acquisition engine. Evaluate the candidate text. "
                "Extract concrete skills, identify institutional backgrounds, and point out distinct domain alignment "
                "strengths based solely on the factual context of the target candidate text profile."
            )
        }
    }

    @classmethod
    def get_domain_property(cls, domain: str, key: str, default: Any = None) -> Any:
        """Safe extraction wrapper to query configuration nested values with reliable fallbacks."""
        domain_profile = cls.DOMAINS.get(domain.lower())
        if not domain_profile:
            # Fall back to legal layout profile if request domain is completely unmapped
            domain_profile = cls.DOMAINS.get("legal", {})
        return domain_profile.get(key, default)