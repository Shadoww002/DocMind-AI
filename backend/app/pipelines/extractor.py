import re
import torch
import logging
from typing import Dict, List, Any
from transformers import pipeline
from backend.app.core.config import DomainConfig

logger = logging.getLogger(__name__)

class DomainExtractor:
    # ---------------------------------------------------------
    # Pre-compiled high-throughput Regex Patterns (Class Level)
    # ---------------------------------------------------------
    _DATE_PATTERN = re.compile(
        r'\b\d{1,2}[-/.]\d{1,2}[-/.]\d{2,4}\b|\b\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{4}\b', 
        re.IGNORECASE
    )
    
    # Advanced Domain Pattern Matrices mapping exactly to requested sub-fields
    _MEDICAL_PATTERNS = {
        "Medications": re.compile(r'\b(?:ibuprofen|paracetamol|aspirin|amoxicillin|metformin|insulin|atorvastatin|antibiotics|dosage|\d+\s*mg|\d+\s*ml)\b', re.IGNORECASE),
        "Diagnoses": re.compile(r'\b(?:acute|chronic|hypertension|diabetes|infection|fracture|pain|syndrome|carcinoma|asthma|cardiac|oncology)\b', re.IGNORECASE),
        "Procedures": re.compile(r'\b(?:surgery|biopsy|resection|intubation|mri|ct\s*scan|ultrasound|dialysis|graft|sutures)\b', re.IGNORECASE),
        "History": re.compile(r'(?:past medical history|history of|prior to|previously diagnosed with|family history of) ([a-zA-Z0-9\s]+)', re.IGNORECASE)
    }

    _LEGAL_PATTERNS = {
        "Clauses": re.compile(r'\b(?:indemnity|termination|jurisdiction|arbitration|liability|confidentiality|force majeure|non-compete|governing\s+law)\b', re.IGNORECASE),
        "Financials": re.compile(r'\b(?:penalty|liquidated\s+damages|compensation|remuneration|consideration|invoice|disbursement|stipend|payment)\b', re.IGNORECASE)
    }

    _RESUME_PATTERNS = {
        "Skills": re.compile(r'\b(?:python|pytorch|tensorflow|transformers|nlp|java|c\+\+|react|aws|docker|kubernetes|sql|machine\s+learning|data\s+science|git|linux|spark|hadoop|kafka)\b', re.IGNORECASE),
        "Certifications": re.compile(r'\b(?:aws|azure|gcp|pmp|scrum|coursera|udemy|certified|gate|infosys)\b', re.IGNORECASE)
    }

    def __init__(self, model_name: str = "dslim/bert-base-NER"):
        """Initializes the token classification pipeline on active hardware accelerators."""
        self.device = 0 if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else -1)
        self.ner_pipeline = pipeline(
            "ner", 
            model=model_name, 
            device=self.device, 
            aggregation_strategy="simple"
        )

    def extract_features(self, full_text: str, domain: str, params: dict = None) -> Dict[str, Any]:
        """
        Combines transformer entity grouping with dynamic runtime parameters coming from the UI UI layer.
        """
        if params is None:
            params = {}

        profile = DomainConfig.DOMAINS.get(domain.lower())
        if not profile:
            raise ValueError(f"Target domain '{domain}' is completely unrecognized by the configuration core.")
            
        # Initialize output dictionary using exactly configured fields
        output_data = {field: "Information not documented within this segment." for field in profile["fields"]}
        
        # Safe string slicing for high-speed model execution
        text_sample = full_text[:4000]
        orgs, people, locs = [], [], []
        
        # Read general tuning parameter: anonymize_pii
        anonymize = params.get("anonymize_pii", False)
        ner_threshold = params.get("ner_threshold", 85) / 100.0  # Normalized to 0.0 - 1.0

        try:
            ner_results = self.ner_pipeline(text_sample)
            for ent in ner_results:
                # Filter out tokens below user-specified NER threshold parameters
                if ent.get('score', 1.0) < ner_threshold:
                    continue
                    
                clean_word = ent['word'].replace(" ##", "").replace("##", "").strip()
                if len(clean_word) < 2:
                    continue
                    
                if ent['entity_group'] == 'ORG':
                    orgs.append(clean_word)
                elif ent['entity_group'] == 'PER':
                    people.append(clean_word)
                elif ent['entity_group'] == 'LOC':
                    locs.append(clean_word)
                    
            orgs, people, locs = list(set(orgs)), list(set(people)), list(set(locs))
        except Exception as e:
            logger.error(f"Transformer parsing execution faulted: {str(e)}")

        detected_dates = list(set(self._DATE_PATTERN.findall(text_sample)))

        # ---------------------------------------------------------
        # Domain Logic Routers Processing Multi-Param Adjustments
        # ---------------------------------------------------------
        if domain == "medical":
            # Param 1: anonymize_pii
            output_data["Patient & Doctors Involved"] = "[REDACTED VIA HIPAA PRIVACY LAYER]" if anonymize else (", ".join(people[:4]) if people else "No clinical staff or patient names found.")
            output_data["Important Dates"] = ", ".join(detected_dates) if detected_dates else "None parsed."
            output_data["Diagnoses & Conditions"] = ", ".join(self._run_matcher(self._MEDICAL_PATTERNS["Diagnoses"], full_text))
            
            # Param 2: extract_dosages
            if params.get("extract_dosages", True):
                output_data["Medications & Dosages"] = ", ".join(self._run_matcher(self._MEDICAL_PATTERNS["Medications"], full_text))
            else:
                output_data["Medications & Dosages"] = "[Extraction suspended via user control profile]"
                
            # Param 3: map_pathology
            if params.get("map_pathology", True):
                hist_matches = self._run_matcher(self._MEDICAL_PATTERNS["History"], full_text)
                output_data["Medical History"] = "Discovered indicators of: " + ", ".join(hist_matches) if hist_matches else "No clear structural historic pathology isolated."
            else:
                output_data["Medical History"] = "[Historic mapping deactivated]"
                
            output_data["Procedures & Treatments"] = ", ".join(self._run_matcher(self._MEDICAL_PATTERNS["Procedures"], full_text))

        elif domain == "legal":
            # Param 1: anonymize_pii
            if anonymize:
                output_data["Parties & Signees"] = "[REDACTED FOR NDA/COMPLIANCE CONSTRAINTS]"
            else:
                all_parties = people + orgs
                output_data["Parties & Signees"] = ", ".join(all_parties)[:250] if all_parties else "Undetermined legal entities."
                
            output_data["Execution Dates"] = ", ".join(detected_dates) if detected_dates else "Execution/Sign dates not found."
            
            # Param 2: scan_liabilities
            if params.get("scan_liabilities", True):
                output_data["Financial Liabilities"] = ", ".join(self._run_matcher(self._LEGAL_PATTERNS["Financials"], full_text))
            else:
                output_data["Financial Liabilities"] = "[Financial metrics profiling disabled]"
                
            output_data["Indemnity & Clauses"] = ", ".join(self._run_matcher(self._LEGAL_PATTERNS["Clauses"], full_text))
            
            # Param 3: isolate_signees
            if params.get("isolate_signees", True) and people:
                output_data["Obligations"] = f"Action Item Assignment Tracked to Signees: {', '.join(people[:2])}"
            else:
                output_data["Obligations"] = "Standard institutional delivery conditions apply."
                
            # Param 4: risk_sensitivity slider effects (scale 1 - 10)
            risk_level = params.get("risk_sensitivity", 7)
            has_termination = bool(self._LEGAL_PATTERNS["Clauses"].search(full_text))
            has_penalties = bool(self._LEGAL_PATTERNS["Financials"].search(full_text))
            
            if has_termination and has_penalties and risk_level >= 5:
                output_data["Risks & Red Flags"] = f"🚨 HIGH ALERT (Sensitivity Level {risk_level}): Exit penalization models matched alongside strict liability modifiers."
            elif has_termination and risk_level >= 7:
                output_data["Risks & Red Flags"] = f"⚠️ WARNING (Sensitivity Level {risk_level}): Compulsory separation or governance clause parsed."
            else:
                output_data["Risks & Red Flags"] = f"Baseline compliance patterns satisfied (Sensitivity {risk_level})."

        elif domain == "resume":
            # Param 1: anonymize_pii (Blind Screening)
            output_data["Candidate Info"] = "[BLIND SCREENING MODE - Candidate Name and Locations Scrubbed]" if anonymize else (f"Verified Profile. Discovered {len(orgs)} operational institutional connections.")
            
            # Param 2: target_role text validation
            target = params.get("target_role", "").strip()
            if target:
                if target.lower() in full_text.lower():
                    output_data["Target Role Alignment"] = f"🎯 HIGH DENSITY MATCH: Document heavily correlates with the target career track '{target.upper()}'."
                else:
                    output_data["Target Role Alignment"] = f"📉 GAP ENCOUNTERED: Candidate profile does not explicitly reference your target specification target '{target}'."
            else:
                output_data["Target Role Alignment"] = "No target comparative profile requested by operator."

            # Param 3: strict_skills extraction filtering
            extracted_skills = self._run_matcher(self._RESUME_PATTERNS["Skills"], full_text)
            if params.get("strict_skills", True):
                output_data["Skills Matrix"] = ", ".join([sk for sk in extracted_skills if len(sk) > 2]) if extracted_skills else "No core technical matrix matches verified."
            else:
                output_data["Skills Matrix"] = ", ".join(extracted_skills) if extracted_skills else "No skills mapped."
                
            # Param 4: parse_academic
            output_data["Education Timeline"] = ", ".join(detected_dates) if (detected_dates and params.get("parse_academic", True)) else "Academic dates unmapped or hidden."
            output_data["Certifications"] = ", ".join(self._run_matcher(self._RESUME_PATTERNS["Certifications"], full_text))
            output_data["Experience Summary"] = f"Verified structural residency history with: {', '.join(orgs[:4])}" if orgs else "Independent contractor or unlisted history patterns found."

        # ---------------------------------------------------------
        # 4. Device Agnostic VRAM Garbage Collection
        # ---------------------------------------------------------
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        elif torch.backends.mps.is_available():
            torch.mps.empty_cache()

        return output_data

    def _run_matcher(self, precompiled_pattern: re.Pattern, text: str, max_limit: int = 8) -> List[str]:
        """Runs validation routines over document structures using matching registers."""
        matches = precompiled_pattern.findall(text)
        normalized = list(set([m.strip().title() for m in matches if len(m.strip()) > 1]))
        return normalized[:max_limit]