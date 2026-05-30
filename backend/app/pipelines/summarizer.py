import logging
import torch
from typing import Dict, List, Any
from transformers import pipeline
from backend.app.core.config import DomainConfig

# Setup enterprise logging
logger = logging.getLogger(__name__)

class DomainSummarizer:
    def __init__(self, model_name: str = "sshleifer/distilbart-cnn-12-6"):
        """Initializes the NLP pipeline and maps it to the optimal hardware accelerator."""
        self.device = 0 if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else -1)
        
        # Load the model directly into hardware memory once upon startup
        self.summarizer = pipeline(
            "summarization", 
            model=model_name, 
            device=self.device,
            framework="pt"
        )

    def summarize(self, chunks: List[str], domain: str, params: Dict[str, Any] = None, max_chunks: int = 15) -> Dict[str, Any]:
        """
        Executes a Map-Reduce pipeline to generate dual-tier summaries.
        Dynamically adjusts generation lengths based on user-defined UI tuning parameters.
        """
        if params is None:
            params = {}
            
        profile = DomainConfig.DOMAINS.get(domain, DomainConfig.DOMAINS.get("legal", {}))
        
        # Pre-filter chunks to eliminate empty or irrelevant spans
        valid_chunks = [chunk for chunk in chunks[:max_chunks] if len(chunk.split()) >= 30]

        if not valid_chunks:
            return {
                "domain_label": profile.get("name", "Unknown"),
                "short_summary": "Insufficient context to generate a reliable summary.",
                "detailed_summary": "No substantial text chunks were identified."
            }

        # Retrieve the length multiplier from UI params (1 = Concise, 2 = Standard, 3 = Comprehensive)
        length_multiplier = params.get("summary_detail", 2)
        
        intermediate_summaries = []
        
        # ==========================================
        # PHASE 1: MAP (Detailed Summary Generation)
        # ==========================================
        for chunk in valid_chunks:
            chunk_word_count = len(chunk.split())
            
            # Dynamic boundaries scaled by the UI Slider Multiplier
            dynamic_max = min(150, int(chunk_word_count * (0.35 * length_multiplier)))
            dynamic_min = min(30, int(chunk_word_count * 0.2))
            
            if dynamic_max <= dynamic_min:
                dynamic_max = dynamic_min + 10
                
            try:
                res = self.summarizer(
                    chunk, 
                    max_length=dynamic_max, 
                    min_length=dynamic_min, 
                    do_sample=False,
                    truncation=True # Strict guard against sequence length overflows
                )
                intermediate_summaries.append(res[0]['summary_text'].strip())
            except Exception as e:
                logger.warning(f"Inference bypassed for a chunk due to tensor exception: {str(e)}")
                continue

        # Format the detailed summary as a readable list of key structural points
        detailed_summary = "\n\n".join([f"• {sec}" for sec in intermediate_summaries])
        combined_text = " ".join(intermediate_summaries)
        input_len = len(combined_text.split())
        
        # ==========================================
        # PHASE 2: REDUCE (Executive Short Summary)
        # ==========================================
        short_summary = combined_text
        if input_len > 60:
            
            # Scale the final reduction step based on the same UI multiplier
            reduce_max = min(180, int(input_len * (0.25 * length_multiplier)))
            reduce_min = min(40, int(input_len * 0.15))
            if reduce_max <= reduce_min: 
                reduce_max = reduce_min + 10
                
            try:
                short_res = self.summarizer(
                    combined_text, 
                    max_length=reduce_max, 
                    min_length=reduce_min, 
                    do_sample=False,
                    truncation=True
                )
                short_summary = short_res[0]['summary_text']
            except Exception as e:
                logger.error(f"Final reduction summarization failed. Falling back to concatenated output. {str(e)}")

        # ==========================================
        # HARDWARE CLEANUP
        # ==========================================
        # Strictly release cached VRAM arrays to prevent out-of-memory (OOM) errors during heavy traffic
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        elif torch.backends.mps.is_available():
            torch.mps.empty_cache()

        return {
            "domain_label": profile.get("name", "Unknown"),
            "short_summary": short_summary.strip(),
            "detailed_summary": detailed_summary if detailed_summary else "Unable to generate detailed breakdown."
        }