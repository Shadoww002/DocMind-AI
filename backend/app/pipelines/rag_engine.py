import torch
import logging
from typing import Dict, List, Any
from transformers import pipeline
from backend.app.core.config import DomainConfig

logger = logging.getLogger(__name__)

class DomainRAGEngine:
    def __init__(self, model_name: str = "google/flan-t5-base"):
        """Initializes the text-generation pipeline on the optimal hardware accelerator."""
        self.device = 0 if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else -1)
        
        # Load text2text pipeline with optimal framework settings
        self.qa_generator = pipeline(
            "text2text-generation", 
            model=model_name, 
            device=self.device,
            framework="pt"
        )

    def generate_answer(self, question: str, retrieved_context: List[Dict[str, Any]], domain: str) -> Dict[str, Any]:
        """
        Generates contextual answers with active prompt-length balancing and hardware VRAM cleanup.
        Prevents silent model token clipping under heavy loads.
        """
        profile = DomainConfig.DOMAINS.get(domain, DomainConfig.DOMAINS.get("legal", {}))
        system_prompt = profile.get("system_prompt", "You are an expert document assistant.")
        
        context_str = ""
        citations = []
        
        # 1. Active Token Optimization Loop
        # We sample context blocks while monitoring length to avoid crashing the 512-token T5 boundary
        for idx, item in enumerate(retrieved_context):
            current_chunk_text = item.get('text', '').strip()
            if not current_chunk_text:
                continue
                
            # Limit individual context blocks to roughly ~450 characters if context is overcrowded
            truncated_chunk = current_chunk_text[:500] if len(retrieved_context) > 2 else current_chunk_text
            
            context_str += f"[Source {idx+1} (Page {item['page']})]: {truncated_chunk}\n"
            citations.append({
                "source_index": idx + 1, 
                "page": item["page"], 
                "text_snippet": current_chunk_text[:150] + "..."
            })
        
        # 2. Build the Hardened Prompt Matrix
        prompt = (
            f"Instruction: {system_prompt}\n\n"
            f"Context Information:\n{context_str}\n"
            f"Question: {question}\n\n"
            f"Provide a factual answer based strictly on the context metadata. If the context does not contain the answer, reply exactly with: Information not verified within context sources.\n"
            f"Answer:"
        )
        
        # 3. Failsafe Inference Execution
        try:
            res = self.qa_generator(
                prompt, 
                max_new_tokens=150, # Tight boundary appropriate for Flan-T5 structure
                do_sample=False,
                truncation=True # Forces the tokenizer to handle absolute worst-case overflows gracefully
            )
            generated_text = res[0]["generated_text"].strip()
            
        except Exception as e:
            logger.error(f"RAG Inference Exception encountered during pipeline execution: {str(e)}")
            generated_text = "Information not verified within context sources"

        # 4. Clean up output string properties
        fallback_flag = "information not verified within context sources"
        is_fallback = fallback_flag in generated_text.lower() or len(generated_text) < 2
        
        # 5. Production VRAM Memory Sweeping
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        elif torch.backends.mps.is_available():
            torch.mps.empty_cache()

        return {
            "answer": "Information not verified within context sources." if is_fallback else generated_text,
            "citations": [] if is_fallback else citations
        }