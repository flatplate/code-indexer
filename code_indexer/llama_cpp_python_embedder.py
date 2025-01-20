from huggingface_hub import hf_hub_download
from llama_cpp import Llama
from typing import List
from .code_transformer_test import CodeDefinition

class LlamaCppCodeEmbedder:
    """Provides embeddings for code definitions using llama.cpp"""
    
    def __init__(self, 
                 model_name: str = "djuna/jina-embeddings-v2-base-code-Q5_K_M-GGUF",
                 model_file: str = "jina-embeddings-v2-base-code-q5_k_m.gguf",
                 max_length: int = 8192,
                 batch_size: int = 32,
                 use_gpu: bool = True,
                 n_threads: int = 32):
        """Initialize the embedder with the specified model"""
        # Download the GGUF model
        model_path = hf_hub_download(model_name, filename=model_file)
        
        # Initialize the model
        self.model = Llama(
            model_path=model_path,
            n_ctx=max_length,
            n_threads=n_threads,
            n_gpu_layers=-1 if use_gpu else 0,
            n_batch=batch_size,
            embedding=True  # Enable embedding mode
        )
        self.max_length = max_length

    def _truncate_code(self, code: str) -> str:
        """Truncate code to maximum length while trying to preserve complete lines"""
        if len(code) <= self.max_length:
            return code
            
        # Split into lines and accumulate until we reach max_length
        lines = code.splitlines()
        truncated_lines = []
        current_length = 0
        
        for line in lines:
            line_length = len(line) + 1  # +1 for newline character
            if current_length + line_length > self.max_length:
                break
            truncated_lines.append(line)
            current_length += line_length
            
        return '\n'.join(truncated_lines)

    def create_embeddings(self, definitions: List[CodeDefinition], batch_size: int = 32) -> List[CodeDefinition]:
        """Create embeddings for a list of definitions"""
        truncated_codes = [self._truncate_code(d.code) for d in definitions]
        response = self.model.create_embedding(truncated_codes)
        
        # Extract the actual embeddings from the response
        embeddings = [item['embedding'] for item in response['data']]

        for embedding, definition in zip(embeddings, definitions):
            definition.embedding = embedding
            
        return definitions

    def embed_single(self, code: str) -> List[float]:
        """Create embedding for a single piece of code"""
        truncated_code = self._truncate_code(code)
        embedding = self.model.embed(truncated_code)
        return embedding
