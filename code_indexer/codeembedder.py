from sentence_transformers import SentenceTransformer
from typing import List
from .code_transformer_test import CodeDefinition

class CodeEmbedder:
    """Provides embeddings for code definitions"""
    
    def __init__(self, model_name: str = "jinaai/jina-embeddings-v2-base-code", max_length=8192):
        """Initialize the embedder with the specified model"""
        self.model = SentenceTransformer(
            model_name,
            trust_remote_code=True,
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
        """Create embeddings for a list of definitions in batches"""

        codes = [self._truncate_code(d.code) for d in definitions]
        embeddings = self.model.encode(codes, batch_size=batch_size)
            
        for definition, embedding in zip(definitions, embeddings):
            definition.embedding = embedding.tolist()
                
        return definitions

    def embed_single(self, code: str) -> List[float]:
        """Create embedding for a single piece of code"""
        truncated_code = self._truncate_code(code)
        embedding = self.model.encode(truncated_code)
        return embedding.tolist()
