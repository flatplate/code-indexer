from dataclasses import dataclass
from typing import List, Optional
import tree_sitter_typescript as ts
from tree_sitter import Language, Parser
from sentence_transformers import SentenceTransformer
from pathlib import Path

@dataclass
class CodeDefinition:
    """Represents a code definition with its location and metadata"""
    file_name: str
    identifier: str
    code: str
    line_start: int
    line_end: int
    char_start: int
    char_end: int
    type: str
    embedding: Optional[List[float]] = None
    parent_identifier: Optional[str] = None
    is_exported: bool = False
    documentation: Optional[str] = None

class TypeScriptEmbeddingProvider:
    """Provides embeddings and code analysis for TypeScript files"""
    
    def __init__(self, model_name: str = "jinaai/jina-embeddings-v2-base-code"):
        """Initialize the provider with the specified embedding model"""
        self.ts_language = Language(ts.language_tsx())
        self.parser = Parser(self.ts_language)
        self.model = SentenceTransformer(model_name, trust_remote_code=True)

    def process_file(self, file_path: str) -> List[CodeDefinition]:
        """Process a TypeScript file and return definitions with embeddings"""
        with open(file_path, "r") as file:
            code = file.read()
        
        code_bytes = bytes(code, "utf8")
        tree = self.parser.parse(code_bytes)
        
        definitions = self._get_definitions(tree, code_bytes, Path(file_path).name)
        
        if not definitions:
            print("No definitions found!")
        
        self._add_embeddings(definitions)
        return definitions

    def _get_definitions(self, tree, code_bytes: bytes, file_name: str) -> List[CodeDefinition]:
        """Get all top level definitions from the TypeScript file"""
        definitions = []
        query_string = """
        (program
            [
                ; Direct declarations
                (class_declaration
                    name: (type_identifier) @name) @definition
                
                (abstract_class_declaration
                    name: (type_identifier) @name) @definition
                    
                (interface_declaration
                    name: (type_identifier) @name) @definition
                    
                (type_alias_declaration
                    name: (type_identifier) @name) @definition
                    
                (enum_declaration
                    name: (identifier) @name) @definition
                    
                (function_declaration
                    name: (identifier) @name) @definition
                    
                (lexical_declaration
                    (variable_declarator
                        name: (identifier) @name
                        value: [(arrow_function) (function_expression) (call_expression) (object)] @value))
                        
                (variable_declaration
                    (variable_declarator
                        name: (identifier) @name
                        value: [(arrow_function) (function_expression) (call_expression) (object)] @value))
                
                ; Exported declarations
                (export_statement
                    [
                        (class_declaration
                            name: (type_identifier) @name) @definition
                        
                        (abstract_class_declaration
                            name: (type_identifier) @name) @definition
                            
                        (interface_declaration
                            name: (type_identifier) @name) @definition
                            
                        (type_alias_declaration
                            name: (type_identifier) @name) @definition
                            
                        (enum_declaration
                            name: (identifier) @name) @definition
                            
                        (function_declaration
                            name: (identifier) @name) @definition
                            
                        (lexical_declaration
                            (variable_declarator
                                name: (identifier) @name
                                value: [(arrow_function) (function_expression) (call_expression) (object)] @value))
                                
                        (variable_declaration
                            (variable_declarator
                                name: (identifier) @name
                                value: [(arrow_function) (function_expression) (call_expression) (object)] @value))
                    ])
            ]) @program
        """
        
        try:
            query = self.ts_language.query(query_string)
            matches = query.matches(tree.root_node)
            
            print("Matches found:", len(matches))
            
            for _, capture_dict in matches:
                if 'name' in capture_dict:
                    name_node = capture_dict['name'][0]
                    name = name_node.text.decode('utf-8')
                    
                    # Get the definition node
                    definition_node = capture_dict['definition'][0] if 'definition' in capture_dict else capture_dict['value'][0]
                    
                    code = code_bytes[definition_node.start_byte:definition_node.end_byte].decode('utf-8')
                    
                    # Determine the type
                    node_type = definition_node.type
                    if 'React.FC' in code:
                        node_type = 'react_component'
                    
                    # Create CodeDefinition instead of dict
                    definition = CodeDefinition(
                        file_name=file_name,
                        identifier=name,
                        code=code,
                        line_start=definition_node.start_point[0],
                        line_end=definition_node.end_point[0],
                        char_start=definition_node.start_point[1],
                        char_end=definition_node.end_point[1],
                        type=node_type,
                        is_exported='export' in capture_dict
                    )
                    definitions.append(definition)
            
            print(f"Found {len(definitions)} definitions")
            for d in definitions:
                print(f"- {d.identifier}: {d.type}")
            
            return definitions
        
        except Exception as e:
            print(f"Error during query: {str(e)}")
            import traceback
            traceback.print_exc()
        
        return definitions

    def _add_embeddings(self, definitions: List[CodeDefinition]):
        """Create embeddings for each definition"""
        if not definitions:
            return
            
        for definition in definitions:
            embedding = self.model.encode(definition.code)
            definition.embedding = embedding.tolist()
