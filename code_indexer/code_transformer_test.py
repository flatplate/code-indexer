from dataclasses import dataclass
from typing import List, Optional
import tree_sitter_typescript as ts
from tree_sitter import Language, Parser
from sentence_transformers import SentenceTransformer
from pathlib import Path

@dataclass
class CodeDefinition:
    file_name: str
    identifier: str
    code: str
    line_start: int
    line_end: int
    char_start: int
    char_end: int
    type: str
    parent_identifier: Optional[str] = None
    is_exported: bool = False
    documentation: Optional[str] = None
    embedding: Optional[List[float]] = None  # Moved to the end as optional


class TypeScriptParser:
    """Parser for TypeScript files that extracts code definitions"""
    
    def __init__(self):
        """Initialize the TypeScript parser"""
        self.ts_language = Language(ts.language_tsx())
        self.parser = Parser(self.ts_language)

    def parse_file(self, file_path: str) -> List[CodeDefinition]:
        """Parse a TypeScript file and return definitions without embeddings"""
        with open(file_path, "r") as file:
            code = file.read()
        
        code_bytes = bytes(code, "utf8")
        tree = self.parser.parse(code_bytes)
        
        return self._get_definitions(tree, code_bytes, Path(file_path).name)

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
              
            ; Regular function declarations (including React components)
            (function_declaration
              name: (identifier) @name
              parameters: (formal_parameters)) @definition
              
            ; Arrow function and other variable declarations
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
                  
                ; Include exported function declarations
                (function_declaration
                  name: (identifier) @name
                  parameters: (formal_parameters)) @definition
                  
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
            
            return definitions
        
        except Exception as e:
            import traceback
            traceback.print_exc()
        
        return definitions

