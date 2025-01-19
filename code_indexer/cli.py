from pathlib import Path
from sentence_transformers import CrossEncoder
import click
from typing import List, Set, Dict, Any
import pathspec
from .code_transformer_test import TypeScriptEmbeddingProvider, CodeDefinition
import sqlite3
import numpy as np
from datetime import datetime
import logging
import json
import torch

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def init_database(db_path: Path):
    """Initialize the SQLite database with the required schema"""
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    
    # Create tables
    c.execute('''
        CREATE TABLE IF NOT EXISTS metadata (
            key TEXT PRIMARY KEY,
            value TEXT
        )
    ''')
    
    c.execute('''
        CREATE TABLE IF NOT EXISTS definitions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            file_name TEXT NOT NULL,
            identifier TEXT NOT NULL,
            code TEXT NOT NULL,
            line_start INTEGER NOT NULL,
            line_end INTEGER NOT NULL,
            char_start INTEGER NOT NULL,
            char_end INTEGER NOT NULL,
            type TEXT NOT NULL,
            parent_identifier TEXT,
            is_exported BOOLEAN NOT NULL,
            documentation TEXT,
            embedding_file TEXT NOT NULL,
            embedding_index INTEGER NOT NULL
        )
    ''')
    
    conn.commit()
    return conn

def save_index(definitions: List[CodeDefinition], output_dir: Path):
    """Save the code definitions to SQLite + NumPy files"""
    output_dir.mkdir(parents=True, exist_ok=True)
    db_path = output_dir / 'index.db'
    embeddings_path = output_dir / 'embeddings.npy'
    
    # Convert embeddings to numpy array
    embeddings = np.array([d.embedding for d in definitions])
    
    # Save embeddings
    np.save(embeddings_path, embeddings)
    
    # Save metadata and definitions to SQLite
    conn = init_database(db_path)
    c = conn.cursor()
    
    # Save metadata
    c.execute("INSERT OR REPLACE INTO metadata VALUES (?, ?)",
              ('created_at', datetime.now().isoformat()))
    c.execute("INSERT OR REPLACE INTO metadata VALUES (?, ?)",
              ('total_definitions', str(len(definitions))))
    
    # Save definitions
    for idx, d in enumerate(definitions):
        c.execute('''
            INSERT INTO definitions (
                file_name, identifier, code, line_start, line_end,
                char_start, char_end, type, parent_identifier,
                is_exported, documentation, embedding_file, embedding_index
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            d.file_name, d.identifier, d.code, d.line_start, d.line_end,
            d.char_start, d.char_end, d.type, d.parent_identifier,
            d.is_exported, d.documentation, 'embeddings.npy', idx
        ))
    
    conn.commit()
    conn.close()

def load_index(index_dir: Path) -> tuple[List[CodeDefinition], np.ndarray]:
    """Load code definitions from SQLite + NumPy files"""
    db_path = index_dir / 'index.db'
    embeddings_path = index_dir / 'embeddings.npy'
    
    # Load embeddings
    embeddings = np.load(embeddings_path)
    
    # Load definitions from SQLite
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    
    definitions = []
    for row in c.execute('SELECT * FROM definitions'):
        embedding_idx = row[13]  # embedding_index column
        definition = CodeDefinition(
            file_name=row[1],
            identifier=row[2],
            code=row[3],
            line_start=row[4],
            line_end=row[5],
            char_start=row[6],
            char_end=row[7],
            type=row[8],
            parent_identifier=row[9],
            is_exported=bool(row[10]),
            documentation=row[11],
            embedding=embeddings[embedding_idx].tolist()
        )
        definitions.append(definition)
    
    conn.close()
    return definitions, embeddings

def load_gitignore(project_root: Path) -> pathspec.PathSpec:
    """Load .gitignore patterns from the project root"""
    gitignore_file = project_root / '.gitignore'
    patterns = []
    
    # Check all parent directories for .gitignore files
    current_dir = project_root
    while current_dir.exists():
        gitignore_path = current_dir / '.gitignore'
        if gitignore_path.is_file():
            with open(gitignore_path, 'r', encoding='utf-8') as f:
                # Add patterns with directory context
                dir_patterns = [line.strip() for line in f.readlines() if line.strip() and not line.startswith('#')]
                patterns.extend(dir_patterns)
        
        # Move to parent directory
        parent_dir = current_dir.parent
        if parent_dir == current_dir:  # Reached root
            break
        current_dir = parent_dir
    
    # Add some default patterns for TypeScript projects
    default_patterns = [
        'node_modules/',
        'build/',
        'dist/',
        '.next/',
        'coverage/',
        '.git/',
        '*.test.ts',
        '*.test.tsx',
        '*.spec.ts',
        '*.spec.tsx',
        '*.d.ts',
        '*.js.map',
        '.env*',
        '.DS_Store',
        'npm-debug.log*',
        'yarn-debug.log*',
        'yarn-error.log*'
    ]
    patterns.extend(default_patterns)
    
    return pathspec.PathSpec.from_lines(pathspec.patterns.GitWildMatchPattern, patterns)

def find_typescript_files(project_root: Path, gitignore_spec: pathspec.PathSpec) -> List[Path]:
    """Find all TypeScript files in the project that aren't ignored"""
    typescript_files = []
    
    try:
        for file_path in project_root.rglob('*'):
            try:
                # Skip if not a file
                if not file_path.is_file():
                    continue
                
                # Skip if not a TypeScript file
                if file_path.suffix not in ['.ts', '.tsx']:
                    continue
                    
                # Get relative path for gitignore matching
                try:
                    relative_path = file_path.relative_to(project_root)
                except ValueError:
                    logger.warning(f"Couldn't get relative path for {file_path}")
                    continue
                
                # Skip files matched by gitignore
                if gitignore_spec.match_file(str(relative_path)):
                    logger.debug(f"Skipping ignored file: {relative_path}")
                    continue
                
                # Skip files that are symlinks to ignored files
                if file_path.is_symlink():
                    real_path = file_path.resolve()
                    try:
                        real_relative = real_path.relative_to(project_root)
                        if gitignore_spec.match_file(str(real_relative)):
                            logger.debug(f"Skipping symlink to ignored file: {relative_path} -> {real_relative}")
                            continue
                    except ValueError:
                        # If the real path is outside the project, skip it
                        logger.debug(f"Skipping external symlink: {relative_path}")
                        continue
                
                typescript_files.append(file_path)
                
            except Exception as e:
                logger.warning(f"Error processing file {file_path}: {str(e)}")
                continue
                
    except Exception as e:
        logger.error(f"Error walking directory {project_root}: {str(e)}")
        raise
    
    return sorted(typescript_files)

@click.group()
def cli():
    """CLI tool for indexing TypeScript projects with embeddings"""
    pass

@cli.command()
@click.argument('project_dir', type=click.Path(exists=True, file_okay=False, dir_okay=True, path_type=Path))
@click.option('--output', '-o', type=click.Path(file_okay=False, dir_okay=True, path_type=Path), 
              default='code_index', help='Output directory for the index')
@click.option('--model', '-m', default="jinaai/jina-embeddings-v2-base-code",
              help='Name of the embedding model to use')
@click.option('--verbose', '-v', is_flag=True, help='Enable verbose logging')
def index(project_dir: Path, output: Path, model: str, verbose: bool):
    """Index a TypeScript project directory"""
    if verbose:
        logger.setLevel(logging.DEBUG)
    
    logger.info(f"Indexing project: {project_dir}")
    
    # Load gitignore patterns
    gitignore_spec = load_gitignore(project_dir)
    
    # Find TypeScript files
    typescript_files = find_typescript_files(project_dir, gitignore_spec)
    logger.info(f"Found {len(typescript_files)} TypeScript files to process")
    
    # Initialize the embedding provider
    provider = TypeScriptEmbeddingProvider(model_name=model)
    
    # Process each file
    all_definitions = []
    for file_path in typescript_files:
        try:
            logger.info(f"Processing {file_path.relative_to(project_dir)}")
            definitions = provider.process_file(str(file_path))
            all_definitions.extend(definitions)
            logger.debug(f"Found {len(definitions)} definitions in file")
        except Exception as e:
            logger.error(f"Error processing {file_path}: {str(e)}")
            continue
    
    logger.info(f"Total definitions found: {len(all_definitions)}")
    
    # Save the index
    save_index(all_definitions, output)
    logger.info(f"Index saved to {output}")

@cli.command()
@click.argument('index_dir', type=click.Path(exists=True, file_okay=False, dir_okay=True, path_type=Path))
@click.argument('query', type=str)
@click.option('--top-k', '-k', default=5, help='Number of results to return')
@click.option('--model', '-m', default="jinaai/jina-embeddings-v2-base-code",
              help='Name of the embedding model to use')
@click.option('--reranker-model', '-r', default="jinaai/jina-reranker-v2-base-multilingual",
              help='Name of the reranker model to use')
@click.option('--candidates', '-c', default=20, 
              help='Number of initial candidates to consider for reranking')
def search(index_dir: Path, query: str, top_k: int, model: str, reranker_model: str, candidates: int):
    """Search the code index for similar definitions"""
    # Load the index
    definitions, embeddings = load_index(index_dir)
    embeddings = embeddings.astype(np.float32)
    logger.info(f"Loaded {len(definitions)} definitions from index")
    
    # Initialize provider and search
    provider = TypeScriptEmbeddingProvider(model_name=model)
    
    # Initialize reranker
    reranker = CrossEncoder(
        reranker_model,
        automodel_args={"torch_dtype": "float16"},
        trust_remote_code=True,
    )
    
    # Convert query to embedding and compute similarities
    query_embedding = provider.model.encode(query)
    similarities = provider.model.similarity(query_embedding, embeddings)[0]
    
    # Get top candidates for reranking
    candidate_indices = np.argsort(-similarities)[:candidates]
    candidate_definitions = [definitions[i] for i in candidate_indices]
    
    # Prepare text pairs for reranking
    # We'll use both the code and documentation (if available) for better matching
    text_pairs = []
    for def_ in candidate_definitions:
        text = def_.code
        if def_.documentation:
            text = f"{def_.documentation}\n{text}"
        text_pairs.append([query, text])
    
    # Rerank candidates
    rerank_scores = reranker.predict(text_pairs, convert_to_tensor=True).to('cpu')
    
    # Get top k results after reranking
    reranked_indices = np.argsort(-rerank_scores)[:top_k]
    results = [(candidate_definitions[i], rerank_scores[i]) for i in reranked_indices]
    
    # Print results
    print("\nSearch Results:")
    print("==============")
    for definition, score in results:
        print(f"\nReranker Score: {score:.3f}")
        print(f"File: {definition.file_name}")
        print(f"Type: {definition.type}")
        print(f"Identifier: {definition.identifier}")
        if definition.documentation:
            print(f"Documentation: {definition.documentation}")
        print(f"Code:\n{definition.code}\n")
        print("-" * 80)

if __name__ == '__main__':
    cli()
