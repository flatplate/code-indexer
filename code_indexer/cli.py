from pathlib import Path
from sentence_transformers import CrossEncoder
from sentence_transformers.util import cos_sim
import click
from typing import List, Set, Dict, Any, Tuple
import pathspec
import numpy as np
from typing import List, Tuple
from .code_transformer_test import TypeScriptParser, CodeDefinition
from .codeembedder import CodeEmbedder
from .llama_cpp_python_embedder import LlamaCppCodeEmbedder
import sqlite3
import numpy as np
from datetime import datetime
import logging
import json
import torch
import os

from contextlib import contextmanager

@contextmanager
def memory_cleanup():
    try:
        yield
    finally:
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        elif hasattr(torch.mps, 'empty_cache'):
            torch.mps.empty_cache()

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
            embedding_file TEXT,
            embedding_index INTEGER
        )
    ''')
    
    conn.commit()
    return conn

def load_index(index_dir: Path) -> Tuple[List[CodeDefinition], np.ndarray]:
    """Load code definitions from SQLite + NumPy files"""
    db_path = index_dir / 'index.db'
    embeddings_path = index_dir / 'embeddings_*.npy'
    
    # Load embeddings with numeric sorting
    files = embeddings_path.parent.glob(embeddings_path.name)
    # Sort files by numeric index
    files = sorted(files, key=lambda x: int(x.stem.split('_')[1]))
    
    embeddings = []
    for file in files:
        embeddings.append(np.load(file))
    embeddings = np.concatenate(embeddings)
    print(f"Loaded {len(embeddings)} embeddings")
    
    # Load definitions from SQLite
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    
    definitions = []
    for row in c.execute('SELECT * FROM definitions where embedding_file IS NOT NULL ORDER BY id;'):
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
            embedding=embeddings[embedding_idx].tolist(),
            embedding_file=row[12],
            embedding_idx=embedding_idx
        )
        definitions.append(definition)
    
    conn.close()
    return definitions, embeddings

def load_gitignore(project_root: Path, config_file: Path = None, additional_patterns: List[str] = None) -> pathspec.PathSpec:
    """Load .gitignore patterns from the project root and additional sources"""
    patterns = []
    
    # Load patterns from .gitignore files
    current_dir = project_root
    while current_dir.exists():
        gitignore_path = current_dir / '.gitignore'
        if gitignore_path.is_file():
            with open(gitignore_path, 'r', encoding='utf-8') as f:
                dir_patterns = [line.strip() for line in f.readlines() 
                              if line.strip() and not line.startswith('#')]
                patterns.extend(dir_patterns)
        
        parent_dir = current_dir.parent
        if parent_dir == current_dir:
            break
        current_dir = parent_dir
    
    # Load patterns from config file if provided
    if config_file and config_file.is_file():
        try:
            with open(config_file, 'r', encoding='utf-8') as f:
                config = json.load(f)
                if 'ignore_patterns' in config:
                    patterns.extend(config['ignore_patterns'])
        except json.JSONDecodeError:
            logger.warning(f"Failed to parse config file: {config_file}")
    
    # Add additional patterns from command line
    if additional_patterns:
        patterns.extend(additional_patterns)
    
    # Add default patterns
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
    
    # Remove duplicates while preserving order
    patterns = list(dict.fromkeys(patterns))
    
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

def save_definitions_without_embeddings(conn, definitions: List[CodeDefinition]):
    """Save definitions to database without embeddings"""
    c = conn.cursor()
    for d in definitions:
        c.execute('''
            INSERT INTO definitions (
                file_name, identifier, code, line_start, line_end,
                char_start, char_end, type, parent_identifier,
                is_exported, documentation
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            d.file_name, d.identifier, d.code, d.line_start, d.line_end,
            d.char_start, d.char_end, d.type, d.parent_identifier,
            d.is_exported, d.documentation
        ))
    conn.commit()

def save_embeddings_batch(output_dir: Path, embeddings: List[np.ndarray], start_idx: int):
    """Save a batch of embeddings to the embeddings file"""
    files = set(os.listdir(output_dir))
    embeddings_path = output_dir / 'embeddings_0.npy'
    file_idx = 0
    while embeddings_path.name in files:
        file_idx += 1
        embeddings_path = output_dir / f'embeddings_{file_idx}.npy'
    embeddings_array = np.array(embeddings)
    
    # For first batch, create new file
    np.save(embeddings_path, embeddings_array)
    return embeddings_path

def save_embeddings_batch_db(conn, definitions: List[Tuple[ CodeDefinition, int ]], embedding_file: str):
    """Save embeddings for a batch of definitions"""
    c = conn.cursor()
    for d, idx in definitions:
        c.execute('''
            UPDATE definitions 
            SET embedding_file = ?, embedding_index = ?
            WHERE identifier = ? AND file_name = ?
        ''', (embedding_file.name, idx, d.identifier, d.file_name))
    conn.commit()

@cli.command()
@click.argument('project_dir', type=click.Path(exists=True, file_okay=False, dir_okay=True, path_type=Path))
@click.option('--use-gpu', '-g', default=True, help='Use GPU for embedding computation')
@click.option('--output', '-o', type=click.Path(file_okay=False, dir_okay=True, path_type=Path), 
              default='code_index', help='Output directory for the index')
@click.option('--batch-size', '-b', default=50, help='Batch size for processing files')
@click.option('--fetch-batch-size', default=500, help='Batch size for fetching definitions from database')
@click.option('--ignore', '-i', default=[], help='Patterns to ignore')
@click.option('--model', '-m', default="jinaai/jina-embeddings-v2-base-code",
              help='Name of the embedding model to use')
@click.option('--skip-parsing', '-s', default=False,
              help='Skip parsing and only create embeddings for existing definitions')
def index(
    project_dir: Path,
    output: Path,
    batch_size: int,
    model: str,
    ignore: List[str],
    skip_parsing: bool,
    fetch_batch_size: int,
    use_gpu: bool
):
    """Index a TypeScript project directory"""
    output.mkdir(parents=True, exist_ok=True)
    db_path = output / 'index.db'
    conn = init_database(db_path)

    if skip_parsing:
        logger.info("Skipping parsing and only creating embeddings for existing definitions")
    else:
        # Phase 1: Parse files and save to database
        parser = TypeScriptParser()
        
        gitignore_spec = load_gitignore(project_dir, additional_patterns=ignore)
        typescript_files = find_typescript_files(project_dir, gitignore_spec)
        
        logger.info("Phase 1: Parsing %d files...", len(typescript_files))
        definitions_without_embeddings = []
        for file_path in typescript_files:
            try:
                definitions = parser.parse_file(str(file_path))
                definitions_without_embeddings.extend(definitions)
            except Exception as e:
                logger.error(f"Error parsing {file_path}: {str(e)}")
        
        # Save definitions without embeddings first
        logger.info("Saving {} definitions to database...".format(len(definitions_without_embeddings)))
        save_definitions_without_embeddings(conn, definitions_without_embeddings)
    embedder = LlamaCppCodeEmbedder(use_gpu=use_gpu, batch_size=batch_size) # TODO allow specifying model

    cursor = conn.cursor()
    cursor.execute('SELECT count(*) FROM definitions WHERE embedding_file IS NULL')
    total_rows = cursor.fetchone()[0]
    logger.info(f"Total rows to process: {total_rows}")
    cursor.execute('SELECT * FROM definitions WHERE embedding_file IS NULL ORDER BY id;')
    
    processed_count = 0
    while True:
        with memory_cleanup():
            batch = cursor.fetchmany(fetch_batch_size)
            if not batch:
                break
            definitions = [
                CodeDefinition(
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
                    embedding=None
                ) for row in batch
            ]

            embedder.create_embeddings(definitions, batch_size=batch_size)

            batch_embeddings = [d.embedding for d in definitions if d.embedding is not None]
            embedding_file = save_embeddings_batch(output, batch_embeddings, processed_count)
            
            definitions_with_ids = [(d, i) for i, d in enumerate(definitions) if d.embedding is not None]
            save_embeddings_batch_db(conn, definitions_with_ids, embedding_file)
            
            processed_count += len(batch_embeddings)
            logger.info(f"Processed {processed_count}/{total_rows} embeddings")

            # Cleanup
            del definitions
            del batch
            del definitions_with_ids
            del batch_embeddings
            # Force garbage collection
            import gc
            gc.collect()
            
            # Clear CUDA cache if using GPU
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            # Clear MPS cache if using Apple Silicon
            elif hasattr(torch.mps, 'empty_cache'):
                torch.mps.empty_cache()
    
    conn.close()
    logger.info("Indexing complete!")

def extract_quoted_terms(query: str) -> Tuple[List[str], str]:
    """Extract quoted terms and return them along with the remaining query"""
    import re
    quoted_terms = re.findall(r'"([^"]*)"', query)
    # Remove quoted terms from the query
    remaining_query = re.sub(r'"[^"]*"', '', query).strip()
    return quoted_terms, remaining_query

@cli.command()
@click.argument('index_dir', type=click.Path(exists=True, file_okay=False, dir_okay=True, path_type=Path))
@click.argument('query', type=str)
@click.option('--top-k', '-k', default=5, help='Number of results to return')
@click.option('--use-gpu', '-g', default=True, help='Use GPU for embedding computation')
@click.option('--use-reranker', '-r', default=True, help='Use reranker')
@click.option('--model', '-m', default="jinaai/jina-embeddings-v2-base-code",
              help='Name of the embedding model to use')
@click.option('--reranker-model', default="jinaai/jina-reranker-v2-base-multilingual",
              help='Name of the reranker model to use')
@click.option('--candidates', '-c', default=20, 
              help='Number of initial candidates to consider for reranking')
def search(
    index_dir: Path,
    query: str,
    top_k: int,
    model: str,
    reranker_model: str,
    candidates: int,
    use_gpu: bool,
    use_reranker: bool
):
    """Search the code index for similar definitions"""
    # Load the index
    definitions, embeddings = load_index(index_dir)
    embeddings = embeddings.astype(np.float32)
    logger.info(f"Loaded {len(definitions)} definitions from index")

    # Extract quoted terms and remaining query
    quoted_terms, remaining_query = extract_quoted_terms(query)
    
    if quoted_terms:
        logger.info(f"Found quoted terms: {quoted_terms}")
        # Filter definitions that contain all quoted terms
        filtered_definitions = []
        filtered_embeddings = []
        
        for definition in definitions:
            matches_all_terms = True
            searchable_text = f"{definition.code} {definition.documentation or ''} {definition.identifier}".lower()
            
            for term in quoted_terms:
                if term.lower() not in searchable_text:
                    matches_all_terms = False
                    break
            
            if matches_all_terms:
                filtered_definitions.append(definition)
                filtered_embeddings.append(embeddings[definition.embedding_idx])
        
        if filtered_definitions:
            logger.info(f"Found {len(filtered_definitions)} definitions matching quoted terms")
            definitions = filtered_definitions
            embeddings = np.array(filtered_embeddings)
        else:
            logger.warning("No definitions found matching all quoted terms")
            return
    
    if not remaining_query and quoted_terms:
        # If only quoted terms were provided, sort by identifier length as a simple ranking
        results = [(d, 1.0) for d in sorted(definitions, key=lambda x: len(x.identifier))][:top_k]
    else:
        # Initialize provider and search
        provider = LlamaCppCodeEmbedder(use_gpu=use_gpu)
        
        # Use remaining query or full query if no quoted terms
        query_embedding = provider.embed_single(query)
        similarities = cos_sim(query_embedding, embeddings)[0]
        
        if use_reranker:
            reranker = CrossEncoder(
                reranker_model,
                automodel_args={"torch_dtype": "float16"},
                trust_remote_code=True,
            )
            
            candidate_indices = np.argsort(-similarities)[:candidates]
            candidate_definitions = [definitions[i] for i in candidate_indices]
            
            text_pairs = []
            for def_ in candidate_definitions:
                text = def_.code
                if def_.documentation:
                    text = f"{def_.documentation}\n{text}"
                text_pairs.append([query, text])
            
            rerank_scores = reranker.predict(text_pairs, convert_to_tensor=True).to('cpu')
            reranked_indices = np.argsort(-rerank_scores)[:top_k]
            results = [(candidate_definitions[i], rerank_scores[i]) for i in reranked_indices]
        else:
            top_indices = np.argsort(-similarities)[:top_k]
            results = [(definitions[i], similarities[i]) for i in top_indices]
    
    # Print results
    print("\nSearch Results:")
    print("==============")
    for definition, score in results:
        print(f"\nScore: {score:.3f}")
        print(f"File: {definition.file_name}")
        print(f"Embedding file: {definition.embedding_file}")
        print(f"Type: {definition.type}")
        print(f"Identifier: {definition.identifier}")
        if definition.documentation:
            print(f"Documentation: {definition.documentation}")
        print(f"Code:\n{definition.code}\n")
        print("-" * 80)

if __name__ == '__main__':
    cli()
