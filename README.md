# TypeScript Code Indexer

A command-line tool for creating searchable embeddings of TypeScript code. This tool indexes TypeScript projects by generating embeddings for code definitions and enables semantic code search capabilities.

## Features

- Indexes TypeScript/TSX files while respecting `.gitignore` patterns
- Generates embeddings for code definitions using the Jina AI embeddings model
- Stores indexed data efficiently using SQLite and NumPy
- Provides semantic code search functionality
- Supports documentation and type information
- Handles symlinks and project-specific ignore patterns

## Prerequisites

- Python 3.8+
- pip (Python package manager)

## Installation

1. Clone the repository:
```bash
git clone git@github.com:flatplate/code-indexer.git
cd git@github.com:flatplate/code-indexer.git
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install the dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Indexing a TypeScript Project

To index a TypeScript project:

```bash
python -m code_indexer index /path/to/typescript/project -o output_directory
```

Options:
- `--output`, `-o`: Output directory for the index (default: 'code_index')
- `--model`, `-m`: Name of the embedding model to use (default: "jinaai/jina-embeddings-v2-base-code")
- `--verbose`, `-v`: Enable verbose logging

### Searching the Index

To search through indexed code:

```bash
python -m code_indexer search /path/to/index "your search query" -k 5
```

Options:
- `--top-k`, `-k`: Number of results to return (default: 5)
- `--model`, `-m`: Name of the embedding model to use (must match the model used for indexing)

## Project Structure

```
.
├── code_indexer/
│   ├── __init__.py
│   ├── cli.py
│   └── code_transformer_test.py
├── README.md
├── requirements.txt
└── setup.py
```

## Output Format

The tool generates two main files in the output directory:
- `index.db`: SQLite database containing code definitions and metadata
- `embeddings.npy`: NumPy file containing the embedding vectors

### Database Schema

The SQLite database contains two tables:

1. `metadata`:
   - `key`: TEXT PRIMARY KEY
   - `value`: TEXT

2. `definitions`:
   - `id`: INTEGER PRIMARY KEY
   - `file_name`: TEXT
   - `identifier`: TEXT
   - `code`: TEXT
   - `line_start`: INTEGER
   - `line_end`: INTEGER
   - `char_start`: INTEGER
   - `char_end`: INTEGER
   - `type`: TEXT
   - `parent_identifier`: TEXT
   - `is_exported`: BOOLEAN
   - `documentation`: TEXT
   - `embedding_file`: TEXT
   - `embedding_index`: INTEGER

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Dependencies

- click: Command line interface creation
- numpy: Numerical computing and array operations
- sqlite3: Database management
- pathspec: .gitignore pattern matching
- typing: Type hints
- logging: Logging functionality
- jinaai/jina-embeddings-v2-base-code: Code embedding model

## Known Limitations

- Currently only supports TypeScript/TSX files
- Requires the specified embedding model to be accessible
- Large projects may require significant memory during indexing

## Support

For issues, questions, or contributions, please create an issue in the GitHub repository.

## Disclaimer

This readme is created by an LLM and I didn't check it so might have a lot of hallucinations.

Also this project is only at testing stage, was written in one afternoon mostly by an LLM.

## Plan

- Performance improvements
    - Performance sucks
    - Especially indexing
    - Search performance can be improved by letting a service run and not have to load the model and embeddings every time
        - Or for very large repositories we can also use some proper nn index
    - Indexing performance can be improved in a few way I think
        - Read and parse files in parallel
        - Parse multiple files before creating the embeddings for bigger batches
        - Use gpu/metal backend for embeddings
        - Keep track of the modification date of the files indexed, and compare for incremental indexing
- Testing
    - The treesitter queries don't match everything I want to match
    - Properly adding some test cases would be the first step to ensure it works
- Search improvements
    - Search for types of symbols, search a specific path, filename etc.
    - Mixed keyword search
    - Maybe add some reranker (jinaai apparently has a good model)
    - Auto document symbols that don't have a doc comment for better indexing performance
    - Rewrite queries to better align with the embedding model's expectations
- Add more metadata
- Integrating into neovim and github.com/flatplate/elelem.nvim
    - The main goal is to integrate this with llm coding assistant for smart context
- Crazy stuff   
    - Git commit search
    - Duplication detection

## License

The MIT License (MIT)

Copyright (c) 2025 Ural Bayhan

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

