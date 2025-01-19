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
git clone [your-repo-url]
cd [repository-name]
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

## License

[Your chosen license]

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

## Troubleshooting

1. If you encounter memory issues during indexing:
   - Try processing fewer files at a time
   - Ensure sufficient RAM is available

2. If search results are unexpected:
   - Verify the same embedding model is used for indexing and searching
   - Check if the code was properly indexed by examining the database

## Support

For issues, questions, or contributions, please create an issue in the GitHub repository.

## Disclaimer

This readme is created by an LLM and I didn't check it so might have a lot of hallucinations.
