# RAG Demo

A demonstration package for Retrieval-Augmented Generation (RAG) implementations using a clean architecture approach.

## Overview

This package provides a structured implementation of RAG systems following clean architecture principles, separating core business logic from delivery mechanisms and implementation details.

## Project Structure

```
.
├── rag_demo/              # Main package directory
│   ├── core/             # Core business logic
│   │   ├── actions/      # Use cases and business operations
│   │   ├── contracts/    # Interfaces and protocols
│   │   ├── entities/     # Business objects and rules
│   │   └── values/       # Value objects and constants
│   ├── delivery/         # Delivery mechanisms
│   │   ├── cli/         # Command-line interface
│   │   └── rest/        # REST API endpoints
│   └── impl/            # Concrete implementations
├── tests/               # Test suite
├── misc/               # Miscellaneous resources, development and testing
    └── demo.ipynb     # Demo notebook
```

## Architecture

The project follows clean architecture principles with clear separation of concerns:

- **Core**: Contains all business logic and rules
  - **Actions**: Business use cases and operations
  - **Contracts**: Interface definitions
  - **Entities**: Business objects
  - **Values**: Value objects and constants
- **Delivery**: Different ways to interact with the system
  - **CLI**: Command-line interface
  - **REST**: REST API implementation
- **Implementation**: Concrete implementations of core contracts

## Prerequisites

Before installing the package, ensure you have the following installed:

- Python 3.13 
- Poetry (package manager)
- Ollama (for running local LLMs)

### Installing Prerequisites

1. Install Python 3.13 from [python.org](https://python.org)
2. Install Poetry:
   ```bash
   curl -sSL https://install.python-poetry.org | python3 -
   ```
3. Install Ollama by following instructions at [ollama.ai](https://ollama.ai)


## Installation

This project uses Poetry for dependency management. To install:

```bash
poetry install
```

To verify the installation, run:

```bash
poetry run python -c "import rag_demo; print('Installation successful!')"
```

## Development Workflow

Use the `misc/` directory for:
   - Creating proof of concepts
   - Building MVPs
   - Testing new features
   - Writing demonstration notebooks

### Code Quality

Before creating a commit, always run:

```bash
# Run ruff for linting and formatting
poetry run ruff check .

# Run mypy for type checking
poetry run mypy .

# Run tests
poetry run pytest
```


## License
[Add your license information here]

## Contributing

[Add contribution guidelines here]

## Contact

[Add contact information here]



## DEmo

API

request

```json
{
  "query": "What is the capital of France?",
}
```

response

```json
{
  "answer": "Paris",
}
```


Front end, its gonna be done with streamlit

