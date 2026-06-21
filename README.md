# AutoPreprocessor

An experimental AutoML preprocessing package that combines rule-based preprocessing with a lightweight Retrieval-Augmented Generation (RAG) system to automatically suggest encoding strategies for tabular datasets.

The project uses Sentence Transformers and ChromaDB locally. It does not rely on external LLM APIs.

## Features

- Mean, median, and mode imputation
- Label encoding for binary columns
- One-hot encoding
- RAG-assisted ordinal encoding
- Standard scaling and robust scaling
- Automatic dropping of columns with excessive missing values

## Installation

Install locally from the project root:

```bash
pip install .
```

For development, you can also install dependencies directly:

```bash
pip install -r requirements.txt
```

## Usage

```python
import pandas as pd
from autopreprocessor import AutoPreprocessor

data = pd.read_csv("sample.csv")

auto = AutoPreprocessor()
X_processed = auto.fit_transform(data)

print(auto.report_)
```

## RAG And ChromaDB

The package preserves the existing RAG behavior and ChromaDB usage. `AutoPreprocessor` loads the persistent ChromaDB collection named `rag_collection` from:

```text
autopreprocessor/chroma_db/
```

The ChromaDB path is resolved with `pathlib` relative to the installed package location, so it can work across machines and operating systems.

## Project Structure

```text
AutoPreprocessor/
|
|-- autopreprocessor/
|   |-- __init__.py
|   |-- auto_preprocessor.py
|   |-- chroma_db/
|
|-- requirements.txt
|-- setup.py
|-- pyproject.toml
|-- README.md
|-- .gitignore
```

## API

```python
from autopreprocessor import AutoPreprocessor
```

`AutoPreprocessor` exposes:

- `fit(data)`
- `transform(data)`
- `fit_transform(data)`
- `report_`

The preprocessing logic is preserved from the original notebook implementation.
