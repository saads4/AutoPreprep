# AutoPreprocessor

An experimental AutoML preprocessing engine that combines traditional rule-based preprocessing with a lightweight Retrieval-Augmented Generation (RAG) system to automatically suggest encoding strategies for tabular datasets.

The goal of this project is to make preprocessing more intelligent and dataset-agnostic without relying on external LLM APIs.

---

## Features

### Missing Value Handling

* Mean Imputation
* Median Imputation
* Mode Imputation

### Encoding

* Label Encoding for binary columns
* One-Hot Encoding
* RAG-assisted Ordinal Encoding

### Scaling

* Standard Scaling
* Robust Scaling
* Detection of columns that do not require scaling

### Feature Elimination

* Automatic dropping of columns with excessive missing values

---

## RAG-Based Ordinal Detection

Instead of using an external LLM, this project uses a custom RAG system built using:

* Sentence Transformers (`all-MiniLM-L6-v2`)
* ChromaDB
* Pattern files containing ordinal relationships

Example patterns:

```text
Poor -> Fair -> Good -> Excellent

Low -> Medium -> High -> Critical

Beginner -> Intermediate -> Advanced -> Expert

Bronze -> Silver -> Gold -> Platinum

Strongly Disagree -> Disagree -> Neutral -> Agree -> Strongly Agree
```

When categorical values resemble known ordinal patterns, the system recommends `OrdinalEncoder`; otherwise, it falls back to `OneHotEncoder`.

---

## Project Structure

```text
AutoPreprocessor/
|
|-- data/                  # Ordinal pattern files
|-- chroma_db/             # Local generated ChromaDB storage
|-- AutoPreprocessor.ipynb # Main preprocessing notebook
|-- rag_ingestion.ipynb    # RAG ingestion notebook
|-- requirements.txt       # Python dependencies
|-- README.md
```

`chroma_db/` is generated locally by `rag_ingestion.ipynb` and is intentionally ignored by Git.

---

## Tech Stack

* Python
* Pandas
* NumPy
* Sentence Transformers
* ChromaDB
* LangChain
* tqdm

---

## Setup

1. Create and activate a virtual environment.

```bash
python -m venv .venv
.venv\Scripts\activate
```

2. Install dependencies.

```bash
pip install -r requirements.txt
```

---

## Usage

1. Run `rag_ingestion.ipynb`.

This loads the `.txt` files from `data/`, generates embeddings with `SentenceTransformer("all-MiniLM-L6-v2")`, recreates the `rag_collection` collection, and stores it in the local `chroma_db/` directory.

2. Run `AutoPreprocessor.ipynb`.

When prompted, enter the path to your CSV file. The notebook reads the dataset, classifies columns, queries the existing ChromaDB collection for ordinal encoding hints, and prints the preprocessing groups.

Re-run `rag_ingestion.ipynb` only when the `.txt` reference files in `data/` change.

---

## Current Status

This project is still under active development.

Current capabilities:

* Rule-based preprocessing
* RAG-based ordinal detection
* Persistent vector database using ChromaDB

Planned improvements:

* Support for one-hot pattern retrieval
* Automatic sklearn Pipeline generation
* `fit()` and `transform()` methods
* Feature type inference
* Time-series support
* Better scaling heuristics
* PyPI package release

---

## Motivation

Most preprocessing libraries rely heavily on manually specified rules.

This project explores whether a lightweight RAG system can help automatically infer preprocessing decisions and generalize across different domains such as:

* Housing
* Finance
* Banking
* Insurance
* Healthcare
* Cybersecurity

without depending on external APIs or large language models.

---

## Contributions

Suggestions, bug reports, and ideas are always welcome.

If you have any feedback or would like to discuss improvements, feel free to:

* Open an issue
* Submit a pull request
* Connect with me on LinkedIn

**LinkedIn:** www.linkedin.com/in/saad-sayed-a04b0932b



---

## Disclaimer

This project is primarily a learning and research project and should not be considered production-ready yet.

The focus is on understanding and experimenting with intelligent preprocessing systems from scratch.

---

## Star the Repository

If you find this project interesting or useful, consider giving it a star. It helps others discover the project and motivates further development.
