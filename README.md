# Embedding Comparison Playground

A small FastAPI application that compares two embedding models:

- `sentence-transformers/all-MiniLM-L6-v2`
- `BAAI/bge-small-en-v1.5`

For three input chunks it calculates:

1. Cosine similarity
2. Euclidean distance
3. Dot product

The UI displays each metric as a 3x3 pairwise matrix for both models.

## Run locally

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
uvicorn main:app --reload
```

Open:

```text
http://127.0.0.1:8000
```

## Why this project?

The goal is to make embedding behavior observable instead of only studying the formulas. The same text chunks are encoded by two models, then their vector relationships are compared using three metrics.
