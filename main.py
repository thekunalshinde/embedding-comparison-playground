from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
import numpy as np

app = FastAPI(title="Embedding Comparison Playground")
app.mount("/static", StaticFiles(directory="static"), name="static")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Two small, practical open-source embedding models.
# MiniLM: 384 dimensions
# BGE-small: 384 dimensions
minilm = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
bge = SentenceTransformer("BAAI/bge-small-en-v1.5")


class EmbeddingRequest(BaseModel):
    chunks: list[str] = Field(..., min_length=2, max_length=3)


def calculate_metrics(embeddings: np.ndarray):
    return {
        "cosine_similarity": np.round(cosine_similarity(embeddings), 4).tolist(),
        "euclidean_distance": np.round(euclidean_distances(embeddings), 4).tolist(),
        "dot_product": np.round(embeddings @ embeddings.T, 4).tolist(),
    }


@app.get("/")
def root():
    return {"message": "Embedding Comparison Playground API is running"}


@app.post("/compare")
def compare_embeddings(request: EmbeddingRequest):
    chunks = [c.strip() for c in request.chunks]

    if any(not c for c in chunks):
        return {"error": "All chunks must contain text."}

    mini_embeddings = minilm.encode(chunks, convert_to_numpy=True)
    bge_embeddings = bge.encode(chunks, convert_to_numpy=True)

    return {
        "chunks": chunks,
        "models": {
            "MiniLM": {
                "dimensions": int(mini_embeddings.shape[1]),
                "metrics": calculate_metrics(mini_embeddings),
            },
            "BGE": {
                "dimensions": int(bge_embeddings.shape[1]),
                "metrics": calculate_metrics(bge_embeddings),
            },
        },
    }
