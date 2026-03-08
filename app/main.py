import numpy as np
import chromadb
from fastapi import FastAPI, HTTPException
from contextlib import asynccontextmanager
from sentence_transformers import SentenceTransformer
from chromadb.api.models.Collection import Collection
from chromadb.api import ClientAPI
from pydantic import BaseModel
from typing import Optional, cast, Any

# Import the custom cache
from .cache import SemanticCache


# --- Internal State Management ---
class AppState:
    def __init__(self):
        self.model: Optional[SentenceTransformer] = None
        self.db_client: Optional[ClientAPI] = None
        self.collection: Optional[Collection] = None
        self.cache = SemanticCache(threshold=0.88)


state = AppState()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Handles startup and shutdown of heavy resources."""
    print("Initializing Semantic Search Service...")

    # Initialize and assign with explicit casting
    state.model = SentenceTransformer("all-MiniLM-L6-v2")
    state.db_client = cast(
        ClientAPI, chromadb.PersistentClient(path="./data/vector_db")
    )
    state.collection = state.db_client.get_collection(name="news_corpus")

    yield
    print("🛑 Shutting down...")


app = FastAPI(title="Trademarkia Semantic Search", lifespan=lifespan)


# --- Schemas ---
class QueryRequest(BaseModel):
    query: str


class QueryResponse(BaseModel):
    query: str
    cache_hit: bool
    matched_query: Optional[str] = None
    similarity_score: Optional[float] = None
    result: str
    dominant_cluster: int


# --- Helper Logic ---
def get_dominant_cluster(query_vec: np.ndarray) -> int:
    """Extracts the cluster ID from the nearest neighbor in ChromaDB."""
    if state.collection is None:
        return 0

    res = state.collection.query(query_embeddings=[query_vec.tolist()], n_results=1)

    if res["metadatas"] and len(res["metadatas"][0]) > 0:
        meta = res["metadatas"][0][0]
        if meta and "dominant_cluster" in meta:
            return int(cast(Any, meta["dominant_cluster"]))
    return 0


# --- Endpoints ---


@app.post("/query", response_model=QueryResponse)
async def perform_query(request: QueryRequest):
    if state.model is None or state.collection is None:
        raise HTTPException(status_code=503, detail="Model/DB not initialized")

    query_text = request.query

    # FORCE conversion to numpy array to satisfy the Cache/GMM functions
    # SentenceTransformer often returns a torch.Tensor by default
    raw_embedding = state.model.encode(query_text, convert_to_numpy=True)
    query_vec = cast(np.ndarray, raw_embedding)

    # 1. Determine Cluster
    cluster_id = get_dominant_cluster(query_vec)

    # 2. Check Semantic Cache
    cache_result = state.cache.lookup(query_vec, cluster_id)

    if cache_result:
        return QueryResponse(
            query=query_text,
            cache_hit=True,
            matched_query=cache_result["matched_query"],
            similarity_score=cache_result["similarity_score"],
            result=cache_result["result"],
            dominant_cluster=cluster_id,
        )

    # 3. Cache Miss: Perform Vector Search
    search_res = state.collection.query(
        query_embeddings=[query_vec.tolist()], n_results=1
    )

    if not search_res["documents"] or not search_res["documents"][0]:
        raise HTTPException(status_code=404, detail="No relevant documents found.")

    final_result = str(search_res["documents"][0][0])

    # 4. Update Cache
    state.cache.update(query_text, query_vec, cluster_id, final_result)

    return QueryResponse(
        query=query_text,
        cache_hit=False,
        result=final_result,
        dominant_cluster=cluster_id,
    )


@app.get("/cache/stats")
async def get_cache_stats():
    return state.cache.get_stats()


@app.delete("/cache")
async def clear_cache():
    state.cache.clear()
    return {"message": "Cache flushed and stats reset."}
