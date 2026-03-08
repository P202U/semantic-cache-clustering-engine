from pydantic import BaseModel
from typing import Optional


class QueryRequest(BaseModel):
    """Input schema for the /query endpoint."""

    query: str


class QueryResponse(BaseModel):
    """
    Output schema as specified in Part 4 requirements.
    Matches the JSON structure:
    { "query": "...", "cache_hit": bool, "matched_query": "...", ... }
    """

    query: str
    cache_hit: bool
    matched_query: Optional[str] = None
    similarity_score: Optional[float] = None
    result: str
    dominant_cluster: int


class CacheStats(BaseModel):
    """Schema for the /cache/stats endpoint."""

    total_entries: int
    hit_count: int
    miss_count: int
    hit_rate: float
