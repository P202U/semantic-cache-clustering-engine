import numpy as np
from typing import Optional, Dict, List, Any


class SemanticCache:
    def __init__(self, threshold: float = 0.85):
        """
        Part 3: Custom Semantic Cache
        - threshold: The cosine similarity cutoff (tunable decision).
        - store: Organized by cluster_id to ensure efficient lookup as cache grows.
        """
        self.threshold = threshold
        # Structure: { cluster_id: [ {"query_vec": np.array, "result": str, "query_text": str}, ... ] }
        self.store: Dict[int, List[Dict[str, Any]]] = {}

        # Stats tracking for Part 4
        self.hit_count = 0
        self.miss_count = 0

    def _cosine_similarity(self, v1: np.ndarray, v2: np.ndarray) -> float:
        """Calculates similarity between two vectors."""
        dot_product = np.dot(v1, v2)
        norm_v1 = np.linalg.norm(v1)
        norm_v2 = np.linalg.norm(v2)
        if norm_v1 == 0 or norm_v2 == 0:
            return 0.0
        return float(dot_product / (norm_v1 * norm_v2))

    def lookup(
        self, query_vector: np.ndarray, cluster_id: int
    ) -> Optional[Dict[str, Any]]:
        """
        Checks the cache for a similar query within the same cluster.
        This is the 'Cluster Membership' optimization required by the prompt.
        """
        if cluster_id not in self.store:
            self.miss_count += 1
            return None

        best_match = None
        highest_score = -1.0

        # Only iterate through queries in the same semantic neighborhood (cluster)
        for entry in self.store[cluster_id]:
            similarity = self._cosine_similarity(query_vector, entry["query_vec"])
            if similarity > highest_score:
                highest_score = similarity
                best_match = entry

        if highest_score >= self.threshold and best_match:
            self.hit_count += 1
            return {
                "result": best_match["result"],
                "matched_query": best_match["query_text"],
                "similarity_score": round(highest_score, 4),
            }

        self.miss_count += 1
        return None

    def update(
        self, query_text: str, query_vector: np.ndarray, cluster_id: int, result: str
    ):
        """Adds a new successful computation to the cache."""
        if cluster_id not in self.store:
            self.store[cluster_id] = []

        self.store[cluster_id].append(
            {"query_text": query_text, "query_vec": query_vector, "result": result}
        )

    def clear(self):
        """Reset for the DELETE /cache endpoint."""
        self.store = {}
        self.hit_count = 0
        self.miss_count = 0

    def get_stats(self) -> Dict[str, Any]:
        total = self.hit_count + self.miss_count
        hit_rate = (self.hit_count / total) if total > 0 else 0

        # Count total entries across all clusters
        total_entries = sum(len(entries) for entries in self.store.values())

        return {
            "total_entries": total_entries,
            "hit_count": self.hit_count,
            "miss_count": self.miss_count,
            "hit_rate": round(hit_rate, 4),
        }
