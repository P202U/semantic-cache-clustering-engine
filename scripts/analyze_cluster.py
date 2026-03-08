import chromadb
import numpy as np
from typing import Dict, List, Any, cast


def analyze():
    client = chromadb.PersistentClient(path="./data/vector_db")
    try:
        collection = client.get_collection(name="news_corpus")
    except Exception:
        print("❌ Collection 'news_corpus' not found. Run ingestion first.")
        return

    results = collection.get(include=["documents", "metadatas"])

    docs = results.get("documents")
    metas = results.get("metadatas")

    if docs is None or metas is None:
        print("📭 The collection is empty or data is missing.")
        return

    clusters: Dict[int, List[str]] = {}

    for doc, meta in zip(docs, metas):
        meta_data = cast(Dict[str, Any], meta)
        c_id = int(meta_data.get("dominant_cluster", 0))

        if c_id not in clusters:
            clusters[c_id] = []
        clusters[c_id].append(doc)

    print("\n" + "=" * 40)
    print("📊 CLUSTER SEMANTIC ANALYSIS")
    print("=" * 40)

    for c_id in sorted(clusters.keys()):
        cluster_docs = clusters[c_id]
        print(f"\n📂 Cluster {c_id} ({len(cluster_docs)} documents)")

        for i in range(min(2, len(cluster_docs))):
            snippet = cluster_docs[i][:90].replace("\n", " ").strip()
            print(f"  - {snippet}...")


if __name__ == "__main__":
    analyze()
