import re
import os
import numpy as np
import chromadb
from typing import cast
from sklearn.datasets import fetch_20newsgroups
from sklearn.utils import Bunch
from sklearn.mixture import GaussianMixture
from sentence_transformers import SentenceTransformer
from tqdm import tqdm


def clean_news_text(text: str) -> str:
    """
    Deliberate cleaning choices:
    1. Strip Headers: Metadata like 'Path:' or 'Lines:' adds zero semantic value.
    2. Remove Quotes: Re-quoted text creates 'semantic echoes' that skew clustering.
    3. Filter Signatures: Removes noise like names/phone numbers.
    """
    # Remove headers (From, Subject, etc.)
    text = re.sub(
        r"(?m)^(From|Subject|Reply-To|Organization|Lines|Nntp-Posting-Host|Distribution|Keywords|Summary|Article-I.D.):.*\n",
        "",
        text,
    )
    # Remove quoting (lines starting with > or |)
    text = re.sub(r"^[>|].*", "", text, flags=re.MULTILINE)
    # Remove common signature delimiters (dash-dash-space)
    text = re.sub(r"--\s*\n.*", "", text, flags=re.DOTALL)
    # Clean whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text


def run_ingestion_and_clustering():
    print("Fetching 20 Newsgroups dataset...")
    raw_data = fetch_20newsgroups(
        subset="all", remove=("headers", "footers", "quotes"), return_X_y=False
    )
    newsgroups = cast(Bunch, raw_data)

    docs = []
    metadata = []

    # 2. CLEAN DATA
    print("🧹 Cleaning and filtering corpus...")
    data_list = newsgroups.data
    target_list = newsgroups.target

    for i, doc in enumerate(tqdm(data_list, desc="Processing text")):
        cleaned = clean_news_text(doc)
        # Decision: Filter short docs (<100 chars) as they lack semantic depth for clustering
        if len(cleaned) > 100:
            docs.append(cleaned)
            metadata.append({"original_label": int(target_list[i]), "doc_id": i})

    print(f"✨ Kept {len(docs)} documents after filtering.")

    # 3. EMBEDDING
    print("🧠 Initializing 'all-MiniLM-L6-v2' and generating embeddings...")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(docs, show_progress_bar=True, batch_size=64)

    # 4. FUZZY CLUSTERING (Part 2)
    # Decision: K=20 to align with ground truth, but using GMM for fuzzy membership
    n_clusters = 20
    print(f"🧬 Fitting Gaussian Mixture Model (K={n_clusters}) for fuzzy clustering...")
    gmm = GaussianMixture(
        n_components=n_clusters, covariance_type="tied", random_state=42
    )
    gmm.fit(embeddings)

    # Get the distribution across clusters for each document
    # probs.shape is (n_samples, 20)
    probs = gmm.predict_proba(embeddings)

    # 5. VECTOR DB STORAGE (Part 1)
    print("Persisting to ChromaDB...")
    client = chromadb.PersistentClient(path="./data/vector_db")
    collection = client.get_or_create_collection(name="news_corpus")

    # Add to DB in batches
    batch_size = 2000
    for i in range(0, len(docs), batch_size):
        end = min(i + batch_size, len(docs))

        # We store the 'dominant_cluster' for efficient filtered retrieval in Part 3
        batch_metadatas = []
        for j in range(i, end):
            meta = metadata[j].copy()
            meta["dominant_cluster"] = int(np.argmax(probs[j]))
            # We can also store the confidence/entropy if needed
            meta["cluster_confidence"] = float(np.max(probs[j]))
            batch_metadatas.append(meta)

        collection.add(
            ids=[str(j) for j in range(i, end)],
            embeddings=embeddings[i:end].tolist(),
            documents=docs[i:end],
            metadatas=batch_metadatas,
        )

    print(
        f"Setup Complete: {len(docs)} documents indexed with fuzzy cluster assignments."
    )


if __name__ == "__main__":
    # Ensure data directory exists
    os.makedirs("./data/vector_db", exist_ok=True)
    run_ingestion_and_clustering()
