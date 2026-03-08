````markdown
# Semantic Cache & Clustering Engine (Trademarkia AI/ML Task)

A lightweight semantic search system built on the **20 Newsgroups** dataset. This project implements fuzzy clustering for topic discovery, a custom "first principles" semantic cache, and a high-performance FastAPI service.

---

## 🚀 Quick Start (Production-Ready)

### Option A: Docker (Recommended)

The easiest way to run the service with all dependencies and pre-indexed data:

```bash
docker build -t trademarkia-engine .
docker run -p 8000:8000 trademarkia-engine
```
````

### Option B: Local Setup (using `uv`)

```bash
# Install dependencies
uv sync

# Build index, generate embeddings, and train GMM clusters
uv run scripts/ingest_and_cluster.py

# Start the FastAPI server
uv run uvicorn app.main:app --host 0.0.0.0 --port 8000

```

---

## 🧠 Architectural Design Decisions

### 1. Data Cleaning & Preprocessing

The 20 Newsgroups dataset is inherently noisy (email headers, PGP signatures, and nested quotes).

- **Deliberate Choice:** I implemented a regex-based pipeline to strip metadata and nested `>` quotes. This ensures the model embeds the **original content** rather than "email echoes," leading to much cleaner clusters.

### 2. Part 2: Fuzzy Clustering (GMM)

Instead of hard assignments like K-Means, I used **Gaussian Mixture Models (GMM)**.

- **The "Fuzzy" Factor:** GMM provides a probability distribution across all 20 clusters.
- **Justification:** Real-world data is messy; a post about "encryption laws" belongs to both _Politics_ and _Technology_. My system reflects this by assigning documents to clusters with varying degrees of membership.

### 3. Part 3: Semantic Cache (First Principles)

Built without Redis or external middleware to demonstrate core algorithmic understanding.

- **Cluster-Aware Efficiency:** The cache is partitioned by the **Dominant Cluster ID**. When a query comes in, the system only performs similarity checks within that cluster's bucket, preventing linear $O(N)$ lookup degradation as the cache grows.
- **The Tunable Decision ($\tau$):** I have set the similarity threshold to **0.88**.
- _Insight:_ A threshold of 0.95 acts like a hash-map (too strict), while 0.75 introduces semantic drift. 0.88 captures synonymous intent (e.g., "fix a car" vs "auto repair") while maintaining precision.

---

## 🛠️ API Endpoints

| Method   | Endpoint       | Description                                              |
| -------- | -------------- | -------------------------------------------------------- |
| `POST`   | `/query`       | Embeds query, checks semantic cache, and returns result. |
| `GET`    | `/cache/stats` | Returns hit/miss counts and cache efficiency.            |
| `DELETE` | `/cache`       | Flushes the in-memory cache and resets stats.            |

---

## 📊 System Validation

To verify the semantic cache and clustering, run the included test suite:

```bash
uv run scripts/test_api.py

```

This script demonstrates a **Cache Miss** followed by a **Semantic Hit** (using different words for the same intent), proving the cache's ability to recognize semantic similarity.

---

## 📂 Project Structure

```text
├── app/
│   ├── main.py          # FastAPI service & state management
│   ├── cache.py         # Custom Semantic Cache (First Principles)
│   └── models.py        # Pydantic schemas
├── scripts/
│   ├── ingest_and_cluster.py  # Part 1 & 2 logic
│   ├── analyze_clusters.py    # Cluster semantic proof
│   └── test_api.py            # API validation
├── data/                      # Vector DB storage (Gitignored)
├── Dockerfile                 # Containerization
└── pyproject.toml             # uv managed dependencies

```

```



### 💡 Final Steps
1. **Sync your `uv.lock`**: Run `uv lock` to make sure the environment is reproducible.
2. **Double-check the Form**: Fill out the [Trademarkia Submission Form](https://forms.gle/4RpHZpAi8rbG9QCE8).
3. **Collaborator Access**: Ensure `recruitments@trademarkia.com` has access to the private or public repo.

**You're all set! Do you need anything else before you hit "Submit"?**

```
