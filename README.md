---
# 🧠 Semantic Cache & Clustering Engine

A production-ready semantic search service built for the **Trademarkia AI/ML Task**. This system leverages the **20 Newsgroups** dataset to provide high-speed, intent-aware retrieval using fuzzy clustering and a "first principles" semantic cache.
---

## 🚀 Quick Start (Production-Ready)

### Option A: Docker Compose (Recommended)

The most robust way to run the service. This handles environment isolation, system dependencies for ChromaDB, and data persistence.

```bash
# Build and start the service
docker-compose up --build

# The service will be available at http://localhost:8000
# Documentation (Swagger) is at http://localhost:8000/docs

```

### Option B: Local Setup (using `uv`)

If you prefer to run natively, ensure you have [uv](https://github.com/astral-sh/uv) installed.

```bash
# 1. Install dependencies
uv sync

# 2. Build index & train Fuzzy Clusters
uv run scripts/ingest_and_cluster.py

# 3. Start the FastAPI server
uv run uvicorn app.main:app --reload

```

---

## 🧠 Architectural Design Decisions

### 1. Data Cleaning & Noise Reduction (Part 1)

The 20 Newsgroups corpus contains significant metadata noise.

- **Choices:** I implemented regex-based stripping of email headers (`Path`, `Lines`, `Organization`), PGP signatures, and nested quote blocks (`>`).
- **Justification:** This prevents the model from clustering documents based on "email style" or shared signatures, forcing it to focus on the **core semantic body** of the post.

### 2. Fuzzy Clustering via GMM (Part 2)

The task required a "distribution, not a label."

- **Model:** I used **Gaussian Mixture Models (GMM)** with a tied covariance matrix.
- **Fuzzy Assignment:** Unlike K-Means, GMM provides a probability vector across all 20 topics. This captures documents that straddle multiple domains (e.g., a post about "Encryption Laws" belonging to both _Politics_ and _Cryptography_).
- **Proof:** Run `uv run scripts/analyze_clusters.py` to see the semantic consistency of the generated clusters.

### 3. "First Principles" Semantic Cache (Part 3)

Built to demonstrate an understanding of algorithmic efficiency without relying on third-party Redis modules.

- **Cluster-Aware Retrieval:** Queries are first assigned to a **Dominant Cluster**. Similarity search is then restricted to that cluster's specific "bucket" in memory.
- **Scalability:** This reduces lookup complexity from $O(N)$ to approximately $O(N/20)$, ensuring the system remains fast as the cache grows.
- **Tunable Decision ($\tau$):** The similarity threshold is set to **0.88**.
- _High (0.95+):_ High precision, but misses synonyms (acts like a hash map).
- _Low (0.75):_ High hit rate, but risks "semantic hallucinations."
- _Sweet Spot (0.88):_ Correct matches "fix a flat tire" with "repairing a punctured wheel."

---

## 🛠️ API Documentation

| Method   | Endpoint       | Description                                               |
| -------- | -------------- | --------------------------------------------------------- |
| `POST`   | `/query`       | Embeds query, checks cluster-aware cache, then Vector DB. |
| `GET`    | `/cache/stats` | Returns hit rate, miss count, and total entries.          |
| `DELETE` | `/cache`       | Flushes the in-memory cache and resets statistics.        |

---

## 📊 System Validation

To verify that the system correctly distinguishes between a **Cache Miss** and a **Semantic Hit**, run:

```bash
uv run scripts/test_api.py

```

This script performs a cold query, then a semantically similar query, and validates that the `hit_rate` updates correctly using the `CacheStats` Pydantic model.

---

## 📂 Project Structure

```text
├── app/
│   ├── main.py           # FastAPI orchestration & lifespan management
│   ├── cache.py          # Custom Semantic Cache logic
│   ├── models.py         # Pydantic Request/Response schemas
│   └── __init__.py
├── scripts/
│   ├── ingest_and_cluster.py  # Data ingestion, GMM training, and DB storage
│   ├── analyze_clusters.py    # Cluster semantic analysis tool
│   └── test_api.py            # Automated API testing suite
├── data/                 # Persistent Vector DB storage (Volume mapped)
├── Dockerfile            # System-level dependencies & build logic
├── docker-compose.yml    # Orchestration & volume configuration
└── pyproject.toml        # Dependency management

```

---
