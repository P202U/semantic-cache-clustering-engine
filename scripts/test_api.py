import requests
import time

BASE_URL = "http://127.0.0.1:8000"


def test_flow():
    # 1. First Query (Will be a CACHE MISS)
    print("🔍 Query 1: 'How do I fix a flat tire?'")
    r1 = requests.post(
        f"{BASE_URL}/query", json={"query": "How do I fix a flat tire?"}
    ).json()
    print(f"   Status: {'✅ HIT' if r1['cache_hit'] else '❌ MISS'}")
    print(f"   Dominant Cluster: {r1['dominant_cluster']}")

    time.sleep(1)  # Dramatic pause

    # 2. Semantic Variation (Should be a CACHE HIT)
    # Different words, same meaning.
    print("\n🔍 Query 2: 'Repairing a punctured car wheel' (Semantic Match)")
    r2 = requests.post(
        f"{BASE_URL}/query", json={"query": "Repairing a punctured car wheel"}
    ).json()
    print(f"   Status: {'✅ HIT' if r2['cache_hit'] else '❌ MISS'}")
    print(f"   Similarity Score: {r2.get('similarity_score')}")
    print(f"   Matched Query: '{r2.get('matched_query')}'")

    # 3. Check Stats
    print("\n📊 Fetching Cache Stats...")
    stats = requests.get(f"{BASE_URL}/cache/stats").json()
    print(f"   Total Entries: {stats['total_entries']}")
    print(f"   Hit Rate: {stats['hit_rate']}")


if __name__ == "__main__":
    try:
        test_flow()
    except Exception as e:
        print(f"Ensure the server is running on {BASE_URL}. Error: {e}")
