import os
from dotenv import load_dotenv
import cohere
from qdrant_client import QdrantClient



# ------------------ CONFIG ------------------

COLLECTION_NAME = "humanoid_book"
TOP_K = 5

TEST_QUERIES = [
    "What is spec-driven development?",
    "Explain humanoid robotics in simple terms",
]

# ------------------ LOAD ENV ------------------

load_dotenv()

COHERE_API_KEY = os.getenv("COHERE_API_KEY")
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

# ------------------ CLIENTS ------------------

co = cohere.Client(COHERE_API_KEY)

qdrant = QdrantClient(
    url=QDRANT_URL,
    api_key=QDRANT_API_KEY
)

# ------------------ HELPERS ------------------

def embed_query(query: str):
    response = co.embed(
        texts=[query],
        model="embed-english-v3.0",
        input_type="search_query"
    )
    return response.embeddings[0]


def search_qdrant(query_vector):
    results = qdrant.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        limit=TOP_K,
        with_payload=True
    )
    return results.points


# ------------------ MAIN TEST ------------------

def run_tests():
    for query in TEST_QUERIES:
        print("\n" + "=" * 80)
        print(f"🔎 QUERY: {query}")

        query_vector = embed_query(query)
        results = search_qdrant(query_vector)

        if not results:
            print("❌ No results found")
            continue

        for i, hit in enumerate(results, start=1):
            payload = hit.payload
            print(f"\n--- Result {i} ---")
            print(f"Score: {hit.score:.4f}")
            print(f"Source: {payload.get('source_url')}")
            print("Text Preview:")
            print(payload.get("text")[:400], "...")
    
    print("\n✅ RETRIEVAL TEST COMPLETED")

# ------------------ ENTRY ------------------

if __name__ == "__main__":
    run_tests()
