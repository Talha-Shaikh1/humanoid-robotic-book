import os
import time
import requests
import warnings
from bs4 import BeautifulSoup, XMLParsedAsHTMLWarning
from uuid import uuid4
from dotenv import load_dotenv

import cohere
from qdrant_client import QdrantClient
from qdrant_client.models import (
    VectorParams,
    Distance,
    Filter,
    FieldCondition,
    MatchValue
)

# ------------------ CONFIG ------------------

BOOK_NAME = "humanoid-robotic-book"
COLLECTION_NAME = "humanoid_book"

CHUNK_SIZE = 500
CHUNK_OVERLAP = 100

SITEMAP_URL = "https://humanoid-robotic-book-eight.vercel.app/sitemap.xml"

# ------------------ SILENCE XML WARNING ------------------

warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)

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

def get_urls_from_sitemap(sitemap_url: str) -> list[str]:
    print(f"🗺️ Fetching sitemap: {sitemap_url}")
    res = requests.get(sitemap_url, timeout=30)
    res.raise_for_status()

    soup = BeautifulSoup(res.text, "html.parser")
    urls = []

    for loc in soup.find_all("loc"):
        url = loc.text.strip()
        if "/docs/" in url:
            urls.append(url)

    print(f"✅ Found {len(urls)} doc URLs from sitemap")
    return urls


def extract_page_text(url: str) -> str:
    print(f"🔍 Fetching page: {url}")
    res = requests.get(url, timeout=30)

    if res.status_code != 200:
        print(f"❌ HTTP {res.status_code} for {url}")
        return ""

    soup = BeautifulSoup(res.text, "html.parser")
    main = soup.find("main")

    if not main:
        print("⚠️ No <main> tag found")
        return ""

    text = main.get_text(separator="\n").strip()

    if "Page Not Found" in text or "404" in text:
        print("❌ 404 detected, skipping")
        return ""

    return text


def chunk_text(text: str) -> list[str]:
    words = text.split()
    chunks = []
    start = 0

    while start < len(words):
        end = start + CHUNK_SIZE
        chunks.append(" ".join(words[start:end]))
        start += CHUNK_SIZE - CHUNK_OVERLAP

    return chunks


def ensure_payload_index():
    qdrant.create_payload_index(
        collection_name=COLLECTION_NAME,
        field_name="source_url",
        field_schema="keyword"
    )


def already_ingested(url: str) -> bool:
    points, _ = qdrant.scroll(
        collection_name=COLLECTION_NAME,
        limit=1,
        with_payload=True,
        scroll_filter=Filter(
            must=[
                FieldCondition(
                    key="source_url",
                    match=MatchValue(value=url)
                )
            ]
        )
    )
    return len(points) > 0


def embed_chunks(chunks: list[str], batch_size: int = 8) -> list[list[float]]:
    all_embeddings = []

    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]

        response = co.embed(
            texts=batch,
            model="embed-english-v3.0",
            input_type="search_document"
        )

        all_embeddings.extend(response.embeddings)

        # ⏸️ rate-limit protection
        time.sleep(1.5)

    return all_embeddings


def setup_collection():
    collections = [c.name for c in qdrant.get_collections().collections]

    if COLLECTION_NAME not in collections:
        print("📦 Creating Qdrant collection...")
        qdrant.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(
                size=1024,
                distance=Distance.COSINE
            )
        )
    else:
        print("✅ Qdrant collection already exists")


def store_embeddings(chunks, embeddings, source_url):
    points = []

    for text, vector in zip(chunks, embeddings):
        points.append({
            "id": str(uuid4()),
            "vector": vector,
            "payload": {
                "text": text,
                "source_url": source_url,
                "book": BOOK_NAME
            }
        })

    qdrant.upsert(
        collection_name=COLLECTION_NAME,
        points=points
    )

# ------------------ MAIN PIPELINE ------------------

def run():
    setup_collection()
    ensure_payload_index()

    urls = get_urls_from_sitemap(SITEMAP_URL)

    for url in urls:
        if already_ingested(url):
            print(f"⏭️ Already ingested, skipping: {url}")
            continue

        text = extract_page_text(url)
        if not text.strip():
            print(f"⚠️ Empty content, skipping: {url}")
            continue

        chunks = chunk_text(text)
        embeddings = embed_chunks(chunks)
        store_embeddings(chunks, embeddings, url)

        print(f"✅ Stored {len(chunks)} chunks from {url}")

    print("\n🎉 INGESTION COMPLETE (SAFE, NO DUPLICATES)")

# ------------------ ENTRY ------------------

if __name__ == "__main__":
    run()
