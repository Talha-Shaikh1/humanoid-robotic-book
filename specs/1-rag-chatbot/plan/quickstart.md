# Quickstart Guide: RAG Chatbot Content Ingestion Pipeline

**Feature**: RAG Chatbot for Book
**Date**: 2025-12-17

## Overview
This guide provides step-by-step instructions to set up and run the content ingestion pipeline for the RAG chatbot. The pipeline will crawl the humanoid robotics book website, extract content, chunk it semantically, generate embeddings, and store them in Qdrant Cloud.

## Prerequisites

### System Requirements
- Python 3.8 or higher
- `uv` package manager (https://github.com/astral-sh/uv)
- Access to Cohere API
- Access to Qdrant Cloud (Free Tier)

### API Access
- Cohere API Key (for embedding generation)
- Qdrant Cloud URL and API Key (for vector storage)

## Setup Instructions

### 1. Clone and Navigate to Project
```bash
# If starting from scratch, create a new backend directory
mkdir rag-backend && cd rag-backend
```

### 2. Install uv Package Manager (if not already installed)
```bash
# Install uv using the official installer
curl -LsSf https://astral.sh/uv/install.sh | sh
# Or on Windows with PowerShell:
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

### 3. Create Python Project with Dependencies
```bash
# Initialize a new Python project
uv init
# Or if you already have a project, just create a virtual environment
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

### 4. Install Required Dependencies
```bash
uv pip install requests beautifulsoup4 cohere qdrant-client python-dotenv
```

### 5. Configure Environment Variables
Create a `.env` file in your project root with the following content:

```env
COHERE_API_KEY=your_cohere_api_key_here
QDRANT_URL=your_qdrant_cloud_url_here
QDRANT_API_KEY=your_qdrant_api_key_here
TARGET_BASE_URL=https://humanoid-robotic-book-eight.vercel.app/
COLLECTION_NAME=robotics_book_chunks
```

**Note**: Replace placeholder values with your actual API keys and URLs.

## Running the Ingestion Pipeline

### 1. Create the Ingestion Script
Create a file named `ingest.py` with the complete ingestion pipeline code (details in the implementation section).

### 2. Execute the Pipeline
```bash
uv run ingest.py
```

### 3. Monitor the Process
The pipeline will:
1. Crawl the website and discover all content pages
2. Extract text content from each page
3. Chunk the content semantically
4. Generate embeddings using Cohere
5. Store vectors in Qdrant Cloud
6. Validate the results

## Expected Output
During execution, you should see:
- Progress indicators for each phase
- Count of discovered URLs
- Count of processed chunks
- Final validation results
- Qdrant collection statistics

## Verification Steps

### 1. Check Qdrant Cloud Dashboard
- Log into your Qdrant Cloud account
- Verify that the collection was created
- Check the number of stored vectors
- Verify the payload schema matches expectations

### 2. Test Similarity Search
Run a simple test to verify retrieval works:
```python
# Example test code
from qdrant_client import QdrantClient
from cohere import Client as CohereClient
import os

# Initialize clients
qdrant_client = QdrantClient(
    url=os.getenv("QDRANT_URL"),
    api_key=os.getenv("QDRANT_API_KEY")
)
cohere_client = CohereClient(api_key=os.getenv("COHERE_API_KEY"))

# Generate a test embedding
test_text = "What is ROS 2?"
embedding = cohere_client.embed(
    texts=[test_text],
    model="embed-english-v3.0",
    input_type="search_query"
).embeddings[0]

# Search in Qdrant
results = qdrant_client.search(
    collection_name=os.getenv("COLLECTION_NAME"),
    query_vector=embedding,
    limit=3
)

print("Top 3 similar chunks:")
for result in results:
    print(f"Score: {result.score}")
    print(f"Content: {result.payload['content'][:200]}...")
    print(f"Source: {result.payload['source_url']}")
    print("---")
```

## Troubleshooting

### Common Issues

1. **API Key Errors**
   - Verify your Cohere and Qdrant API keys are correct
   - Check that your `.env` file is properly formatted

2. **Crawling Issues**
   - Verify the target website is accessible
   - Check for robots.txt restrictions
   - Ensure the base URL is correct

3. **Rate Limiting**
   - If you encounter rate limit errors, add delays between API calls
   - Consider upgrading your API tier if processing large amounts of content

4. **Memory Issues**
   - For large sites, process in batches
   - Monitor memory usage during embedding generation

### Getting Help
- Check the implementation plan in `specs/1-rag-chatbot/plan.md` for detailed architecture
- Review the data models in `specs/1-rag-chatbot/plan/data-model.md`
- Examine the research outcomes in `specs/1-rag-chatbot/plan/research.md`