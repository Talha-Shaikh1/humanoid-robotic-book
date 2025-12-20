---
id: 2
title: "RAG Chatbot Implementation Plan"
stage: "plan"
date: "2025-12-17"
surface: "agent"
model: "claude-sonnet-4-5-20250929"
feature: "rag-chatbot"
branch: "1-rag-chatbot"
user: "user"
command: "/sp.plan"
labels:
  - "plan"
  - "architecture"
  - "ingestion"
  - "rag"
tests: []
links:
  spec: "specs/1-rag-chatbot/spec.md"
  ticket: null
  adr: null
  pr: null
files:
  - "specs/1-rag-chatbot/plan.md"
  - "specs/1-rag-chatbot/plan/research.md"
  - "specs/1-rag-chatbot/plan/data-model.md"
  - "specs/1-rag-chatbot/plan/quickstart.md"
---

# RAG Chatbot Implementation Plan

## Prompt Text

Create:
An end-to-end implementation plan for Part 2 (RAG pipeline), starting with content ingestion and vector database setup.

---

Phase 1: Project Initialization (Foundation)
- Initialize a new backend project using `uv`.
- Create a single Python entry file (e.g., ingest.py) for the entire ingestion pipeline.
- Configure environment variables for:
  - Cohere API Key
  - Qdrant Cloud URL
  - Qdrant API Key
- Use a clean, minimal dependency set (requests/httpx, BeautifulSoup, cohere, qdrant-client).

Acceptance Criteria:
- Project runs via `uv run`.
- Environment loads correctly.
- Single-file ingestion script executes without errors.

---

Phase 2: URL Discovery & Crawling (Research + Implementation)
Target Website:
https://humanoid-robotic-book-eight.vercel.app/

Steps:
- Fetch the homepage HTML.
- Extract all internal links belonging to the same domain.
- Normalize and deduplicate URLs.
- Filter out non-content pages (e.g., assets, search, anchors).
- Store discovered URLs in memory.

Decisions to Document:
- Crawl depth (single-level vs recursive).
- URL filtering rules.
- Handling relative vs absolute URLs.

Acceptance Criteria:
- All book-related pages are discovered.
- No duplicate or external URLs included.

---

Phase 3: Content Extraction & Cleaning
Steps:
- Fetch HTML content for each discovered URL.
- Extract main readable content (headings, paragraphs, lists).
- Remove navigation, footer, sidebar, and script/style tags.
- Preserve heading hierarchy (H1–H3) for later chunk metadata.

Decisions to Document:
- HTML selectors used for content extraction.
- How heading hierarchy is preserved.

Acceptance Criteria:
- Clean, readable text extracted per page.
- Content matches what a human sees on the page.

---

Phase 4: Chunking Strategy
Steps:
- Split extracted text into semantic chunks.
- Chunk rules:
  - Heading-aware splitting.
  - Fixed token/character window with overlap.
- Attach metadata to each chunk:
  - source_url
  - page_title
  - heading
  - chunk_index

Decisions to Document:
- Chunk size and overlap.
- Why heading-aware chunking was chosen.

Acceptance Criteria:
- Chunks are neither too large nor too small.
- Each chunk is independently meaningful.

---

Phase 5: Qdrant Vector Database Setup
Steps:
- Create a Qdrant collection (if not exists).
- Use cosine similarity.
- Define vector size based on Cohere embedding model.
- Define payload schema for metadata and raw text.

Decisions to Document:
- Collection naming strategy.
- Vector configuration.

Acceptance Criteria:
- Collection successfully created.
- Collection visible in Qdrant dashboard.

---

Phase 6: Embedding Generation (Cohere)
Steps:
- Initialize Cohere embedding model.
- Generate embeddings for each chunk.
- Ensure batch embedding for efficiency.
- Validate embedding vector dimensions.

Decisions to Document:
- Selected Cohere model.
- Batch size tradeoffs.

Acceptance Criteria:
- Embeddings generated successfully.
- No dimension mismatch errors.

---

Phase 7: Vector Upsert to Qdrant
Steps:
- Upsert each chunk + embedding into Qdrant.
- Attach full metadata payload.
- Handle retries and partial failures gracefully.

Acceptance Criteria:
- All chunks stored in Qdrant.
- Sample similarity search returns relevant results.

---

Phase 8: Pipeline Testing & Validation
Steps:
- Run similarity search with known queries.
- Verify returned chunks match the book content.
- Log number of URLs processed, chunks created, and vectors stored.

Acceptance Criteria:
- Retrieval accuracy is acceptable.
- End-to-end pipeline works in a single run.

---

Technical Constraints:
- All ingestion logic must live in a single Python file.
- Use `uv` for environment and execution.
- No frontend or agent logic in this phase.
- Code must be readable, modular (via functions), and production-oriented.

---

Deliverable for This Plan:
- One executable ingestion script:
  - Crawls website
  - Extracts content
  - Chunks text
  - Generates Cohere embeddings
  - Stores vectors in Qdrant
- Documented decisions and validation results.

## Response Text

Successfully created a comprehensive implementation plan for the RAG chatbot content ingestion pipeline. The plan includes:

1. **Technical Context**: Outlined the system requirements and resolved all unknowns through research
2. **Constitution Check**: Verified alignment with project constitutional principles
3. **Gates**: Confirmed all pre-implementation requirements are met
4. **Research Outcomes**: Completed research on Cohere models, chunking strategies, Qdrant configuration, and batch processing
5. **Data Model**: Defined the EmbeddingChunk entity with proper fields and validation rules
6. **Quickstart Guide**: Created detailed setup and execution instructions
7. **Implementation Steps**: Detailed the 8-phase implementation process

Key decisions documented:
- Cohere embed-english-v3.0 model for technical content
- 512 token chunks with 128 token overlap for semantic chunking
- Cosine similarity with 1024 dimensions for Qdrant configuration
- Batch size of 96 for optimal Cohere API usage

The plan is structured as a single-file Python script using uv for dependency management and follows all technical constraints specified in the requirements.

## Outcome

The implementation plan is complete and ready for the next phase. All architectural decisions have been made, research completed, and documentation created. The plan provides a clear roadmap for implementing the content ingestion pipeline with all necessary details for successful execution.