# Implementation Plan: RAG Chatbot Content Ingestion Pipeline

**Feature**: RAG Chatbot for Book
**Branch**: 1-rag-chatbot
**Created**: 2025-12-17
**Status**: Draft
**Author**: Claude Code

## Technical Context

This plan outlines the implementation of the content ingestion pipeline for the RAG chatbot. The pipeline will crawl the humanoid robotics book website, extract content, chunk it semantically, generate embeddings using Cohere, and store the vectors in Qdrant Cloud.

### Target System
- **Website**: https://humanoid-robotic-book-eight.vercel.app/
- **Technology Stack**: Python with uv, Cohere embeddings, Qdrant Cloud
- **Architecture**: Single-file ingestion script with modular functions

### Known Requirements
- Crawl all public URLs of the deployed Docusaurus website
- Generate semantic embeddings for content chunks
- Store embeddings, metadata, and raw text in Qdrant Cloud
- Support two retrieval modes: full-book search and user-selected-text-only search

### Unknowns (NEEDS CLARIFICATION)
- All unknowns have been resolved through research phase.

## Constitution Check

### Alignment with Constitutional Principles

✅ **Technical Accuracy and Documentation Compliance**: The ingestion pipeline will extract content exactly as presented on the website, maintaining the integrity of the educational material.

✅ **Strict Information Grounding**: The RAG system will be built to reference ONLY book content for answers, preventing hallucination.

✅ **AI-Native Content Generation**: This implementation follows the spec.md → plan.md → tasks.md → implementation pipeline.

✅ **Reproducibility and Testing Standards**: The pipeline will be built with testing and validation components to ensure consistent results.

### Potential Violations
None identified - the implementation aligns with all constitutional principles.

## Gates

### Pre-Implementation Gates

1. **Environment Setup**: Ensure `uv` is available and dependencies can be installed
2. **API Access**: Verify Cohere and Qdrant Cloud credentials work
3. **Website Accessibility**: Confirm target website is accessible and crawlable
4. **Architecture Review**: Single-file design meets technical constraints

**Status**: All gates pass - proceeding with implementation.

## Phase 0: Research & Discovery

### Research Tasks

1. **Cohere Model Selection**
   - Task: Research optimal Cohere embedding model for technical documentation
   - Expected outcome: Select most appropriate model for book content

2. **Chunking Strategy**
   - Task: Research semantic chunking strategies for technical documentation
   - Expected outcome: Define optimal chunk size and overlap parameters

3. **Qdrant Configuration**
   - Task: Research optimal Qdrant collection settings for this use case
   - Expected outcome: Define vector dimensions and collection parameters

4. **Batch Processing**
   - Task: Research optimal batch sizes for Cohere embeddings
   - Expected outcome: Define efficient batch processing parameters

### Research Outcomes (research.md)

#### Decision: Cohere Embedding Model Selection
- **Chosen**: Cohere embed-english-v3.0
- **Rationale**: Best suited for technical documentation and educational content, optimized for retrieval tasks
- **Alternatives considered**: embed-multilingual-v3.0 (for potential Urdu support), embed-english-light-v3.0 (for performance)

#### Decision: Chunking Strategy
- **Chosen**: 512 token chunks with 128 token overlap, heading-aware splitting
- **Rationale**: Balances semantic coherence with retrieval precision for technical content
- **Alternatives considered**: Fixed character chunks, sentence-based chunks

#### Decision: Qdrant Collection Configuration
- **Chosen**: Cosine similarity, vector size 1024 (for embed-english-v3.0), HNSW indexing
- **Rationale**: Standard configuration for semantic search with good performance
- **Alternatives considered**: Euclidean distance, different vector sizes

#### Decision: Batch Processing Parameters
- **Chosen**: Batch size of 96 for Cohere embeddings
- **Rationale**: Optimizes API usage while respecting rate limits
- **Alternatives considered**: Various batch sizes from 10 to 100

## Phase 1: Design & Architecture

### Data Model (data-model.md)

#### EmbeddingChunk Entity
- **id**: Unique identifier for the chunk
- **source_url**: URL of the source page
- **page_title**: Title of the source page
- **heading**: Main heading associated with this chunk
- **chunk_index**: Position of this chunk within the page
- **content**: The actual text content of the chunk
- **embedding**: Vector representation of the content
- **created_at**: Timestamp when chunk was created

#### Validation Rules
- source_url must be a valid URL from the target domain
- content must not be empty
- embedding must have correct dimensions (1024 for Cohere embed-english-v3.0)
- chunk_index must be non-negative integer

### API Contracts

Since this is an ingestion pipeline, there are no external API contracts. The internal functions will follow these patterns:

- `crawl_website(base_url: str) -> List[str]` - Discover all URLs
- `extract_content(url: str) -> Dict[str, str]` - Extract content from a URL
- `chunk_content(text: str, url: str, title: str) -> List[Dict]` - Create semantic chunks
- `generate_embeddings(chunks: List[Dict]) -> List[Dict]` - Add embeddings to chunks
- `store_in_qdrant(chunks: List[Dict]) -> None` - Store in vector database

### Quickstart Guide (quickstart.md)

#### Setting Up the Ingestion Pipeline

1. **Install Dependencies**
   ```bash
   uv venv
   source .venv/bin/activate  # On Windows: .venv\\Scripts\\activate
   uv pip install requests beautifulsoup4 cohere qdrant-client python-dotenv
   ```

2. **Configure Environment Variables**
   Create `.env` file with:
   ```
   COHERE_API_KEY=your_cohere_api_key
   QDRANT_URL=your_qdrant_cloud_url
   QDRANT_API_KEY=your_qdrant_api_key
   ```

3. **Run the Ingestion Pipeline**
   ```bash
   uv run ingest.py
   ```

4. **Verify Results**
   - Check Qdrant Cloud dashboard for created collection
   - Verify vector count matches expected number of chunks
   - Test similarity search with sample queries

## Phase 2: Implementation Plan

### Project Structure
```
backend/
├── ingest.py                 # Main ingestion script
├── requirements.txt          # Dependencies
└── .env                     # Environment variables
```

### Implementation Steps

#### Step 1: Project Initialization
- Set up uv project
- Create single ingestion script (ingest.py)
- Add dependency management
- Configure environment loading

#### Step 2: URL Discovery & Crawling
- Implement web crawler for target domain
- Extract and normalize internal links
- Filter out non-content pages
- Store discovered URLs

#### Step 3: Content Extraction
- Fetch HTML content for each URL
- Extract main content using appropriate selectors
- Remove navigation, footer, and other non-content elements
- Preserve heading hierarchy

#### Step 4: Semantic Chunking
- Implement heading-aware chunking strategy
- Apply token/character limits with overlap
- Attach metadata to each chunk

#### Step 5: Qdrant Setup
- Create collection with proper configuration
- Define payload schema
- Set up similarity search parameters

#### Step 6: Embedding Generation
- Initialize Cohere client
- Generate embeddings for all chunks
- Handle batch processing efficiently

#### Step 7: Vector Storage
- Upsert embeddings to Qdrant
- Attach full metadata payload
- Handle errors and retries

#### Step 8: Validation & Testing
- Implement sample similarity searches
- Verify content accuracy
- Log processing metrics

### Dependencies
- requests: For HTTP requests
- beautifulsoup4: For HTML parsing
- cohere: For embedding generation
- qdrant-client: For vector database operations
- python-dotenv: For environment variable management

## Success Criteria

### Technical Success
- [ ] All book pages crawled successfully
- [ ] Content extracted accurately without noise
- [ ] Chunks created with appropriate size and overlap
- [ ] Embeddings generated without errors
- [ ] Vectors stored in Qdrant collection
- [ ] Similarity search returns relevant results

### Performance Success
- [ ] Pipeline completes within reasonable time
- [ ] Proper error handling and retry logic
- [ ] Memory usage remains reasonable for large sites
- [ ] API rate limits respected

### Quality Success
- [ ] Content fidelity maintained from source to chunks
- [ ] Metadata properly attached to each chunk
- [ ] Duplicate content handled appropriately
- [ ] Invalid content filtered out