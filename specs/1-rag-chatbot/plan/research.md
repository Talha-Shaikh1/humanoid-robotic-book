# Research Outcomes: RAG Chatbot Content Ingestion

**Feature**: RAG Chatbot for Book
**Date**: 2025-12-17

## Cohere Model Selection

### Decision: Cohere embed-english-v3.0
**Rationale**: After researching Cohere's embedding models, embed-english-v3.0 is the optimal choice for technical documentation like a robotics textbook. This model is specifically optimized for retrieval tasks and performs well with educational content.

**Alternatives Considered**:
- `embed-multilingual-v3.0`: Could support Urdu translation in the future, but embed-english-v3.0 has better performance for English technical content
- `embed-english-light-v3.0`: Faster and less expensive, but lower quality for complex technical concepts
- `embed-3.0-multilingual`: Similar to embed-multilingual-v3.0 but with different performance characteristics

**Vector Dimensions**: 1024 dimensions for optimal performance with technical content.

## Chunking Strategy

### Decision: 512 token chunks with 128 token overlap, heading-aware splitting
**Rationale**: Based on research into semantic chunking for technical documentation, this configuration provides the best balance between:
- Semantic coherence (chunks contain complete thoughts/sections)
- Retrieval precision (not too large to dilute relevance)
- Processing efficiency (manageable chunk sizes)

**Technical Implementation**:
- Use heading hierarchy to maintain context (H1, H2, H3)
- Apply sliding window with overlap to preserve context across chunks
- Target ~512 tokens (approximately 384-512 words) per chunk
- 128 token overlap to maintain semantic continuity

**Alternatives Considered**:
- Fixed character chunks: Less semantic coherence
- Sentence-based chunks: Too granular for technical content
- Recursive splitting: May break semantic boundaries
- Header-based only: May create very large chunks

## Qdrant Collection Configuration

### Decision: Cosine similarity, 1024 dimensions, HNSW indexing
**Rationale**: This is the standard configuration for semantic search with Cohere embeddings. Cosine similarity is ideal for high-dimensional vector spaces like embeddings, and HNSW provides efficient approximate nearest neighbor search.

**Configuration Details**:
- Similarity: cosine
- Vector size: 1024 (matches Cohere embed-english-v3.0)
- Indexing: HNSW for performance
- Payload schema: URL, title, content, metadata

**Alternatives Considered**:
- Euclidean distance: Less appropriate for high-dimensional embedding spaces
- Dot product: Different characteristics but cosine is standard for embeddings
- Different indexing methods: HNSW offers best balance of speed and accuracy

## Batch Processing Parameters

### Decision: Batch size of 96 for Cohere embeddings
**Rationale**: Cohere's API has a limit of 96 texts per batch for most embedding models. This maximizes throughput while respecting API constraints. Testing shows this provides optimal balance between efficiency and rate limit compliance.

**Considerations**:
- API rate limits: 5 calls per second, 1000 texts per minute for free tier
- Memory usage: Larger batches require more memory
- Error handling: Smaller batches allow more granular error recovery

**Alternatives Considered**:
- Batch size 1: Simple but inefficient
- Batch size 50: Conservative approach but suboptimal efficiency
- Batch size 100: Exceeds API limits for most Cohere models