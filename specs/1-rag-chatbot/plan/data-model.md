# Data Model: RAG Chatbot Content Ingestion

**Feature**: RAG Chatbot for Book
**Date**: 2025-12-17

## Entity: EmbeddingChunk

### Description
Represents a semantically meaningful chunk of content from the robotics textbook, with associated embedding vector and metadata for retrieval.

### Fields

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| id | String | Yes | Unique identifier for the chunk (UUID) |
| source_url | String | Yes | URL of the original source page |
| page_title | String | Yes | Title of the source page |
| heading | String | No | Main heading associated with this chunk (H1, H2, or H3) |
| chunk_index | Integer | Yes | Sequential index of this chunk within the page |
| content | String | Yes | The actual text content of the chunk |
| embedding | Array[Float] | Yes | Vector representation of the content (1024 dimensions) |
| created_at | DateTime | Yes | Timestamp when chunk was created |
| word_count | Integer | Yes | Number of words in the content |
| heading_hierarchy | Array[String] | No | Full heading hierarchy (e.g., ["Chapter 1", "Section 1.2", "Subsection 1.2.1"]) |

### Relationships
- None (standalone entity stored in vector database)

### Validation Rules
1. `source_url` must be a valid URL from the target domain (https://humanoid-robotic-book-eight.vercel.app/)
2. `content` must be non-empty and contain at least 10 words
3. `embedding` must have exactly 1024 dimensions (for Cohere embed-english-v3.0)
4. `chunk_index` must be a non-negative integer
5. `word_count` must be consistent with actual content length
6. `id` must be unique across all chunks

### Constraints
- Content should not exceed 1024 tokens to maintain semantic coherence
- Embedding vectors must be normalized for cosine similarity search
- Metadata fields should be properly escaped for JSON storage

## Entity: ProcessingLog

### Description
Tracks the ingestion process for monitoring and debugging purposes.

### Fields

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| id | String | Yes | Unique identifier for the log entry |
| url | String | Yes | URL being processed |
| status | String | Yes | Processing status (pending, success, failed) |
| error_message | String | No | Error details if processing failed |
| processed_at | DateTime | Yes | Timestamp when processing occurred |
| chunk_count | Integer | No | Number of chunks created from this URL |

### Validation Rules
1. `status` must be one of: "pending", "success", "failed"
2. `error_message` required when status is "failed"
3. `chunk_count` must be non-negative when status is "success"

## Entity: CrawlResult

### Description
Temporary entity for tracking crawled URLs during the discovery phase.

### Fields

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| url | String | Yes | The discovered URL |
| title | String | No | Page title (if available) |
| discovered_at | DateTime | Yes | When the URL was discovered |
| processed | Boolean | Yes | Whether this URL has been processed |

### Validation Rules
1. `url` must be a valid, well-formed URL
2. `processed` defaults to false
3. `url` must belong to the target domain