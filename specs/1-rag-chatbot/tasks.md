# Tasks: RAG Chatbot for Book - Content Ingestion & Embeddings

**Feature**: RAG Chatbot for Book
**Branch**: 1-rag-chatbot
**Created**: 2025-12-17
**Status**: Draft
**Author**: Claude Code

## Overview

This document outlines the implementation tasks for the RAG Chatbot content ingestion and embeddings system. The system will crawl the humanoid robotics book website, extract content, chunk it semantically, generate embeddings using Cohere, and store the vectors in Qdrant Cloud.

### Tech Stack
- Python with uv for dependency management
- requests for HTTP requests
- BeautifulSoup4 for HTML parsing
- Cohere for embedding generation
- Qdrant-client for vector database operations
- python-dotenv for environment management

### Project Structure
```
backend/
├── ingest.py                 # Main ingestion script
├── requirements.txt          # Dependencies
└── .env                     # Environment variables
```

## Dependencies

- Plan: specs/1-rag-chatbot/plan.md
- Spec: specs/1-rag-chatbot/spec.md
- Research: specs/1-rag-chatbot/plan/research.md
- Data Model: specs/1-rag-chatbot/plan/data-model.md

## Implementation Strategy

### MVP Scope
- Basic web crawling functionality
- Content extraction from HTML
- Simple embedding generation and storage
- Minimum viable ingestion pipeline

### Delivery Approach
- Incremental development following user story priorities
- Each user story delivers independently testable functionality
- Parallel execution where possible to optimize development time

## Phase 1: Setup Tasks

### Goal
Initialize project structure and configure environment for the ingestion pipeline.

- [X] T001 Create backend directory structure
- [X] T002 Initialize uv project in backend directory
- [X] T003 Create requirements.txt with dependencies: requests, beautifulsoup4, cohere, qdrant-client, python-dotenv
- [X] T004 Create .env file template with COHERE_API_KEY, QDRANT_URL, QDRANT_API_KEY, TARGET_BASE_URL, COLLECTION_NAME
- [X] T005 Create initial ingest.py file with imports and basic structure
- [X] T006 Set up logging configuration in ingest.py
- [X] T007 Create configuration module to handle environment variables

## Phase 2: Foundational Tasks

### Goal
Implement core utilities and foundational components needed by all user stories.

- [X] T010 Create web crawler utility function to fetch and parse HTML
- [X] T011 Implement URL normalization and validation functions
- [X] T012 Create content extraction utility using BeautifulSoup
- [X] T013 Implement semantic chunking function with heading-aware splitting
- [X] T014 Set up Qdrant client connection and collection creation
- [X] T015 Create Cohere client initialization function
- [X] T016 Implement embedding generation function with batching
- [X] T017 Create data models for EmbeddingChunk based on data-model.md

## Phase 3: [US1] Full Book Search Implementation

### User Story
As a reader of the humanoid robotics book, I want to ask questions about the book content using a chat interface so that I can quickly find relevant information without manually searching through pages.

### Goal
Implement the core ingestion pipeline that crawls the website, extracts content, chunks it, generates embeddings, and stores in Qdrant.

### Independent Test Criteria
- The system can crawl all book pages successfully
- Content is extracted accurately without noise
- Chunks are created with appropriate size and overlap
- Embeddings are generated without errors
- Vectors are stored in Qdrant collection
- Sample similarity search returns relevant results

### Tasks

- [X] T020 [US1] Implement website crawling function to discover all URLs from https://humanoid-robotic-book-eight.vercel.app/
- [X] T021 [US1] Create URL filtering function to exclude non-content pages (assets, search, anchors)
- [X] T022 [US1] Implement content extraction function to get main readable content from each URL
- [X] T023 [US1] Create HTML cleaning function to remove navigation, footer, sidebar, and script/style tags
- [X] T024 [US1] Implement heading hierarchy preservation in content extraction
- [X] T025 [US1] Create semantic chunking function with 512 token chunks and 128 token overlap
- [X] T026 [US1] Implement chunk metadata attachment (source_url, page_title, heading, chunk_index)
- [X] T027 [US1] Create Qdrant collection setup function with cosine similarity and 1024 dimensions
- [X] T028 [US1] Implement embedding generation using Cohere embed-english-v3.0 model
- [X] T029 [US1] Create batch embedding function with batch size of 96 for efficiency
- [X] T030 [US1] Implement vector upsert function to store chunks and embeddings in Qdrant
- [X] T031 [US1] Add full metadata payload to Qdrant including URL, title, content, and chunk information
- [X] T032 [US1] Implement error handling and retry logic for API calls
- [X] T033 [US1] Create validation function to test similarity search with known queries
- [X] T034 [US1] Implement logging for processing metrics (URLs processed, chunks created, vectors stored)

## Phase 4: [US2] Context-Restricted Search Foundation

### User Story
As a reader studying specific sections of the book, I want to select text on the page and ask questions about only that selected text so that I can get focused answers without interference from other parts of the book.

### Goal
Prepare the ingestion pipeline to support context-restricted search by properly structuring chunks with metadata that enables later retrieval isolation.

### Independent Test Criteria
- Chunks contain proper metadata for source identification
- Individual chunks can be retrieved independently
- Chunk boundaries preserve semantic meaning
- Metadata enables later context restriction

### Tasks

- [ ] T040 [US2] Enhance chunk metadata to include heading hierarchy information
- [ ] T041 [US2] Implement chunk validation to ensure semantic coherence
- [ ] T042 [US2] Add chunk indexing within pages to enable range selection
- [ ] T043 [US2] Create chunk boundary preservation function to maintain context
- [ ] T044 [US2] Implement source attribution tracking for each chunk
- [ ] T045 [US2] Add quality validation for chunk content to ensure relevance

## Phase 5: [US3] Chat Session Persistence Foundation

### User Story
As a returning user, I want my chat history to be preserved between sessions so that I can continue conversations where I left off.

### Goal
Set up foundational components that will support chat session persistence in later phases.

### Independent Test Criteria
- System can identify and track different content sources
- Metadata is properly structured for later session management
- Content chunks are uniquely identifiable
- Source attribution is preserved

### Tasks

- [ ] T050 [US3] Enhance EmbeddingChunk model with additional metadata fields for session tracking
- [ ] T051 [US3] Implement unique chunk identification system
- [ ] T052 [US3] Add content fingerprinting to detect changes in source material
- [ ] T053 [US3] Create content versioning system to handle book updates

## Phase 6: Polish & Cross-Cutting Concerns

### Goal
Complete the ingestion pipeline with monitoring, validation, and optimization features.

- [ ] T060 Implement comprehensive error logging and reporting
- [ ] T061 Add performance monitoring for ingestion pipeline
- [ ] T062 Create ingestion pipeline configuration validation
- [ ] T063 Implement duplicate content detection and handling
- [ ] T064 Add content quality checks to filter low-value chunks
- [ ] T065 Create ingestion pipeline health check function
- [ ] T066 Implement backup and recovery procedures for the pipeline
- [ ] T067 Add progress tracking and reporting for long-running ingestion jobs
- [ ] T068 Create documentation for the ingestion pipeline setup and operation
- [ ] T069 Perform end-to-end testing of the complete ingestion pipeline
- [ ] T070 Optimize ingestion pipeline for performance and cost efficiency

## Parallel Execution Opportunities

### Within User Story 1:
- [P] T020 (crawling) and T027 (Qdrant setup) can run in parallel
- [P] T022 (content extraction) and T025 (chunking) can run in parallel
- [P] T028 (embedding) and T030 (upsert) can run in parallel

### Across User Stories:
- [P] US2 tasks can be implemented after foundational tasks are complete
- [P] US3 tasks can be implemented in parallel with US2 tasks

## Task Validation Checklist

- [x] All tasks follow the required format: `- [ ] T### [US#] Task description with file path`
- [x] Task IDs are sequential and in execution order
- [x] User story tasks have proper [US#] labels
- [x] Parallelizable tasks have [P] markers
- [x] Each task includes specific file paths where applicable
- [x] Dependencies between tasks are properly sequenced
- [x] Each user story has independently testable criteria
- [x] MVP scope includes core functionality from US1