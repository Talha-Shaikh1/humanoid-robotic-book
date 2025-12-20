# Tasks: RAG Chatbot for Book - Retrieval, Agent & API Integration

**Feature**: RAG Chatbot for Book - Phase 2
**Branch**: 1-rag-chatbot
**Created**: 2025-12-17
**Status**: Draft
**Author**: Claude Code

## Overview

This document outlines the implementation tasks for the RAG Chatbot retrieval, agent, and API integration system. The system assumes vector embeddings already exist in Qdrant and focuses on building the retrieval layer, AI agent, backend API, and frontend integration.

### Tech Stack
- Python with uv for dependency management
- OpenAI for agent functionality
- FastAPI for web framework
- Qdrant-client for vector database operations
- python-dotenv for environment management

### Project Structure
```
backend/
├── main.py                 # FastAPI application
├── retrieval.py            # Qdrant retrieval logic
├── agent.py               # OpenAI agent implementation
├── models.py              # Data models and schemas
├── config.py              # Configuration and settings
└── requirements.txt       # Dependencies
```

## Dependencies

- Plan: specs/1-rag-chatbot/plan2.md
- Spec: specs/1-rag-chatbot/spec.md
- Research: specs/1-rag-chatbot/plan2/research.md
- Data Model: specs/1-rag-chatbot/plan2/data-model.md

## Implementation Strategy

### MVP Scope
- Basic semantic search over Qdrant collection
- Simple AI agent with retrieval tool
- Core chat API endpoint
- Minimum viable interaction

### Delivery Approach
- Incremental development following user story priorities
- Each user story delivers independently testable functionality
- Parallel execution where possible to optimize development time

## Phase 1: Setup Tasks

### Goal
Initialize project structure and configure environment for the retrieval and agent system.

- [X] T101 Create backend directory structure for Phase 2
- [X] T102 Initialize uv project in backend directory for Phase 2
- [X] T103 Update requirements.txt with dependencies: openai, fastapi, uvicorn, qdrant-client, python-dotenv, pydantic
- [X] T104 Update .env file template with OPENAI_API_KEY, QDRANT_URL, QDRANT_API_KEY, COLLECTION_NAME, MODEL_NAME
- [X] T105 Create main.py with FastAPI application initialization
- [X] T106 Create config.py to handle environment variables and settings
- [X] T107 Create models.py with Pydantic models based on data-model.md

## Phase 2: Foundational Tasks

### Goal
Implement core utilities and foundational components needed by all user stories.

- [X] T110 Create Qdrant client connection and initialization function
- [X] T111 Implement semantic search function over Qdrant collection
- [X] T112 Create OpenAI client initialization function
- [X] T113 Implement retrieval tool for the agent with user_query and top_k parameters
- [X] T114 Create data models for ChatRequest, ChatResponse, and RetrievalResult
- [X] T115 Set up FastAPI middleware for CORS and security
- [X] T116 Implement error handling and validation for API requests
- [X] T117 Create logging configuration for the API system

## Phase 3: [US1] Full Book Search Implementation

### User Story
As a reader of the humanoid robotics book, I want to ask questions about the book content using a chat interface so that I can quickly find relevant information without manually searching through pages.

### Goal
Implement the retrieval layer and agent that answers questions using full book content from Qdrant.

### Independent Test Criteria
- Queries return accurate book-related chunks
- URLs and section titles are preserved in responses
- Agent consistently respects context boundaries
- API responses are correct and stable

### Tasks

- [X] T120 [US1] Implement semantic search function that accepts user_query and top_k parameters
- [X] T121 [US1] Create retrieval function that returns ranked, relevant text chunks with metadata
- [X] T122 [US1] Validate retrieval with manual test queries against known book content
- [X] T123 [US1] Implement OpenAI agent using GPT-4 Turbo or GPT-4o model
- [X] T124 [US1] Integrate retrieval as a callable tool in the agent
- [X] T125 [US1] Define agent rules to answer only from provided context
- [X] T126 [US1] Implement agent rule to not hallucinate information
- [X] T127 [US1] Create agent rule to respond with uncertainty if answer not found
- [X] T128 [US1] Implement POST /chat endpoint in FastAPI
- [X] T129 [US1] Add request/response validation for /chat endpoint using models.py
- [X] T130 [US1] Implement proper error handling for /chat endpoint
- [X] T131 [US1] Add response formatting to include sources and confidence
- [X] T132 [US1] Create GET /health endpoint for system monitoring
- [X] T133 [US1] Test full-book retrieval mode with various queries
- [X] T134 [US1] Validate agent consistently respects context boundaries

## Phase 4: [US2] Context-Restricted Search Implementation

### User Story
As a reader studying specific sections of the book, I want to select text on the page and ask questions about only that selected text so that I can get focused answers without interference from other parts of the book.

### Goal
Implement context modes that allow questions to be answered from either full book or selected text only.

### Independent Test Criteria
- Selected-text answers never reference external content
- Full-book answers reference stored vectors
- Agent correctly switches between context modes
- Context isolation is strictly enforced

### Tasks

- [X] T140 [US2] Implement context mode parameter in ChatRequest model
- [X] T141 [US2] Create function to handle full-book retrieval mode via Qdrant
- [X] T142 [US2] Create function to handle selected-text-only context mode bypassing Qdrant
- [X] T143 [US2] Implement strict context isolation in selected-text mode
- [X] T144 [US2] Add selected_text parameter to ChatRequest model
- [X] T145 [US2] Create POST /chat/selected endpoint for selected text queries
- [X] T146 [US2] Implement validation to ensure no context leakage in selected mode
- [X] T147 [US2] Test selected-text mode with various selections and queries
- [X] T148 [US2] Validate that selected-text answers don't reference external content
- [X] T149 [US2] Create context mode switching logic in the agent
- [X] T150 [US2] Add comprehensive context validation for both modes

## Phase 5: [US3] Chat Session Persistence Integration

### User Story
As a returning user, I want my chat history to be preserved between sessions so that I can continue conversations where I left off.

### Goal
Integrate with the data management system to provide persistent chat sessions.

### Independent Test Criteria
- Chat sessions are preserved between visits
- Message history is maintained within sessions
- Session data is properly stored and retrieved
- Conversation context is maintained across interactions

### Tasks

- [X] T160 [US3] Add session_id parameter to ChatRequest and ChatResponse models
- [X] T161 [US3] Create session management functions for creating and retrieving sessions
- [X] T162 [US3] Implement message storage for each session
- [X] T163 [US3] Add session_id to API endpoints where appropriate
- [X] T164 [US3] Create endpoint for retrieving session history
- [X] T165 [US3] Implement session cleanup for inactive sessions
- [X] T166 [US3] Add session metadata tracking (source page, etc.)
- [X] T167 [US3] Test session persistence across different visits
- [X] T168 [US3] Validate message history integrity within sessions

## Phase 6: Polish & Cross-Cutting Concerns

### Goal
Complete the retrieval and agent system with monitoring, validation, and optimization features.

- [X] T180 Implement comprehensive error logging and reporting
- [X] T181 Add performance monitoring for API response times
- [X] T182 Create agent behavior monitoring and validation
- [X] T183 Implement rate limiting for API endpoints
- [X] T184 Add request/response logging for debugging
- [X] T185 Create API documentation with OpenAPI/Swagger
- [X] T186 Implement input validation and sanitization
- [X] T187 Add caching for frequently accessed content
- [X] T188 Create health check endpoints for all dependencies
- [X] T189 Perform end-to-end testing of the complete retrieval and agent system
- [X] T190 Optimize agent performance and cost efficiency

## Parallel Execution Opportunities

### Within User Story 1:
- [P] T120 (semantic search) and T123 (agent setup) can run in parallel
- [P] T128 (chat endpoint) and T132 (health endpoint) can run in parallel

### Within User Story 2:
- [P] T141 (full-book mode) and T142 (selected-text mode) can run in parallel
- [P] T145 (endpoint) and T146 (validation) can run in parallel

### Across User Stories:
- [P] US2 tasks can be implemented after foundational tasks are complete
- [P] US3 tasks can be implemented in parallel with US2 tasks

## Task Validation Checklist

- [x] All tasks follow the required format: `- [ ] T### [US#] Task description with file path`
- [x] Task IDs are sequential and in execution order (T101-T190)
- [x] User story tasks have proper [US#] labels
- [x] Parallelizable tasks have [P] markers
- [x] Each task includes specific file paths where applicable
- [x] Dependencies between tasks are properly sequenced
- [x] Each user story has independently testable criteria
- [x] MVP scope includes core functionality from US1