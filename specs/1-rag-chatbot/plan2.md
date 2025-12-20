# Implementation Plan: RAG Chatbot Retrieval, Agent & API Integration

**Feature**: RAG Chatbot for Book - Phase 2
**Branch**: 1-rag-chatbot
**Created**: 2025-12-17
**Status**: Draft
**Author**: Claude Code

## Technical Context

This plan outlines the implementation of the retrieval, agent, and API integration layer for the RAG chatbot. The system assumes that vector embeddings already exist in Qdrant and focuses on building the retrieval layer, AI agent, backend API, and frontend integration.

### Target System
- **Assumptions**: Vector embeddings already exist in Qdrant, Cohere was used for embeddings, book content is fully indexed
- **Technology Stack**: Python with uv, OpenAI Agents SDK, FastAPI, Qdrant client
- **Architecture**: Multi-layered system with retrieval, agent, and API layers

### Known Requirements
- Semantic search over existing Qdrant collection
- Two context modes: full-book retrieval and selected-text-only context
- AI agent with strict context boundaries and no hallucination
- FastAPI backend with chat endpoints
- Frontend integration with text selection feature

### Unknowns (NEEDS CLARIFICATION)
- All unknowns have been resolved through research phase.

## Constitution Check

### Alignment with Constitutional Principles

✅ **Technical Accuracy and Documentation Compliance**: The system will provide accurate answers based on book content only.

✅ **Strict Information Grounding**: The agent will be built to answer ONLY from provided context, preventing hallucination.

✅ **AI-Native Content Generation**: This implementation follows the spec.md → plan.md → tasks.md → implementation pipeline.

✅ **Multi-Modal Learning Experience**: The chatbot will enhance the learning experience with interactive Q&A capabilities.

### Potential Violations
None identified - the implementation aligns with all constitutional principles.

## Gates

### Pre-Implementation Gates

1. **Environment Setup**: Ensure `uv` is available and dependencies can be installed
2. **API Access**: Verify OpenAI and Qdrant Cloud credentials work
3. **Database Access**: Confirm access to existing Qdrant collection
4. **Architecture Review**: Multi-layered design meets technical constraints

**Status**: All gates pass - proceeding with implementation.

## Phase 0: Research & Discovery

### Research Tasks

1. **OpenAI Model Selection**
   - Task: Research optimal OpenAI model for RAG agent applications
   - Expected outcome: Select most appropriate model for book content Q&A

2. **Retrieval Parameters**
   - Task: Research optimal top-k and similarity threshold values
   - Expected outcome: Define parameters for best retrieval quality

3. **Qdrant Collection Details**
   - Task: Determine exact collection name, vector dimensions, and payload schema
   - Expected outcome: Accurate configuration for retrieval layer

4. **Frontend Integration**
   - Task: Research best practices for embedding chatbot in Docusaurus
   - Expected outcome: Optimal integration approach with text selection

### Research Outcomes (research.md)

#### Decision: OpenAI Model Selection
- **Chosen**: gpt-4-turbo or gpt-4o
- **Rationale**: Best balance of intelligence, accuracy, and cost for educational content. These models perform well with tool usage and context following.
- **Alternatives considered**: gpt-3.5-turbo (less capable), o1-preview (more expensive, less suitable for Q&A)

#### Decision: Retrieval Parameters
- **Chosen**: top_k=5, similarity threshold=0.3
- **Rationale**: Top-5 provides good context without overwhelming the agent; 0.3 threshold filters out irrelevant results while preserving relevant ones
- **Alternatives considered**: Various top_k values (3-10), different similarity thresholds

#### Decision: Qdrant Collection Configuration
- **Chosen**: Use existing collection from Phase 1, verify vector dimensions (1024), payload schema
- **Rationale**: Leverages existing work from Phase 1 with Cohere embeddings
- **Alternatives considered**: Creating new collection (unnecessary duplication)

#### Decision: Frontend Integration Approach
- **Chosen**: React component embedded in Docusaurus layout with floating chat widget
- **Rationale**: Provides seamless user experience while maintaining site design
- **Alternatives considered**: Separate chat page, iframe integration

## Phase 1: Design & Architecture

### Data Model (data-model.md)

#### ChatRequest Entity
- **query**: User's question text
- **context_mode**: "full_book" or "selected_text"
- **selected_text**: Text selected by user (for selected_text mode)
- **top_k**: Number of results to retrieve (default: 5)
- **session_id**: Identifier for conversation context

#### ChatResponse Entity
- **answer**: Agent's response to the query
- **sources**: List of source URLs/references used
- **confidence**: Confidence level of the answer (0-1)
- **context_used**: Text that was provided to the agent
- **query_id**: Unique identifier for the query

#### RetrievalResult Entity
- **id**: Unique identifier for the result
- **content**: Text content of the chunk
- **source_url**: URL of the source page
- **page_title**: Title of the source page
- **score**: Similarity score from vector search
- **metadata**: Additional metadata from Qdrant

### API Contracts

#### Chat Endpoint
```
POST /chat
Content-Type: application/json

Request:
{
  "query": "What is ROS 2?",
  "context_mode": "full_book",
  "session_id": "session-123",
  "top_k": 5
}

Response:
{
  "answer": "ROS 2 is a robotics framework...",
  "sources": ["https://..."],
  "confidence": 0.8,
  "query_id": "query-456"
}
```

#### Selected Text Chat Endpoint
```
POST /chat/selected
Content-Type: application/json

Request:
{
  "query": "What does this mean?",
  "selected_text": "ROS 2 is a flexible framework for robotics development...",
  "session_id": "session-123"
}

Response:
{
  "answer": "This refers to ROS 2 being a flexible framework...",
  "sources": [],
  "confidence": 0.9,
  "query_id": "query-789"
}
```

#### Health Check Endpoint
```
GET /health

Response:
{
  "status": "healthy",
  "timestamp": "2025-12-17T10:00:00Z"
}
```

### Quickstart Guide (quickstart.md)

#### Setting Up the Retrieval & Agent System

1. **Install Dependencies**
   ```bash
   uv venv
   source .venv/bin/activate  # On Windows: .venv\\Scripts\\activate
   uv pip install openai fastapi uvicorn qdrant-client python-dotenv
   ```

2. **Configure Environment Variables**
   Create `.env` file with:
   ```
   OPENAI_API_KEY=your_openai_api_key
   QDRANT_URL=your_qdrant_cloud_url
   QDRANT_API_KEY=your_qdrant_api_key
   COLLECTION_NAME=robotics_book_chunks
   ```

3. **Run the Backend Server**
   ```bash
   uv run uvicorn main:app --reload
   ```

4. **Verify Endpoints**
   - Test health endpoint: `GET /health`
   - Test chat endpoint with sample query
   - Verify agent responses are grounded in context

## Phase 2: Implementation Plan

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

### Implementation Steps

#### Step 1: Project Initialization
- Set up uv project
- Create FastAPI application structure
- Add dependency management
- Configure environment loading

#### Step 2: Retrieval Layer Implementation
- Implement semantic search over Qdrant collection
- Create retrieval function with user_query and top_k parameters
- Return ranked chunks with metadata preserved
- Add validation for retrieval quality

#### Step 3: Context Modes Implementation
- Implement full-book retrieval mode using Qdrant
- Implement selected-text-only mode bypassing Qdrant
- Enforce strict context isolation in selected-text mode
- Add validation to ensure no context leakage

#### Step 4: Agent Construction
- Build AI agent using OpenAI Agents SDK
- Integrate retrieval as a callable tool
- Implement agent rules: context-only answers, no hallucination, uncertainty responses
- Add tool usage observability

#### Step 5: FastAPI Backend
- Wrap agent and retrieval logic in FastAPI endpoints
- Implement POST /chat endpoint
- Implement POST /chat/selected endpoint
- Implement GET /health endpoint
- Add proper error handling and validation

#### Step 6: Frontend Integration
- Create React chat component for Docusaurus
- Implement text selection functionality
- Add "Ask from selected text" feature
- Handle CORS and environment configuration
- Connect to FastAPI backend

### Dependencies
- openai: For AI agent functionality
- fastapi: For web framework
- uvicorn: For ASGI server
- qdrant-client: For vector database operations
- python-dotenv: For environment variable management

## Success Criteria

### Technical Success
- [ ] Semantic search returns accurate book-related chunks
- [ ] URLs and section titles preserved in results
- [ ] Context isolation enforced in selected-text mode
- [ ] Agent respects context boundaries consistently
- [ ] API endpoints respond correctly and stably
- [ ] Frontend chatbot works inside Docusaurus site

### Functional Success
- [ ] Full-book mode retrieves from Qdrant collection
- [ ] Selected-text mode bypasses Qdrant and uses provided context only
- [ ] Agent does not hallucinate information
- [ ] Agent responds with uncertainty when answer not found
- [ ] Text selection → "Ask from selected text" works correctly

### Performance Success
- [ ] Response times under 5 seconds
- [ ] Proper error handling and retry logic
- [ ] Memory usage remains reasonable
- [ ] API rate limits respected