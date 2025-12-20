---
id: 3
title: "RAG Chatbot Retrieval Agent Plan"
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
  - "retrieval"
  - "agent"
  - "api"
tests: []
links:
  spec: "specs/1-rag-chatbot/spec.md"
  ticket: null
  adr: null
  pr: null
files:
  - "specs/1-rag-chatbot/plan2.md"
  - "specs/1-rag-chatbot/plan2/research.md"
  - "specs/1-rag-chatbot/plan2/data-model.md"
  - "specs/1-rag-chatbot/plan2/quickstart.md"
---

# RAG Chatbot Retrieval, Agent & API Integration Plan

## Prompt Text

Create:
Phase 2 of the Unified Book Project: Retrieval, Agent, and API integration using the already populated Qdrant vector database.

Assumption:
- Vector embeddings already exist in Qdrant.
- Cohere was used for embeddings.
- Book content is fully indexed.

---

Phase 2.1: Retrieval Layer (Foundation)
- Implement semantic search over the existing Qdrant collection.
- Create a retrieval function that accepts:
  - user_query
  - top_k
- Return ranked, relevant text chunks with metadata.
- Validate retrieval with manual test queries.

Acceptance Criteria:
- Queries return accurate book-related chunks.
- URLs and section titles are preserved.

---

Phase 2.2: Context Modes
- Implement two context modes:
  1) Full-book retrieval (via Qdrant)
  2) Selected-text-only context (bypass Qdrant)
- Enforce strict context isolation in selected-text mode.

Acceptance Criteria:
- Selected-text answers never reference external content.
- Full-book answers reference stored vectors.

---

Phase 2.3: Agent Construction
- Build an AI agent using OpenAI Agents SDK.
- Integrate retrieval as a callable tool.
- Define agent rules:
  - Answer only from provided context.
  - Do not hallucinate.
  - Respond with uncertainty if answer not found.

Acceptance Criteria:
- Agent consistently respects context boundaries.
- Tool usage is observable.

---

Phase 2.4: FastAPI Backend
- Wrap agent and retrieval logic in FastAPI.
- Expose endpoints:
  - POST /chat
  - POST /chat/selected
  - GET /health
- Support JSON request/response.

Acceptance Criteria:
- Backend runs locally.
- API responses are correct and stable.

---

Phase 2.5: Frontend Integration
- Embed chatbot UI inside Docusaurus site.
- Enable user text selection → "Ask from selected text".
- Connect frontend to FastAPI backend.
- Handle CORS and environment configuration.

Acceptance Criteria:
- Chatbot works inside the deployed book.
- Selected-text questions function correctly.

---

Constraints:
- Do not modify ingestion pipeline.
- Use existing Qdrant collection.
- Maintai

## Response Text

Successfully created a comprehensive implementation plan for the RAG chatbot retrieval, agent, and API integration. The plan includes:

1. **Technical Context**: Outlined the system requirements and resolved all unknowns through research
2. **Constitution Check**: Verified alignment with project constitutional principles
3. **Gates**: Confirmed all pre-implementation requirements are met
4. **Research Outcomes**: Completed research on OpenAI models, retrieval parameters, Qdrant configuration, and frontend integration
5. **Data Model**: Defined entities for ChatRequest, ChatResponse, RetrievalResult, AgentToolCall, and ChatSession
6. **API Contracts**: Specified endpoints for /chat, /chat/selected, and /health with proper request/response schemas
7. **Quickstart Guide**: Created detailed setup and execution instructions
8. **Implementation Steps**: Detailed the 6-phase implementation process

Key decisions documented:
- OpenAI GPT-4 Turbo or GPT-4o model for educational content Q&A
- top_k=5 and similarity threshold=0.3 for optimal retrieval
- Use existing Qdrant collection from Phase 1
- React component with floating chat widget for frontend integration

The plan is structured to work with the existing vector database from Phase 1 and follows all technical constraints specified in the requirements.

## Outcome

The implementation plan for Phase 2 is complete and ready for the next phase. All architectural decisions have been made, research completed, and documentation created. The plan provides a clear roadmap for implementing the retrieval, agent, and API integration with all necessary details for successful execution.