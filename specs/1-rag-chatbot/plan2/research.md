# Research Outcomes: RAG Chatbot Retrieval, Agent & API Integration

**Feature**: RAG Chatbot for Book - Phase 2
**Date**: 2025-12-17

## OpenAI Model Selection

### Decision: OpenAI GPT-4 Turbo or GPT-4o
**Rationale**: After researching OpenAI models for RAG applications, GPT-4 Turbo or GPT-4o are optimal choices for educational content Q&A. These models offer:
- Strong tool usage capabilities for retrieval integration
- Good context following for strict grounding requirements
- Appropriate balance of intelligence and cost for educational applications
- Reliable response quality for technical content

**Alternatives Considered**:
- `gpt-3.5-turbo`: Less capable for complex technical questions, but more cost-effective
- `o1-preview`: More expensive and designed for reasoning rather than Q&A
- `gpt-4`: Older model, being phased out in favor of newer variants

## Retrieval Parameters

### Decision: top_k=5, similarity threshold=0.3
**Rationale**: Based on research into RAG retrieval parameters, this configuration provides the best balance between:
- Providing sufficient context for the agent to answer comprehensively
- Not overwhelming the agent with too much information
- Filtering out irrelevant results while preserving relevant ones
- Maintaining response quality and accuracy

**Technical Implementation**:
- top_k=5: Provides 5 relevant chunks as context, sufficient for most queries
- similarity threshold=0.3: Filters out very low-relevance results while keeping relevant content
- Allows for flexibility based on query complexity

**Alternatives Considered**:
- top_k=3: More conservative but might miss relevant information
- top_k=10: More comprehensive but could overwhelm the agent
- similarity threshold=0.5: More restrictive but might exclude relevant results
- similarity threshold=0.1: More permissive but includes potentially irrelevant content

## Qdrant Collection Configuration

### Decision: Use existing collection from Phase 1 with verification
**Rationale**: Since Phase 1 already created the Qdrant collection with Cohere embeddings, the optimal approach is to leverage this existing work. This ensures consistency and avoids duplication of effort.

**Configuration Details**:
- Collection name: robotics_book_chunks (from Phase 1)
- Vector dimensions: 1024 (for Cohere embed-english-v3.0)
- Payload schema: {content, source_url, page_title, heading, chunk_index, ...}
- Similarity: cosine (as configured in Phase 1)

**Verification Steps**:
- Confirm collection exists and is populated
- Verify vector dimensions match expected values
- Test sample queries to ensure functionality

## Frontend Integration Approach

### Decision: React component with floating chat widget
**Rationale**: For integrating the chatbot into the Docusaurus site, a floating chat widget provides the best user experience while maintaining the existing site design. This approach allows users to access the chatbot from anywhere on the site while preserving the educational content's focus.

**Technical Implementation**:
- Create React component for chat interface
- Implement as floating widget that can be toggled open/closed
- Add text selection functionality that appears contextually
- Integrate with Docusaurus layout without disrupting existing design

**Alternatives Considered**:
- Separate chat page: Would require navigation away from content
- Inline chat at bottom of each page: Would clutter content pages
- Iframe integration: Would create separation from main site
- Dedicated sidebar: Would reduce content viewing area