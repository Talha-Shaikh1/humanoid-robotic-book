# Implementation Plan: RAG Chatbot Frontend Integration & Data Management

**Feature**: RAG Chatbot for Book - Phase 3
**Branch**: 1-rag-chatbot
**Created**: 2025-12-17
**Status**: Draft
**Author**: Claude Code

## Technical Context

This plan outlines the implementation of the frontend integration and data management layer for the RAG chatbot. This includes embedding the chatbot UI in the Docusaurus site, implementing text selection functionality, and managing chat sessions in Neon Postgres.

### Target System
- **Integration**: Docusaurus-based book deployed on GitHub Pages
- **Technology Stack**: React for frontend, Neon Serverless Postgres for data management
- **Architecture**: Client-side chat widget with backend session persistence

### Known Requirements
- Embed chatbot UI inside Docusaurus site
- Enable text selection → "Ask from selected text" feature
- Store chat sessions, user messages, and timestamps in Neon Postgres
- Handle CORS and security considerations
- Support production deployment with environment-based configuration

### Unknowns (NEEDS CLARIFICATION)
- All unknowns have been resolved through research phase.

## Constitution Check

### Alignment with Constitutional Principles

✅ **Multi-Modal Learning Experience**: The chatbot will enhance the learning experience with interactive Q&A capabilities.

✅ **AI-Native Content Generation**: This implementation follows the spec.md → plan.md → tasks.md → implementation pipeline.

✅ **Pedagogical Progression and Accessibility**: The chatbot provides an additional learning pathway for students.

✅ **Reproducibility and Testing Standards**: The system will be built with proper testing and validation components.

### Potential Violations
None identified - the implementation aligns with all constitutional principles.

## Gates

### Pre-Implementation Gates

1. **Environment Setup**: Ensure development environment supports React and database integration
2. **Database Access**: Verify Neon Postgres credentials and connection work
3. **Frontend Integration**: Confirm Docusaurus site can accommodate embedded components
4. **Security Review**: Ensure proper CORS and security configurations

**Status**: All gates pass - proceeding with implementation.

## Phase 0: Research & Discovery

### Research Tasks

1. **React Component Architecture**
   - Task: Research best practices for chatbot UI components in React
   - Expected outcome: Select optimal component structure for the chatbot widget

2. **Session Management**
   - Task: Research session identification and management strategies
   - Expected outcome: Define approach for maintaining conversation context

3. **Database Configuration**
   - Task: Determine optimal Neon Postgres connection and schema setup
   - Expected outcome: Proper database configuration for chat persistence

4. **Docusaurus Integration**
   - Task: Research best practices for embedding React components in Docusaurus
   - Expected outcome: Optimal integration method without disrupting site design

### Research Outcomes (research.md)

#### Decision: React Component Architecture
- **Chosen**: Floating widget with toggle functionality
- **Rationale**: Provides non-intrusive user experience while maintaining accessibility
- **Alternatives considered**: Inline component, dedicated chat page, iframe integration

#### Decision: Session Management Approach
- **Chosen**: UUID-based session IDs stored in browser localStorage with backend persistence
- **Rationale**: Balances user experience with data persistence across visits
- **Alternatives considered**: Server-only sessions, URL parameters, cookies

#### Decision: Database Configuration
- **Chosen**: Neon Serverless Postgres with connection pooling and SSL
- **Rationale**: Scales automatically and provides reliable persistence for chat history
- **Alternatives considered**: Other cloud databases, local storage only, file-based storage

#### Decision: Docusaurus Integration Method
- **Chosen**: Custom React component injected via Docusaurus theme layout
- **Rationale**: Maintains site design while providing seamless integration
- **Alternatives considered**: Plugin approach, external script injection

## Phase 1: Design & Architecture

### Data Model (data-model.md)

#### ChatSession Entity
- **session_id**: Unique identifier for the session (UUID)
- **user_id**: Identifier for the user (optional, for registered users)
- **created_at**: Timestamp when session was created
- **last_activity**: Timestamp of last interaction
- **metadata**: Additional session metadata (source page, etc.)

#### Message Entity
- **message_id**: Unique identifier for the message (UUID)
- **session_id**: Reference to parent session
- **role**: "user" or "assistant"
- **content**: The actual message content
- **timestamp**: When the message was created
- **sources**: For assistant messages, list of sources used

#### ChatHistory Entity
- **session_id**: Reference to the session
- **messages**: Array of messages in the session
- **title**: Auto-generated title for the session based on first query

### API Contracts

#### Session Creation Endpoint
```
POST /sessions
Content-Type: application/json

Request:
{
  "user_id": "user-123",
  "metadata": {
    "source_page": "https://...",
    "referrer": "..."
  }
}

Response:
{
  "session_id": "session-456",
  "created_at": "2025-12-17T10:00:00Z"
}
```

#### Message Storage Endpoint
```
POST /sessions/{session_id}/messages
Content-Type: application/json

Request:
{
  "role": "user",
  "content": "What is ROS 2?",
  "sources": []
}

Response:
{
  "message_id": "msg-789",
  "timestamp": "2025-12-17T10:01:00Z"
}
```

#### Session History Endpoint
```
GET /sessions/{session_id}/history

Response:
{
  "session_id": "session-456",
  "messages": [
    {
      "message_id": "msg-789",
      "role": "user",
      "content": "What is ROS 2?",
      "timestamp": "2025-12-17T10:01:00Z"
    },
    {
      "message_id": "msg-901",
      "role": "assistant",
      "content": "ROS 2 is a robotics framework...",
      "sources": ["https://..."],
      "timestamp": "2025-12-17T10:02:00Z"
    }
  ]
}
```

### Quickstart Guide (quickstart.md)

#### Setting Up the Frontend Integration & Data Management

1. **Install Dependencies**
   ```bash
   # For backend (Node.js/Express or continue with FastAPI)
   npm install pg @neondatabase/serverless  # For Node.js
   # OR for Python with asyncpg
   uv pip install asyncpg python-dotenv

   # For frontend React component
   npm install @docusaurus/core react react-dom
   ```

2. **Configure Environment Variables**
   Create `.env` file with:
   ```
   NEON_DATABASE_URL=your_neon_database_url
   FRONTEND_URL=your_docusaurus_site_url
   BACKEND_URL=your_fastapi_backend_url
   ```

3. **Run the System**
   ```bash
   # Backend: uv run uvicorn main:app --reload
   # Frontend: npm run start (or integrated with Docusaurus)
   ```

4. **Verify Integration**
   - Test chat session creation and persistence
   - Verify message storage and retrieval
   - Confirm text selection functionality works

## Phase 2: Implementation Plan

### Project Structure
```
frontend/
├── components/
│   └── ChatWidget.jsx        # Main chat widget component
├── hooks/
│   ├── useChatSession.js     # Session management hook
│   └── useMessageStorage.js  # Message storage hook
├── services/
│   ├── api.js               # API client
│   └── sessionManager.js    # Session management
└── styles/
    └── chat.css             # Chat widget styles

backend/
├── database/
│   ├── models.py           # Database models
│   └── connection.py       # Database connection
├── routes/
│   └── sessions.py         # Session management endpoints
└── middleware/
    └── cors.py            # CORS configuration

docusaurus/
├── src/
│   └── theme/
│       └── Layout.js      # Integration point for chat widget
└── static/
    └── css/
        └── chat-integration.css
```

### Implementation Steps

#### Step 1: Database Setup
- Set up Neon Postgres schema for chat sessions and messages
- Create tables for ChatSession and Message entities
- Implement connection pooling and error handling

#### Step 2: Backend Session Management
- Implement session creation endpoint
- Create message storage and retrieval endpoints
- Add proper authentication and validation

#### Step 3: Frontend Chat Widget
- Create React component for the floating chat widget
- Implement toggle functionality to show/hide the chat
- Add styling that matches Docusaurus design

#### Step 4: Session Management Hooks
- Create React hooks for session creation and management
- Implement localStorage for session persistence across visits
- Add automatic session cleanup

#### Step 5: Text Selection Integration
- Implement text selection detection on the page
- Add "Ask from selected text" context menu
- Connect to the appropriate backend endpoint

#### Step 6: Docusaurus Integration
- Integrate the chat widget into the Docusaurus layout
- Ensure proper styling and positioning
- Handle mobile responsiveness

#### Step 7: Testing & Validation
- Test session persistence across page reloads
- Verify message storage and retrieval
- Validate text selection functionality
- Test cross-browser compatibility

### Dependencies
- @neondatabase/serverless: For Neon Postgres connection
- react: For frontend components
- @docusaurus/core: For Docusaurus integration
- Additional UI libraries as needed (e.g., styled-components)

## Success Criteria

### Technical Success
- [ ] Chat widget embedded in Docusaurus site seamlessly
- [ ] Text selection → "Ask from selected text" functionality works
- [ ] Chat sessions stored and retrieved from Neon Postgres
- [ ] Proper CORS and security configurations implemented
- [ ] Production deployment configuration works

### Functional Success
- [ ] Sessions persist across page visits
- [ ] Message history maintained within sessions
- [ ] Text selection feature intuitive and responsive
- [ ] Chat widget doesn't interfere with reading experience
- [ ] Mobile-responsive design maintained

### Performance Success
- [ ] Sub-200ms response times for UI interactions
- [ ] Efficient database queries with proper indexing
- [ ] Minimal impact on page load times
- [ ] Proper error handling and fallbacks