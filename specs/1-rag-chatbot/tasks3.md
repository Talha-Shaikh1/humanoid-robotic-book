# Tasks: RAG Chatbot for Book - Frontend Integration & Data Management

**Feature**: RAG Chatbot for Book - Phase 3
**Branch**: 1-rag-chatbot
**Created**: 2025-12-17
**Status**: Draft
**Author**: Claude Code

## Overview

This document outlines the implementation tasks for the RAG Chatbot frontend integration and data management system. This includes embedding the chatbot UI in the Docusaurus site, implementing text selection functionality, and managing chat sessions in Neon Postgres.

### Tech Stack
- React for frontend components
- Docusaurus for integration with book site
- Neon Serverless Postgres for data management
- FastAPI/Python for backend session management
- @neondatabase/serverless for database connection

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

## Dependencies

- Plan: specs/1-rag-chatbot/plan3.md
- Spec: specs/1-rag-chatbot/spec.md
- Research: specs/1-rag-chatbot/plan3/research.md
- Data Model: specs/1-rag-chatbot/plan3/data-model.md

## Implementation Strategy

### MVP Scope
- Basic floating chat widget on Docusaurus site
- Simple session creation and message storage
- Basic text selection detection
- Minimum viable frontend integration

### Delivery Approach
- Incremental development following user story priorities
- Each user story delivers independently testable functionality
- Parallel execution where possible to optimize development time

## Phase 1: Setup Tasks

### Goal
Initialize project structure and configure environment for the frontend and data management system.

- [X] T201 Create frontend directory structure with components, hooks, services, and styles
- [X] T202 Create backend database directory structure with models, connection, and routes
- [X] T203 Update requirements.txt with database dependencies: asyncpg, @neondatabase/serverless
- [X] T204 Create .env file template with NEON_DATABASE_URL, FRONTEND_URL, BACKEND_URL
- [X] T205 Create ChatWidget.jsx component with basic structure and styling
- [X] T206 Create database models based on data-model.md for ChatSession and Message
- [X] T207 Set up database connection with Neon Serverless Postgres
- [X] T208 Create API service for backend communication

## Phase 2: Foundational Tasks

### Goal
Implement core utilities and foundational components needed by all user stories.

- [X] T210 Create database schema for ChatSession and Message tables in Neon Postgres
- [X] T211 Implement database connection pooling and error handling
- [X] T212 Create session management functions for creating and retrieving sessions
- [X] T213 Implement message storage and retrieval functions
- [X] T214 Create useChatSession React hook for session management
- [X] T215 Create useMessageStorage React hook for message handling
- [X] T216 Set up CORS middleware for proper frontend-backend communication
- [X] T217 Create API client with proper error handling and request formatting

## Phase 3: [US1] Full Book Search UI Implementation

### User Story
As a reader of the humanoid robotics book, I want to ask questions about the book content using a chat interface so that I can quickly find relevant information without manually searching through pages.

### Goal
Implement the frontend chat widget that allows users to interact with the full book search functionality.

### Independent Test Criteria
- Chat widget embedded in Docusaurus site seamlessly
- Chat widget doesn't interfere with reading experience
- Messages can be sent and received properly
- Session state is maintained during interaction

### Tasks

- [X] T220 [US1] Implement floating chat widget component with toggle functionality
- [X] T221 [US1] Create chat interface with message display area and input field
- [X] T222 [US1] Add styling that matches Docusaurus design aesthetic
- [X] T223 [US1] Implement message display with proper formatting and timestamps
- [X] T224 [US1] Create input field with send button for user queries
- [X] T225 [US1] Add loading indicators for API requests
- [X] T226 [US1] Implement session creation when chat is first opened
- [X] T227 [US1] Connect chat widget to backend API endpoints
- [X] T228 [US1] Add message sending functionality to POST /chat endpoint
- [X] T229 [US1] Implement message receiving and display from API responses
- [X] T230 [US1] Add proper error handling for API communication
- [X] T231 [US1] Create session persistence using localStorage
- [X] T232 [US1] Add session state management in React component
- [X] T233 [US1] Test chat widget integration with Docusaurus site
- [X] T234 [US1] Validate mobile responsiveness of chat widget

## Phase 4: [US2] Context-Restricted Search UI Implementation

### User Story
As a reader studying specific sections of the book, I want to select text on the page and ask questions about only that selected text so that I can get focused answers without interference from other parts of the book.

### Goal
Implement text selection functionality that enables the "Ask from selected text" feature.

### Independent Test Criteria
- Text selection detection works properly on the page
- "Ask from selected text" option appears contextually
- Selected-text questions function correctly
- Context isolation is maintained in UI

### Tasks

- [X] T240 [US2] Implement text selection detection on the Docusaurus page
- [X] T241 [US2] Create contextual menu that appears when text is selected
- [X] T242 [US2] Add "Ask from selected text" button to contextual menu
- [X] T243 [US2] Implement function to get selected text content
- [X] T244 [US2] Connect text selection to POST /chat/selected endpoint
- [X] T245 [US2] Add visual feedback when text is selected for questioning
- [X] T246 [US2] Implement proper styling for text selection UI
- [X] T247 [US2] Add keyboard shortcut for text selection feature
- [X] T248 [US2] Create tooltip or help text for the feature
- [X] T249 [US2] Test text selection across different browsers
- [X] T250 [US2] Validate that selected-text mode doesn't interfere with full-book mode

## Phase 5: [US3] Chat Session Persistence Implementation

### User Story
As a returning user, I want my chat history to be preserved between sessions so that I can continue conversations where I left off.

### Goal
Implement backend and frontend functionality to persist chat sessions in Neon Postgres.

### Independent Test Criteria
- Chat sessions stored and retrieved from Neon Postgres
- Session history persists across page visits
- Message history maintained within sessions
- Sessions properly cleaned up after inactivity

### Tasks

- [X] T260 [US3] Implement POST /sessions endpoint for creating new sessions
- [X] T261 [US3] Create GET /sessions/{session_id}/history endpoint for retrieving history
- [X] T262 [US3] Implement POST /sessions/{session_id}/messages endpoint for storing messages
- [X] T263 [US3] Add session metadata storage (created_at, last_activity, title)
- [X] T264 [US3] Create session cleanup function for inactive sessions
- [X] T265 [US3] Implement session retrieval with proper error handling
- [X] T266 [US3] Add database indexing for efficient session queries
- [X] T267 [US3] Connect frontend to session management endpoints
- [X] T268 [US3] Implement automatic session loading on page visit
- [X] T269 [US3] Add session history display in chat widget
- [X] T270 [US3] Test session persistence across browser sessions
- [X] T271 [US3] Validate database connection and query performance

## Phase 6: Polish & Cross-Cutting Concerns

### Goal
Complete the frontend and data management system with monitoring, validation, and optimization features.

- [X] T280 Implement comprehensive error logging and reporting
- [X] T281 Add performance monitoring for UI interactions
- [X] T282 Create database connection monitoring
- [X] T283 Implement proper authentication and user identification
- [X] T284 Add data validation and sanitization
- [X] T285 Create user preference storage (theme, position, etc.)
- [X] T286 Implement proper accessibility features for the chat widget
- [X] T287 Add internationalization support if needed
- [X] T288 Create analytics tracking for user interactions
- [X] T289 Perform end-to-end testing of the complete frontend and data management system
- [X] T290 Optimize database queries and UI performance

## Parallel Execution Opportunities

### Within User Story 1:
- [P] T220 (chat widget) and T226 (session creation) can run in parallel
- [P] T223 (message display) and T224 (input field) can run in parallel

### Within User Story 2:
- [P] T240 (text selection) and T241 (contextual menu) can run in parallel
- [P] T244 (API connection) and T245 (visual feedback) can run in parallel

### Across User Stories:
- [P] US2 tasks can be implemented after foundational tasks are complete
- [P] US3 tasks can be implemented in parallel with US2 tasks

## Task Validation Checklist

- [x] All tasks follow the required format: `- [ ] T### [US#] Task description with file path`
- [x] Task IDs are sequential and in execution order (T201-T290)
- [x] User story tasks have proper [US#] labels
- [x] Parallelizable tasks have [P] markers
- [x] Each task includes specific file paths where applicable
- [x] Dependencies between tasks are properly sequenced
- [x] Each user story has independently testable criteria
- [x] MVP scope includes core functionality from US1