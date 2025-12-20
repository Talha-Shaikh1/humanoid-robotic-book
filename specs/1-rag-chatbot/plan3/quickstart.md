# Quickstart Guide: RAG Chatbot Frontend Integration & Data Management

**Feature**: RAG Chatbot for Book - Phase 3
**Date**: 2025-12-17

## Overview
This guide provides step-by-step instructions to set up and run the frontend integration and data management layer for the RAG chatbot. This system handles the chatbot UI embedding in Docusaurus, text selection functionality, and chat session persistence in Neon Postgres.

## Prerequisites

### System Requirements
- Node.js 16+ (for Docusaurus integration)
- Python 3.8+ (for backend database operations)
- Access to Neon Serverless Postgres
- Existing Docusaurus book site

### Database Setup
- Neon Postgres database instance
- Required tables for ChatSession and Message entities (schema in data-model.md)

## Setup Instructions

### 1. Backend Database Setup
```bash
# Navigate to backend directory
cd backend

# Install Python dependencies
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
uv pip install asyncpg python-dotenv fastapi uvicorn
```

### 2. Frontend Component Setup
```bash
# Navigate to Docusaurus directory
cd docusaurus

# Install React dependencies
npm install
npm install @neondatabase/serverless  # For database connection
```

### 3. Configure Environment Variables
Create `.env` file in your backend directory:

```env
NEON_DATABASE_URL=your_neon_database_connection_string
FRONTEND_URL=https://your-docusaurus-site.com
BACKEND_URL=http://localhost:8000
ALLOWED_ORIGINS=http://localhost:3000,https://your-docusaurus-site.com
```

### 4. Database Initialization
Run the database schema initialization script to create required tables:

```sql
-- Create ChatSession table
CREATE TABLE chat_sessions (
    session_id UUID PRIMARY KEY,
    user_id VARCHAR(255),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    last_activity TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    title VARCHAR(200),
    metadata JSONB,
    is_active BOOLEAN DEFAULT TRUE
);

-- Create Message table
CREATE TABLE messages (
    message_id UUID PRIMARY KEY,
    session_id UUID REFERENCES chat_sessions(session_id),
    role VARCHAR(10) NOT NULL,  -- 'user' or 'assistant'
    content TEXT NOT NULL,
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    sources JSONB,
    parent_message_id UUID REFERENCES messages(message_id),
    INDEX idx_session_id (session_id),
    INDEX idx_timestamp (timestamp)
);
```

## Frontend Integration

### 1. Create Chat Widget Component
Create `src/components/ChatWidget.jsx` in your Docusaurus project:

```jsx
import React, { useState, useEffect } from 'react';
import './ChatWidget.css';

const ChatWidget = () => {
  const [isOpen, setIsOpen] = useState(false);
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState('');

  // Component implementation here
  // Handles chat functionality, session management, etc.
};

export default ChatWidget;
```

### 2. Integrate with Docusaurus Layout
Modify `src/theme/Layout.js` to include the chat widget:

```jsx
import React from 'react';
import OriginalLayout from '@theme-original/Layout';
import ChatWidget from '../components/ChatWidget';

export default function Layout(props) {
  return (
    <>
      <OriginalLayout {...props} />
      <ChatWidget />
    </>
  );
}
```

### 3. Text Selection Feature
The component should automatically detect text selection and provide an "Ask from selected text" option that appears contextually when text is selected.

## Backend API Endpoints

### Session Management
```
POST /sessions
GET /sessions/{session_id}/history
POST /sessions/{session_id}/messages
```

### Running the Integrated System

1. **Start the backend server:**
```bash
cd backend
uv run uvicorn main:app --reload
```

2. **Start the Docusaurus site:**
```bash
cd docusaurus
npm run start
```

## Expected Output
During execution, you should see:
- Chat widget appearing on all pages of your Docusaurus site
- Session data being stored in Neon Postgres
- Text selection detection working properly
- Messages being saved and retrieved correctly

## Verification Steps

### 1. Test Database Connection
Verify that the backend can connect to Neon Postgres and create/read data.

### 2. Test Frontend Integration
- Verify the chat widget appears on all pages
- Test opening and closing the chat
- Confirm text selection detection works

### 3. Test Session Persistence
- Start a conversation
- Navigate to different pages
- Verify the session and messages persist
- Close browser and reopen to test localStorage persistence

### 4. Test Message Storage
- Send messages and verify they're stored in the database
- Retrieve message history for a session
- Confirm sources are properly stored for assistant messages

## Troubleshooting

### Common Issues

1. **Database Connection Errors**
   - Verify your Neon database URL is correct
   - Check that the required tables exist
   - Ensure proper SSL configuration

2. **CORS Issues**
   - Verify ALLOWED_ORIGINS includes your Docusaurus site
   - Check backend CORS configuration

3. **Text Selection Not Working**
   - Ensure proper event listeners are attached
   - Check for conflicts with other page interactions

4. **Session Persistence Issues**
   - Verify localStorage is working in the browser
   - Check that session IDs are properly maintained

### Getting Help
- Check the implementation plan in `specs/1-rag-chatbot/plan3.md` for detailed architecture
- Review the data models in `specs/1-rag-chatbot/plan3/data-model.md`
- Examine the research outcomes in `specs/1-rag-chatbot/plan3/research.md`