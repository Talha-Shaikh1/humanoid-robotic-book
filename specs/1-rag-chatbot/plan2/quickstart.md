# Quickstart Guide: RAG Chatbot Retrieval, Agent & API Integration

**Feature**: RAG Chatbot for Book - Phase 2
**Date**: 2025-12-17

## Overview
This guide provides step-by-step instructions to set up and run the retrieval, agent, and API integration layer for the RAG chatbot. This system assumes that vector embeddings already exist in Qdrant and focuses on building the retrieval layer, AI agent, backend API, and frontend integration.

## Prerequisites

### System Requirements
- Python 3.8 or higher
- `uv` package manager (https://github.com/astral-sh/uv)
- Access to OpenAI API
- Access to existing Qdrant Cloud collection from Phase 1

### API Access
- OpenAI API Key (for agent functionality)
- Qdrant Cloud URL and API Key (for retrieval)
- Existing Qdrant collection from Phase 1

## Setup Instructions

### 1. Clone and Navigate to Project
```bash
# If starting from scratch, create a new backend directory
mkdir rag-backend-phase2 && cd rag-backend-phase2
```

### 2. Install uv Package Manager (if not already installed)
```bash
# Install uv using the official installer
curl -LsSf https://astral.sh/uv/install.sh | sh
# Or on Windows with PowerShell:
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

### 3. Create Python Project with Dependencies
```bash
# Initialize a new Python project
uv init
# Or if you already have a project, just create a virtual environment
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

### 4. Install Required Dependencies
```bash
uv pip install openai fastapi uvicorn qdrant-client python-dotenv pydantic
```

### 5. Configure Environment Variables
Create a `.env` file in your project root with the following content:

```env
OPENAI_API_KEY=your_openai_api_key_here
QDRANT_URL=your_qdrant_cloud_url_here
QDRANT_API_KEY=your_qdrant_api_key_here
COLLECTION_NAME=robotics_book_chunks
MODEL_NAME=gpt-4-turbo
TOP_K=5
SIMILARITY_THRESHOLD=0.3
```

**Note**: Replace placeholder values with your actual API keys and URLs. Use the same QDRANT credentials and COLLECTION_NAME from Phase 1.

## Running the Backend Server

### 1. Create the Application Files
Create the following files for the complete backend system:

**main.py** - FastAPI application
**retrieval.py** - Qdrant retrieval logic
**agent.py** - OpenAI agent implementation
**models.py** - Data models and schemas
**config.py** - Configuration and settings

### 2. Execute the Server
```bash
uv run uvicorn main:app --reload
```

### 3. Monitor the Process
The server will start and you should see:
- FastAPI startup messages
- Available endpoints
- Server running on localhost:8000 (default)

## API Endpoints

### Health Check
```
GET http://localhost:8000/health
```

### Chat (Full Book Mode)
```
POST http://localhost:8000/chat
Content-Type: application/json

{
  "query": "What is ROS 2?",
  "context_mode": "full_book",
  "session_id": "session-123",
  "top_k": 5
}
```

### Chat (Selected Text Mode)
```
POST http://localhost:8000/chat/selected
Content-Type: application/json

{
  "query": "What does this mean?",
  "selected_text": "ROS 2 is a flexible framework for robotics development...",
  "session_id": "session-123"
}
```

## Expected Output
During execution, you should see:
- Server startup confirmation
- API endpoint availability
- Successful response to sample queries
- Proper agent behavior with context boundaries

## Frontend Integration

### 1. Docusaurus Integration
Add the chatbot component to your Docusaurus site by adding it to the layout or as a floating widget.

### 2. Text Selection Feature
The frontend should implement:
- Text selection detection
- "Ask from selected text" button that appears on selection
- API call to the `/chat/selected` endpoint

### 3. Configuration
Ensure CORS is properly configured to allow communication between Docusaurus frontend and FastAPI backend.

## Verification Steps

### 1. Test API Endpoints
Use curl or a tool like Postman to test the endpoints:

```bash
# Test health endpoint
curl http://localhost:8000/health

# Test chat endpoint
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is ROS 2?",
    "context_mode": "full_book",
    "top_k": 5
  }'
```

### 2. Verify Agent Behavior
- Test that agent answers only from provided context
- Verify that agent responds with uncertainty when answer not found
- Confirm that selected-text mode doesn't reference external content

### 3. Check Retrieval Quality
- Verify that retrieved results are relevant to queries
- Confirm that URLs and section titles are preserved
- Test that similarity thresholds work correctly

## Troubleshooting

### Common Issues

1. **API Key Errors**
   - Verify your OpenAI and Qdrant API keys are correct
   - Check that your `.env` file is properly formatted

2. **Qdrant Connection Issues**
   - Verify the QDRANT_URL and API_KEY match Phase 1 settings
   - Confirm the COLLECTION_NAME exists and is populated

3. **Agent Hallucination**
   - Ensure proper context is being passed to the agent
   - Verify agent instructions are properly configured

4. **CORS Issues**
   - Configure CORS middleware in FastAPI to allow your Docusaurus domain
   - Check frontend network requests for CORS errors

### Getting Help
- Check the implementation plan in `specs/1-rag-chatbot/plan2.md` for detailed architecture
- Review the data models in `specs/1-rag-chatbot/plan2/data-model.md`
- Examine the research outcomes in `specs/1-rag-chatbot/plan2/research.md`