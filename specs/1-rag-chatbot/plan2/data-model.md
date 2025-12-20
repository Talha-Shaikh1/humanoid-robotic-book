# Data Model: RAG Chatbot Retrieval, Agent & API Integration

**Feature**: RAG Chatbot for Book - Phase 2
**Date**: 2025-12-17

## Entity: ChatRequest

### Description
Represents a user's request to the chatbot system, containing the query and context parameters.

### Fields

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| query | String | Yes | The user's question or query text |
| context_mode | String | Yes | Context mode: "full_book" or "selected_text" |
| selected_text | String | No | Text selected by user for selected_text mode |
| top_k | Integer | No | Number of results to retrieve (default: 5) |
| session_id | String | No | Identifier for conversation session |
| user_id | String | No | Identifier for the user (if available) |
| timestamp | DateTime | Yes | When the request was made |

### Validation Rules
1. `context_mode` must be either "full_book" or "selected_text"
2. `query` must be non-empty and less than 2000 characters
3. `top_k` must be between 1 and 20 if provided
4. `selected_text` required when context_mode is "selected_text"
5. `session_id` should be a valid UUID format

## Entity: ChatResponse

### Description
Represents the system's response to a user's chat request, containing the answer and metadata.

### Fields

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| answer | String | Yes | The agent's response to the query |
| sources | Array[String] | Yes | List of source URLs referenced in the answer |
| confidence | Float | Yes | Confidence level of the answer (0-1) |
| context_used | String | No | Text that was provided to the agent |
| query_id | String | Yes | Unique identifier for the query |
| timestamp | DateTime | Yes | When the response was generated |
| tokens_used | Integer | No | Number of tokens used in the response |

### Validation Rules
1. `confidence` must be between 0 and 1
2. `answer` must be non-empty
3. `query_id` must be a valid UUID
4. `sources` should contain valid URLs when provided
5. `tokens_used` must be non-negative if provided

## Entity: RetrievalResult

### Description
Represents a result from the semantic search in the Qdrant vector database.

### Fields

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| id | String | Yes | Unique identifier for the result |
| content | String | Yes | Text content of the retrieved chunk |
| source_url | String | Yes | URL of the source page |
| page_title | String | Yes | Title of the source page |
| score | Float | Yes | Similarity score from vector search (0-1) |
| heading | String | No | Main heading associated with this chunk |
| chunk_index | Integer | No | Position of this chunk within the page |
| metadata | Object | No | Additional metadata from Qdrant payload |

### Validation Rules
1. `score` must be between 0 and 1
2. `content` must be non-empty
3. `source_url` must be a valid URL from the target domain
4. `chunk_index` must be non-negative if provided
5. `id` must be unique within the search results

## Entity: AgentToolCall

### Description
Represents a tool call made by the AI agent during response generation, specifically for retrieval.

### Fields

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| tool_name | String | Yes | Name of the tool called (e.g., "retrieve_from_qdrant") |
| parameters | Object | Yes | Parameters passed to the tool |
| result | Object | No | Result returned by the tool |
| timestamp | DateTime | Yes | When the tool was called |

### Validation Rules
1. `tool_name` must be a valid tool name in the system
2. `parameters` must match the expected schema for the tool
3. `timestamp` must be current or past time

## Entity: ChatSession

### Description
Represents a conversation session between user and chatbot for maintaining context.

### Fields

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| session_id | String | Yes | Unique identifier for the session |
| user_id | String | No | Identifier for the user |
| created_at | DateTime | Yes | When the session was created |
| last_activity | DateTime | Yes | When the session was last used |
| message_count | Integer | Yes | Number of messages in the session |

### Validation Rules
1. `session_id` must be a valid UUID
2. `message_count` must be non-negative
3. `last_activity` must be same or after `created_at`