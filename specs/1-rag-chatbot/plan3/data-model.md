# Data Model: RAG Chatbot Frontend Integration & Data Management

**Feature**: RAG Chatbot for Book - Phase 3
**Date**: 2025-12-17

## Entity: ChatSession

### Description
Represents a conversation session between user and chatbot, storing metadata and managing conversation context.

### Fields

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| session_id | String (UUID) | Yes | Unique identifier for the session |
| user_id | String | No | Identifier for the user (optional, for registered users) |
| created_at | DateTime | Yes | Timestamp when session was created |
| last_activity | DateTime | Yes | Timestamp of last interaction |
| title | String | No | Auto-generated title based on first query |
| metadata | JSON | No | Additional session metadata (source page, referrer, etc.) |
| is_active | Boolean | Yes | Whether the session is currently active |

### Validation Rules
1. `session_id` must be a valid UUID format
2. `created_at` must be before or equal to `last_activity`
3. `is_active` defaults to true
4. `user_id` should be a valid identifier format if provided
5. `title` must be less than 200 characters if provided

### Relationships
- One ChatSession contains many Messages

## Entity: Message

### Description
Represents an individual message within a chat session, either from the user or the assistant.

### Fields

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| message_id | String (UUID) | Yes | Unique identifier for the message |
| session_id | String (UUID) | Yes | Reference to parent session |
| role | String | Yes | Message role: "user" or "assistant" |
| content | String | Yes | The actual message content |
| timestamp | DateTime | Yes | When the message was created |
| sources | Array[Object] | No | For assistant messages, list of sources used |
| parent_message_id | String (UUID) | No | Reference to parent message for threading |

### Validation Rules
1. `role` must be either "user" or "assistant"
2. `message_id` must be a valid UUID format
3. `session_id` must reference an existing ChatSession
4. `content` must be non-empty and less than 10,000 characters
5. `sources` required when role is "assistant" and sources exist
6. `parent_message_id` must reference an existing message in the same session if provided

### Relationships
- Many Messages belong to one ChatSession

## Entity: UserInteraction

### Description
Tracks user interactions with the chatbot interface for analytics and improvement purposes.

### Fields

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| interaction_id | String (UUID) | Yes | Unique identifier for the interaction |
| session_id | String (UUID) | Yes | Reference to the session |
| interaction_type | String | Yes | Type: "open_chat", "send_message", "close_chat", "text_selection" |
| element_target | String | No | UI element targeted by the interaction |
| timestamp | DateTime | Yes | When the interaction occurred |
| metadata | JSON | No | Additional interaction metadata |

### Validation Rules
1. `interaction_type` must be one of the allowed values
2. `session_id` must reference an existing ChatSession
3. `element_target` should be a valid UI element identifier if provided

## Entity: ChatPreferences

### Description
Stores user preferences for the chatbot interface and behavior.

### Fields

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| user_id | String | Yes | Identifier for the user |
| theme | String | No | UI theme: "light", "dark", "auto" (default: "auto") |
| position | String | No | Widget position: "bottom-right", "bottom-left" (default: "bottom-right") |
| is_expanded | Boolean | No | Whether chat is expanded by default (default: false) |
| created_at | DateTime | Yes | When preferences were first set |
| updated_at | DateTime | Yes | When preferences were last updated |

### Validation Rules
1. `theme` must be one of "light", "dark", or "auto"
2. `position` must be one of "bottom-right" or "bottom-left"
3. `user_id` should be a valid identifier format
4. `updated_at` must be same or after `created_at`

### Relationships
- One User can have one ChatPreference