# Data Model: RAG Chatbot Evaluation, Hardening & Documentation

**Feature**: RAG Chatbot for Book - Phase 5
**Date**: 2025-12-17

## Entity: EvaluationResult

### Description
Represents the result of evaluating a single query-response interaction, capturing accuracy and relevance metrics.

### Fields

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| evaluation_id | String (UUID) | Yes | Unique identifier for the evaluation |
| query | String | Yes | The test query used |
| query_type | String | Yes | Type: "factual", "conceptual", "edge_case" |
| retrieved_context | Array[Object] | Yes | Context provided to the agent |
| agent_response | String | Yes | The agent's response |
| expected_answer | String | No | Expected response for accuracy evaluation |
| accuracy_score | Float | Yes | Numerical score for answer accuracy (0-1) |
| relevance_score | Float | Yes | Score for context relevance (0-1) |
| hallucination_detected | Boolean | Yes | Whether hallucination was detected |
| sources_used | Array[String] | Yes | Sources referenced in the response |
| timestamp | DateTime | Yes | When the evaluation was performed |
| session_id | String | No | Session identifier if applicable |

### Validation Rules
1. `query_type` must be one of "factual", "conceptual", "edge_case"
2. `accuracy_score` and `relevance_score` must be between 0 and 1
3. `evaluation_id` must be a valid UUID format
4. `query` must be non-empty
5. `retrieved_context` must contain at least one item

## Entity: LogEntry

### Description
Represents a structured log entry for observability and debugging of the RAG system.

### Fields

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| log_id | String (UUID) | Yes | Unique identifier for the log entry |
| timestamp | DateTime | Yes | When the event occurred |
| level | String | Yes | Log level: "DEBUG", "INFO", "WARN", "ERROR" |
| message | String | Yes | The log message |
| context | JSON | No | Additional context information |
| user_id | String | No | User identifier if applicable |
| session_id | String | No | Session identifier |
| query_id | String | No | Query identifier |
| component | String | No | Which system component generated the log |
| pii_protected | Boolean | Yes | Whether PII has been removed from context |

### Validation Rules
1. `level` must be one of "DEBUG", "INFO", "WARN", "ERROR"
2. `log_id` must be a valid UUID format
3. `message` must be non-empty
4. `pii_protected` defaults to true for security
5. `component` should be a valid system component name if provided

## Entity: PerformanceMetric

### Description
Represents a performance metric collected from the RAG system for monitoring and optimization.

### Fields

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| metric_id | String (UUID) | Yes | Unique identifier for the metric |
| metric_name | String | Yes | Name of the metric (e.g., "response_time", "accuracy") |
| value | Float | Yes | The metric value |
| timestamp | DateTime | Yes | When the metric was collected |
| component | String | No | Which component the metric relates to |
| unit | String | No | Unit of measurement (e.g., "seconds", "percentage") |
| tags | JSON | No | Additional tags for categorization |
| query_type | String | No | Type of query if metric is query-specific |

### Validation Rules
1. `metric_name` must be a valid metric identifier
2. `value` must be non-negative
3. `unit` should be a standard unit of measurement if provided
4. `metric_id` must be a valid UUID format
5. `query_type` must be one of the defined query types if provided

## Entity: FailureCase

### Description
Documents a specific failure case or limitation discovered during evaluation.

### Fields

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| failure_id | String (UUID) | Yes | Unique identifier for the failure case |
| failure_type | String | Yes | Type: "hallucination", "retrieval_error", "context_boundary", "performance" |
| query | String | Yes | The query that caused the failure |
| system_response | String | Yes | How the system responded |
| expected_behavior | String | No | What the expected behavior should be |
| severity | String | Yes | Severity: "low", "medium", "high", "critical" |
| status | String | Yes | Status: "open", "investigating", "resolved", "wont_fix" |
| resolution_notes | String | No | Notes about how the issue was resolved |
| timestamp | DateTime | Yes | When the failure was discovered |

### Validation Rules
1. `failure_type` must be one of the allowed values
2. `severity` must be one of "low", "medium", "high", "critical"
3. `status` must be one of "open", "investigating", "resolved", "wont_fix"
4. `failure_id` must be a valid UUID format
5. `query` must be non-empty

## Entity: HardeningRule

### Description
Represents a specific hardening rule or configuration for preventing hallucination and ensuring proper behavior.

### Fields

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| rule_id | String (UUID) | Yes | Unique identifier for the rule |
| rule_name | String | Yes | Name of the hardening rule |
| rule_type | String | Yes | Type: "context_boundary", "response_validation", "content_filter" |
| configuration | JSON | Yes | Rule-specific configuration parameters |
| enabled | Boolean | Yes | Whether the rule is currently active |
| priority | Integer | Yes | Priority level for rule execution |
| created_at | DateTime | Yes | When the rule was created |
| updated_at | DateTime | Yes | When the rule was last updated |

### Validation Rules
1. `rule_type` must be one of the allowed values
2. `enabled` defaults to true
3. `priority` must be a positive integer
4. `rule_id` must be a valid UUID format
5. `updated_at` must be same or after `created_at`