# Data Model: RAG Chatbot Testing & Validation

**Feature**: RAG Chatbot for Book - Phase 4
**Date**: 2025-12-17

## Entity: TestResult

### Description
Represents the result of a single test execution, capturing the outcome and relevant metadata.

### Fields

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| test_id | String (UUID) | Yes | Unique identifier for the test |
| test_name | String | Yes | Name/description of the test |
| test_type | String | Yes | Type: "unit", "integration", "e2e", "performance", "validation" |
| result | String | Yes | Result: "pass", "fail", "skip", "error" |
| timestamp | DateTime | Yes | When the test was executed |
| duration | Float | No | Time taken to execute the test in seconds |
| details | String | No | Additional information about the test result |
| component | String | No | Which system component was tested |
| environment | String | No | Environment where test was run |

### Validation Rules
1. `test_type` must be one of the allowed values
2. `result` must be one of "pass", "fail", "skip", "error"
3. `duration` must be non-negative if provided
4. `test_id` must be a valid UUID format
5. `environment` should be a valid environment identifier

## Entity: ValidationReport

### Description
Represents the results of a validation test, such as accuracy or relevance validation.

### Fields

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| report_id | String (UUID) | Yes | Unique identifier for the report |
| validation_type | String | Yes | Type: "accuracy", "relevance", "context-boundary", "hallucination" |
| score | Float | Yes | Numerical score for the validation (0-1) |
| details | JSON | No | Detailed results of the validation |
| timestamp | DateTime | Yes | When the validation was performed |
| sample_size | Integer | Yes | Number of samples tested |
| pass_threshold | Float | No | Threshold that determines pass/fail |
| test_queries | Array[String] | No | Queries used for the validation |

### Validation Rules
1. `validation_type` must be one of the allowed values
2. `score` must be between 0 and 1
3. `sample_size` must be positive
4. `pass_threshold` must be between 0 and 1 if provided
5. `report_id` must be a valid UUID format

## Entity: LogEntry

### Description
Represents a single log entry from the application, capturing system events and information.

### Fields

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| log_id | String (UUID) | Yes | Unique identifier for the log entry |
| timestamp | DateTime | Yes | When the event occurred |
| level | String | Yes | Log level: "DEBUG", "INFO", "WARN", "ERROR" |
| message | String | Yes | The log message |
| context | JSON | No | Additional context information |
| component | String | No | Which system component generated the log |
| user_id | String | No | User identifier if applicable |
| session_id | String | No | Session identifier if applicable |
| query_id | String | No | Query identifier if applicable |

### Validation Rules
1. `level` must be one of "DEBUG", "INFO", "WARN", "ERROR"
2. `log_id` must be a valid UUID format
3. `message` must be non-empty
4. `component` should be a valid system component name if provided

## Entity: PerformanceMetric

### Description
Represents a performance metric collected from the system for monitoring and analysis.

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

### Validation Rules
1. `metric_name` must be a valid metric identifier
2. `value` must be non-negative
3. `unit` should be a standard unit of measurement if provided
4. `metric_id` must be a valid UUID format

## Entity: TestSuiteResult

### Description
Represents the aggregated results of a complete test suite execution.

### Fields

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| suite_id | String (UUID) | Yes | Unique identifier for the test suite |
| suite_name | String | Yes | Name of the test suite |
| total_tests | Integer | Yes | Total number of tests in the suite |
| passed_tests | Integer | Yes | Number of tests that passed |
| failed_tests | Integer | Yes | Number of tests that failed |
| skipped_tests | Integer | Yes | Number of tests that were skipped |
| execution_time | Float | Yes | Total time to execute the suite in seconds |
| timestamp | DateTime | Yes | When the suite was executed |
| environment | String | Yes | Environment where suite was run |

### Validation Rules
1. `total_tests` must equal `passed_tests` + `failed_tests` + `skipped_tests`
2. `execution_time` must be non-negative
3. `suite_id` must be a valid UUID format
4. All count fields must be non-negative