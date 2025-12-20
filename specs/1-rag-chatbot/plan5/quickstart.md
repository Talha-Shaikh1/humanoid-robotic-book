# Quickstart Guide: RAG Chatbot Evaluation, Hardening & Documentation

**Feature**: RAG Chatbot for Book - Phase 5
**Date**: 2025-12-17

## Overview
This guide provides step-by-step instructions to set up and run evaluation, hardening, and documentation for the RAG chatbot system. This includes comprehensive evaluation of the RAG pipeline, hallucination prevention, observability enhancements, and performance optimization.

## Prerequisites

### System Requirements
- Complete RAG system (ingestion, retrieval, agent, frontend) deployed
- Python 3.8 or higher
- `uv` package manager
- Access to all system components (OpenAI, Qdrant, FastAPI, etc.)

### Dependencies
- pytest for evaluation framework
- openai for agent interaction
- qdrant-client for retrieval evaluation
- fastapi for metrics endpoints

## Setup Instructions

### 1. Install Dependencies
```bash
# Navigate to project directory
cd rag-backend

# Install evaluation and monitoring dependencies
uv pip install pytest openai qdrant-client fastapi uvicorn
```

### 2. Configure Environment Variables
Create `.env` file in your project root:

```env
OPENAI_API_KEY=your_openai_api_key
QDRANT_URL=your_qdrant_cloud_url
QDRANT_API_KEY=your_qdrant_api_key
COLLECTION_NAME=robotics_book_chunks
LOG_LEVEL=INFO
METRICS_ENABLED=true
EVALUATION_ENABLED=true
```

### 3. Create Evaluation Directory Structure
```bash
mkdir -p evaluation hardening monitoring docs
```

## Running Evaluation

### 1. Execute Evaluation Suite
Run comprehensive evaluation tests:
```bash
python -m evaluation.run_tests
```

### 2. Run Specific Evaluation Categories
```bash
# Factual questions
python -m evaluation.run_tests --category factual

# Conceptual explanations
python -m evaluation.run_tests --category conceptual

# Edge cases
python -m evaluation.run_tests --category edge_case
```

### 3. Generate Evaluation Reports
```bash
python -m evaluation.report_generator
```

## Enabling Hardening Features

### 1. Activate Hallucination Prevention
```bash
python -m hardening.activate_protection
```

### 2. Configure Context Boundary Rules
```bash
python -m hardening.context_enforcer --enable-boundaries
```

### 3. Validate Response Compliance
```bash
python -m hardening.response_validator --check-compliance
```

## Monitoring System Behavior

### 1. Enable Structured Logging
Configure logging in your application:

```python
from monitoring.structured_logger import StructuredLogger
logger = StructuredLogger("rag-chatbot")
```

### 2. Collect Performance Metrics
Access metrics via the API:

```bash
# Get performance metrics
curl http://localhost:8000/metrics/performance

# Get hardening status
curl http://localhost:8000/hardening/status

# Get evaluation results
curl http://localhost:8000/evaluate/results
```

### 3. Monitor Query-Response Traces
Check structured logs for detailed query-response traces that enable full traceability.

## Expected Output
During execution, you should see:
- Evaluation results with accuracy and relevance scores
- Hardening status confirming protection is active
- Structured logs with complete query-response traces
- Performance metrics showing system behavior
- Failure cases documented with severity levels

## Verification Steps

### 1. Evaluation Verification
- Run test queries covering factual, conceptual, and edge cases
- Verify accuracy scores meet requirements (>95% for factual)
- Check that majority of answers are grounded in retrieved context

### 2. Hardening Verification
- Test that agent responds with "not found in book" when appropriate
- Verify hallucination prevention is actively blocking out-of-context responses
- Confirm context boundary enforcement is working correctly

### 3. Observability Verification
- Check that structured logs contain query, chunks, and response information
- Verify metrics are being collected and accessible
- Confirm system behavior is fully traceable

### 4. Performance Verification
- Measure response latency and confirm it meets benchmarks
- Verify system handles expected load without degradation
- Check that performance metrics are within acceptable ranges

## Troubleshooting

### Common Issues

1. **Evaluation Failures**
   - Verify all system components are running
   - Check API keys and connection strings
   - Ensure test data is properly formatted

2. **Hardening Not Working**
   - Confirm agent instructions are properly configured
   - Check that context boundary rules are enabled
   - Verify response validation is active

3. **Logging Issues**
   - Ensure structured logging is properly configured
   - Check that PII protection is working
   - Verify log format is consistent

4. **Performance Problems**
   - Monitor system resource usage
   - Check for bottlenecks in the RAG pipeline
   - Review API rate limits and timeouts

### Getting Help
- Check the implementation plan in `specs/1-rag-chatbot/plan5.md` for detailed architecture
- Review the data models in `specs/1-rag-chatbot/plan5/data-model.md`
- Examine the research outcomes in `specs/1-rag-chatbot/plan5/research.md`