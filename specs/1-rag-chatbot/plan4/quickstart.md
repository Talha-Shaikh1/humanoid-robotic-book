# Quickstart Guide: RAG Chatbot Testing & Validation

**Feature**: RAG Chatbot for Book - Phase 4
**Date**: 2025-12-17

## Overview
This guide provides step-by-step instructions to set up and run comprehensive testing and validation for the RAG chatbot system. This includes unit tests, integration tests, end-to-end tests, accuracy validation, and monitoring implementation.

## Prerequisites

### System Requirements
- Python 3.8 or higher
- `uv` package manager
- Access to the complete RAG system (ingestion, retrieval, agent, frontend)
- Test data for validation

### Dependencies
- pytest for testing framework
- pytest-cov for coverage analysis
- pytest-asyncio for async testing
- httpx for API testing

## Setup Instructions

### 1. Install Testing Dependencies
```bash
# Navigate to project directory
cd rag-backend

# Install testing dependencies
uv pip install pytest pytest-asyncio pytest-cov httpx
```

### 2. Configure Test Environment
Create `pytest.ini` in your project root:

```ini
[tool:pytest]
testpaths = tests
python_files = test_*.py
python_classes = Test*
python_functions = test_*
addopts =
    --verbose
    --cov=src
    --cov-report=html
    --cov-report=term-missing
    --strict-markers
markers =
    unit: Unit tests
    integration: Integration tests
    e2e: End-to-end tests
    validation: Validation tests
    slow: Slow running tests
```

### 3. Create Test Directory Structure
```bash
mkdir -p tests/{unit,integration,e2e,validation}
mkdir -p validation monitoring
```

### 4. Configure Logging
Set up structured logging in your application:

```python
# In logger.py
import logging
import json
from datetime import datetime

class StructuredLogger:
    def __init__(self, name):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.INFO)

    def _log(self, level, message, **context):
        log_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": level,
            "message": message,
            "context": context
        }
        self.logger.log(getattr(logging, level.upper()), json.dumps(log_entry))
```

## Running Tests

### 1. Unit Tests
Run core logic unit tests:
```bash
pytest tests/unit/ -v
```

### 2. Integration Tests
Run API and database integration tests:
```bash
pytest tests/integration/ -v
```

### 3. End-to-End Tests
Run complete workflow tests:
```bash
pytest tests/e2e/ -v
```

### 4. Validation Tests
Run accuracy and relevance validation:
```bash
python -m validation.accuracy_test
python -m validation.relevance_test
python -m validation.context_boundary_test
```

### 5. All Tests with Coverage
Run complete test suite with coverage:
```bash
pytest --cov=src --cov-report=html
```

## Validation Procedures

### 1. Accuracy Validation
Test agent responses against known book content:
```python
# Example validation script
from validation.accuracy import validate_accuracy

results = validate_accuracy(
    test_queries=[
        {"question": "What is ROS 2?", "expected_answer": "ROS 2 is a robotics framework..."}
    ],
    sample_size=100
)
print(f"Accuracy: {results['accuracy']:.2%}")
```

### 2. Context Boundary Validation
Ensure selected-text mode doesn't leak external content:
```python
from validation.context_boundary import validate_context_isolation

results = validate_context_isolation(
    test_queries=[
        {"question": "What does this mean?", "selected_text": "ROS 2 is a framework..."}
    ]
)
print(f"Context compliance: {results['compliance_rate']:.2%}")
```

### 3. Performance Testing
Test system performance under load:
```bash
# Example using a load testing tool
pytest tests/performance/ --benchmark-only
```

## Monitoring Setup

### 1. Metrics Collection
Implement metrics collection in your application:
```python
# In metrics.py
from monitoring.metrics import MetricsCollector

metrics = MetricsCollector()
metrics.record_response_time(1.2)  # seconds
metrics.record_error_count(1)
```

### 2. Health Checks
Verify system health:
```bash
curl http://localhost:8000/health
curl http://localhost:8000/metrics
```

## Expected Output
During testing, you should see:
- Test results with pass/fail status
- Coverage reports showing code coverage percentages
- Validation accuracy and compliance rates
- Performance metrics and response times
- Structured log entries for debugging

## Verification Steps

### 1. Test Coverage Verification
- Verify unit tests cover 80% of core logic
- Confirm integration tests validate all endpoints
- Check that end-to-end tests cover complete workflows

### 2. Accuracy Validation
- Run accuracy tests against known book content
- Verify response accuracy exceeds 90%
- Check that answers are properly sourced

### 3. Context Boundary Validation
- Test selected-text mode for external content leakage
- Verify 99% compliance with context boundaries
- Confirm full-book mode works correctly

### 4. Performance Verification
- Measure response times under 2 seconds
- Verify system handles expected concurrent users
- Check error rates remain below 1%

## Troubleshooting

### Common Issues

1. **Test Failures**
   - Verify all dependencies are installed
   - Check that the RAG system is running
   - Review test configuration and fixtures

2. **Low Coverage**
   - Identify uncovered code paths
   - Write additional tests for missing coverage
   - Focus on critical validation and data handling code

3. **Performance Issues**
   - Check system resource usage
   - Verify database and vector store connections
   - Review API rate limits and timeouts

4. **Validation Problems**
   - Verify test data accuracy
   - Check that validation thresholds are appropriate
   - Review agent configuration and instructions

### Getting Help
- Check the implementation plan in `specs/1-rag-chatbot/plan4.md` for detailed architecture
- Review the data models in `specs/1-rag-chatbot/plan4/data-model.md`
- Examine the research outcomes in `specs/1-rag-chatbot/plan4/research.md`