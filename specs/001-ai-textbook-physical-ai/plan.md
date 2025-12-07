# Implementation Plan: AI-Native Textbook on Physical AI & Humanoid Robotics

**Branch**: `001-ai-textbook-physical-ai` | **Date**: 2025-12-07 | **Spec**: [link]
**Input**: Feature specification from `/specs/001-ai-textbook-physical-ai/spec.md`

**Note**: This template is filled in by the `/sp.plan` command. See `.specify/templates/commands/plan.md` for the execution workflow.

## Summary

Create a comprehensive AI-Native Textbook for Physical AI & Humanoid Robotics using Docusaurus with MDX format. The system will provide structured content for four core modules (ROS 2, Gazebo & Unity, NVIDIA Isaac, VLA) with interactive elements like RAG chatbot, code examples, URDF snippets, and assessments. Includes bonus features: authentication, personalization, and Urdu translation.

## Technical Context

**Language/Version**: Python 3.11, Node.js 18+ (for Docusaurus)
**Primary Dependencies**: Docusaurus, FastAPI, Qdrant, React, OpenAI API or alternatives
**Storage**: Qdrant Cloud Free (vector database), Neon Serverless Postgres (user data), GitHub Pages (static hosting)
**Testing**: pytest for backend, Playwright for e2e tests, Jest for frontend
**Target Platform**: Web application (GitHub Pages hosting)
**Project Type**: Web/single - Docusaurus frontend with FastAPI backend services
**Performance Goals**: Fast loading of content pages, <2s response for chatbot queries, responsive UI
**Constraints**: Must deploy to GitHub Pages, use free tiers (Qdrant Cloud Free, Neon Serverless Postgres), <90s demo video
**Scale/Scope**: Target students, educators, professionals in AI/robotics; support multiple content formats and interactive elements

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

- Educational Excellence: All content must be technically accurate and pedagogically sound per constitution ✓ RESOLVED
- Interactivity and Usability: Must use MDX format for Docusaurus with interactive elements per constitution ✓ RESOLVED
- AI Integration: Must leverage RAG chatbot capabilities per constitution ✓ RESOLVED
- Modularity: Architecture must support bonus features seamlessly per constitution ✓ RESOLVED
- Code Quality Standards: All code must be type-safe, error-handled, production-ready per constitution ✓ RESOLVED
- Bonus Feature Implementation: Must prioritize bonus features for maximum impact per constitution ✓ RESOLVED
- Deployment Requirements: Must deploy to GitHub Pages or Vercel per constitution ✓ RESOLVED
- Free Tier Usage: Must use Qdrant Cloud Free, Neon Serverless Postgres per constitution ✓ RESOLVED

## Post-Design Constitution Check

All design decisions comply with constitutional principles:
- Technical stack (FastAPI, Docusaurus, Qdrant) supports educational excellence and interactivity
- Architecture is modular to support bonus features
- Code quality standards maintained through type hints and error handling
- AI integration achieved through RAG chatbot and translation services
- Free tier constraints respected (Qdrant Cloud Free, Neon Postgres)

## Project Structure

### Documentation (this feature)

```text
specs/001-ai-textbook-physical-ai/
├── plan.md              # This file (/sp.plan command output)
├── research.md          # Phase 0 output (/sp.plan command)
├── data-model.md        # Phase 1 output (/sp.plan command)
├── quickstart.md        # Phase 1 output (/sp.plan command)
├── contracts/           # Phase 1 output (/sp.plan command)
└── tasks.md             # Phase 2 output (/sp.tasks command - NOT created by /sp.plan)
```

### Source Code (repository root)

```text
# Single project with Docusaurus frontend and backend services
docs/
├── src/
│   ├── components/      # Custom React components (chatbot, interactive elements)
│   ├── pages/           # Static pages
│   ├── theme/           # Custom theme components
│   └── modules/         # Content modules (ROS 2, Gazebo, etc.)
├── docs/                # MDX content for textbook
│   ├── week1-13/        # Weekly breakdown content
│   ├── modules/         # Core modules content (ROS 2, Gazebo & Unity, NVIDIA Isaac, VLA)
│   └── assessments/     # Quiz and exercise content
├── static/              # Static assets (images, diagrams)
└── docusaurus.config.js # Docusaurus configuration

backend/
├── src/
│   ├── main.py          # FastAPI application entry point
│   ├── api/             # API endpoints
│   │   ├── v1/
│   │   │   ├── chatbot/ # RAG chatbot endpoints
│   │   │   ├── auth/    # Authentication endpoints
│   │   │   └── personalization/ # Personalization endpoints
│   │   └── models/      # Data models and schemas
│   ├── services/        # Business logic
│   │   ├── rag_service.py # RAG service for chatbot
│   │   ├── content_service.py # Content management
│   │   └── translation_service.py # Translation service
│   ├── database/        # Database interactions
│   └── utils/           # Utility functions
├── tests/               # Backend tests
└── requirements.txt     # Python dependencies

api/
├── contracts/           # API contracts and schemas
└── openapi.yaml         # OpenAPI specification

# Translation and internationalization
i18n/
└── ur/
    └── docusaurus-plugin-content-docs/
        └── current/     # Urdu translation content

# Configuration and deployment
.github/
└── workflows/
    └── deploy.yml       # GitHub Actions for deployment to GitHub Pages

# Testing
tests/
├── e2e/                 # End-to-end tests
├── integration/         # Integration tests
└── unit/                # Unit tests
```

**Structure Decision**: Single project with Docusaurus frontend and backend services, organized to support modularity for bonus features and proper separation of concerns between content, API, and UI.

## Complexity Tracking

> **Fill ONLY if Constitution Check has violations that must be justified**

| Violation | Why Needed | Simpler Alternative Rejected Because |
|-----------|------------|-------------------------------------|
| [e.g., 4th project] | [current need] | [why 3 projects insufficient] |
| [e.g., Repository pattern] | [specific problem] | [why direct DB access insufficient] |