# Research: AI-Native Textbook on Physical AI & Humanoid Robotics

## Tech Stack Decisions

### Backend Framework Choice
**Decision**: FastAPI for backend services
**Rationale**: FastAPI offers excellent performance, automatic API documentation (OpenAPI/Swagger), strong typing support, and async capabilities. It's well-suited for ML/AI integration and has excellent community support for building APIs that serve AI features.
**Alternatives considered**:
- Flask: More familiar but less performant and lacks automatic docs
- Django: Heavy for this use case, overkill for API services
- Express.js: Node.js option but Python ecosystem better for AI/ML features

### Authentication System
**Decision**: Better Auth or custom JWT-based authentication
**Rationale**: Better Auth provides good security practices, OAuth integration, and session management. If customization is needed, a JWT-based system would provide flexibility for personalization features.
**Alternatives considered**:
- NextAuth.js: Only for Next.js (not Docusaurus)
- Auth0/Firebase: Would exceed free tier constraints
- Simple JWT: Less secure without proper refresh token handling

### Translation Service
**Decision**: OpenAI API for translation or dedicated translation API (like LibreTranslate)
**Rationale**: OpenAI provides high-quality translation with context awareness, but LibreTranslate is open source and could work within free tier constraints. For Urdu specifically, OpenAI has good support.
**Alternatives considered**:
- Google Translate API: Cost may exceed free tier
- AWS Translate: Same cost concern
- Manual translations: Doesn't meet AI integration requirement

### Vector Database for RAG
**Decision**: Qdrant Cloud Free tier for vector storage
**Rationale**: Qdrant is specifically designed for vector similarity search, has good performance, and offers a free tier that meets the constraints. It integrates well with Python backends.
**Alternatives considered**:
- Pinecone: Good but might exceed free tier
- Weaviate: Alternative vector DB but Qdrant meets requirements
- Chroma: Self-hosted option but requires more infrastructure

## Content Structure Research

### Docusaurus Configuration
**Decision**: Standard Docusaurus setup with custom MDX components for interactive elements
**Rationale**: Docusaurus provides excellent documentation site capabilities, supports MDX natively, and can be deployed to GitHub Pages. Custom components can handle interactive elements like code examples and chatbot integration.
**Alternatives considered**:
- Gatsby: More complex setup
- Nuxt/VuePress: Less familiar ecosystem
- Custom React app: More work for similar functionality

### Module Organization
**Decision**: Organize content by the four core modules (ROS 2, Gazebo & Unity, NVIDIA Isaac, VLA) with weekly breakdowns as sub-sections
**Rationale**: This matches the course structure and user requirements. Each module gets its own section with practical examples and interactive elements.
**Alternatives considered**:
- Chronological approach: Less aligned with course modules
- Skill-based organization: Doesn't match specified structure

## Bonus Features Research

### Personalization Implementation
**Decision**: Use user profile data and interaction history to customize content recommendations
**Rationale**: Personalization can be implemented by tracking user progress, preferences, and performance on assessments to suggest relevant content or highlight important concepts.
**Alternatives considered**:
- Basic progress tracking: Less sophisticated
- Static user categories: Less dynamic adaptation

### Assessment System
**Decision**: Interactive quizzes with immediate feedback, integrated into MDX content
**Rationale**: Docusaurus supports custom React components that can render interactive quizzes with immediate feedback, meeting the assessment requirement.
**Alternatives considered**:
- Static assessment pages: Less interactive
- External assessment tools: More complex integration

## Deployment Strategy

### GitHub Pages Deployment
**Decision**: Use GitHub Actions to build and deploy Docusaurus site to GitHub Pages
**Rationale**: Meets the constraint of deploying to GitHub Pages, integrates well with version control, and is free. Backend services can be deployed separately or integrated via API calls.
**Alternatives considered**:
- Vercel: Also meets constraints but GitHub Pages was specified
- Netlify: Similar to Vercel

### Backend Hosting
**Decision**: Self-hosted FastAPI on a free tier service (like Railway, Render) or serverless functions
**Rationale**: GitHub Pages can't host backend services, so a separate hosting solution is needed for the API services (chatbot, auth, personalization).
**Alternatives considered**:
- Serverless functions: Good for specific endpoints
- Railway/Render: Good free tiers that meet constraints