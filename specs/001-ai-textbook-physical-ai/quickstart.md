# Quickstart Guide: AI-Native Textbook on Physical AI & Humanoid Robotics

## Prerequisites

- Node.js 18+ and npm/yarn
- Python 3.11+
- Git
- Access to OpenAI API (or alternative) for RAG and translation features
- Qdrant Cloud Free account for vector database
- Neon Serverless Postgres account for user data

## Setup Instructions

### 1. Clone the Repository
```bash
git clone <repository-url>
cd humanoid-robotics-book
```

### 2. Setup Backend (FastAPI)

```bash
cd backend
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt

# Set environment variables
cp .env.example .env
# Edit .env with your API keys and database URLs
```

### 3. Setup Frontend (Docusaurus)

```bash
cd docs  # or the frontend directory
npm install
```

### 4. Environment Configuration

Create `.env` files in both backend and docs directories:

**Backend `.env`:**
```
DATABASE_URL=postgresql://user:password@ep-xxx.us-east-1.aws.neon.tech/dbname
QDRANT_URL=https://your-cluster.qdrant.tech
QDRANT_API_KEY=your-api-key
OPENAI_API_KEY=your-openai-key
JWT_SECRET=your-jwt-secret
```

**Frontend `.env`:**
```
REACT_APP_API_BASE_URL=http://localhost:8000/api/v1
REACT_APP_QDRANT_URL=https://your-cluster.qdrant.tech
```

### 5. Run the Applications

**Backend:**
```bash
cd backend
uvicorn src.main:app --reload --port 8000
```

**Frontend:**
```bash
cd docs
npm start
```

## Development Workflow

### Adding New Content Modules

1. Create MDX files in `docs/docs/modules/` for each module (ROS 2, Gazebo & Unity, NVIDIA Isaac, VLA)
2. Update `docs/sidebars.js` to include new content in navigation
3. Add module data to the database using the admin interface or seed scripts

### Adding Interactive Elements

1. Create custom React components in `docs/src/components/`
2. Use the components in MDX files using standard Docusaurus syntax
3. Ensure components follow accessibility guidelines

### Running Tests

**Backend tests:**
```bash
cd backend
pytest
```

**Frontend tests:**
```bash
cd docs
npm test
```

**E2E tests:**
```bash
cd tests
npx playwright test
```

## API Endpoints

- Backend API: `http://localhost:8000/api/v1`
- Documentation: `http://localhost:8000/docs` (Swagger UI)
- Frontend: `http://localhost:3000`

## Deployment

### GitHub Pages Deployment

1. Build the Docusaurus site:
```bash
cd docs
npm run build
```

2. The site will be built to the `build` directory and can be deployed to GitHub Pages

3. Set up GitHub Actions workflow for automated deployment (see `.github/workflows/deploy.yml`)

### Backend Deployment

Deploy the FastAPI backend to a service that supports Python applications (e.g., Railway, Render, or AWS/Azure).

## Bonus Features Setup

### Authentication
- The auth system supports both email/password and OAuth providers
- Configure providers in the admin panel

### Personalization
- User progress tracking is enabled by default
- Personalization algorithms use progress data to suggest content

### Translation
- Urdu translation is available through the translation service
- Content can be translated on-demand or pre-translated for better performance

## Troubleshooting

- If you get CORS errors, ensure your backend and frontend are configured to allow the correct origins
- For database connection issues, verify your Neon Postgres connection string
- For vector search issues, check your Qdrant connection and API key