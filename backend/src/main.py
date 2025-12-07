"""
Main FastAPI application for the AI-Native Textbook on Physical AI & Humanoid Robotics
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import API routers
from .api.v1.modules import router as modules_router
from .api.v1.assessments import router as assessments_router
from .api.v1.progress import router as progress_router
from .api.v1.chatbot import router as chatbot_router
from .api.v1.auth import router as auth_router
from .api.v1.translation import router as translation_router

# Create FastAPI app
app = FastAPI(
    title="AI-Native Textbook API",
    description="API for the Physical AI & Humanoid Robotics textbook application",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routers
app.include_router(modules_router, prefix="/api/v1", tags=["modules"])
app.include_router(assessments_router, prefix="/api/v1", tags=["assessments"])
app.include_router(progress_router, prefix="/api/v1", tags=["progress"])
app.include_router(chatbot_router, prefix="/api/v1", tags=["chatbot"])
app.include_router(auth_router, prefix="/api/v1", tags=["auth"])
app.include_router(translation_router, prefix="/api/v1", tags=["translation"])

@app.get("/")
def read_root():
    return {"message": "AI-Native Textbook API for Physical AI & Humanoid Robotics"}

@app.get("/health")
def health_check():
    return {"status": "healthy", "service": "AI-Native Textbook API"}