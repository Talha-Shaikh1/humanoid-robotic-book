"""
Modules API routes for the AI-Native Textbook application
"""
from fastapi import APIRouter, Depends, HTTPException, status
from typing import List, Optional
from uuid import UUID
from pydantic import BaseModel
from ...services.content_service import ContentService
from ...database import get_db
from sqlalchemy.orm import Session

# Create the router
router = APIRouter()

# Pydantic models for request/response
class ContentModuleCreate(BaseModel):
    title: str
    description: Optional[str] = None
    module_type: str
    week_number: Optional[int] = None
    content: str
    prerequisites: Optional[str] = None
    learning_outcomes: Optional[str] = None

class ContentModuleResponse(BaseModel):
    id: UUID
    title: str
    description: Optional[str]
    module_type: str
    week_number: Optional[int]
    content: str
    prerequisites: Optional[str]
    learning_outcomes: Optional[str]

# Initialize service
content_service = ContentService()

@router.get("/modules", response_model=List[ContentModuleResponse])
def get_all_modules(
    module_type: Optional[str] = None,
    week_number: Optional[int] = None,
    db: Session = Depends(get_db)
):
    """
    Get all content modules with optional filtering
    """
    return content_service.get_all_modules(db, module_type, week_number)

@router.get("/modules/{module_id}", response_model=ContentModuleResponse)
def get_module(
    module_id: UUID,
    db: Session = Depends(get_db)
):
    """
    Get a specific content module
    """
    module = content_service.get_module_by_id(db, module_id)
    if not module:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Module not found"
        )
    return module

@router.post("/modules", response_model=ContentModuleResponse)
def create_module(
    module: ContentModuleCreate,
    db: Session = Depends(get_db)
):
    """
    Create a new content module
    """
    return content_service.create_module(db, module)

@router.put("/modules/{module_id}", response_model=ContentModuleResponse)
def update_module(
    module_id: UUID,
    module: ContentModuleCreate,
    db: Session = Depends(get_db)
):
    """
    Update an existing content module
    """
    updated_module = content_service.update_module(db, module_id, module)
    if not updated_module:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Module not found"
        )
    return updated_module

@router.delete("/modules/{module_id}")
def delete_module(
    module_id: UUID,
    db: Session = Depends(get_db)
):
    """
    Delete a content module
    """
    success = content_service.delete_module(db, module_id)
    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Module not found"
        )
    return {"message": "Module deleted successfully"}