"""
InteractiveElement model for the AI-Native Textbook application
"""
from sqlalchemy import Column, Integer, String, Text, DateTime, ForeignKey
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.sql import func
import uuid
from enum import Enum
from . import Base

class ElementType(str, Enum):
    code_example = "code_example"
    urdf_snippet = "urdf_snippet"
    quiz = "quiz"
    exercise = "exercise"

class DifficultyLevel(str, Enum):
    beginner = "beginner"
    intermediate = "intermediate"
    advanced = "advanced"

class InteractiveElement(Base):
    __tablename__ = "interactive_elements"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    title = Column(String, nullable=False)
    type = Column(String, nullable=False)  # Using string instead of SQLEnum for flexibility
    content = Column(Text, nullable=False)  # The actual code/quiz content
    module_id = Column(UUID(as_uuid=True), ForeignKey("content_modules.id"))
    difficulty = Column(String)  # Using string instead of SQLEnum for flexibility
    language = Column(String)  # For code examples
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())