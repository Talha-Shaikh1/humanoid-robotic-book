"""
Assessment model for the AI-Native Textbook application
"""
from sqlalchemy import Column, Integer, String, Text, DateTime, ForeignKey
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.sql import func
import uuid
from . import Base

class Assessment(Base):
    __tablename__ = "assessments"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    title = Column(String, nullable=False)
    description = Column(Text)
    module_id = Column(UUID(as_uuid=True), ForeignKey("content_modules.id"))
    questions = Column(String)  # JSON string for questions with answers
    max_score = Column(Integer, nullable=False)
    time_limit = Column(Integer)  # In minutes, nullable for untimed
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())