"""
TranslationCache model for the AI-Native Textbook application
"""
from sqlalchemy import Column, String, Text, DateTime
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.sql import func
import uuid
from . import Base

class TranslationCache(Base):
    __tablename__ = "translation_cache"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    original_content_id = Column(UUID(as_uuid=True))  # ID of original content
    original_content_type = Column(String)  # Type of content being translated
    original_text = Column(Text, nullable=False)
    translated_text = Column(Text, nullable=False)
    target_language = Column(String, nullable=False)  # e.g., 'ur' for Urdu
    created_at = Column(DateTime(timezone=True), server_default=func.now())