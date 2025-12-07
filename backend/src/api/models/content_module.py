"""
ContentModule model for the AI-Native Textbook application
"""
from sqlalchemy import Column, Integer, String, Text, DateTime, ForeignKey
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.sql import func
import uuid
from enum import Enum
from . import Base

class ModuleType(str, Enum):
    ros2 = "ros2"
    gazebo_unity = "gazebo_unity"
    nvidia_isaac = "nvidia_isaac"
    vla = "vla"

class ContentModule(Base):
    __tablename__ = "content_modules"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    title = Column(String, nullable=False)
    description = Column(Text)
    module_type = Column(String, nullable=False)  # Using string instead of SQLEnum for flexibility
    week_number = Column(Integer)  # 1-13 for weekly breakdown
    content = Column(Text)  # MDX content
    prerequisites = Column(String)  # JSON string for prerequisites
    learning_outcomes = Column(String)  # JSON string for learning outcomes
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())