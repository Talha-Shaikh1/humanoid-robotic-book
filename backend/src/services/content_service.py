"""
Content service for the AI-Native Textbook application
"""
from typing import List, Optional
from uuid import UUID
from sqlalchemy.orm import Session
from ..api.models.content_module import ContentModule
from ..api.models.interactive_element import InteractiveElement
from ..api.v1.modules.routes import ContentModuleCreate

class ContentService:
    def __init__(self):
        pass

    def get_all_modules(
        self,
        db: Session,
        module_type: Optional[str] = None,
        week_number: Optional[int] = None
    ) -> List[ContentModule]:
        """
        Get all content modules with optional filtering
        """
        query = db.query(ContentModule)

        if module_type:
            query = query.filter(ContentModule.module_type == module_type)

        if week_number is not None:
            query = query.filter(ContentModule.week_number == week_number)

        return query.all()

    def get_module_by_id(self, db: Session, module_id: UUID) -> Optional[ContentModule]:
        """
        Get a content module by ID
        """
        return db.query(ContentModule).filter(ContentModule.id == module_id).first()

    def create_module(self, db: Session, module_data: ContentModuleCreate) -> ContentModule:
        """
        Create a new content module
        """
        db_module = ContentModule(
            title=module_data.title,
            description=module_data.description,
            module_type=module_data.module_type,
            week_number=module_data.week_number,
            content=module_data.content,
            prerequisites=module_data.prerequisites,
            learning_outcomes=module_data.learning_outcomes
        )
        db.add(db_module)
        db.commit()
        db.refresh(db_module)
        return db_module

    def update_module(
        self,
        db: Session,
        module_id: UUID,
        module_data: ContentModuleCreate
    ) -> Optional[ContentModule]:
        """
        Update an existing content module
        """
        db_module = self.get_module_by_id(db, module_id)
        if db_module:
            for field, value in module_data.dict().items():
                setattr(db_module, field, value)
            db.commit()
            db.refresh(db_module)
        return db_module

    def delete_module(self, db: Session, module_id: UUID) -> bool:
        """
        Delete a content module
        """
        db_module = self.get_module_by_id(db, module_id)
        if db_module:
            db.delete(db_module)
            db.commit()
            return True
        return False

    def get_interactive_elements_for_module(
        self,
        db: Session,
        module_id: UUID
    ) -> List[InteractiveElement]:
        """
        Get all interactive elements for a specific module
        """
        return (
            db.query(InteractiveElement)
            .filter(InteractiveElement.module_id == module_id)
            .all()
        )