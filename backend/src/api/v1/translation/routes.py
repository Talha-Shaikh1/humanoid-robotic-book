from fastapi import APIRouter

router = APIRouter()

@router.get("/translation")
def get_translation():
    return {"message": "Translation endpoint"}