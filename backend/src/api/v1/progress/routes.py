from fastapi import APIRouter

router = APIRouter()

@router.get("/progress")
def get_progress():
    return {"message": "Progress endpoint"}