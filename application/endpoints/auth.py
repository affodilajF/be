from fastapi import APIRouter, Body, Depends
from sqlalchemy.orm import Session
from database.database import get_db
from database.db_schemas import UserSignupSchema, UserLoginSchema, RefreshSchema
from service.auth_service import signup_user, login_user, refresh_user_token

router = APIRouter()

@router.post("/api/signup")
async def user_signup(user: UserSignupSchema = Body(...), db: Session = Depends(get_db)):
    return signup_user(user, db)

@router.post("/api/login")
async def user_login(user: UserLoginSchema = Body(...), db: Session = Depends(get_db)):
    return login_user(user, db)

@router.post("/api/refresh")
async def user_refresh(payload: RefreshSchema = Body(...), db: Session = Depends(get_db)):
    return refresh_user_token(payload, db)
