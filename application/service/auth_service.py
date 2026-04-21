from datetime import datetime, timezone, timedelta
from sqlalchemy.orm import Session
from app.auth_handler import signJWT, get_password_hash, verify_password, decodeJWT
from database.db_models import User
from database.db_schemas import UserSignupSchema, UserLoginSchema, RefreshSchema
from fastapi.responses import JSONResponse

WIB = timezone(timedelta(hours=7))

def signup_user(user: UserSignupSchema, db: Session):
    # Check if user already exists
    existing_user = db.query(User).filter(User.email == user.email).first()
    if existing_user:
        return {
            "success": False,
            "message": "Email already registered"
        }
    
    # Create new user
    new_user = User(
        username=user.username,
        email=user.email,
        password=get_password_hash(user.password),
        created_at=datetime.now(WIB).replace(microsecond=0)
    )
    db.add(new_user)
    db.commit()
    db.refresh(new_user)
    
    return {
        "success": True,
        "message": "User created successfully",
        "data": {
            "user_id": new_user.id,
            "username": new_user.username,
            "email": new_user.email
        }
    }

def login_user(user: UserLoginSchema, db: Session):
    db_user = db.query(User).filter(User.email == user.email).first()
    if db_user and verify_password(user.password, db_user.password):
        return signJWT(str(db_user.id), db_user.username)
    
    return JSONResponse(
        status_code=401,
        content={
            "error": {
                "code": 401,
                "message": "Wrong login details!",
                "data": user.email
            }
        }
    )

def refresh_user_token(payload: RefreshSchema, db: Session):
    token_data = decodeJWT(payload.refresh_token)
    if token_data:
        user_id = token_data.get("user_id")
        db_user = db.query(User).filter(User.id == int(user_id)).first()
        if db_user:
            return signJWT(str(db_user.id), db_user.username)

    return JSONResponse(
        status_code=401,
        content={
            "error": {
                "code": 401,
                "message": "Invalid or expired refresh token",
                "data": payload.refresh_token
            }
        }
    )
