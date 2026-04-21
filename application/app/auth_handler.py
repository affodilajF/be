import time
from typing import Dict
from jose import jwt
from passlib.context import CryptContext

# Configuration
JWT_SECRET = "8f03125e98f0322521743d839da46505" # should be from env
JWT_ALGORITHM = "HS256"

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# Expiry times
ACCESS_TOKEN_EXPIRE_MINUTES = 30
REFRESH_TOKEN_EXPIRE_DAYS = 7

def signJWT(user_id: str, name: str) -> dict:
    # Access Token
    access_payload = {
        "user_id": user_id,
        "name": name,
        "expires": time.time() + (ACCESS_TOKEN_EXPIRE_MINUTES * 60)
        # "expires": time.time() + 5
    }
    access_token = jwt.encode(access_payload, JWT_SECRET, algorithm=JWT_ALGORITHM)
    
    # Refresh Token
    refresh_payload = {
        "user_id": user_id,
        "expires": time.time() + (REFRESH_TOKEN_EXPIRE_DAYS * 24 * 3600)
    }
    refresh_token = jwt.encode(refresh_payload, JWT_SECRET, algorithm=JWT_ALGORITHM)

    return {
        "success": True,
        "message": "Login success",
        "data": {
            "access_token": access_token,
            "refresh_token": refresh_token,
            "name": name
        }
    }

def decodeJWT(token: str) -> dict:
    try:
        decoded_token = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
        if decoded_token.get("expires") >= time.time():
            return decoded_token
        return {} # Token expired
    except Exception:
        return {} # Invalid token

def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password):
    return pwd_context.hash(password)
