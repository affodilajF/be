from fastapi import Body
from pydantic import BaseModel

class UserSignupSchema(BaseModel):
    username: str = Body(...)
    email: str = Body(...)
    password: str = Body(...)

class UserLoginSchema(BaseModel):
    email: str = Body(...)
    password: str = Body(...)

class RefreshSchema(BaseModel):
    refresh_token: str = Body(...)

class DetectionParameter(BaseModel):
    top_roi: int
    bottom_roi: int
    frame_interval: int