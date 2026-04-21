from fastapi import FastAPI
from endpoints.api import router as api_router
from endpoints.auth import router as auth_router
from fastapi.middleware.cors import CORSMiddleware
from database.database import engine, Base
from database import db_models

# Create database tables
Base.metadata.create_all(bind=engine)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # frontend kamu
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(api_router)
app.include_router(auth_router)


