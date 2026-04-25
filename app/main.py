"""FastAPI entrypoint for future API expansion."""

from fastapi import FastAPI

from app.api.endpoints import router as api_router

app = FastAPI(title="FindMyBall AIM240 MVP", version="0.1.0")
app.include_router(api_router)

