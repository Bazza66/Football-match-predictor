from __future__ import annotations

import os
import time

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import router
from app.db.base import Base
from app.db.session import engine

app = FastAPI(title="Football Predictor")

origins = [origin.strip() for origin in os.getenv("CORS_ORIGINS", "http://localhost:5173").split(",")]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(router)


@app.on_event("startup")
def on_startup() -> None:
    for _ in range(10):
        try:
            Base.metadata.create_all(bind=engine)
            return
        except Exception:
            time.sleep(2)
    Base.metadata.create_all(bind=engine)
