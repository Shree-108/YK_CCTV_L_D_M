"""CCTV Litter Monitor - FastAPI backend.

Run from the project root:
    uvicorn backend.main:app --host 0.0.0.0 --port 8000
"""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.api.routes import router
from backend.api.routes_lab import router as lab_router
from backend.core.config import AppConfig
from backend.services.state import AppState

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(name)s: %(message)s")


def create_app(cfg: Optional[AppConfig] = None, state: Optional[AppState] = None) -> FastAPI:
    cfg = cfg or AppConfig()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        st = state or AppState(cfg)
        app.state.app_state = st
        st.db.log_event("info", "system", "backend started",
                        data={"model": st.engine.version if st.engine else None})
        st.sync_cameras()
        yield
        st.shutdown()
        st.db.log_event("info", "system", "backend stopped")

    app = FastAPI(title="CCTV Litter Monitor API", version="3.0.0", lifespan=lifespan)
    app.add_middleware(CORSMiddleware, allow_origins=cfg.cors_origins, allow_methods=["*"],
                       allow_headers=["Authorization", "Content-Type", "X-Admin-Token"])
    app.include_router(lab_router)
    app.include_router(router)
    return app


app = create_app()
