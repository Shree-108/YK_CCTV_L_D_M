"""CCTV Litter Monitor - FastAPI backend.

Run from the project root (local only - there is no login yet, see backend/api/local_only.py):
    uvicorn backend.main:app --host 127.0.0.1 --port 8000
"""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.api.local_only import LocalOnlyMiddleware
from backend.api.routes_auth import router as auth_router
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

    docs = cfg.enable_api_docs   # off by default: LM_ENABLE_API_DOCS=true turns /docs, /redoc, /openapi.json on
    app = FastAPI(title="CCTV Litter Monitor API", version="3.0.0", lifespan=lifespan,
                  docs_url="/docs" if docs else None, redoc_url="/redoc" if docs else None,
                  openapi_url="/openapi.json" if docs else None)
    app.add_middleware(CORSMiddleware, allow_origins=cfg.cors_origins, allow_methods=["*"],
                       allow_headers=["Authorization", "Content-Type", "X-Admin-Token"])
    # Added last = outermost: no login exists yet, so only this computer may call the API (any --host).
    app.add_middleware(LocalOnlyMiddleware, extra_hosts=cfg.local_extra_hosts)
    # Login/sessions (Step 2): the auth database is only opened/created when LM_AUTH_MODE=on. No route is
    # protected by it yet (Step 4); require_admin and ADMIN_API_TOKEN work exactly as before.
    app.state.auth_cfg = cfg
    app.state.auth = None
    if cfg.auth_mode == "on":
        from backend.auth import build_service
        app.state.auth = build_service(cfg)
    app.include_router(auth_router)
    app.include_router(lab_router)
    app.include_router(router)
    return app


app = create_app()
