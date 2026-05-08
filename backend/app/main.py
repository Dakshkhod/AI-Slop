"""FastAPI application entry point for TruthLens."""

from __future__ import annotations

import logging
import sys

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from . import __version__
from .config import get_settings
from .routes import analyze as analyze_routes
from .routes import extension as extension_routes
from .routes import health as health_routes


def _configure_logging(level: str) -> None:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        logging.Formatter(
            "%(asctime)s %(levelname)-7s %(name)s — %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )
    root = logging.getLogger()
    root.setLevel(level)
    root.handlers = [handler]


def create_app() -> FastAPI:
    settings = get_settings()
    _configure_logging(settings.log_level)

    settings.data_dir.mkdir(parents=True, exist_ok=True)
    settings.cache_dir.mkdir(parents=True, exist_ok=True)

    app = FastAPI(
        title=f"{settings.service_name} API",
        version=__version__,
        description=(
            "Multi-modal AI slop detection with explainable, physics-based "
            "signals. POST /api/analyze with a file."
        ),
    )

    origins = list(settings.cors_origins)
    if settings.frontend_url and settings.frontend_url not in origins:
        origins.append(settings.frontend_url)

    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(health_routes.router)
    app.include_router(analyze_routes.router)
    app.include_router(extension_routes.router)

    @app.get("/")
    def root():
        return JSONResponse(
            {
                "name": settings.service_name,
                "version": __version__,
                "docs": "/docs",
                "endpoints": ["/healthz", "/api/info", "/api/analyze", "/api/analyze-url", "/api/check_hash", "/api/analyze_url", "/api/report_wrong"],
            }
        )

    return app


app = create_app()
