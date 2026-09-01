"""Version-one API router."""

from fastapi import APIRouter

from app.api.v1 import analyses, artifacts, health, reports, uploads, validation

router = APIRouter()
router.include_router(health.router)
router.include_router(uploads.router)
router.include_router(validation.router)
router.include_router(analyses.router)
router.include_router(artifacts.router)
router.include_router(reports.router)
