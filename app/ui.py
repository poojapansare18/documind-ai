from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

router = APIRouter()
ASSET_DIR = Path(__file__).parent / "static"
ALLOWED_ASSETS = {"app.css", "app.js"}


def render_home() -> str:
    return (ASSET_DIR / "index.html").read_text(encoding="utf-8")


@router.get("/static/{asset_name}")
def serve_ui_asset(asset_name: str) -> FileResponse:
    if asset_name not in ALLOWED_ASSETS:
        raise HTTPException(status_code=404, detail="Static asset not found.")
    asset_path = ASSET_DIR / asset_name
    if not asset_path.is_file():
        raise HTTPException(status_code=404, detail="Static asset not found.")
    return FileResponse(asset_path)
