"""CrowdSense AI HTTP API (FastAPI).

Serves the React SPA and three analysis endpoints backed by the project's
existing Python pipeline. Heavy work runs in FastAPI's threadpool because the
endpoints are plain ``def`` functions.
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

# Make the project root importable when uvicorn is launched from elsewhere.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend import service  # noqa: E402
from backend.schemas import (  # noqa: E402
    FrameDetectionResponse,
    HealthResponse,
    ImageDetectionResponse,
    ModelInfo,
    VideoDetectionResponse,
)

logger = logging.getLogger("crowdsense.api")

API_VERSION = "1.0.0"
STATIC_DIR = Path(__file__).resolve().parent / "static"

app = FastAPI(
    title="CrowdSense AI API",
    description=(
        "Crowd photo/video analysis: YOLO person detection, ByteTrack tracking, "
        "window features, and optional anomaly scoring from a trained bundle."
    ),
    version=API_VERSION,
)

_origins = os.getenv(
    "CROWDSENSE_CORS_ORIGINS",
    "http://localhost:5173,http://127.0.0.1:5173",
).split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[origin.strip() for origin in _origins if origin.strip()],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(service.InvalidUploadError)
async def invalid_upload_handler(_request, exc: service.InvalidUploadError):
    return JSONResponse(status_code=400, content={"detail": str(exc)})


@app.exception_handler(Exception)
async def unhandled_exception_handler(_request, exc: Exception):
    logger.exception("Unhandled API error")
    return JSONResponse(status_code=500, content={"detail": "Internal server error"})


@app.get("/api/health", response_model=HealthResponse)
def health() -> dict:
    return {"status": "ok", "service": "crowdsense-ai", "version": API_VERSION}


@app.get("/api/model", response_model=ModelInfo)
def model_status() -> dict:
    return service.model_info()


@app.post("/api/detect-image", response_model=ImageDetectionResponse)
def detect_image(
    file: UploadFile = File(..., description="Crowd photo (JPG/PNG/TIF/BMP)"),
    conf: float = Form(0.05, ge=0.01, le=0.9),
    imgsz: int = Form(640),
) -> dict:
    if imgsz not in {320, 640, 1280}:
        raise HTTPException(status_code=422, detail="imgsz must be one of 320, 640, 1280")
    data = file.file.read()
    return service.detect_image(data, conf=conf, imgsz=imgsz)


@app.post("/api/detect-frame", response_model=FrameDetectionResponse)
def detect_frame(
    frame: UploadFile = File(..., description="Single webcam frame (PNG/JPEG)"),
    conf: float = Form(0.05, ge=0.01, le=0.9),
    imgsz: int = Form(640),
) -> dict:
    if imgsz not in {320, 640, 1280}:
        raise HTTPException(status_code=422, detail="imgsz must be one of 320, 640, 1280")
    data = frame.file.read()
    return service.detect_frame(data, conf=conf, imgsz=imgsz)


@app.post("/api/detect-video", response_model=VideoDetectionResponse)
def detect_video(
    file: UploadFile = File(..., description="Crowd video (MP4/AVI/MOV/MKV/WEBM)"),
    max_frames: int = Form(150, ge=30, le=600),
    stride: int = Form(1, ge=1, le=5),
    conf: float = Form(0.05, ge=0.01, le=0.9),
    imgsz: int = Form(640),
    window: int = Form(30, ge=10, le=60),
    window_stride: int = Form(5, ge=1, le=30),
) -> dict:
    if imgsz not in {320, 640, 1280}:
        raise HTTPException(status_code=422, detail="imgsz must be one of 320, 640, 1280")
    if window_stride > window:
        raise HTTPException(status_code=422, detail="window_stride must be <= window")
    data = file.file.read()
    return service.analyze_video(
        data,
        file.filename or "upload.mp4",
        max_frames=max_frames,
        frame_stride=stride,
        conf=conf,
        imgsz=imgsz,
        window=window,
        window_stride=window_stride,
    )


if STATIC_DIR.is_dir():
    # Hashed assets are immutable; the three SPA routes return index.html so a
    # browser refresh on /video or /live-camera works. Unknown paths stay 404.
    assets_dir = STATIC_DIR / "assets"
    if assets_dir.is_dir():
        app.mount("/assets", StaticFiles(directory=assets_dir), name="assets")

    def _spa_index() -> FileResponse:
        return FileResponse(STATIC_DIR / "index.html")

    for route in ("/", "/video", "/live-camera"):
        app.add_api_route(route, _spa_index, include_in_schema=False)


def main() -> None:
    import uvicorn

    uvicorn.run(
        "backend.app:app",
        host=os.getenv("CROWDSENSE_HOST", "127.0.0.1"),
        port=int(os.getenv("CROWDSENSE_PORT", "8000")),
        reload=os.getenv("CROWDSENSE_RELOAD", "0") == "1",
    )


if __name__ == "__main__":
    main()