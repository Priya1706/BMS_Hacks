import os
from pathlib import Path
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from app.api.routes import router as api_router

app = FastAPI(
    title="RootCause API",
    description="Offline Crop Disease Detection and Advisory System for Low-End Android Phones",
    version="1.0.0"
)

# Set ROOTCAUSE_CORS_ORIGINS to a comma-separated list for a separately hosted frontend.
cors_origins = [origin.strip() for origin in os.getenv(
    "ROOTCAUSE_CORS_ORIGINS",
    "http://localhost:8000,http://127.0.0.1:8000,http://localhost:5500,http://127.0.0.1:5500"
).split(",") if origin.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routes
app.include_router(api_router)

# Mount static frontend directories if present
BASE_DIR = Path(__file__).resolve().parent.parent

if (BASE_DIR / "css").is_dir():
    app.mount("/css", StaticFiles(directory=str(BASE_DIR / "css")), name="css")
if (BASE_DIR / "js").is_dir():
    app.mount("/js", StaticFiles(directory=str(BASE_DIR / "js")), name="js")
if (BASE_DIR / "assets").is_dir():
    app.mount("/assets", StaticFiles(directory=str(BASE_DIR / "assets")), name="assets")


@app.get("/")
def home():
    return {
        "message": "Welcome to RootCause!",
        "status": "Backend is running"
    }


@app.get("/health")
def health_check():
    return {
        "status": "healthy"
    }


@app.get("/index.html")
@app.get("/app")
def serve_app():
    index_file = BASE_DIR / "index.html"
    if index_file.is_file():
        return FileResponse(str(index_file))
    return {"message": "RootCause frontend not found"}


@app.get("/manifest.json")
def serve_manifest():
    manifest_file = BASE_DIR / "manifest.json"
    if manifest_file.is_file():
        return FileResponse(str(manifest_file), media_type="application/manifest+json")
    return {"error": "manifest.json not found"}


@app.get("/sw.js")
def serve_sw():
    sw_file = BASE_DIR / "sw.js"
    if sw_file.is_file():
        return FileResponse(str(sw_file), media_type="application/javascript")
    return {"error": "sw.js not found"}
