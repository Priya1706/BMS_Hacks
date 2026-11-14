from fastapi import FastAPI
from app.api.routes import router as api_router

app = FastAPI(
    title="RootCause API",
    description="Offline Crop Disease Detection and Advisory System for Low-End Android Phones",
    version="1.0.0"
)

# Include API routes
app.include_router(api_router)


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
