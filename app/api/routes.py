"""API Routes for RootCause.

This module exposes REST endpoints for:
1. Crop disease diagnosis and quality verification: POST /predict
2. Standalone image quality assessment: POST /quality-check
3. Model availability status: GET /models/status
4. Crop disease advisory information: GET /advisory/{crop_type}
5. Disease catalog listing: GET /diseases
"""

from typing import Optional

from fastapi import APIRouter, File, Form, UploadFile, status
from fastapi.responses import JSONResponse

from app.services.advisory import get_disease_advisory, list_supported_diseases
from app.services.image_quality import assess_image_quality
from app.services.predictor import get_model_status, predict_crop_disease

router = APIRouter(tags=["Crop Disease Detection & Advisory"])

SUPPORTED_CROPS = ("rice", "wheat")


def _read_upload_file_bytes(upload_file: UploadFile) -> bytes:
    """Read bytes from an UploadFile safely across environments."""
    if hasattr(upload_file, "file") and hasattr(upload_file.file, "read"):
        if hasattr(upload_file.file, "seek"):
            upload_file.file.seek(0)
        data = upload_file.file.read()
        return data
    if hasattr(upload_file, "read"):
        return upload_file.read()
    return b""


@router.post(
    "/predict",
    summary="Predict crop disease from leaf image",
    description=(
        "Upload a leaf image and specify the crop type ('rice' or 'wheat'). "
        "The system first performs image-quality verification (resolution, blur, lighting). "
        "If the image passes quality checks, it proceeds to disease prediction and advisory."
    ),
)
def predict(
    crop_type: str = Form(
        ...,
        description="Target crop type to diagnose ('rice' or 'wheat')",
        examples=["rice", "wheat"],
    ),
    image: UploadFile = File(
        ...,
        description="Photo of the crop leaf (JPEG, PNG, etc.)",
    ),
):
    """Diagnose crop disease from an uploaded leaf image."""
    # 1. Validate crop type parameter
    normalized_crop = crop_type.strip().lower() if crop_type else ""
    if normalized_crop not in SUPPORTED_CROPS:
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={
                "status": "error",
                "error_code": "INVALID_CROP_TYPE",
                "message": (
                    f"Unsupported crop type '{crop_type}'. "
                    f"Supported crop types are: {list(SUPPORTED_CROPS)}."
                ),
            },
        )

    # 2. Safely read uploaded file contents
    try:
        image_bytes = _read_upload_file_bytes(image)
    except Exception as exc:
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={
                "status": "error",
                "error_code": "FILE_READ_ERROR",
                "message": f"Failed to read uploaded image: {str(exc)}",
            },
        )

    if not image_bytes or len(image_bytes) == 0:
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={
                "status": "error",
                "error_code": "EMPTY_IMAGE_FILE",
                "message": "Uploaded image file is empty (0 bytes).",
            },
        )

    # 3. Perform image quality assessment before inference
    quality_result = assess_image_quality(image_bytes)

    # If the image is unsuitable, return explanation and retake suggestions
    if not quality_result["passed"]:
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={
                "status": "quality_check_failed",
                "crop_type": normalized_crop,
                "filename": image.filename,
                "message": (
                    "Image failed quality assessment. "
                    "The image is unsuitable for reliable disease diagnosis."
                ),
                "issues": quality_result["issues"],
                "suggestions": quality_result["suggestions"],
                "quality_assessment": quality_result,
            },
        )

    # 4. Image passed quality checks: proceed to prediction
    prediction_result = predict_crop_disease(normalized_crop, image_bytes)

    # 5. Fetch structured advisory data
    predicted_disease = prediction_result.get("prediction")
    advisory_result = get_disease_advisory(normalized_crop, predicted_disease)

    # Formulate response
    return JSONResponse(
        status_code=status.HTTP_200_OK,
        content={
            "status": "success",
            "crop_type": normalized_crop,
            "filename": image.filename,
            "quality_assessment": {
                "passed": quality_result["passed"],
                "message": quality_result["message"],
                "metrics": quality_result["metrics"],
            },
            "prediction": prediction_result,
            "advisory": advisory_result,
        },
    )


@router.post(
    "/quality-check",
    summary="Evaluate image quality without prediction",
    description="Check resolution, blur, and lighting of a crop leaf photo before diagnosis.",
)
def check_quality(
    image: UploadFile = File(..., description="Crop leaf photo to assess"),
):
    """Perform standalone image quality validation."""
    image_bytes = _read_upload_file_bytes(image)
    if not image_bytes:
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={"error": "Uploaded image file is empty."},
        )

    result = assess_image_quality(image_bytes)
    return JSONResponse(
        status_code=status.HTTP_200_OK if result["passed"] else status.HTTP_400_BAD_REQUEST,
        content=result,
    )


@router.get(
    "/models/status",
    summary="Check deployed model status",
    description="Inspect whether trained TensorFlow/TFLite models are deployed on the server.",
)
def model_status():
    """Return model availability and expected file paths."""
    return get_model_status()


@router.get(
    "/advisory/{crop_type}",
    summary="Get disease advisory for a crop",
    description="Retrieve structured agronomic advisory guidance for rice or wheat diseases.",
)
def get_advisory_route(crop_type: str, disease: Optional[str] = None):
    """Retrieve disease advisory guidance."""
    return get_disease_advisory(crop_type, disease)


@router.get(
    "/diseases",
    summary="List supported diseases",
    description="List all cataloged crop diseases for rice and wheat.",
)
def list_diseases_route(crop: Optional[str] = None):
    """List cataloged diseases by crop."""
    return list_supported_diseases(crop)
