"""Prediction Service for RootCause.

This module manages deep learning model loading and inference for crop disease detection.
It is architected to support separate offline TensorFlow/Keras or TFLite models for
each crop type ('rice' and 'wheat').

IMPORTANT DEVELOPMENT POLICY:
- If no trained model weights exist on disk, this service returns a clearly labelled,
  informative 'model_not_available' response.
- Fake or simulated ML predictions are never returned as real results.
- Model loading is lazy and modular, ready for dropping in trained .keras or .h5 weights.
"""

import io
from pathlib import Path
from typing import Any, BinaryIO, Dict, List, Optional, Tuple, Union

import numpy as np
from PIL import Image

# Directory where trained model weights are located
MODELS_DIR = Path(__file__).resolve().parent.parent / "models"

SUPPORTED_CROPS = ("rice", "wheat")

# Target image dimensions expected by standard lightweight backbones (e.g. MobileNetV2)
TARGET_IMAGE_SIZE = (224, 224)

# Global in-memory cache to avoid reloading models on every inference request
_MODEL_CACHE: Dict[str, Any] = {}

# Default disease class labels mapped to model output indices
CROP_CLASS_LABELS: Dict[str, List[str]] = {
    "rice": [
        "Bacterial Leaf Blight",
        "Brown Spot",
        "Healthy Rice Leaf",
        "Leaf Blast",
        "Leaf scald",
        "Sheath Blight",
    ],
    "wheat": [
        "Black Rust",
        "Brown Rust",
        "Yellow Rust",
        "Powdery Mildew",
    ],
}


def get_class_labels_for_crop(crop_type: str) -> List[str]:
    """Retrieve class labels for a crop, checking for a saved JSON mapping first."""
    normalized_crop = crop_type.strip().lower()
    candidate_classes_file = MODELS_DIR / f"{normalized_crop}_mobilenet_candidate_classes.json"
    if candidate_classes_file.exists():
        try:
            import json
            with open(candidate_classes_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, list) and len(data) > 0:
                    return data
        except Exception:
            pass

    classes_file = MODELS_DIR / f"{normalized_crop}_classes.json"
    if classes_file.exists():
        try:
            import json
            with open(classes_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, list) and len(data) > 0:
                    return data
        except Exception:
            pass

    mapping_file = MODELS_DIR / f"{normalized_crop}_class_mapping.json"
    if mapping_file.exists():
        try:
            import json
            with open(mapping_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, dict) and len(data) > 0:
                    return [data[str(i)] for i in range(len(data))]
        except Exception:
            pass

    return CROP_CLASS_LABELS.get(normalized_crop, [])


def get_model_path(crop_type: str) -> Optional[Path]:
    """Find the path to the trained model file for a given crop if it exists.

    Checks for supported formats: .keras, .h5, and .tflite.

    Args:
        crop_type: 'rice' or 'wheat'.

    Returns:
        Path if a model file is found, else None.
    """
    normalized_crop = crop_type.strip().lower()
    # The transfer-learning candidate was promoted after beating the fixed
    # held-out baseline without collapsing predictions. Keep the original
    # rice_model.keras as a fallback and preserve all wheat lookup behavior.
    candidate_names = (
        ["rice_mobilenet_candidate.keras", "rice_model.keras", "rice_model.h5", "rice_model.tflite"]
        if normalized_crop == "rice" else
        ["wheat_mobilenet_candidate.keras", "wheat_model.keras", "wheat_model.h5", "wheat_model.tflite"]
        if normalized_crop == "wheat" else
        [f"{normalized_crop}_model{ext}" for ext in (".keras", ".h5", ".tflite")]
    )

    for name in candidate_names:
        model_path = MODELS_DIR / name
        if model_path.exists() and model_path.is_file():
            return model_path

    return None


def is_model_available(crop_type: str) -> bool:
    """Check if a trained model file exists on disk for the specified crop."""
    return get_model_path(crop_type) is not None


def get_model_status() -> Dict[str, Any]:
    """Inspect model availability and paths for all supported crops.

    Returns:
        dict: Status report showing whether weights are present for each crop.
    """
    status: Dict[str, Any] = {
        "models_directory": str(MODELS_DIR),
        "crops": {},
    }

    for crop in SUPPORTED_CROPS:
        model_path = get_model_path(crop)
        status["crops"][crop] = {
            "available": model_path is not None,
            "path": str(model_path) if model_path else None,
            "expected_file": model_path.name if model_path else (
                "rice_mobilenet_candidate.keras" if crop == "rice" else f"{crop}_model.keras"
            ),
            "framework": "TensorFlow / Keras",
            "target_input_shape": [1, TARGET_IMAGE_SIZE[0], TARGET_IMAGE_SIZE[1], 3],
        }

    return status


def preprocess_image(
    image_input: Union[bytes, BinaryIO, str, Path, Image.Image],
    target_size: Tuple[int, int] = TARGET_IMAGE_SIZE,
) -> np.ndarray:
    """Preprocess an input image for TensorFlow model inference.

    Pipeline:
    1. Load image and convert to 3-channel RGB.
    2. Resize to target dimensions (224x224) using bilinear interpolation.
    3. Convert to float32 NumPy array normalized to [0.0, 1.0].
    4. Add batch dimension -> shape (1, 224, 224, 3).

    Args:
        image_input: Raw bytes, file buffer, path, or PIL Image.
        target_size: Desired (width, height) tuple.

    Returns:
        np.ndarray: Batch array of shape (1, height, width, 3), dtype float32.
    """
    if isinstance(image_input, Image.Image):
        pil_img = image_input.convert("RGB")
    elif isinstance(image_input, bytes):
        pil_img = Image.open(io.BytesIO(image_input)).convert("RGB")
    elif isinstance(image_input, (str, Path)):
        pil_img = Image.open(image_input).convert("RGB")
    elif hasattr(image_input, "read"):
        if hasattr(image_input, "seek"):
            image_input.seek(0)
        data = image_input.read()
        pil_img = Image.open(io.BytesIO(data)).convert("RGB")
    else:
        raise ValueError(f"Unsupported image input type: {type(image_input)}")

    # Resize to target CNN input size
    resized_img = pil_img.resize(target_size, Image.Resampling.BILINEAR)

    # Convert to normalized float32 array
    img_array = np.array(resized_img, dtype=np.float32) / 255.0

    # Expand dims to create batch: (1, 224, 224, 3)
    batch_array = np.expand_dims(img_array, axis=0)
    return batch_array


def load_model_for_crop(crop_type: str) -> Optional[Any]:
    """Load and cache the TensorFlow model for the given crop if weights exist.

    Uses lazy importing of TensorFlow to avoid unnecessary overhead during startup.

    Args:
        crop_type: 'rice' or 'wheat'.

    Returns:
        Loaded model object, or None if no model file exists.
    """
    normalized_crop = crop_type.strip().lower()
    if normalized_crop in _MODEL_CACHE:
        return _MODEL_CACHE[normalized_crop]

    model_path = get_model_path(normalized_crop)
    if model_path is None:
        return None

    # Model file exists: load using TensorFlow
    try:
        import tensorflow as tf

        if model_path.suffix == ".tflite":
            # TFLite interpreter for low-end mobile deployment
            interpreter = tf.lite.Interpreter(model_path=str(model_path))
            interpreter.allocate_tensors()
            _MODEL_CACHE[normalized_crop] = interpreter
            return interpreter

        model = tf.keras.models.load_model(str(model_path))
        _MODEL_CACHE[normalized_crop] = model
        return model
    except Exception as exc:
        # If loading fails (corrupted model file, etc.), log and return None
        return None


def predict_crop_disease(
    crop_type: str,
    image_input: Union[bytes, BinaryIO, str, Path, Image.Image],
) -> Dict[str, Any]:
    """Perform crop disease prediction on a pre-validated image.

    If trained model weights are not present in app/models/, this function
    returns an informative 'model_not_available' response.

    Args:
        crop_type: 'rice' or 'wheat'.
        image_input: Image data.

    Returns:
        dict: Prediction result or model_not_available response.
    """
    normalized_crop = crop_type.strip().lower() if crop_type else ""

    if normalized_crop not in SUPPORTED_CROPS:
        return {
            "status": "error",
            "crop_type": crop_type,
            "prediction": None,
            "confidence": None,
            "message": f"Unsupported crop type '{crop_type}'. Supported crops are: {list(SUPPORTED_CROPS)}.",
        }

    # Check if a trained model is deployed
    model = load_model_for_crop(normalized_crop)

    if model is None:
        # NO MODEL WEIGHTS FOUND: Return informative placeholder response
        expected_file = (
            "rice_mobilenet_candidate.keras" if normalized_crop == "rice"
            else f"{normalized_crop}_model.keras"
        )
        return {
            "status": "model_not_available",
            "crop_type": normalized_crop,
            "prediction": None,
            "confidence": None,
            "is_model_loaded": False,
            "message": (
                f"Trained TensorFlow model for '{normalized_crop}' is not currently available on the server. "
                f"Please place the trained weights file ('{expected_file}') in the 'app/models/' directory."
            ),
            "model_info": {
                "crop": normalized_crop,
                "expected_file": expected_file,
                "model_directory": str(MODELS_DIR),
                "framework": "TensorFlow / Keras",
                "target_input_shape": [1, TARGET_IMAGE_SIZE[0], TARGET_IMAGE_SIZE[1], 3],
                "status": "pending_deployment",
            },
        }

    # REAL MODEL IS DEPLOYED: Perform actual inference
    try:
        batch_data = preprocess_image(image_input)

        # Check if TFLite or Keras model
        if hasattr(model, "get_input_details"):
            # TFLite inference
            input_details = model.get_input_details()
            output_details = model.get_output_details()
            model.set_tensor(input_details[0]["index"], batch_data)
            model.invoke()
            raw_predictions = model.get_tensor(output_details[0]["index"])[0]
        else:
            # Keras model inference
            raw_predictions = model.predict(batch_data, verbose=0)[0]

        predicted_idx = int(np.argmax(raw_predictions))
        confidence = float(raw_predictions[predicted_idx])

        # Map to class label if available
        class_labels = get_class_labels_for_crop(normalized_crop)
        predicted_disease = (
            class_labels[predicted_idx]
            if predicted_idx < len(class_labels)
            else f"class_{predicted_idx}"
        )

        return {
            "status": "success",
            "crop_type": normalized_crop,
            "prediction": predicted_disease,
            "confidence": round(confidence, 4),
            "is_model_loaded": True,
            "raw_scores": [round(float(s), 4) for s in raw_predictions],
            "message": f"Disease predicted: {predicted_disease} ({confidence * 100:.1f}% confidence)",
        }
    except Exception as exc:
        return {
            "status": "error",
            "crop_type": normalized_crop,
            "prediction": None,
            "confidence": None,
            "message": f"Inference execution failed: {str(exc)}",
        }
