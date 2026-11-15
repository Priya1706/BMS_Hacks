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
import logging
from pathlib import Path
from typing import Any, BinaryIO, Dict, List, Optional, Tuple, Union

import numpy as np
from PIL import Image

# Directory where trained model weights are located
MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
logger = logging.getLogger(__name__)

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

# Explicit selected checkpoints. Experimental candidates at the model root are
# intentionally not used as fallbacks: doing so can silently pair a checkpoint
# with an incompatible class map.
SELECTED_MODEL_RELATIVE_PATHS: Dict[str, Path] = {
    "rice": Path("rice_accuracy_experiments/label_smoothing_ft30/rice_5class_label_smoothing_ft30.keras"),
    "wheat": Path("wheat_accuracy_experiments/finetune_last20_smoothing/wheat_finetune_last20_smoothing.keras"),
}
_MODEL_LOAD_ERRORS: Dict[str, str] = {}


def get_class_labels_for_crop(crop_type: str) -> List[str]:
    """Return labels for the selected checkpoint, in its trained output order."""
    normalized_crop = str(crop_type or "").strip().lower()
    return list(CROP_CLASS_LABELS.get(normalized_crop, []))


def get_model_path(crop_type: str) -> Optional[Path]:
    """Return the explicitly selected production checkpoint for a crop."""
    normalized_crop = str(crop_type or "").strip().lower()
    relative_path = SELECTED_MODEL_RELATIVE_PATHS.get(normalized_crop)
    model_path = MODELS_DIR / relative_path if relative_path is not None else None
    return model_path if model_path is not None and model_path.is_file() else None


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
            "expected_file": str(SELECTED_MODEL_RELATIVE_PATHS[crop]),
            "framework": "TensorFlow / Keras",
            "target_input_shape": [1, TARGET_IMAGE_SIZE[0], TARGET_IMAGE_SIZE[1], 3],
            "output_classes": len(CROP_CLASS_LABELS[crop]),
            "class_labels": get_class_labels_for_crop(crop),
            "loaded": crop in _MODEL_CACHE,
            "load_error": _MODEL_LOAD_ERRORS.get(crop),
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

        expected_classes = len(get_class_labels_for_crop(normalized_crop))
        if model_path.suffix == ".tflite":
            # TFLite interpreter for low-end mobile deployment
            interpreter = tf.lite.Interpreter(model_path=str(model_path))
            interpreter.allocate_tensors()
            input_shape = tuple(interpreter.get_input_details()[0]["shape"])
            output_shape = tuple(interpreter.get_output_details()[0]["shape"])
            if input_shape != (1, *TARGET_IMAGE_SIZE, 3) or output_shape[-1] != expected_classes:
                raise ValueError(
                    f"Selected {normalized_crop} TFLite model has incompatible shapes "
                    f"{input_shape}->{output_shape}; expected (1,224,224,3)->{expected_classes} classes"
                )
            _MODEL_CACHE[normalized_crop] = interpreter
            _MODEL_LOAD_ERRORS.pop(normalized_crop, None)
            return interpreter

        model = tf.keras.models.load_model(str(model_path))
        input_shape = tuple(model.input_shape)
        output_shape = tuple(model.output_shape)
        if input_shape != (None, *TARGET_IMAGE_SIZE, 3) or output_shape[-1] != expected_classes:
            raise ValueError(
                f"Selected {normalized_crop} model has incompatible shapes "
                f"{input_shape}->{output_shape}; expected (None,224,224,3)->{expected_classes} classes"
            )
        _MODEL_CACHE[normalized_crop] = model
        _MODEL_LOAD_ERRORS.pop(normalized_crop, None)
        return model
    except Exception as exc:
        _MODEL_LOAD_ERRORS[normalized_crop] = str(exc)
        logger.exception("Failed to load selected %s model at %s", normalized_crop, model_path)
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
        expected_file = str(SELECTED_MODEL_RELATIVE_PATHS[normalized_crop])
        load_error = _MODEL_LOAD_ERRORS.get(normalized_crop)
        return {
            "status": "model_not_available",
            "crop_type": normalized_crop,
            "prediction": None,
            "confidence": None,
            "is_model_loaded": False,
            "message": (
                f"Selected model for '{normalized_crop}' is unavailable or could not be loaded. "
                f"Expected checkpoint: 'app/models/{expected_file}'."
                + (f" Load error: {load_error}" if load_error else "")
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

        raw_predictions = np.asarray(raw_predictions, dtype=np.float32).reshape(-1)
        class_labels = get_class_labels_for_crop(normalized_crop)
        if raw_predictions.size != len(class_labels) or not np.isfinite(raw_predictions).all():
            raise ValueError(
                f"Model returned {raw_predictions.size} scores for {len(class_labels)} configured classes"
            )
        predicted_idx = int(np.argmax(raw_predictions))
        confidence = float(raw_predictions[predicted_idx])

        # Map to class label if available
        predicted_disease = class_labels[predicted_idx]

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
