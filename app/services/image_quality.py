"""Image Quality Assessment Service for RootCause.

This module evaluates images captured by low-end Android smartphones to ensure
they meet the minimum visual standards needed for accurate crop disease diagnosis:
1. Valid, uncorrupted image formats (JPEG, PNG, etc.) via Pillow.
2. Adequate resolution to distinguish leaf disease lesions.
3. Sharp focus (non-blurry) using OpenCV's Laplacian variance metric.
4. Optimal lighting (not underexposed/dark or overexposed/glare) using pixel statistics.
"""

import io
from pathlib import Path
from typing import Any, BinaryIO, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
from PIL import Image, UnidentifiedImageError

# Default threshold configurations (optimized for mobile leaf photography)
DEFAULT_MIN_WIDTH = 150
DEFAULT_MIN_HEIGHT = 150
# A 100-point cutoff rejected many clean held-out crop photos even though the
# classifier resizes all inputs to 224x224. Keep the gate for near-uniform or
# truly defocused captures, rather than natural low-texture leaves.
DEFAULT_BLUR_THRESHOLD = 10.0
DEFAULT_MIN_BRIGHTNESS = 45.0   # Mean grayscale intensity (0-255); below is underexposed
DEFAULT_MAX_BRIGHTNESS = 215.0  # Mean grayscale intensity (0-255); above is overexposed


def _get_image_bytes(image_input: Union[bytes, BinaryIO, str, Path, Image.Image]) -> bytes:
    """Helper function to extract raw bytes from various input types safely.

    Supports bytes, file-like objects (e.g. io.BytesIO, UploadFile.file),
    file paths, and PIL Image instances.
    """
    if isinstance(image_input, bytes):
        return image_input

    if isinstance(image_input, (str, Path)):
        with open(image_input, "rb") as f:
            return f.read()

    if isinstance(image_input, Image.Image):
        buffer = io.BytesIO()
        image_format = image_input.format or "JPEG"
        image_input.save(buffer, format=image_format)
        return buffer.getvalue()

    # File-like object (has .read())
    if hasattr(image_input, "read"):
        # Save current position if seekable
        if hasattr(image_input, "seek"):
            current_pos = image_input.tell() if hasattr(image_input, "tell") else 0
            image_input.seek(0)
            data = image_input.read()
            # Reset position for subsequent consumers
            image_input.seek(current_pos)
            return data
        return image_input.read()

    raise ValueError(f"Unsupported image input type: {type(image_input)}")


def validate_image(image_input: Union[bytes, BinaryIO, str, Path]) -> Dict[str, Any]:
    """Validate whether an input file is an uncorrupted, decodable image.

    Uses Pillow's verify() and load() to inspect file headers and structure.

    Args:
        image_input: Raw image bytes, file-like object, or file path.

    Returns:
        dict: Structured validation result containing 'valid', 'message',
              'format', and 'size'.
    """
    try:
        raw_bytes = _get_image_bytes(image_input)
        if not raw_bytes or len(raw_bytes) == 0:
            return {
                "valid": False,
                "message": "Image file is empty (0 bytes).",
                "format": None,
                "size": None,
            }

        # Step 1: Open and verify file header integrity
        stream = io.BytesIO(raw_bytes)
        img = Image.open(stream)
        img_format = img.format
        img_size = img.size

        # verify() confirms structural integrity without decoding all pixel data
        img.verify()

        # Step 2: verify() may leave file stream exhausted; re-open and load pixels
        # to ensure the entire image stream is decodable (not truncated)
        stream.seek(0)
        img_reloaded = Image.open(stream)
        img_reloaded.load()

        return {
            "valid": True,
            "message": "Image is valid and decodable.",
            "format": img_format,
            "size": img_size,
        }

    except (UnidentifiedImageError, OSError, ValueError, SyntaxError) as exc:
        return {
            "valid": False,
            "message": f"Invalid or corrupted image: {str(exc)}",
            "format": None,
            "size": None,
        }
def check_resolution(
    width: int,
    height: int,
    min_width: int = DEFAULT_MIN_WIDTH,
    min_height: int = DEFAULT_MIN_HEIGHT,
) -> Dict[str, Any]:
    """Check if the image meets the minimum spatial resolution.

    Args:
        width: Image width in pixels.
        height: Image height in pixels.
        min_width: Minimum acceptable width.
        min_height: Minimum acceptable height.

    Returns:
        dict: Evaluation result with passed status, dimensions, and advice.
    """
    passed = (width >= min_width) and (height >= min_height)

    if passed:
        message = f"Resolution ({width}x{height}) meets the minimum required ({min_width}x{min_height})."
        suggestion = None
    else:
        message = (
            f"Image resolution ({width}x{height}) is too low. "
            f"Minimum required is {min_width}x{min_height} pixels."
        )
        suggestion = "Please hold the camera closer to the affected leaf or take a higher resolution photo."

    return {
        "passed": passed,
        "width": int(width),
        "height": int(height),
        "min_width": int(min_width),
        "min_height": int(min_height),
        "message": message,
        "suggestion": suggestion,
    }


def check_blur(
    gray_image: np.ndarray,
    blur_threshold: float = DEFAULT_BLUR_THRESHOLD,
) -> Dict[str, Any]:
    """Detect motion blur or defocus using OpenCV's Laplacian variance method.

    The Laplacian operator calculates the 2nd spatial derivative of an image.
    Sharp edges produce high variance; smooth, blurry regions produce low variance.

    Args:
        gray_image: Grayscale NumPy image array (uint8).
        blur_threshold: Threshold below which an image is considered blurry.

    Returns:
        dict: Blur assessment containing 'passed', 'is_blurry', 'score',
              'threshold', and retake suggestion.
    """
    # Ensure single channel grayscale uint8
    if len(gray_image.shape) == 3:
        gray_image = cv2.cvtColor(gray_image, cv2.COLOR_RGB2GRAY)

    # Compute variance of the Laplacian
    laplacian = cv2.Laplacian(gray_image, cv2.CV_64F)
    variance = float(laplacian.var())
    blur_score = round(variance, 2)

    is_blurry = blur_score < blur_threshold
    passed = not is_blurry

    if passed:
        message = f"Image focus is sharp and acceptable (blur score: {blur_score})."
        suggestion = None
    else:
        message = (
            f"Image is too blurry (blur score: {blur_score:.1f}, "
            f"minimum required: {blur_threshold:.1f})."
        )
        suggestion = "Please hold the phone steady, tap to focus on the leaf, and retake the photo."

    return {
        "passed": passed,
        "is_blurry": is_blurry,
        "score": blur_score,
        "threshold": float(blur_threshold),
        "message": message,
        "suggestion": suggestion,
    }


def check_brightness(
    gray_image: np.ndarray,
    min_brightness: float = DEFAULT_MIN_BRIGHTNESS,
    max_brightness: float = DEFAULT_MAX_BRIGHTNESS,
) -> Dict[str, Any]:
    """Assess lighting conditions from image pixel intensity statistics.

    Args:
        gray_image: Grayscale NumPy image array (uint8).
        min_brightness: Lower limit for mean pixel intensity (0-255).
        max_brightness: Upper limit for mean pixel intensity (0-255).

    Returns:
        dict: Lighting status ('normal', 'dark', 'overexposed') with metrics.
    """
    if len(gray_image.shape) == 3:
        gray_image = cv2.cvtColor(gray_image, cv2.COLOR_RGB2GRAY)

    mean_val = float(np.mean(gray_image))
    std_val = float(np.std(gray_image))
    mean_brightness = round(mean_val, 2)
    std_brightness = round(std_val, 2)

    if mean_val < min_brightness:
        status = "dark"
        passed = False
        message = (
            f"Image is too dark (average brightness: {mean_brightness:.1f}/255, "
            f"acceptable range: {min_brightness}-{max_brightness})."
        )
        suggestion = "Please move to an area with better lighting or turn on your phone's flashlight."
    elif mean_val > max_brightness:
        status = "overexposed"
        passed = False
        message = (
            f"Image is overexposed/too bright (average brightness: {mean_brightness:.1f}/255, "
            f"acceptable range: {min_brightness}-{max_brightness})."
        )
        suggestion = "Please shield the leaf from direct sunlight glare and disable extreme flash."
    else:
        status = "normal"
        passed = True
        message = f"Image brightness is optimal (average brightness: {mean_brightness:.1f}/255)."
        suggestion = None

    return {
        "passed": passed,
        "status": status,
        "mean": mean_brightness,
        "std": std_brightness,
        "min_threshold": float(min_brightness),
        "max_threshold": float(max_brightness),
        "message": message,
        "suggestion": suggestion,
    }


def assess_image_quality(
    image_input: Union[bytes, BinaryIO, str, Path, Image.Image],
    min_width: int = DEFAULT_MIN_WIDTH,
    min_height: int = DEFAULT_MIN_HEIGHT,
    blur_threshold: float = DEFAULT_BLUR_THRESHOLD,
    min_brightness: float = DEFAULT_MIN_BRIGHTNESS,
    max_brightness: float = DEFAULT_MAX_BRIGHTNESS,
) -> Dict[str, Any]:
    """Execute a complete image quality pipeline on a crop photo.

    Evaluates:
    1. Validity and format integrity
    2. Resolution
    3. Blur / focus
    4. Lighting / brightness

    Args:
        image_input: Image bytes, file buffer, path, or PIL Image.
        min_width: Minimum required width in pixels.
        min_height: Minimum required height in pixels.
        blur_threshold: Minimum Laplacian variance for sharpness.
        min_brightness: Minimum mean intensity for adequate exposure.
        max_brightness: Maximum mean intensity to prevent glare.

    Returns:
        dict: Complete assessment including 'passed', 'is_valid', 'issues',
              'suggestions', and nested 'metrics'.
    """
    # 1. Validation check
    val_result = validate_image(image_input)
    if not val_result["valid"]:
        return {
            "passed": False,
            "is_valid": False,
            "message": "Image quality assessment failed: file is corrupted or not a recognized image format.",
            "issues": [val_result["message"]],
            "suggestions": [
                "Please upload a standard uncorrupted photo file in JPEG or PNG format."
            ],
            "metrics": None,
        }

    # 2. Convert to RGB PIL Image and NumPy Grayscale for CV checks
    try:
        raw_bytes = _get_image_bytes(image_input)
        pil_img = Image.open(io.BytesIO(raw_bytes)).convert("RGB")
        width, height = pil_img.size

        rgb_np = np.array(pil_img)
        gray_np = cv2.cvtColor(rgb_np, cv2.COLOR_RGB2GRAY)
    except Exception as exc:
        return {
            "passed": False,
            "is_valid": False,
            "message": f"Failed to decode image data for quality analysis: {str(exc)}",
            "issues": [str(exc)],
            "suggestions": ["Please capture and upload a fresh photo."],
            "metrics": None,
        }

    # 3. Individual quality metric assessments
    resolution_res = check_resolution(width, height, min_width=min_width, min_height=min_height)
    # The classifier sees a 224x224 resized image. Scoring sharpness at the
    # original camera resolution made detailed high-resolution leaves appear
    # blurry because fine lesion edges occupy only a few source pixels.
    blur_input = cv2.resize(gray_np, (224, 224), interpolation=cv2.INTER_LINEAR)
    blur_res = check_blur(blur_input, blur_threshold=blur_threshold)
    brightness_res = check_brightness(
        gray_np, min_brightness=min_brightness, max_brightness=max_brightness
    )

    # 4. Aggregate results
    issues: List[str] = []
    suggestions: List[str] = []

    if not resolution_res["passed"]:
        issues.append(resolution_res["message"])
        if resolution_res["suggestion"]:
            suggestions.append(resolution_res["suggestion"])

    if not blur_res["passed"]:
        issues.append(blur_res["message"])
        if blur_res["suggestion"]:
            suggestions.append(blur_res["suggestion"])

    if not brightness_res["passed"]:
        issues.append(brightness_res["message"])
        if brightness_res["suggestion"]:
            suggestions.append(brightness_res["suggestion"])

    passed = len(issues) == 0

    if passed:
        summary_message = "Image passed all quality checks and is suitable for diagnosis."
    else:
        summary_message = f"Image failed quality assessment ({len(issues)} issue{'s' if len(issues) > 1 else ''} detected)."

    return {
        "passed": passed,
        "is_valid": True,
        "message": summary_message,
        "issues": issues,
        "suggestions": suggestions,
        "metrics": {
            "resolution": resolution_res,
            "blur": blur_res,
            "brightness": brightness_res,
            "format": val_result.get("format"),
        },
    }

def check_resolution(
    width: int,
    height: int,
    min_width: int = DEFAULT_MIN_WIDTH,
    min_height: int = DEFAULT_MIN_HEIGHT,
) -> Dict[str, Any]:
    """Check if the image meets the minimum spatial resolution.

    Args:
        width: Image width in pixels.
        height: Image height in pixels.
        min_width: Minimum acceptable width.
        min_height: Minimum acceptable height.

    Returns:
        dict: Evaluation result with passed status, dimensions, and advice.
    """
    passed = (width >= min_width) and (height >= min_height)

    if passed:
        message = f"Resolution ({width}x{height}) meets the minimum required ({min_width}x{min_height})."
        suggestion = None
    else:
        message = (
            f"Image resolution ({width}x{height}) is too low. "
            f"Minimum required is {min_width}x{min_height} pixels."
        )
        suggestion = "Please hold the camera closer to the affected leaf or take a higher resolution photo."

    return {
        "passed": passed,
        "width": int(width),
        "height": int(height),
        "min_width": int(min_width),
        "min_height": int(min_height),
        "message": message,
        "suggestion": suggestion,
    }


def check_blur(
    gray_image: np.ndarray,
    blur_threshold: float = DEFAULT_BLUR_THRESHOLD,
) -> Dict[str, Any]:
    """Detect motion blur or defocus using OpenCV's Laplacian variance method.

    The Laplacian operator calculates the 2nd spatial derivative of an image.
    Sharp edges produce high variance; smooth, blurry regions produce low variance.

    Args:
        gray_image: Grayscale NumPy image array (uint8).
        blur_threshold: Threshold below which an image is considered blurry.

    Returns:
        dict: Blur assessment containing 'passed', 'is_blurry', 'score',
              'threshold', and retake suggestion.
    """
    # Ensure single channel grayscale uint8
    if len(gray_image.shape) == 3:
        gray_image = cv2.cvtColor(gray_image, cv2.COLOR_RGB2GRAY)

    # Compute variance of the Laplacian
    laplacian = cv2.Laplacian(gray_image, cv2.CV_64F)
    variance = float(laplacian.var())
    blur_score = round(variance, 2)

    is_blurry = blur_score < blur_threshold
    passed = not is_blurry

    if passed:
        message = f"Image focus is sharp and acceptable (blur score: {blur_score})."
        suggestion = None
    else:
        message = (
            f"Image is too blurry (blur score: {blur_score:.1f}, "
            f"minimum required: {blur_threshold:.1f})."
        )
        suggestion = "Please hold the phone steady, tap to focus on the leaf, and retake the photo."

    return {
        "passed": passed,
        "is_blurry": is_blurry,
        "score": blur_score,
        "threshold": float(blur_threshold),
        "message": message,
        "suggestion": suggestion,
    }


def check_brightness(
    gray_image: np.ndarray,
    min_brightness: float = DEFAULT_MIN_BRIGHTNESS,
    max_brightness: float = DEFAULT_MAX_BRIGHTNESS,
) -> Dict[str, Any]:
    """Assess lighting conditions from image pixel intensity statistics.

    Args:
        gray_image: Grayscale NumPy image array (uint8).
        min_brightness: Lower limit for mean pixel intensity (0-255).
        max_brightness: Upper limit for mean pixel intensity (0-255).

    Returns:
        dict: Lighting status ('normal', 'dark', 'overexposed') with metrics.
    """
    if len(gray_image.shape) == 3:
        gray_image = cv2.cvtColor(gray_image, cv2.COLOR_RGB2GRAY)

    mean_val = float(np.mean(gray_image))
    std_val = float(np.std(gray_image))
    mean_brightness = round(mean_val, 2)
    std_brightness = round(std_val, 2)

    if mean_val < min_brightness:
        status = "dark"
        passed = False
        message = (
            f"Image is too dark (average brightness: {mean_brightness:.1f}/255, "
            f"acceptable range: {min_brightness}-{max_brightness})."
        )
        suggestion = "Please move to an area with better lighting or turn on your phone's flashlight."
    elif mean_val > max_brightness:
        status = "overexposed"
        passed = False
        message = (
            f"Image is overexposed/too bright (average brightness: {mean_brightness:.1f}/255, "
            f"acceptable range: {min_brightness}-{max_brightness})."
        )
        suggestion = "Please shield the leaf from direct sunlight glare and disable extreme flash."
    else:
        status = "normal"
        passed = True
        message = f"Image brightness is optimal (average brightness: {mean_brightness:.1f}/255)."
        suggestion = None

    return {
        "passed": passed,
        "status": status,
        "mean": mean_brightness,
        "std": std_brightness,
        "min_threshold": float(min_brightness),
        "max_threshold": float(max_brightness),
        "message": message,
        "suggestion": suggestion,
    }


def assess_image_quality(
    image_input: Union[bytes, BinaryIO, str, Path, Image.Image],
    min_width: int = DEFAULT_MIN_WIDTH,
    min_height: int = DEFAULT_MIN_HEIGHT,
    blur_threshold: float = DEFAULT_BLUR_THRESHOLD,
    min_brightness: float = DEFAULT_MIN_BRIGHTNESS,
    max_brightness: float = DEFAULT_MAX_BRIGHTNESS,
) -> Dict[str, Any]:
    """Execute a complete image quality pipeline on a crop photo.

    Evaluates:
    1. Validity and format integrity
    2. Resolution
    3. Blur / focus
    4. Lighting / brightness

    Args:
        image_input: Image bytes, file buffer, path, or PIL Image.
        min_width: Minimum required width in pixels.
        min_height: Minimum required height in pixels.
        blur_threshold: Minimum Laplacian variance for sharpness.
        min_brightness: Minimum mean intensity for adequate exposure.
        max_brightness: Maximum mean intensity to prevent glare.

    Returns:
        dict: Complete assessment including 'passed', 'is_valid', 'issues',
              'suggestions', and nested 'metrics'.
    """
    # 1. Validation check
    val_result = validate_image(image_input)
    if not val_result["valid"]:
        return {
            "passed": False,
            "is_valid": False,
            "message": "Image quality assessment failed: file is corrupted or not a recognized image format.",
            "issues": [val_result["message"]],
            "suggestions": [
                "Please upload a standard uncorrupted photo file in JPEG or PNG format."
            ],
            "metrics": None,
        }

    # 2. Convert to RGB PIL Image and NumPy Grayscale for CV checks
    try:
        raw_bytes = _get_image_bytes(image_input)
        pil_img = Image.open(io.BytesIO(raw_bytes)).convert("RGB")
        width, height = pil_img.size

        rgb_np = np.array(pil_img)
        gray_np = cv2.cvtColor(rgb_np, cv2.COLOR_RGB2GRAY)
    except Exception as exc:
        return {
            "passed": False,
            "is_valid": False,
            "message": f"Failed to decode image data for quality analysis: {str(exc)}",
            "issues": [str(exc)],
            "suggestions": ["Please capture and upload a fresh photo."],
            "metrics": None,
        }

    # 3. Individual quality metric assessments
    resolution_res = check_resolution(width, height, min_width=min_width, min_height=min_height)
    # Match the classifier's input scale so resolution alone does not change
    # the meaning of the Laplacian threshold.
    blur_input = cv2.resize(gray_np, (224, 224), interpolation=cv2.INTER_LINEAR)
    blur_res = check_blur(blur_input, blur_threshold=blur_threshold)
    brightness_res = check_brightness(
        gray_np, min_brightness=min_brightness, max_brightness=max_brightness
    )

    # 4. Aggregate results
    issues: List[str] = []
    suggestions: List[str] = []

    if not resolution_res["passed"]:
        issues.append(resolution_res["message"])
        if resolution_res["suggestion"]:
            suggestions.append(resolution_res["suggestion"])

    if not blur_res["passed"]:
        issues.append(blur_res["message"])
        if blur_res["suggestion"]:
            suggestions.append(blur_res["suggestion"])

    if not brightness_res["passed"]:
        issues.append(brightness_res["message"])
        if brightness_res["suggestion"]:
            suggestions.append(brightness_res["suggestion"])

    passed = len(issues) == 0

    if passed:
        summary_message = "Image passed all quality checks and is suitable for diagnosis."
    else:
        summary_message = f"Image failed quality assessment ({len(issues)} issue{'s' if len(issues) > 1 else ''} detected)."

    return {
        "passed": passed,
        "is_valid": True,
        "message": summary_message,
        "issues": issues,
        "suggestions": suggestions,
        "metrics": {
            "resolution": resolution_res,
            "blur": blur_res,
            "brightness": brightness_res,
            "format": val_result.get("format"),
        },
    }
