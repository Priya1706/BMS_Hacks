"""Unit tests for Image Quality Assessment service.

Covers:
- Valid images
- Corrupted and invalid format images
- Low-resolution images
- Blurry images (Laplacian variance)
- Dark and overexposed images (pixel intensity statistics)
- Full assessment pipeline and suggestions
"""

import io
import unittest
import numpy as np
from PIL import Image

from app.services.image_quality import (
    DEFAULT_BLUR_THRESHOLD,
    DEFAULT_MAX_BRIGHTNESS,
    DEFAULT_MIN_BRIGHTNESS,
    DEFAULT_MIN_HEIGHT,
    DEFAULT_MIN_WIDTH,
    assess_image_quality,
    check_blur,
    check_brightness,
    check_resolution,
    validate_image,
)


def make_in_memory_image(
    width: int = 300,
    height: int = 300,
    pattern: str = "sharp",
    fill_value: int = 128,
    img_format: str = "JPEG",
) -> io.BytesIO:
    """Helper to generate in-memory synthetic test images."""
    if pattern == "sharp":
        # Create high-contrast grid/checkerboard with sharp edges
        arr = np.full((height, width, 3), fill_value, dtype=np.uint8)
        arr[::10, :, :] = 255
        arr[:, ::10, :] = 0
        img = Image.fromarray(arr)
    elif pattern == "solid":
        # Flat, uniform solid image (zero edge variance, perfectly blurry)
        arr = np.full((height, width, 3), fill_value, dtype=np.uint8)
        img = Image.fromarray(arr)
    else:
        img = Image.new("RGB", (width, height), color=(fill_value, fill_value, fill_value))

    buf = io.BytesIO()
    img.save(buf, format=img_format)
    buf.seek(0)
    return buf


class TestImageValidation(unittest.TestCase):
    """Tests for raw image validation and corruption detection."""

    def test_valid_jpeg_image(self):
        """Test that a well-formed JPEG image passes validation."""
        buf = make_in_memory_image(300, 300, pattern="sharp")
        result = validate_image(buf)
        self.assertTrue(result["valid"])
        self.assertEqual(result["format"], "JPEG")
        self.assertEqual(result["size"], (300, 300))

    def test_valid_png_image(self):
        """Test that a well-formed PNG image passes validation."""
        buf = make_in_memory_image(250, 250, pattern="sharp", img_format="PNG")
        result = validate_image(buf)
        self.assertTrue(result["valid"])
        self.assertEqual(result["format"], "PNG")
        self.assertEqual(result["size"], (250, 250))

    def test_corrupted_image_bytes(self):
        """Test that random non-image bytes fail validation safely."""
        corrupt_data = b"NOT_AN_IMAGE_HEADER_1234567890_GARBAGE"
        result = validate_image(corrupt_data)
        self.assertFalse(result["valid"])
        self.assertIn("Invalid or corrupted image", result["message"])

    def test_empty_image(self):
        """Test that an empty buffer fails validation safely."""
        result = validate_image(b"")
        self.assertFalse(result["valid"])
        self.assertIn("empty", result["message"].lower())


class TestResolutionCheck(unittest.TestCase):
    """Tests for resolution threshold enforcement."""

    def test_sufficient_resolution(self):
        """Image >= 224x224 should pass."""
        res = check_resolution(width=300, height=400, min_width=224, min_height=224)
        self.assertTrue(res["passed"])
        self.assertIsNone(res["suggestion"])

    def test_exact_minimum_resolution(self):
        """Image exactly 224x224 should pass."""
        res = check_resolution(width=224, height=224, min_width=224, min_height=224)
        self.assertTrue(res["passed"])

    def test_low_width(self):
        """Image with width < 224 should fail with a clear suggestion."""
        res = check_resolution(width=180, height=300, min_width=224, min_height=224)
        self.assertFalse(res["passed"])
        self.assertIn("too low", res["message"])
        self.assertIsNotNone(res["suggestion"])

    def test_low_height(self):
        """Image with height < 224 should fail."""
        res = check_resolution(width=300, height=150, min_width=224, min_height=224)
        self.assertFalse(res["passed"])
        self.assertIn("too low", res["message"])

    def test_tiny_image(self):
        """Thumbnail / low-res image (e.g. 64x64) should fail."""
        res = check_resolution(width=64, height=64)
        self.assertFalse(res["passed"])


class TestBlurDetection(unittest.TestCase):
    """Tests for OpenCV Laplacian variance blur detection."""

    def test_sharp_image(self):
        """High-contrast image with crisp lines should pass blur check."""
        # Create sharp grid
        arr = np.zeros((300, 300), dtype=np.uint8)
        arr[::10, :] = 255
        arr[:, ::10] = 255

        res = check_blur(arr, blur_threshold=100.0)
        self.assertTrue(res["passed"])
        self.assertFalse(res["is_blurry"])
        self.assertGreater(res["score"], 100.0)
        self.assertIsNone(res["suggestion"])

    def test_blurry_solid_image(self):
        """Flat solid image has variance ~0.0 and should be flagged as blurry."""
        arr = np.full((300, 300), 128, dtype=np.uint8)
        res = check_blur(arr, blur_threshold=100.0)
        self.assertFalse(res["passed"])
        self.assertTrue(res["is_blurry"])
        self.assertLess(res["score"], 100.0)
        self.assertIsNotNone(res["suggestion"])
        self.assertIn("steady", res["suggestion"].lower())


class TestBrightnessAssessment(unittest.TestCase):
    """Tests for pixel intensity lighting checks."""

    def test_optimal_lighting(self):
        """Moderate gray intensity should pass as 'normal'."""
        arr = np.full((300, 300), 128, dtype=np.uint8)
        res = check_brightness(arr, min_brightness=45.0, max_brightness=215.0)
        self.assertTrue(res["passed"])
        self.assertEqual(res["status"], "normal")
        self.assertIsNone(res["suggestion"])

    def test_dark_underexposed_image(self):
        """Mean intensity < 45.0 should be flagged as 'dark'."""
        arr = np.full((300, 300), 20, dtype=np.uint8)
        res = check_brightness(arr, min_brightness=45.0, max_brightness=215.0)
        self.assertFalse(res["passed"])
        self.assertEqual(res["status"], "dark")
        self.assertIn("dark", res["message"].lower())
        self.assertIsNotNone(res["suggestion"])
        self.assertIn("lighting", res["suggestion"].lower())

    def test_overexposed_image(self):
        """Mean intensity > 215.0 should be flagged as 'overexposed'."""
        arr = np.full((300, 300), 240, dtype=np.uint8)
        res = check_brightness(arr, min_brightness=45.0, max_brightness=215.0)
        self.assertFalse(res["passed"])
        self.assertEqual(res["status"], "overexposed")
        self.assertIn("overexposed", res["message"].lower())
        self.assertIsNotNone(res["suggestion"])
        self.assertIn("glare", res["suggestion"].lower())


class TestCombinedQualityAssessment(unittest.TestCase):
    """Tests for full assess_image_quality pipeline."""

    def test_high_quality_image_passes(self):
        """Valid, high-resolution, sharp, well-lit image passes all checks."""
        buf = make_in_memory_image(width=300, height=300, pattern="sharp", fill_value=128)
        result = assess_image_quality(buf)

        self.assertTrue(result["passed"])
        self.assertTrue(result["is_valid"])
        self.assertEqual(len(result["issues"]), 0)
        self.assertEqual(len(result["suggestions"]), 0)
        self.assertIsNotNone(result["metrics"])
        self.assertTrue(result["metrics"]["resolution"]["passed"])
        self.assertTrue(result["metrics"]["blur"]["passed"])
        self.assertTrue(result["metrics"]["brightness"]["passed"])

    def test_corrupted_image_fails_safely(self):
        """Corrupted data fails gracefully without unhandled exceptions."""
        result = assess_image_quality(b"GARBAGE_BYTES_THAT_ARE_NOT_AN_IMAGE")
        self.assertFalse(result["passed"])
        self.assertFalse(result["is_valid"])
        self.assertGreater(len(result["issues"]), 0)
        self.assertGreater(len(result["suggestions"]), 0)

    def test_low_res_image_fails_with_suggestion(self):
        """Low resolution image is rejected with actionable advice."""
        buf = make_in_memory_image(width=100, height=100, pattern="sharp")
        result = assess_image_quality(buf)
        self.assertFalse(result["passed"])
        self.assertTrue(result["is_valid"])
        self.assertFalse(result["metrics"]["resolution"]["passed"])
        self.assertTrue(any("resolution" in issue.lower() for issue in result["issues"]))

    def test_blurry_image_fails_with_suggestion(self):
        """Solid uniform image fails blur check with retake suggestion."""
        buf = make_in_memory_image(width=300, height=300, pattern="solid", fill_value=128)
        result = assess_image_quality(buf)
        self.assertFalse(result["passed"])
        self.assertTrue(result["is_valid"])
        self.assertFalse(result["metrics"]["blur"]["passed"])
        self.assertTrue(any("blurry" in issue.lower() for issue in result["issues"]))

    def test_dark_image_fails_with_suggestion(self):
        """Dark image fails brightness check."""
        buf = make_in_memory_image(width=300, height=300, pattern="solid", fill_value=15)
        result = assess_image_quality(buf)
        self.assertFalse(result["passed"])
        self.assertEqual(result["metrics"]["brightness"]["status"], "dark")
        self.assertTrue(any("dark" in issue.lower() for issue in result["issues"]))

    def test_overexposed_image_fails_with_suggestion(self):
        """Overexposed image fails brightness check."""
        buf = make_in_memory_image(width=300, height=300, pattern="solid", fill_value=245)
        result = assess_image_quality(buf)
        self.assertFalse(result["passed"])
        self.assertEqual(result["metrics"]["brightness"]["status"], "overexposed")
        self.assertTrue(any("overexposed" in issue.lower() for issue in result["issues"]))


if __name__ == "__main__":
    unittest.main()
