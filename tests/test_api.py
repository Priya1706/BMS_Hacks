"""API integration and unit tests for RootCause FastAPI endpoints.

Covers:
- Root and health endpoints (/ and /health)
- POST /predict with:
  - Valid images (sharp, well-lit, adequate resolution)
  - Corrupted images
  - Low-resolution images
  - Blurry images
  - Dark images
  - Overexposed images
  - Unsupported crop types
  - Empty image upload
  - Case-insensitive crop types
- Standalone POST /quality-check
- GET /models/status
- GET /advisory/{crop_type}
- GET /diseases
"""

import io
import json
import unittest
import numpy as np
from PIL import Image
from fastapi import UploadFile

from app.api.routes import (
    check_quality,
    get_advisory_route,
    list_diseases_route,
    model_status,
    predict,
)
from app.main import app, health_check, home


def create_test_upload_file(
    filename: str = "test_leaf.jpg",
    width: int = 300,
    height: int = 300,
    pattern: str = "sharp",
    fill_value: int = 128,
    raw_bytes: bytes = None,
) -> UploadFile:
    """Helper to generate an in-memory UploadFile object."""
    if raw_bytes is not None:
        buf = io.BytesIO(raw_bytes)
    elif pattern == "sharp":
        # Sharp high-contrast grid
        arr = np.full((height, width, 3), fill_value, dtype=np.uint8)
        arr[::10, :, :] = 255
        arr[:, ::10, :] = 0
        img = Image.fromarray(arr)
        buf = io.BytesIO()
        img.save(buf, format="JPEG")
        buf.seek(0)
    elif pattern == "solid":
        arr = np.full((height, width, 3), fill_value, dtype=np.uint8)
        img = Image.fromarray(arr)
        buf = io.BytesIO()
        img.save(buf, format="JPEG")
        buf.seek(0)
    else:
        img = Image.new("RGB", (width, height), color=(fill_value, fill_value, fill_value))
        buf = io.BytesIO()
        img.save(buf, format="JPEG")
        buf.seek(0)

    return UploadFile(filename=filename, file=buf)


def parse_response(response):
    """Helper to extract status code and JSON content from a Response/JSONResponse."""
    if hasattr(response, "status_code"):
        status_code = response.status_code
    else:
        status_code = 200

    if hasattr(response, "body"):
        data = json.loads(response.body.decode("utf-8"))
    elif isinstance(response, dict):
        data = response
    else:
        data = response

    return status_code, data


class TestAPIEndpoints(unittest.TestCase):
    """Test suite for FastAPI endpoints."""

    def test_root_endpoint(self):
        """Test GET / returns welcome message."""
        res = home()
        self.assertEqual(res["status"], "Backend is running")
        self.assertIn("RootCause", res["message"])

    def test_health_check_endpoint(self):
        """Test GET /health returns status healthy."""
        res = health_check()
        self.assertEqual(res["status"], "healthy")

    def test_predict_valid_image_rice(self):
        """Test POST /predict with a valid sharp image of rice."""
        upload = create_test_upload_file(width=300, height=300, pattern="sharp")
        response = predict(crop_type="rice", image=upload)
        status_code, data = parse_response(response)

        self.assertEqual(status_code, 200)
        self.assertEqual(data["status"], "success")
        self.assertEqual(data["crop_type"], "rice")
        self.assertTrue(data["quality_assessment"]["passed"])
        self.assertEqual(data["prediction"]["status"], "success")
        self.assertTrue(data["prediction"]["is_model_loaded"])
        self.assertIn(data["prediction"]["prediction"], {
            "Bacterial Leaf Blight", "Brown Spot", "Healthy Rice Leaf",
            "Leaf Blast", "Leaf scald", "Sheath Blight",
        })
        # Verify advisory is returned with placeholder disclaimer
        self.assertTrue(data["advisory"]["is_placeholder"])
        self.assertIn("DISCLAIMER", data["advisory"]["disclaimer"])

    def test_predict_valid_image_wheat(self):
        """Test POST /predict with a valid sharp image of wheat."""
        upload = create_test_upload_file(width=300, height=300, pattern="sharp")
        response = predict(crop_type="wheat", image=upload)
        status_code, data = parse_response(response)

        self.assertEqual(status_code, 200)
        self.assertEqual(data["crop_type"], "wheat")
        self.assertEqual(data["prediction"]["status"], "success")
        self.assertTrue(data["prediction"]["is_model_loaded"])
        self.assertIn(data["prediction"]["prediction"], {
            "Black Rust", "Brown Rust", "Yellow Rust", "Powdery Mildew",
        })

    def test_predict_case_insensitive_crop(self):
        """Test that crop type 'Rice' or 'WHEAT' is accepted and normalized."""
        upload = create_test_upload_file(width=300, height=300, pattern="sharp")
        response = predict(crop_type="  RICE  ", image=upload)
        status_code, data = parse_response(response)

        self.assertEqual(status_code, 200)
        self.assertEqual(data["crop_type"], "rice")

    def test_predict_unsupported_crop(self):
        """Test POST /predict with unsupported crop (e.g. 'corn') returns 400 Bad Request."""
        upload = create_test_upload_file(width=300, height=300, pattern="sharp")
        response = predict(crop_type="corn", image=upload)
        status_code, data = parse_response(response)

        self.assertEqual(status_code, 400)
        self.assertEqual(data["status"], "error")
        self.assertEqual(data["error_code"], "INVALID_CROP_TYPE")
        self.assertIn("corn", data["message"])
        self.assertIn("rice", data["message"])
        self.assertIn("wheat", data["message"])

    def test_predict_corrupted_image(self):
        """Test POST /predict with corrupted bytes returns 400 quality_check_failed."""
        upload = create_test_upload_file(raw_bytes=b"CORRUPTED_FILE_NOT_AN_IMAGE")
        response = predict(crop_type="rice", image=upload)
        status_code, data = parse_response(response)

        self.assertEqual(status_code, 400)
        self.assertEqual(data["status"], "quality_check_failed")
        self.assertIn("failed", data["message"])
        self.assertGreater(len(data["issues"]), 0)
        self.assertGreater(len(data["suggestions"]), 0)

    def test_predict_low_resolution_image(self):
        """Test POST /predict with low-resolution image (< 224x224) returns 400."""
        upload = create_test_upload_file(width=120, height=120, pattern="sharp")
        response = predict(crop_type="rice", image=upload)
        status_code, data = parse_response(response)

        self.assertEqual(status_code, 400)
        self.assertEqual(data["status"], "quality_check_failed")
        self.assertTrue(any("resolution" in issue.lower() for issue in data["issues"]))

    def test_predict_blurry_image(self):
        """Test POST /predict with blurry image returns 400 with retake suggestion."""
        upload = create_test_upload_file(width=300, height=300, pattern="solid", fill_value=128)
        response = predict(crop_type="rice", image=upload)
        status_code, data = parse_response(response)

        self.assertEqual(status_code, 400)
        self.assertEqual(data["status"], "quality_check_failed")
        self.assertTrue(any("blurry" in issue.lower() for issue in data["issues"]))
        self.assertTrue(any("focus" in sug.lower() or "steady" in sug.lower() for sug in data["suggestions"]))

    def test_predict_dark_image(self):
        """Test POST /predict with underexposed image returns 400 with lighting suggestion."""
        upload = create_test_upload_file(width=300, height=300, pattern="solid", fill_value=15)
        response = predict(crop_type="wheat", image=upload)
        status_code, data = parse_response(response)

        self.assertEqual(status_code, 400)
        self.assertEqual(data["status"], "quality_check_failed")
        self.assertTrue(any("dark" in issue.lower() for issue in data["issues"]))

    def test_predict_overexposed_image(self):
        """Test POST /predict with overexposed image returns 400 with glare suggestion."""
        upload = create_test_upload_file(width=300, height=300, pattern="solid", fill_value=245)
        response = predict(crop_type="rice", image=upload)
        status_code, data = parse_response(response)

        self.assertEqual(status_code, 400)
        self.assertEqual(data["status"], "quality_check_failed")
        self.assertTrue(any("overexposed" in issue.lower() for issue in data["issues"]))

    def test_predict_empty_file(self):
        """Test POST /predict with 0-byte upload returns 400."""
        upload = create_test_upload_file(raw_bytes=b"")
        response = predict(crop_type="rice", image=upload)
        status_code, data = parse_response(response)

        self.assertEqual(status_code, 400)
        self.assertEqual(data["error_code"], "EMPTY_IMAGE_FILE")

    def test_standalone_quality_check_valid(self):
        """Test POST /quality-check with sharp image returns 200 passed=True."""
        upload = create_test_upload_file(width=300, height=300, pattern="sharp")
        response = check_quality(image=upload)
        status_code, data = parse_response(response)

        self.assertEqual(status_code, 200)
        self.assertTrue(data["passed"])

    def test_standalone_quality_check_blurry(self):
        """Test POST /quality-check with blurry image returns 400 passed=False."""
        upload = create_test_upload_file(width=300, height=300, pattern="solid", fill_value=128)
        response = check_quality(image=upload)
        status_code, data = parse_response(response)

        self.assertEqual(status_code, 400)
        self.assertFalse(data["passed"])

    def test_models_status_endpoint(self):
        """Test GET /models/status reports availability for all crops."""
        data = model_status()
        self.assertIn("crops", data)
        self.assertIn("rice", data["crops"])
        self.assertIn("wheat", data["crops"])

    def test_advisory_route(self):
        """Test GET /advisory/rice returns structured advisory with disclaimer."""
        data = get_advisory_route("rice", "leaf_blast")
        self.assertEqual(data["disease_id"], "leaf_blast")
        self.assertTrue(data["is_placeholder"])
        self.assertIn("DISCLAIMER", data["disclaimer"])

    def test_diseases_catalog_route(self):
        """Test GET /diseases returns catalog for both crops."""
        data = list_diseases_route()
        self.assertIn("rice", data)
        self.assertIn("wheat", data)


if __name__ == "__main__":
    unittest.main()
