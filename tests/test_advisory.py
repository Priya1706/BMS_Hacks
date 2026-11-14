"""Unit tests for Disease Advisory service.

Covers:
- Presence of placeholder flags on all data
- Structured agronomic fields for rice and wheat
- Absence of unverified chemical dosages / presence of advisory disclaimers
- General advisory fallbacks
- Unsupported crop handling
"""

import unittest

from app.services.advisory import (
    ALL_DISEASES,
    DISCLAIMER_TEXT,
    get_disease_advisory,
    list_supported_diseases,
)


class TestAdvisoryService(unittest.TestCase):
    """Test suite for the crop disease advisory service."""

    def test_all_diseases_marked_as_placeholder(self):
        """Every single disease entry in the catalog must be flagged as a placeholder."""
        for crop, diseases in ALL_DISEASES.items():
            for d_id, data in diseases.items():
                self.assertTrue(
                    data.get("is_placeholder"),
                    f"Disease '{d_id}' in crop '{crop}' must have is_placeholder=True",
                )

    def test_rice_leaf_blast_structure(self):
        """Verify rice blast advisory contains all required structured fields."""
        adv = get_disease_advisory("rice", "leaf_blast")
        self.assertEqual(adv["disease_id"], "leaf_blast")
        self.assertEqual(adv["crop_type"], "rice")
        self.assertTrue(adv["is_placeholder"])
        self.assertIsInstance(adv["symptoms"], list)
        self.assertGreater(len(adv["symptoms"]), 0)
        self.assertIsInstance(adv["preventive_measures"], list)
        self.assertGreater(len(adv["preventive_measures"]), 0)
        self.assertIsInstance(adv["organic_controls"], list)
        self.assertGreater(len(adv["organic_controls"]), 0)
        self.assertIn("disclaimer", adv)
        self.assertIn("DISCLAIMER", adv["disclaimer"])
        # Ensure chemical guidance directs to local agricultural extension, not invented dosage
        self.assertIn("consult", adv["chemical_guidance"].lower())

    def test_wheat_rust_structure(self):
        """Verify wheat leaf rust advisory contains all required fields."""
        adv = get_disease_advisory("wheat", "leaf_rust")
        self.assertEqual(adv["disease_id"], "leaf_rust")
        self.assertEqual(adv["crop_type"], "wheat")
        self.assertTrue(adv["is_placeholder"])
        self.assertGreater(len(adv["symptoms"]), 0)
        self.assertIn("consult", adv["chemical_guidance"].lower())

    def test_general_guidance_fallback(self):
        """When disease_id is None, return general crop health guidance."""
        adv = get_disease_advisory("rice", None)
        self.assertEqual(adv["disease_id"], "general_guidance")
        self.assertTrue(adv["is_placeholder"])
        self.assertIn("General", adv["disease_name"])

    def test_unknown_disease_id_fallback(self):
        """When an unknown disease_id is requested, return general crop health guidance."""
        adv = get_disease_advisory("wheat", "non_existent_fungus")
        self.assertEqual(adv["disease_id"], "non_existent_fungus")
        self.assertTrue(adv["is_placeholder"])
        self.assertIn("General", adv["disease_name"])

    def test_unsupported_crop(self):
        """Unsupported crop should return an error message in advisory."""
        adv = get_disease_advisory("sugarcane", "red_rot")
        self.assertTrue(adv["is_placeholder"])
        self.assertIn("Unsupported crop", adv["message"])

    def test_list_supported_diseases_all(self):
        """Listing supported diseases returns records for both crops."""
        catalog = list_supported_diseases()
        self.assertIn("rice", catalog)
        self.assertIn("wheat", catalog)
        self.assertGreaterEqual(len(catalog["rice"]), 4)
        self.assertGreaterEqual(len(catalog["wheat"]), 4)

    def test_list_supported_diseases_filtered(self):
        """Listing supported diseases filtered by crop returns only that crop."""
        rice_catalog = list_supported_diseases("rice")
        self.assertIn("rice", rice_catalog)
        self.assertNotIn("wheat", rice_catalog)


if __name__ == "__main__":
    unittest.main()
