"""Static integration contracts for the browser-only questionnaire flow.

The project has no JavaScript test runner or installed JS runtime in its venv,
so these tests guard the UI wiring and ML separation without adding a runtime
dependency. Browser interaction remains a separate manual check.
"""

from html.parser import HTMLParser
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]


class IdParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.ids = []
        self.visible_text = []

    def handle_starttag(self, tag, attrs):
        values = dict(attrs)
        if "id" in values:
            self.ids.append(values["id"])

    def handle_data(self, data):
        if data.strip():
            self.visible_text.append(data.strip())


class QuestionnaireFrontendContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.html = (ROOT / "index.html").read_text(encoding="utf-8")
        cls.app = (ROOT / "js/app.js").read_text(encoding="utf-8")
        cls.assessment = (ROOT / "js/services/contextualAssessment.js").read_text(encoding="utf-8")
        cls.service_worker = (ROOT / "sw.js").read_text(encoding="utf-8")

    def test_prompt_and_crop_specific_screens_exist(self):
        parser = IdParser()
        parser.feed(self.html)
        self.assertEqual(len(parser.ids), len(set(parser.ids)), "HTML element IDs must be unique")
        for element_id in (
            "screen-questionnaire-consent", "screen-field-questionnaire",
            "field-questionnaire-form", "questionnaire-fields", "screen-results",
            "results-context-section", "advisory-field-context",
        ):
            self.assertIn(element_id, parser.ids)
        self.assertIn("Would you like to provide additional information about your crop?", parser.visible_text)
        self.assertIn("Your observations can help us provide a more contextual assessment.", parser.visible_text)
        self.assertIn("Yes, provide additional information", parser.visible_text)
        self.assertIn("No, continue to results", parser.visible_text)

    def test_yes_no_navigation_and_session_reset_are_wired(self):
        self.assertRegex(self.app, r"btn-questionnaire-yes[\s\S]{0,220}navigateTo\('screen-field-questionnaire'\)")
        self.assertRegex(self.app, r"btn-questionnaire-no[\s\S]{0,320}answers: null[\s\S]{0,120}navigateTo\('screen-results'\)")
        reset_start = self.app.index("  onEnterCaptureScreen() {")
        reset_end = self.app.index("  setImagePreview(", reset_start)
        self.assertIn("this.questionnaireSession = null;", self.app[reset_start:reset_end])
        self.assertIn("this.renderQuestionnaire();", self.app)
        self.assertIn("this.navigateTo('screen-results');", self.app)

    def test_answers_stay_in_separate_session_and_survive_rerender(self):
        self.assertIn("session.answers = {", self.app)
        self.assertIn("answers.symptoms || []", self.app)
        self.assertIn("select.value = answers[key] || 'unsure'", self.app)
        self.assertIn("assessment: assessFieldObservations(this.currentDisease, answers)", self.app)
        self.assertIn("fieldObservations: this.questionnaireSession?.answers || null", self.app)
        self.assertIn("diseaseId: this.currentDisease?.id || null", self.app)

    def test_prediction_confidence_and_advisory_source_remain_original(self):
        diagnosis_start = self.app.index("this.currentDiagnosis = Object.freeze({")
        diagnosis_end = self.app.index("this.currentDisease = result.disease;", diagnosis_start)
        assignment = self.app[diagnosis_start:diagnosis_end]
        self.assertIn("...result", assignment)
        self.assertNotRegex(assignment, r"confidence\s*[:=]")
        self.assertNotRegex(self.app, r"currentDiagnosis\.confidence\s*=")
        self.assertIn("Original model confidence: ${diagnosis.confidence}%", self.app)
        self.assertIn("this.renderAdvisoryScreen(this.currentDisease);", self.app)
        self.assertIn("this.renderFieldContext('advisory-field-context'", self.app)

    def test_rice_and_wheat_questions_and_qualitative_statuses_exist(self):
        self.assertIn("rice: {", self.assessment)
        self.assertIn("wheat: {", self.assessment)
        for cue in ("edge_stripes", "brown_spots", "sheath_lesions", "scattered_brown_pustules", "yellow_stripes", "dark_stem_pustules", "white_powder"):
            self.assertIn(cue, self.assessment)
        for status in ("consistent", "misaligned", "insufficient"):
            self.assertIn(f"status: '{status}'", self.assessment)
        self.assertNotRegex(self.assessment, r"confidence\s*[:=]")

    def test_new_module_is_precached_and_prediction_module_is_untouched(self):
        self.assertIn("./js/services/contextualAssessment.js", self.service_worker)
        self.assertIn("rootcause-v1.3.0", self.service_worker)
        # The questionnaire change must not edit the protected prediction code.
        import subprocess

        result = subprocess.run(
            ["git", "diff", "--quiet", "--", "app/services/predictor.py", "app/api/routes.py"],
            cwd=ROOT,
            check=False,
        )
        self.assertEqual(result.returncode, 0, "predictor.py and API routes must remain unchanged")


if __name__ == "__main__":
    unittest.main()
