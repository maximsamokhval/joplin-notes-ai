import unittest

from pydantic import ValidationError

from joplin_notes_ai.models import TransformationResult


class LlmSchemaTestCase(unittest.TestCase):
    def test_transformation_result_valid_json(self):
        payload = """
        {
          "new_title": "Title",
          "content": "Body",
          "logical_gaps": ["g1"],
          "further_questions": ["q1"],
          "target_notebook": "Inbox",
          "suggested_tags": ["a", "b", "c"]
        }
        """
        result = TransformationResult.model_validate_json(payload)
        self.assertEqual(result.new_title, "Title")
        self.assertEqual(result.target_notebook, "Inbox")

    def test_transformation_result_invalid_json(self):
        payload = '{"new_title":"Title","content":"Body"}'
        with self.assertRaises(ValidationError):
            TransformationResult.model_validate_json(payload)


if __name__ == "__main__":
    unittest.main()
