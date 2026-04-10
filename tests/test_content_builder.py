import unittest

from joplin_notes_ai.models import RelatedNote, TransformationResult
from joplin_notes_ai.services.content_builder import build_final_content


class ContentBuilderTestCase(unittest.TestCase):
    def test_build_final_content_includes_blocks(self):
        result = TransformationResult(
            new_title="Clean title",
            content="## Rewritten",
            logical_gaps=["Gap 1"],
            further_questions=["Question 1"],
            target_notebook="Inbox",
            suggested_tags=["tag1", "tag2", "tag3"],
        )
        related = [RelatedNote(note_id="n2", title="Other", similarity=0.91)]

        final_content = build_final_content(
            result,
            related,
            similarity_threshold=0.8,
            machine_marker="<!-- ai_audited_v1 -->",
        )

        self.assertIn("## Rewritten", final_content)
        self.assertIn("Семантичні зв'язки", final_content)
        self.assertIn("AI Audit Notes", final_content)
        self.assertIn("<!-- ai_audited_v1 -->", final_content)
        self.assertIn("схожість: 91%", final_content)


if __name__ == "__main__":
    unittest.main()
