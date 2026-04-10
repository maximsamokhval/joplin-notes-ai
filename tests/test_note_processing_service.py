import unittest
from unittest.mock import Mock

from joplin_notes_ai.config import Settings
from joplin_notes_ai.exceptions import (
    JoplinApiError,
    LlmResponseValidationError,
    VectorStoreError,
)
from joplin_notes_ai.models import NoteDetails, NoteSummary, RelatedCandidate, TransformationResult
from joplin_notes_ai.services.note_processing import NoteProcessingService


def make_settings() -> Settings:
    return Settings.model_validate(
        {
            "JOPLIN_TOKEN": "token",
            "LLM_API_KEY": "llm-key",
            "PAUSE_BETWEEN_NOTES": 0,
        }
    )


class NoteProcessingServiceTestCase(unittest.TestCase):
    def setUp(self):
        self.settings = make_settings()
        self.joplin = Mock()
        self.llm = Mock()
        self.vector_store = Mock()
        self.service = NoteProcessingService(
            settings=self.settings,
            joplin_client=self.joplin,
            llm_client=self.llm,
            vector_store=self.vector_store,
        )
        self.note_summary = NoteSummary(id="n1", title="Draft")
        self.note = NoteDetails(id="n1", title="Draft", body="raw body")
        self.result = TransformationResult(
            new_title="New title",
            content="Rewritten content",
            logical_gaps=[],
            further_questions=[],
            target_notebook="Tech",
            suggested_tags=["python", "ai", "notes"],
        )
        self.joplin.get_note_tag_titles.return_value = {"ai-audited", "python", "ai", "notes"}
        self.vector_store.search_candidates.return_value = []

    def test_successful_processing(self):
        self.joplin.get_note.return_value = self.note
        self.llm.transform_note.return_value = self.result
        self.joplin.update_note.return_value = True

        outcome = self.service.process(self.note_summary, {"Tech": "folder-1"})

        self.assertEqual(outcome.status, "processed")
        self.joplin.update_note.assert_called_once()
        update = self.joplin.update_note.call_args.args[1]
        self.assertEqual(update.parent_id, "folder-1")
        self.assertIn("<!-- ai_audited_v1 -->", update.body)

    def test_skip_empty_note(self):
        self.joplin.get_note.return_value = NoteDetails(id="n1", title="Draft", body=" ")

        outcome = self.service.process(self.note_summary, {"Tech": "folder-1"})

        self.assertEqual(outcome.status, "skipped_empty")
        self.llm.transform_note.assert_not_called()

    def test_skip_already_transformed(self):
        self.joplin.get_note.return_value = NoteDetails(
            id="n1",
            title="Draft",
            body="some text\n<!-- ai_audited_v1 -->\n",
        )

        outcome = self.service.process(self.note_summary, {"Tech": "folder-1"})

        self.assertEqual(outcome.status, "skipped_already_transformed")
        self.joplin.add_tag_to_note_by_title.assert_called_once()

    def test_skip_llm_failed_on_invalid_json(self):
        self.joplin.get_note.return_value = self.note
        self.llm.transform_note.side_effect = LlmResponseValidationError("invalid response")

        outcome = self.service.process(self.note_summary, {"Tech": "folder-1"})

        self.assertEqual(outcome.status, "skipped_llm_failed")
        self.joplin.add_tag_to_note_by_title.assert_called_with("n1", "ai-failed")
        self.joplin.update_note.assert_not_called()

    def test_missing_notebook_keeps_parent_id_empty(self):
        self.joplin.get_note.return_value = self.note
        self.llm.transform_note.return_value = self.result
        self.joplin.update_note.return_value = True

        outcome = self.service.process(self.note_summary, {"Other": "folder-2"})

        self.assertEqual(outcome.status, "processed")
        update = self.joplin.update_note.call_args.args[1]
        self.assertIsNone(update.parent_id)

    def test_partial_failures_in_indexing_and_tagging(self):
        self.joplin.get_note.return_value = self.note
        self.llm.transform_note.return_value = self.result
        self.vector_store.upsert_note.side_effect = VectorStoreError("index unavailable")
        self.joplin.update_note.return_value = True
        self.joplin.add_tag_to_note_by_title.side_effect = [
            JoplinApiError("tag fail"),
            None,
            None,
            None,
        ]

        outcome = self.service.process(self.note_summary, {"Tech": "folder-1"})

        self.assertEqual(outcome.status, "processed")
        self.joplin.update_note.assert_called_once()

    def test_dry_run_processes_without_writes(self):
        self.joplin.get_note.return_value = self.note
        self.llm.transform_note.return_value = self.result

        outcome = self.service.process(self.note_summary, {"Tech": "folder-1"}, dry_run=True)

        self.assertEqual(outcome.status, "dry_run")
        self.joplin.update_note.assert_not_called()
        self.vector_store.upsert_note.assert_not_called()

    def test_candidates_below_threshold_are_not_published_in_note(self):
        self.joplin.get_note.return_value = self.note
        self.llm.transform_note.return_value = self.result
        self.vector_store.search_candidates.return_value = [
            RelatedCandidate(
                note_id="n2",
                title="Near miss",
                distance=0.31,
                similarity=0.69,
                accepted=False,
                rejection_reason="below_threshold",
                rank=1,
            )
        ]
        self.joplin.update_note.return_value = True

        self.service.process(self.note_summary, {"Tech": "folder-1"})

        update = self.joplin.update_note.call_args.args[1]
        self.assertNotIn("Семантичні зв'язки", update.body)
        self.assertNotIn("Near miss", update.body)

    def test_normalizes_note_before_llm_call(self):
        raw_body = (
            "Line 1\r\n\r\n\r\n"
            "<details><summary>Оригінальна чернетка (Backup)</summary>trash</details>\n"
            "<!-- ai_audited_v1 -->\n"
            "Line 2"
        )

        normalized_body = NoteProcessingService._normalize_note_for_llm(raw_body)
        self.assertNotIn("<details>", normalized_body)
        self.assertNotIn("<!-- ai_audited_v1 -->", normalized_body)
        self.assertIn("Line 1", normalized_body)
        self.assertIn("Line 2", normalized_body)


if __name__ == "__main__":
    unittest.main()
