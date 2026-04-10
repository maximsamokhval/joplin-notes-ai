import unittest
from unittest.mock import Mock

from joplin_notes_ai.config import Settings
from joplin_notes_ai.exceptions import (
    JoplinApiError,
    LlmResponseValidationError,
    VectorStoreError,
)
from joplin_notes_ai.models import NoteDetails, NoteSummary, TransformationResult
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

    def test_successful_processing(self):
        self.joplin.get_note.return_value = self.note
        self.llm.transform_note.return_value = self.result
        self.vector_store.find_related.return_value = []
        self.joplin.update_note.return_value = True

        outcome = self.service.process(self.note_summary, {"Tech": "folder-1"})

        self.assertEqual(outcome.status, "processed")
        self.joplin.update_note.assert_called_once()
        update = self.joplin.update_note.call_args.args[1]
        self.assertEqual(update.parent_id, "folder-1")

    def test_skip_empty_note(self):
        self.joplin.get_note.return_value = NoteDetails(id="n1", title="Draft", body=" ")

        outcome = self.service.process(self.note_summary, {"Tech": "folder-1"})

        self.assertEqual(outcome.status, "skipped_empty")
        self.llm.transform_note.assert_not_called()

    def test_skip_already_transformed(self):
        self.joplin.get_note.return_value = NoteDetails(
            id="n1",
            title="Draft",
            body="some text\nОригінальна чернетка\n",
        )

        outcome = self.service.process(self.note_summary, {"Tech": "folder-1"})

        self.assertEqual(outcome.status, "skipped_already_transformed")
        self.joplin.add_tag_to_note_by_title.assert_called_once()

    def test_skip_llm_failed_on_invalid_json(self):
        self.joplin.get_note.return_value = self.note
        self.llm.transform_note.side_effect = LlmResponseValidationError("invalid response")

        outcome = self.service.process(self.note_summary, {"Tech": "folder-1"})

        self.assertEqual(outcome.status, "skipped_llm_failed")
        self.joplin.update_note.assert_not_called()

    def test_missing_notebook_keeps_parent_id_empty(self):
        self.joplin.get_note.return_value = self.note
        self.llm.transform_note.return_value = self.result
        self.vector_store.find_related.return_value = []
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
        self.vector_store.find_related.return_value = []

        outcome = self.service.process(self.note_summary, {"Tech": "folder-1"}, dry_run=True)

        self.assertEqual(outcome.status, "dry_run")
        self.joplin.update_note.assert_not_called()
        self.vector_store.upsert_note.assert_not_called()


if __name__ == "__main__":
    unittest.main()
