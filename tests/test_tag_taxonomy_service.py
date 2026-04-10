import unittest
from unittest.mock import Mock

from joplin_notes_ai.config import Settings
from joplin_notes_ai.models import (
    NoteSummary,
    TagInfo,
    TagTaxonomyAssignment,
    TagTaxonomyPlan,
)
from joplin_notes_ai.services.tag_taxonomy import TagTaxonomyService


def make_settings() -> Settings:
    return Settings.model_validate(
        {
            "JOPLIN_TOKEN": "token",
            "LLM_API_KEY": "llm-key",
        }
    )


class TagTaxonomyServiceTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.settings = make_settings()
        self.joplin = Mock()
        self.llm = Mock()
        self.service = TagTaxonomyService(
            settings=self.settings,
            joplin_client=self.joplin,
            llm_client=self.llm,
        )
        self.tags = [
            TagInfo(id="t1", title="AI", note_count=2, sample_note_titles=["One", "Two"]),
            TagInfo(id="t2", title="artificial-intelligence", note_count=1, sample_note_titles=["One"]),
            TagInfo(id="t3", title="tmp", note_count=1, sample_note_titles=["Draft"]),
            TagInfo(id="t4", title="ai-audited", note_count=3, sample_note_titles=["Protected"]),
        ]
        self.joplin.list_tags.return_value = self.tags
        self.joplin.list_notes_by_tag.side_effect = self._list_notes_by_tag

    def _list_notes_by_tag(self, tag_id: str) -> list[NoteSummary]:
        mapping = {
            "t1": [NoteSummary(id="n1", title="One"), NoteSummary(id="n2", title="Two")],
            "t2": [NoteSummary(id="n1", title="One")],
            "t3": [NoteSummary(id="n3", title="Draft")],
            "t4": [NoteSummary(id="n4", title="Protected")],
        }
        return mapping.get(tag_id, [])

    def test_dry_run_reports_plan_without_mutations(self) -> None:
        self.llm.generate_tag_taxonomy.return_value = TagTaxonomyPlan(
            canonical_tags=["ai"],
            assignments=[
                TagTaxonomyAssignment(
                    current_title="artificial-intelligence",
                    canonical_title="AI",
                    action="merge",
                    reason="same meaning",
                )
            ],
            taxonomy_summary="Consolidated synonyms",
        )

        outcome = self.service.organize(dry_run=True)

        self.assertEqual(outcome.status, "dry_run")
        self.assertEqual(outcome.analyzed_tags, 3)
        self.assertEqual(outcome.changed_tags, 1)
        self.joplin.replace_tag_in_notes.assert_not_called()
        self.joplin.delete_tag.assert_not_called()
        call = self.llm.generate_tag_taxonomy.call_args.args
        analyzed_titles = [tag.title for tag in call[0]]
        self.assertNotIn("ai-audited", analyzed_titles)

    def test_merge_replaces_old_tag_and_deletes_source(self) -> None:
        self.llm.generate_tag_taxonomy.return_value = TagTaxonomyPlan(
            canonical_tags=["AI"],
            assignments=[
                TagTaxonomyAssignment(
                    current_title="artificial-intelligence",
                    canonical_title="AI",
                    action="merge",
                    reason="same concept",
                )
            ],
            taxonomy_summary="Merged synonyms",
        )
        self.joplin.replace_tag_in_notes.return_value = ("t1", 1)
        self.joplin.delete_tag.return_value = True

        outcome = self.service.organize(dry_run=False)

        self.assertEqual(outcome.status, "completed")
        self.joplin.replace_tag_in_notes.assert_called_once_with("t2", "AI")
        self.joplin.delete_tag.assert_called_once_with("t2")

    def test_delete_detaches_tag_from_notes(self) -> None:
        self.llm.generate_tag_taxonomy.return_value = TagTaxonomyPlan(
            canonical_tags=["AI"],
            assignments=[
                TagTaxonomyAssignment(
                    current_title="tmp",
                    canonical_title=None,
                    action="delete",
                    reason="noise tag",
                )
            ],
            taxonomy_summary="Removed noise",
        )
        self.joplin.delete_tag.return_value = True

        outcome = self.service.organize(dry_run=False)

        self.assertEqual(outcome.changed_tags, 1)
        self.joplin.detach_tag_from_note.assert_called_once_with("n3", "t3")
        self.joplin.delete_tag.assert_called_once_with("t3")


if __name__ == "__main__":
    unittest.main()
