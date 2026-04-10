from loguru import logger

from joplin_notes_ai.clients import JoplinClient, LlmClient
from joplin_notes_ai.config import Settings
from joplin_notes_ai.models import (
    NoteSummary,
    TagInfo,
    TagOrganizationOutcome,
    TagTaxonomyAssignment,
    TagTaxonomyPlan,
)


class TagTaxonomyService:
    def __init__(
        self,
        settings: Settings,
        joplin_client: JoplinClient,
        llm_client: LlmClient,
    ) -> None:
        self._settings = settings
        self._joplin = joplin_client
        self._llm = llm_client

    def organize(self, dry_run: bool = False) -> TagOrganizationOutcome:
        protected_tags = self._protected_tags()
        tag_infos = self._collect_tag_infos(protected_tags)
        if not tag_infos:
            return TagOrganizationOutcome(
                status="skipped",
                message="Немає тегів для аналізу таксономії.",
                analyzed_tags=0,
                changed_tags=0,
            )

        plan = self._llm.generate_tag_taxonomy(tag_infos, protected_tags)
        changed_tags = self._apply_plan(plan, tag_infos, dry_run=dry_run)
        status = "dry_run" if dry_run else "completed"
        return TagOrganizationOutcome(
            status=status,
            message=plan.taxonomy_summary or "Таксономію тегів сформовано.",
            analyzed_tags=len(tag_infos),
            changed_tags=changed_tags,
        )

    def _collect_tag_infos(self, protected_tags: set[str]) -> list[TagInfo]:
        collected: list[TagInfo] = []
        for tag in self._joplin.list_tags():
            normalized = tag.title.strip().lower()
            if not normalized or normalized in protected_tags:
                continue
            notes = self._joplin.list_notes_by_tag(tag.id)
            collected.append(
                TagInfo(
                    id=tag.id,
                    title=tag.title,
                    note_count=len(notes),
                    sample_note_titles=self._sample_note_titles(notes),
                )
            )
        return collected

    def _apply_plan(
        self,
        plan: TagTaxonomyPlan,
        tags: list[TagInfo],
        dry_run: bool,
    ) -> int:
        tags_by_title = {tag.title.strip().lower(): tag for tag in tags}
        changed_tags = 0

        for assignment in plan.assignments:
            source = tags_by_title.get(assignment.current_title.strip().lower())
            if source is None:
                logger.warning("Тег '{}' з плану не знайдено серед існуючих.", assignment.current_title)
                continue

            if assignment.action == "keep":
                logger.info(
                    "tag_taxonomy_keep current_title={!r} canonical_title={!r} reason={!r}",
                    source.title,
                    assignment.canonical_title or source.title,
                    assignment.reason,
                )
                continue

            changed_tags += 1
            canonical_title = (assignment.canonical_title or "").strip()
            if dry_run:
                logger.info(
                    "tag_taxonomy_plan action={} source={!r} canonical={!r} notes={} reason={!r}",
                    assignment.action,
                    source.title,
                    canonical_title,
                    source.note_count,
                    assignment.reason,
                )
                continue

            if assignment.action == "merge" and canonical_title:
                self._merge_tag(source, canonical_title, assignment)
                continue

            if assignment.action == "delete":
                self._delete_tag(source, assignment)
                continue

            logger.warning(
                "Пропущено невідому операцію таксономії для тегу {!r}: {}",
                source.title,
                assignment.action,
            )
            changed_tags -= 1

        return changed_tags

    def _merge_tag(
        self,
        source: TagInfo,
        canonical_title: str,
        assignment: TagTaxonomyAssignment,
    ) -> None:
        target_tag_id, replaced_count = self._joplin.replace_tag_in_notes(source.id, canonical_title)
        logger.info(
            "tag_taxonomy_merge source={!r} canonical={!r} replaced_notes={} reason={!r}",
            source.title,
            canonical_title,
            replaced_count,
            assignment.reason,
        )
        if target_tag_id and target_tag_id != source.id:
            deleted = self._joplin.delete_tag(source.id)
            if not deleted:
                logger.warning("Не вдалося видалити вихідний тег {!r} після merge.", source.title)

    def _delete_tag(
        self,
        source: TagInfo,
        assignment: TagTaxonomyAssignment,
    ) -> None:
        detached_count = 0
        for note in self._joplin.list_notes_by_tag(source.id):
            self._joplin.detach_tag_from_note(note.id, source.id)
            detached_count += 1
        deleted = self._joplin.delete_tag(source.id)
        logger.info(
            "tag_taxonomy_delete source={!r} detached_notes={} deleted={} reason={!r}",
            source.title,
            detached_count,
            deleted,
            assignment.reason,
        )

    def _protected_tags(self) -> set[str]:
        return {
            self._settings.processed_tag.strip().lower(),
            self._settings.failed_tag.strip().lower(),
        }

    @staticmethod
    def _sample_note_titles(notes: list[NoteSummary]) -> list[str]:
        return [note.title for note in notes[:3] if note.title.strip()]
