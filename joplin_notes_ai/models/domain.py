from typing import Literal

from pydantic import BaseModel, Field


class NoteSummary(BaseModel):
    id: str
    title: str = ""


class NoteDetails(BaseModel):
    id: str
    title: str
    body: str
    parent_id: str | None = None
    created_time: int | None = None
    updated_time: int | None = None
    user_updated_time: int | None = None
    is_todo: int | None = None
    source_url: str | None = None


class Notebook(BaseModel):
    id: str
    title: str


class TagInfo(BaseModel):
    id: str
    title: str
    note_count: int = 0
    sample_note_titles: list[str] = Field(default_factory=list)


class TagTaxonomyAssignment(BaseModel):
    current_title: str
    canonical_title: str | None = None
    action: Literal["keep", "merge", "delete"]
    reason: str = ""


class TagTaxonomyPlan(BaseModel):
    canonical_tags: list[str] = Field(default_factory=list)
    assignments: list[TagTaxonomyAssignment] = Field(default_factory=list)
    taxonomy_summary: str = ""


class TagOrganizationOutcome(BaseModel):
    status: Literal["completed", "dry_run", "skipped"]
    message: str = ""
    analyzed_tags: int = 0
    changed_tags: int = 0


class RelatedNote(BaseModel):
    note_id: str
    title: str
    similarity: float


class RelatedCandidate(BaseModel):
    note_id: str
    title: str
    distance: float
    similarity: float
    accepted: bool = False
    rejection_reason: str | None = None
    rank: int


class WarmupResult(BaseModel):
    enabled: bool
    success: bool
    duration_ms: float
    degraded: bool = False
    message: str = ""


class ProcessedNoteUpdate(BaseModel):
    title: str
    body: str
    parent_id: str | None = None


class ProcessingOutcome(BaseModel):
    note_id: str
    status: Literal[
        "processed",
        "skipped_empty",
        "skipped_already_transformed",
        "skipped_llm_failed",
        "dry_run",
        "failed",
    ]
    message: str = Field(default="")
