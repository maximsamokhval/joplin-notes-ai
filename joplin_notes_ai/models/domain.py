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
