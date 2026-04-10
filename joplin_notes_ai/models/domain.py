from typing import Literal

from pydantic import BaseModel, Field


class NoteSummary(BaseModel):
    id: str
    title: str = ""


class NoteDetails(BaseModel):
    id: str
    title: str
    body: str


class Notebook(BaseModel):
    id: str
    title: str


class RelatedNote(BaseModel):
    note_id: str
    title: str
    similarity: float


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
