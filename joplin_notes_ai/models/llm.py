from pydantic import BaseModel, Field


class TransformationResult(BaseModel):
    """Schema expected from LLM."""

    new_title: str = Field(description="Новий, осмислений та лаконічний заголовок замітки")
    content: str = Field(
        description="Повністю переписаний, структурований текст замітки у форматі Markdown"
    )
    logical_gaps: list[str] = Field(default_factory=list, description="Що було упущено в чернетці")
    further_questions: list[str] = Field(
        default_factory=list, description="Питання для розкриття теми"
    )
    target_notebook: str = Field(
        description="Точна назва існуючого блокнота, куди найбільше підходить ця замітка"
    )
    suggested_tags: list[str] = Field(
        default_factory=list, description="Список з 3-5 релевантних тегів для замітки"
    )
