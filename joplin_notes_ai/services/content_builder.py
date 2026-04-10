from joplin_notes_ai.models import RelatedNote, TransformationResult


def build_related_links_block(related: list[RelatedNote], similarity_threshold: float) -> str:
    if not related:
        return ""

    lines = [
        f"- [{item.title}](:/{item.note_id}) (схожість: {round(item.similarity * 100)}%)"
        for item in related
    ]
    header = f"\n\n### Семантичні зв'язки (similarity >= {int(similarity_threshold * 100)}%)\n"
    return header + "\n".join(lines)


def build_audit_details(result: TransformationResult) -> str:
    if not result.logical_gaps and not result.further_questions:
        return ""

    parts = ["\n\n---\n### AI Audit Notes\n"]
    if result.logical_gaps:
        parts.append("**Що варто додати згодом:**\n")
        parts.extend([f"- {item}\n" for item in result.logical_gaps])
    if result.further_questions:
        parts.append("**Питання для роздумів:**\n")
        parts.extend([f"- {item}\n" for item in result.further_questions])
    return "".join(parts)


def build_final_content(
    result: TransformationResult,
    related: list[RelatedNote],
    similarity_threshold: float,
    machine_marker: str,
) -> str:
    related_block = build_related_links_block(related, similarity_threshold)
    audit_details = build_audit_details(result)
    content = result.content + related_block + audit_details
    if machine_marker and machine_marker not in content:
        content = content.rstrip() + f"\n\n{machine_marker}\n"
    return content
