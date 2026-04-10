import time

import requests
from loguru import logger

from joplin_notes_ai.config import Settings
from joplin_notes_ai.exceptions import JoplinApiError
from joplin_notes_ai.models import Notebook, NoteDetails, NoteSummary, ProcessedNoteUpdate, TagInfo


class JoplinClient:
    def __init__(self, settings: Settings):
        self._settings = settings

    def _request(
        self,
        endpoint: str,
        method: str = "get",
        data: dict | None = None,
        params: dict | None = None,
    ) -> dict | None:
        url = f"{self._settings.joplin_base_url}/{endpoint}"
        query_params = {"token": self._settings.joplin_token}
        if params:
            query_params.update(params)

        last_exc: Exception | None = None
        for attempt in range(self._settings.max_retries + 1):
            try:
                response = requests.request(
                    method=method.upper(),
                    url=url,
                    params=query_params,
                    json=data,
                    timeout=self._settings.request_timeout,
                )

                if response.status_code == 500:
                    logger.debug(f"Joplin 500 на {endpoint}. Можливо, об'єкт вже існує.")
                    return None

                response.raise_for_status()
                return response.json() if response.content else {"status": "ok"}
            except requests.RequestException as exc:
                last_exc = exc
                if attempt >= self._settings.max_retries:
                    break
                sleep_time = self._settings.retry_backoff_seconds * (attempt + 1)
                logger.warning(
                    f"Joplin запит не вдався (спроба {attempt + 1}), повтор через {sleep_time:.1f}с"
                )
                time.sleep(sleep_time)

        raise JoplinApiError(f"API Joplin Error ({endpoint}): {last_exc}")

    def list_notebooks(self) -> list[Notebook]:
        notebooks: list[Notebook] = []
        page = 1
        while True:
            res = self._request("folders", params={"page": page, "fields": "id,title"})
            if not res or not res.get("items"):
                break
            for item in res["items"]:
                notebooks.append(Notebook(id=item["id"], title=item["title"]))
            if not res.get("has_more"):
                break
            page += 1
        return notebooks

    def search_unprocessed_notes(self, processed_tag: str) -> list[NoteSummary]:
        notes: list[NoteSummary] = []
        page = 1
        query = f"-tag:{processed_tag} type:note"
        while True:
            res = self._request(
                "search",
                params={"query": query, "page": page, "fields": "id,title"},
            )
            if not res or not res.get("items"):
                break
            notes.extend(NoteSummary(**item) for item in res["items"])
            if not res.get("has_more"):
                break
            page += 1
        return notes

    def get_note(self, note_id: str) -> NoteDetails | None:
        fields = (
            "id,title,body,parent_id,created_time,updated_time,"
            "user_updated_time,is_todo,source_url"
        )
        res = self._request(f"notes/{note_id}", params={"fields": fields})
        if not res:
            return None
        return NoteDetails(
            id=note_id,
            title=res.get("title", ""),
            body=res.get("body", ""),
            parent_id=res.get("parent_id"),
            created_time=res.get("created_time"),
            updated_time=res.get("updated_time"),
            user_updated_time=res.get("user_updated_time"),
            is_todo=res.get("is_todo"),
            source_url=res.get("source_url"),
        )

    def list_notes_for_indexing(self) -> list[NoteDetails]:
        notes: list[NoteDetails] = []
        page = 1
        fields = (
            "id,title,body,parent_id,created_time,updated_time,"
            "user_updated_time,is_todo,source_url"
        )
        while True:
            res = self._request("notes", params={"page": page, "fields": fields})
            if not res or not res.get("items"):
                break
            for item in res["items"]:
                notes.append(
                    NoteDetails(
                        id=item["id"],
                        title=item.get("title", ""),
                        body=item.get("body", ""),
                        parent_id=item.get("parent_id"),
                        created_time=item.get("created_time"),
                        updated_time=item.get("updated_time"),
                        user_updated_time=item.get("user_updated_time"),
                        is_todo=item.get("is_todo"),
                        source_url=item.get("source_url"),
                    )
                )
            if not res.get("has_more"):
                break
            page += 1
        return notes

    def ensure_tag_exists(self, tag_title: str) -> str | None:
        for tag in self.list_tags():
            if tag.title.lower() == tag_title.lower():
                return tag.id

        res = self._request("tags", method="post", data={"title": tag_title})
        return res.get("id") if res else None

    def list_tags(self) -> list[TagInfo]:
        tags: list[TagInfo] = []
        page = 1
        while True:
            res = self._request("tags", params={"page": page, "fields": "id,title"})
            if not res or not res.get("items"):
                break
            tags.extend(
                TagInfo(
                    id=item["id"],
                    title=item.get("title", ""),
                )
                for item in res["items"]
            )
            if not res.get("has_more"):
                break
            page += 1
        return tags

    def list_notes_by_tag(self, tag_id: str) -> list[NoteSummary]:
        notes: list[NoteSummary] = []
        page = 1
        while True:
            res = self._request(
                f"tags/{tag_id}/notes",
                params={"page": page, "fields": "id,title"},
            )
            if not res or not res.get("items"):
                break
            notes.extend(
                NoteSummary(
                    id=item["id"],
                    title=item.get("title", ""),
                )
                for item in res["items"]
            )
            if not res.get("has_more"):
                break
            page += 1
        return notes

    def attach_tag_to_note(self, note_id: str, tag_id: str) -> bool:
        res = self._request(f"tags/{tag_id}/notes", method="post", data={"id": note_id})
        return bool(res)

    def detach_tag_from_note(self, note_id: str, tag_id: str) -> bool:
        res = self._request(f"tags/{tag_id}/notes/{note_id}", method="delete")
        return bool(res)

    def delete_tag(self, tag_id: str) -> bool:
        res = self._request(f"tags/{tag_id}", method="delete")
        return bool(res)

    def add_tag_to_note_by_title(self, note_id: str, tag_title: str) -> bool:
        tag_id = self.ensure_tag_exists(tag_title)
        if not tag_id:
            return False
        return self.attach_tag_to_note(note_id, tag_id)

    def replace_tag_in_notes(
        self,
        source_tag_id: str,
        target_tag_title: str,
    ) -> tuple[str | None, int]:
        target_tag_id = self.ensure_tag_exists(target_tag_title)
        if not target_tag_id:
            return None, 0
        if target_tag_id == source_tag_id:
            return target_tag_id, 0

        replaced = 0
        for note in self.list_notes_by_tag(source_tag_id):
            self.attach_tag_to_note(note.id, target_tag_id)
            self.detach_tag_from_note(note.id, source_tag_id)
            replaced += 1
        return target_tag_id, replaced

    def get_note_tag_titles(self, note_id: str) -> set[str]:
        res = self._request(f"notes/{note_id}/tags", params={"fields": "title"})
        if not res or "items" not in res:
            return set()
        return {
            item["title"].strip().lower()
            for item in res["items"]
            if item.get("title") and item["title"].strip()
        }

    def update_note(self, note_id: str, update: ProcessedNoteUpdate) -> bool:
        payload = update.model_dump(exclude_none=True)
        res = self._request(f"notes/{note_id}", method="put", data=payload)
        return bool(res)
