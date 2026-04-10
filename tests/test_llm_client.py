import unittest
from unittest.mock import Mock, patch

from joplin_notes_ai.clients.llm import LlmClient
from joplin_notes_ai.config import Settings
from joplin_notes_ai.exceptions import LlmResponseValidationError
from joplin_notes_ai.repositories import PromptLoader


def make_settings() -> Settings:
    return Settings.model_validate(
        {
            "JOPLIN_TOKEN": "token",
            "LLM_API_KEY": "llm-key",
            "LLM_MAX_TOKENS": 4000,
        }
    )


def make_response(content: str, finish_reason: str) -> Mock:
    response = Mock()
    response.raise_for_status.return_value = None
    response.json.return_value = {
        "choices": [
            {
                "finish_reason": finish_reason,
                "message": {"content": content},
            }
        ],
        "usage": {"prompt_tokens": 111, "completion_tokens": 222},
    }
    return response


class LlmClientTestCase(unittest.TestCase):
    @patch("joplin_notes_ai.clients.llm.requests.post")
    def test_retries_with_compact_prompt_after_truncated_json(self, post_mock: Mock) -> None:
        settings = make_settings()
        prompt_loader = Mock(spec=PromptLoader)
        prompt_loader.load.return_value = "system prompt"
        client = LlmClient(settings, prompt_loader)
        post_mock.side_effect = [
            make_response('{"new_title":"Title","content":"Body","logical_gaps":["cut', "length"),
            make_response(
                (
                    '{"new_title":"Title","content":"Body","logical_gaps":["g1"],'
                    '"further_questions":["q1"],"target_notebook":"Inbox",'
                    '"suggested_tags":["a","b","c"]}'
                ),
                "stop",
            ),
        ]

        result = client.transform_note("Draft", "Some body", ["Inbox"])

        self.assertEqual(result.new_title, "Title")
        self.assertEqual(post_mock.call_count, 2)
        retry_payload = post_mock.call_args_list[1].kwargs["json"]
        self.assertEqual(retry_payload["max_tokens"], 4000)
        self.assertIn("попередня відповідь була надто довгою", retry_payload["messages"][1]["content"])

    @patch("joplin_notes_ai.clients.llm.requests.post")
    def test_raises_error_when_invalid_json_persists(self, post_mock: Mock) -> None:
        settings = make_settings()
        prompt_loader = Mock(spec=PromptLoader)
        prompt_loader.load.return_value = "system prompt"
        client = LlmClient(settings, prompt_loader)
        post_mock.side_effect = [
            make_response('{"new_title":"Title"', "length"),
            make_response('{"new_title":"Still broken"', "stop"),
        ]

        with self.assertRaises(LlmResponseValidationError):
            client.transform_note("Draft", "Some body", ["Inbox"])


if __name__ == "__main__":
    unittest.main()
