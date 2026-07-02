import unittest

import _path  # noqa: F401

from voice_agent.providers.llm_foundry import ChatMessage
from voice_agent.providers.llm_llama_cpp import _parse_completion_line, _prompt_from_messages


class LlamaCppProviderTests(unittest.TestCase):
    def test_gemma_prompt_uses_turn_boundaries(self):
        prompt = _prompt_from_messages(
            [
                ChatMessage("system", "Be concise."),
                ChatMessage("system", "Context:\nlocal assistant"),
                ChatMessage("user", "你好"),
            ]
        )

        self.assertEqual(
            prompt,
            "<start_of_turn>system\nBe concise.\n<end_of_turn>\n"
            "<start_of_turn>system\nContext:\nlocal assistant\n<end_of_turn>\n"
            "<start_of_turn>user\n你好\n<end_of_turn>\n"
            "<start_of_turn>model\n",
        )

    def test_parse_llama_cpp_sse_content(self):
        self.assertEqual(_parse_completion_line('data: {"content":"你"}'), "你")
        self.assertIsNone(_parse_completion_line("data: [DONE]"))

    def test_prepare_prompt_is_prefix_of_decode_prompt(self):
        messages = [ChatMessage("system", "Be concise."), ChatMessage("user", "你好")]

        prepared_prompt = _prompt_from_messages(messages[:1], add_generation_prompt=False)
        decode_prompt = _prompt_from_messages(messages)

        self.assertTrue(decode_prompt.startswith(prepared_prompt))
        self.assertTrue(decode_prompt.endswith("<start_of_turn>model\n"))


if __name__ == "__main__":
    unittest.main()