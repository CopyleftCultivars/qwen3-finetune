from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]


def load_module():
    spec = importlib.util.spec_from_file_location(
        "score_ladder_openai", ROOT / "scripts/score_ladder_openai.py"
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class OpenAILadderScorerTests(unittest.TestCase):
    def test_payload_is_greedy_and_disables_thinking_by_default(self):
        module = load_module()
        payload = module.build_payload(
            "Qwen/Qwen3-8B",
            [{"role": "user", "content": "question"}],
            8,
            42,
            False,
            "vllm",
        )
        self.assertEqual(payload["temperature"], 0.0)
        self.assertEqual(payload["top_p"], 1.0)
        self.assertEqual(payload["seed"], 42)
        self.assertEqual(payload["chat_template_kwargs"], {"enable_thinking": False})

        ollama_payload = module.build_payload(
            "qwen3:8b",
            [{"role": "user", "content": "question"}],
            8,
            42,
            False,
            "ollama",
        )
        self.assertEqual(ollama_payload["reasoning_effort"], "none")

    def test_one_item_scores_through_chat_completion_transport(self):
        module = load_module()
        with tempfile.TemporaryDirectory() as directory:
            benchmark = Path(directory) / "benchmark.jsonl"
            benchmark.write_text(
                json.dumps(
                    {
                        "question_id": "smoke-1",
                        "question": "Which option is correct?",
                        "A": "First",
                        "B": "Second",
                        "C": "Third",
                        "D": "Fourth",
                        "answer": "C",
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            args = module.parse_args(
                [
                    "--base-url",
                    "http://127.0.0.1:11434/v1",
                    "--benchmark",
                    str(benchmark),
                    "--limit",
                    "1",
                    "--out-prefix",
                    str(Path(directory) / "smoke"),
                ]
            )
            response = {
                "model": "qwen3:8b",
                "choices": [{"message": {"role": "assistant", "content": "C"}}],
            }
            system = "You are a benchmark assistant."
            template = "Question: {question}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer:"
            extract = lambda text: text.strip() if text.strip() in "ABCD" else None
            with mock.patch.object(module, "_post_with_retries", return_value=response) as post:
                results = module.score_rows(
                    module.load_rows(benchmark, 1),
                    system,
                    template,
                    extract,
                    args,
                    Path(directory) / "smoke.jsonl",
                )
            self.assertEqual(len(results), 1)
            self.assertTrue(results[0]["correct"])
            post.assert_called_once()
            self.assertTrue(post.call_args.args[0].endswith("/v1/chat/completions"))
            self.assertEqual(post.call_args.args[1]["temperature"], 0.0)


if __name__ == "__main__":
    unittest.main()
