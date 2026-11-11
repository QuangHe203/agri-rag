import csv
import json
import tempfile
import sys
from types import ModuleType, SimpleNamespace
import unittest
from pathlib import Path
from unittest.mock import patch

from eval import compare_llm_rag
from eval.compare_llm_rag import load_dataset, parse_judgement, save_reports


class ParseJudgementTests(unittest.TestCase):
    def test_reads_valid_judgement_json(self):
        self.assertEqual(
            parse_judgement('{"winner": "A", "reason": "More accurate."}'),
            {"winner": "A", "reason": "More accurate."},
        )

    def test_reads_json_code_fence(self):
        self.assertEqual(
            parse_judgement('```json\n{"winner": "tie", "reason": "Same."}\n```'),
            {"winner": "tie", "reason": "Same."},
        )

    def test_rejects_unknown_winner(self):
        with self.assertRaises(ValueError):
            parse_judgement('{"winner": "neither", "reason": "Invalid label."}')

    def test_rejects_non_string_winner(self):
        with self.assertRaises(ValueError):
            parse_judgement('{"winner": [], "reason": "Invalid label."}')


class DatasetTests(unittest.TestCase):
    def test_requires_reference_answer(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "dataset.json"
            path.write_text(
                json.dumps({"samples": [{"question": "Question?"}]}),
                encoding="utf-8",
            )
            with self.assertRaises(SystemExit):
                load_dataset(path)


class ComparisonTests(unittest.TestCase):
    def test_maps_judge_choice_back_to_rag(self):
        class FakeLLM:
            def __init__(self):
                self.calls = 0

            def generate(self, system_prompt, user_prompt):
                self.calls += 1
                if self.calls == 1:
                    return "Plain answer"
                return '{"winner": "B", "reason": "RAG answer matches the reference."}'

        fake_llm = FakeLLM()
        gemini_module = ModuleType("app.providers.llm.gemini")
        gemini_module.get_gemini_provider = lambda: fake_llm
        rag_module = ModuleType("app.services.rag_service")
        rag_module.RAGService = lambda llm_provider: SimpleNamespace(
            answer=lambda question, top_k: SimpleNamespace(answer="RAG answer")
        )

        with (
            patch.dict(
                sys.modules,
                {
                    "app.providers.llm.gemini": gemini_module,
                    "app.services.rag_service": rag_module,
                },
            ),
            patch("eval.compare_llm_rag.random.shuffle"),
        ):
            rows = compare_llm_rag.compare_answers(
                [{"question": "Question?", "ground_truth": "Reference."}],
                top_k=5,
            )

        self.assertEqual(rows[0]["winner"], "RAG")
        self.assertEqual(rows[0]["plain_llm_answer"], "Plain answer")
        self.assertEqual(rows[0]["rag_answer"], "RAG answer")


class ReportTests(unittest.TestCase):
    def test_saves_rows_and_winner_summary(self):
        rows = [
            {"question": "Q1", "winner": "RAG"},
            {"question": "Q2", "winner": "LLM"},
            {"question": "Q3", "winner": "tie"},
        ]
        with tempfile.TemporaryDirectory() as directory:
            with patch("eval.compare_llm_rag._RESULTS_DIR", Path(directory)):
                csv_path, json_path = save_reports(rows)

            with json_path.open(encoding="utf-8") as file:
                report = json.load(file)
            with csv_path.open(encoding="utf-8-sig", newline="") as file:
                csv_rows = list(csv.DictReader(file))

        self.assertEqual(report["summary"]["wins"], {"RAG": 1, "LLM": 1, "tie": 1})
        self.assertEqual(report["summary"]["rag_win_rate"], 1 / 3)
        self.assertEqual(len(csv_rows), 3)


if __name__ == "__main__":
    unittest.main()
