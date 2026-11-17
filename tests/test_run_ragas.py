import sys
import unittest
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

from eval.run_ragas import collect_predictions


class CollectPredictionsTests(unittest.TestCase):
    def test_uses_reference_answer_from_dataset(self):
        rag_module = ModuleType("app.services.rag_service")
        rag_module.RAGService = lambda: SimpleNamespace(
            answer=lambda question, top_k: SimpleNamespace(
                answer="Generated answer",
                sources=[SimpleNamespace(excerpt="Retrieved context")],
            )
        )

        with patch.dict(sys.modules, {"app.services.rag_service": rag_module}):
            rows = collect_predictions(
                [{"question": "Question?", "reference_answer": "Reference answer."}],
                top_k=5,
            )

        self.assertEqual(rows[0]["reference"], "Reference answer.")


if __name__ == "__main__":
    unittest.main()
