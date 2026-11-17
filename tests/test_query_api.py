import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

from app.api.routes import query as query_route
from app.schemas.query import EnrichedQueryResponse, QueryRequest, QueryResponse
from app.services.rag_service import RAGService


class QueryApiTests(unittest.IsolatedAsyncioTestCase):
    async def test_query_endpoint_does_not_forward_top_k_from_request(self):
        response = EnrichedQueryResponse(
            question="Question?",
            answer="Answer",
            sources=[],
            model="test-model",
        )
        run_in_threadpool = AsyncMock(return_value=response)

        with patch.object(query_route, "run_in_threadpool", run_in_threadpool):
            await query_route.query(QueryRequest.model_validate({
                "question": "Question?",
                "top_k": 2,
            }))

        run_in_threadpool.assert_awaited_once_with(
            query_route._answer_query,
            "Question?",
        )

    def test_query_request_does_not_expose_top_k(self):
        self.assertNotIn("top_k", QueryRequest.model_fields)
        self.assertNotIn("top_k", QueryRequest.model_json_schema()["properties"])

    def test_answer_query_uses_configured_default_top_k(self):
        response = QueryResponse(
            question="Question?",
            answer="Answer",
            sources=[],
            model="test-model",
        )
        rag = Mock(spec=RAGService)
        rag.answer.return_value = response

        with (
            patch.object(query_route, "_get_rag_service", return_value=rag),
            patch.object(query_route, "enrich_response", return_value=response),
            patch.object(
                query_route,
                "get_settings",
                return_value=SimpleNamespace(DEFAULT_TOP_K=5),
            ),
        ):
            query_route._answer_query("Question?")

        rag.answer.assert_called_once_with(question="Question?", top_k=5)


if __name__ == "__main__":
    unittest.main()
