"""Offline RAGAS evaluation harness for the Agriculture RAG system.

What it does
------------
1. Loads a golden Q&A set (``eval/dataset.json``).
2. For every question, runs the *real* retrieval + generation pipeline
   (:class:`RetrievalService` + :class:`RAGService`) to collect the generated
   answer and the retrieved contexts.
3. Scores the run with RAGAS, using the project's own Gemini model as the
   judge LLM and the ``AITeamVN/Vietnamese_Embedding`` model for the embedding
   based metrics.
4. Prints the aggregate scores and writes a timestamped JSON + CSV report to
   ``eval/results/``.

Metrics
-------
* faithfulness        — is the answer grounded in the retrieved context?
* answer_relevancy    — does the answer address the question?
* context_precision   — are the retrieved chunks relevant (ranked well)?
* context_recall      — did retrieval cover the ground-truth answer?

Usage
-----
    pip install -r requirements.txt -r requirements-eval.txt
    # Make sure GEMINI_API_KEY is set (e.g. in .env) and documents are indexed.
    python -m eval.run_ragas
    python -m eval.run_ragas --dataset eval/dataset.json --top-k 5 --limit 20

This script makes real Gemini API calls (one per question for generation, plus
several per question for the RAGAS judge) — it will consume quota.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List

# Ensure the project root is importable when run as a plain script.
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from app.core.config import get_settings  # noqa: E402
from app.core.logging import get_logger, setup_logging  # noqa: E402

logger = get_logger(__name__)

_RESULTS_DIR = _PROJECT_ROOT / "eval" / "results"


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

def load_dataset(path: Path) -> List[Dict]:
    """Load the golden Q&A samples from *path*."""
    with path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    samples = data.get("samples", data) if isinstance(data, dict) else data
    if not samples:
        raise SystemExit(f"No samples found in {path}")
    for i, s in enumerate(samples):
        if "question" not in s:
            raise SystemExit(f"Sample #{i} is missing required field 'question'.")
    return samples


# ---------------------------------------------------------------------------
# Run the RAG pipeline to collect answers + contexts
# ---------------------------------------------------------------------------

def collect_predictions(samples: List[Dict], top_k: int) -> List[Dict]:
    """Run retrieval + generation for each sample and attach results."""
    from app.services.rag_service import RAGService

    rag = RAGService()
    rows: List[Dict] = []
    for i, s in enumerate(samples, start=1):
        question = s["question"]
        logger.info("[%d/%d] Querying: %s", i, len(samples), question)
        response = rag.answer(question=question, top_k=top_k)
        contexts = [src.excerpt for src in response.sources] or [""]
        rows.append(
            {
                "user_input": question,
                "response": response.answer,
                "retrieved_contexts": contexts,
                "reference": s.get("ground_truth", ""),
            }
        )
    return rows


# ---------------------------------------------------------------------------
# RAGAS evaluation
# ---------------------------------------------------------------------------

def build_ragas_models():
    """Wrap the project's Gemini LLM + HF embeddings for RAGAS."""
    from langchain_google_genai import ChatGoogleGenerativeAI
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas.llms import LangchainLLMWrapper

    from app.providers.embedding.vietnamese_embedding import get_embedding_model

    settings = get_settings()
    if not settings.GEMINI_API_KEY:
        raise SystemExit(
            "GEMINI_API_KEY is not configured. Set it in .env before running eval."
        )

    judge = ChatGoogleGenerativeAI(
        model=settings.GEMINI_MODEL,
        google_api_key=settings.GEMINI_API_KEY,
        temperature=0.0,  # deterministic judging
    )
    llm = LangchainLLMWrapper(judge)
    embeddings = LangchainEmbeddingsWrapper(get_embedding_model())
    return llm, embeddings


def run_evaluation(rows: List[Dict]):
    """Score *rows* with RAGAS and return the result object."""
    from ragas import EvaluationDataset, evaluate
    from ragas.metrics import (
        answer_relevancy,
        context_precision,
        context_recall,
        faithfulness,
    )

    llm, embeddings = build_ragas_models()

    # context_recall needs a reference answer; drop it if the dataset has none.
    has_reference = all(r.get("reference") for r in rows)
    metrics = [faithfulness, answer_relevancy, context_precision]
    if has_reference:
        metrics.append(context_recall)
    else:
        logger.warning(
            "Some samples have no 'ground_truth' — skipping context_recall. "
            "Add reference answers for a complete evaluation."
        )

    dataset = EvaluationDataset.from_list(rows)
    logger.info("Running RAGAS on %d samples with %d metrics...", len(rows), len(metrics))
    return evaluate(dataset=dataset, metrics=metrics, llm=llm, embeddings=embeddings)


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def save_reports(result) -> Path:
    _RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    df = result.to_pandas()
    csv_path = _RESULTS_DIR / f"ragas_{stamp}.csv"
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")

    # Aggregate mean per metric (ignore non-numeric columns).
    numeric = df.select_dtypes("number")
    summary = {col: float(numeric[col].mean()) for col in numeric.columns}
    json_path = _RESULTS_DIR / f"ragas_{stamp}.json"
    with json_path.open("w", encoding="utf-8") as fh:
        json.dump({"summary": summary, "n_samples": len(df)}, fh, ensure_ascii=False, indent=2)

    print("\n=== RAGAS summary ===")
    for metric, value in summary.items():
        print(f"  {metric:22s}: {value:.4f}")
    print(f"\nPer-sample CSV : {csv_path}")
    print(f"Summary JSON   : {json_path}")
    return json_path


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Run RAGAS evaluation on the Agriculture RAG system.")
    parser.add_argument(
        "--dataset",
        default=str(_PROJECT_ROOT / "eval" / "dataset.json"),
        help="Path to the golden Q&A JSON file.",
    )
    parser.add_argument("--top-k", type=int, default=5, help="Chunks to retrieve per question.")
    parser.add_argument("--limit", type=int, default=None, help="Evaluate only the first N samples.")
    args = parser.parse_args()

    setup_logging(level="INFO")

    samples = load_dataset(Path(args.dataset))
    if args.limit:
        samples = samples[: args.limit]
    logger.info("Loaded %d samples from %s", len(samples), args.dataset)

    rows = collect_predictions(samples, top_k=args.top_k)
    result = run_evaluation(rows)
    save_reports(result)


if __name__ == "__main__":
    main()
