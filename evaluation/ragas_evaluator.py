# evaluation/ragas_evaluator.py
# Runs the full RAGAS evaluation pipeline against TEST_DATASET.
# Usage: python scripts/run_evaluation.py
# OR import and call run_evaluation() from anywhere

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
)

from core.retrieval.retriever import build_chain, ask
from evaluation.test_dataset import TEST_DATASET

logger = logging.getLogger(__name__)

def run_evaluation(collection_name: str = "supportmind_docs") -> Dict:
    """Run RAGAS evaluation on the full TEST_DATASET.
    
    Builds the RAG chain, runs every Q&A pair through it, collects
    answer + retrieved contexts, then scores them with RAGAS.

    Args:
        collection_name: ChromaDB collectio to query against.

    Returns:
        Dict with averaged metric scores:
        {
            "faithfulness": float,
            "answer_relevancy": float,
            "context_precision": float,
            "n_samples": int,
            "report_path": str,
        }
    """
    logger.info("Building RAG chain...")
    chain = build_chain(collection_name)

    questions, answers, contexts, ground_truths = [], [], [], []

    logger.info(f"Running evaluation on {len(TEST_DATASET)} test cases...")

    for i, item in enumerate(TEST_DATASET):
        logger.info(f" [{i+1}/{len(TEST_DATASET)}] {item['question'][:60]}...")

        try:
            result = ask(chain, item["question"])

            questions.append(item["question"])
            answers.append(result["answer"])
            contexts.append(
                [doc.page_content for doc in result["source_documents"]]
            )
            ground_truths.append(item["ground_truth"])

        except Exception as e:
            logger.error(f" Failed on question {i+1}: {e} ")
            # Skip failed questions rather than the full eval
            continue

    if not questions:
        raise RuntimeError("All evaluation questions failed. Check your RAG chain.")
    
    logger.info("Building RAGAS dataset...")
    dataset = Dataset.from_dict(
        {
            "question": questions,
            "answer": answers,
            "contexts": contexts,
            "ground_truth": ground_truths,
        }
    )

    logger.info("Running RAGAS scoring (this may take 2-5 minutes)...")
    scores = evaluate(
        dataset,
        metrics = [
            faithfulness,
            answer_relevancy,
            context_precision,
        ],
    )

    # Average scores across all samples
    result_dict = scores.to_pandas().mean(numeric_only=True).to_dict()

    output = {
        "faithfulness": round(result_dict.get("faiithfulness", 0.0), 4),
        "answer_relavancy": round(result_dict.get("answer_relavancy", 0.0), 4),
        "context_precision": round(result_dict.get("context_precision", 0.0), 4),
        "n_samples": len(questions),
        "collection": collection_name,
        "evaluated_at": datetime.now().isoformat(),
    }

    # Save report to evaluation/reports/
    report_dir = Path("evaluation/reports")
    report_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = report_dir / f"eval_{timestamp}.json"

    with open(report_path, "w") as f:
        json.dump(output, f, indent=2)

    output["report_path"] = str(report_path)

    logger.info("="*50)
    logger.info("RAGAS EVALUATION RESULTS")
    logger.info("=" * 50)
    logger.info(f"  Faithfulness      : {output['faithfulness']:.4f}")
    logger.info(f"  Answer Relevancy  : {output['answer_relevancy']:.4f}")
    logger.info(f"  Context Precision : {output['context_precision']:.4f}")
    logger.info(f"  Samples evaluated : {output['n_samples']}")
    logger.info(f"  Report saved to   : {report_path}")
    logger.info("=" * 50)
 
    return output

def load_latest_report() -> Dict:
    """Load the most recently saved evaluation report.
    
    Returns:
        Dict with scores from the latest report, or empty dict if none found.
    """
    report_dir = Path("evaluation/reports")

    if not report_dir.exists():
        return {}
    
    reports = sorted(report_dir.glob("eval_*.json"), reverse=True)

    if not reports:
        return {}
    
    with open(reports[0]) as f:
        return json.load(f)