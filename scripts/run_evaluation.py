# scripts/run_evaluation.py
# Standalone script to run RAGAS evaluation.
# Usage: python scripts/run_evaluation.py
#        make eval

import logging
import sys
from pathlib import Path

# Ensure project root is on the path when run as a script
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv
load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)

from evaluation.ragas_evaluator import run_evaluation

if __name__=="__main__":
    results = run_evaluation()

    print("\n📊 Final Scores:")
    print(f"   Faithfulness      : {results['faithfulness']:.4f}")
    print(f"   Answer Relevancy  : {results['answer_relevancy']:.4f}")
    print(f"   Context Precision : {results['context_precision']:.4f}")
    print(f"\n✅ Report saved to  : {results['report_path']}")