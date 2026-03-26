import logging
import os
import sys
from pathlib import Path
from core.ingestion.pipeline import ingest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)
logger = logging.getLogger(__name__)

PDF_PATHS = [
    "data/raw/razorpay_support.pdf",
]

URLS = [
    "https://www.swiggy.com/support"
]

def main():
    logger.info("Starting sample data ingestion...")

    for path in PDF_PATHS:
        if not Path(path).exists():
            logger.warning(f"PDF not found, skipping: {path}")

    existing_pdfs = [p for p in PDF_PATHS if Path(p).exists()]

    result = ingest(
        file_paths=existing_pdfs,
        urls=URLS,
        collection_name="supportmind_docs"
    )

    print("\n" + "=" * 50)
    print("✅ Ingestion Complete!")
    print(f"   Collection : {result['collection_name']}")
    print(f"   Total chunks stored : {result['total_chunks']}")
    print(f"   Sources ingested : {', '.join(result['sources'])}")
    print("=" * 50)
    print("\nTest your vectorstore with:")
    print("  python -c \"from core.vectorstore.chroma_store import similarity_search; "
          "print(similarity_search('refund policy', k=3))\"")
    
if __name__=="__main__":
    main()