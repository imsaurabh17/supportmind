import hashlib
import logging
from pathlib import Path
from typing import List

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document

logger = logging.getLogger(__name__)

def load_pdf(file_path: str) -> List[Document]:
    """Load a PDF and return list of documents with enriched metadata.

    Args:
        file_path: Absolute or relative path to the PDF file.

    Returns:
        List of Document objects, one per page.

    Raises:
        FileNotFoundError: If the PDF does not exists.
        ValueError: If the file is not a valid PDF.
    """

    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"PDF not found: {file_path}")
    
    if path.suffix.lower() != ".pdf":
        raise ValueError(f"Expected .pdf file, got: {path.suffix}")
    
    logger.info(f"Loading PDF: {path.name}")

    try:
        loader = PyPDFLoader(str(path))
        documents = loader.load()
    except Exception as e:
        logger.error(f"Failed to load PDF {file_path}: {e}")
        raise

    for i, doc in enumerate(documents):
        doc.metadata.update({
            "source": path.name.split("_", 1)[-1] if "_" in path.name else path.name,
            "file_path": str(path.absolute()),
            "page": doc.metadata.get("page", i),
            "total_pages": len(documents),
            "doc_type": "pdf",
        })

    logger.info(f"Loaded {len(documents)} pages from {path.name}")
    return documents

if __name__=="__main__":
    print(load_pdf("D:/supportmind/supportmind/data/raw/razorpay_refund_policy.pdf"))