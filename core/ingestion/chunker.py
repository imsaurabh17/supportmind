import hashlib
import logging
from typing import List

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

logger = logging.getLogger(__name__)

def chunk_documents(
        documents: List[Document],
        chunk_size: int = 1000,
        chunk_overlap: int = 200
) -> List[Document]:
    """Split documents into overlapping chunks with unique IDs.

    Args:
        documents: List of Document objects to split.
        chunk_size: Maximum character length of each chunk.
        chunk_overlap: Number of overlapping characters between chunks.

    Returns:
        List of chunked Document objects with enriched metadata.
    """

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len,
    )

    chunks = splitter.split_documents(documents)

    # Assign unique chunk_id to each chunk
    for i, chunk in enumerate(chunks):
        source = chunk.metadata.get("source", "unknown")
        page = chunk.metadata.get("page", 0)
        raw = f"{source}-page{page}-{chunk.page_content[:50]}"
        chunk.metadata["chunk_id"] = hashlib.md5(raw.encode()).hexdigest()[:12]

        chunk.metadata["chunk_index"] = i

    logger.info(f"Created {len(chunks)} chunks from {len(documents)} documents")
    return chunks

if __name__=="__main__":
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).resolve().parents[2]))

    from core.ingestion.pdf_loader import load_pdf

    docs = load_pdf("data/raw/razorpay_refund_policy.pdf")
    chunks = chunk_documents(docs)

    print(f"Content : {chunks[0].page_content[:120]}...")
    print(f"Metadata: {chunks[0].metadata}")