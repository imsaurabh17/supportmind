import logging
from pathlib import Path
from typing import List, Dict, Any
from core.ingestion.pdf_loader import load_pdf
from core.ingestion.url_loader import load_url
from core.ingestion.chunker import chunk_documents
from core.vectorstore.chroma_store import add_document

logger = logging.getLogger(__name__)

def ingest(
        file_paths: List[str] = None,
        urls: List[str] = None,
        collection_name: str = "supportmind_docs",
) -> Dict[str, Any]:
    """Orchestrate the full ingestion pipeline.

    Loads PDFs and/or URLs -> chunks -> embeds -> stores in ChromaDB.

    Args:
        file_paths: List of absolute or relative paths to PDF files.
        urls: List of public URLs to load.
        collection_name: ChromaDB collection to store chunks in.

    Returns:
        Dict with keys: collection_name, total_chunks, sources.

    Raises:
        ValueError: If neither file_paths nor urls are provided.
    """
    if not file_paths and not urls:
        raise ValueError('Provide at least one file_path or URL.')
    
    file_paths = file_paths or []
    urls = urls or []
    all_documents = []

    # --- Step 1: Load ---
    logger.info("Step 1/4 - Loading documents...")

    for path in file_paths:
        logger.info(f"Loading PDF: {path}")
        docs = load_pdf(path)
        all_documents.extend(docs)

    for url in urls:
        logger.info(f"Loading URL: {url}")
        docs = load_url(url)
        all_documents.extend(docs)

    logger.info(f"Loaded {len(all_documents)} raw document (s) total.")

    # --- Step 2: Chunk ---
    logger.info("Step 2/4 - Chunking documents...")
    chunks = chunk_documents(all_documents)
    logger.info(f" Created {len(chunks)} chunks.")

    # --- Step 3: Embed (logged inside embedder) ---
    logger.info("Step 3/4 - Embedding chunks (this may take a moment)...")

    # ---Step 4: Store ---
    logger.info("Step 4/4 - Storing in ChromaDB...")
    count = add_document(chunks, collection_name=collection_name)
    logger.info(f" Stored {count} chunks in collection: {collection_name}.")

    sources = list({
        doc.metadata.get("source", "unknown")
        for doc in all_documents
    })

    result = {
        "collection_name" : collection_name,
        "total_chunks" : count,
        "sources" : sources,
    }

    logger.info(f"Ingestion complete: {result}")
    return result

def run_pipeline(
        pdf_paths: List[str] = None,
        urls: List[str] = None,
        collection_name: str = "supportmind_docs",
) -> Dict[str, Any]:
    """Alias for ingest() with pdf_paths parameter name.
    
    Used by the streamlit sidebar component.
    """

    return ingest(
        file_paths=pdf_paths,
        urls=urls,
        collection_name=collection_name,
    )