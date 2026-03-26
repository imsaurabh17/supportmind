import logging
from pathlib import Path
from typing import List, Optional
from langchain_core.documents import Document
from langchain_chroma import Chroma
from core.embeddings.embedder import get_embeddings

logger = logging.getLogger(__name__)

CHROMA_PATH = "data/chroma_db"
DEFAULT_COLLECTION = "supportmind_docs"

def get_vectorstore(collection_name: str=DEFAULT_COLLECTION) -> Chroma:
    """Get or create a persistent Chroma vectorstore.

    Args:
        collection_name: Name of the ChromaDB collection to use.

    Returns:
        Chroma vectorstore instance connected to local storage.
    """

    Path(CHROMA_PATH).mkdir(parents=True, exist_ok=True)
    return Chroma(
        collection_name=collection_name,
        embedding_function=get_embeddings(),
        persist_directory=CHROMA_PATH,
    )

def add_document(
        documents: List[Document],
        collection_name: str = DEFAULT_COLLECTION,
) -> int:
    """Add chuncked documents to the vectorstore.

    Args:
        documents: List of Document chunks to embed and store.
        collection_name: Target collection in ChromaDB.

    Returns:
        Number of documents successfully added.

    Raises:
        Exception: If Chroma write fails.
    """

    try:
        store = get_vectorstore(collection_name)
        store.add_documents(documents)
        logger.info(f"Added {len(documents)} chunks to '{collection_name}'")
        return len(documents)
    except Exception as e:
        logger.error(f"Failed to add documents to ChromaDB:{e}")
        raise

def similarity_search(
        query: str,
        k: int = 5,
        collection_name: str = DEFAULT_COLLECTION,
) -> List[Document]:
    """Retrieve top-k semantically similar chunks for a query.

    Args:
        query: Natural language search query.
        k: Number of top results to return.
        collection_name: Collection to search in.

    Returns:
        List of top-k Document chunks, most relavant first.
    """
    store = get_vectorstore(collection_name)
    results = store.similarity_search(query, k=k)
    logger.info(f"Query '{query[:40]}...' returned {len(results)} results")
    return results

def get_collection_count(collection_name: str = DEFAULT_COLLECTION) -> int:
    """Return total number of chunks stored in a collection.

    Useful for verifying ingestion worked correctly.
    """
    store = get_vectorstore(collection_name)
    return store._collection.count()

if __name__=="__main__":
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).resolve().parents[2]))

    logging.basicConfig(level=logging.INFO)

    # Test 1: Store documents
    test_docs = [
        Document(
            page_content="Razorpay refunds take 5 to 7 business days to process.",
            metadata={"source": "test.pdf", "page": 1, "chunk_id": "abc001"},
        ),
        Document(
            page_content="To raise a refund request, go to the Razorpay dashboard and click Transactions.",
            metadata={"source": "test.pdf", "page": 2, "chunk_id": "abc002"},
        ),
        Document(
            page_content="Swiggy orders can be cancelled within 2 minutes of placing the order.",
            metadata={"source": "swiggy.pdf", "page": 1, "chunk_id": "abc003"},
        ),
    ]

    count = add_document(test_docs, collection_name="test_collection")
    print(f"Stored {count} documents")

    # Test 2: Semantic search
    results = similarity_search(
        "how long does a refund take",
        k=2,
        collection_name="test_collection",
    )
    print(f"Got {len(results)} results for refund query:")
    for i, r in enumerate(results):
        print(f"   [{i+1}] source={r.metadata['source']} | {r.page_content[:70]}...")

    # Test 3: Count check
    total = get_collection_count("test_collection")
    print(f"Total chunks in collection: {total}")

    # Test 4: Relevance check
    sources = [r.metadata["source"] for r in results]
    if "swiggy.pdf" not in sources:
        print(" Semantic search is working - Swiggy chunk correctly excluded")
    else:
        print("Check your embeddings - unrelated chunk is appearing in results")