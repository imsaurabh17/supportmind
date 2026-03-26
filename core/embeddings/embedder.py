import logging
from functools import lru_cache
from langchain_huggingface import HuggingFaceEmbeddings

logger = logging.getLogger(__name__)
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

@lru_cache(maxsize=1)
def get_embeddings() -> HuggingFaceEmbeddings:
    """Singleton: load embeddings model once, reuse everywhere.

    The @lru_cache ensures the 80MB model is downloaded and loaded
    only once per process, even if called from multiple modules.

    Returns:
        HuggingFaceEmbeddigs instance ready to embed text. 
    """
    logger.info(f"Loading embedding model: {MODEL_NAME}")
    embeddings = HuggingFaceEmbeddings(
        model_name=MODEL_NAME,
        model_kwargs={"device":"cpu"},
        encode_kwargs={"normalize_embeddings":True},
    )
    logger.info("Embedding model loaded successfully")
    return embeddings

if __name__=="__main__":
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).resolve().parents[2]))

    logging.basicConfig(level=logging.INFO)

    emb = get_embeddings()

    # Test 1 - shape check
    vecotr = emb.embed_query("what is the refund policy?")
    print(f"Vector shape: {len(vecotr)}")

    # Test 2 - same model returned (singleton check)
    emb2 = get_embeddings()
    print(f"Singleton works: {emb is emb2}")