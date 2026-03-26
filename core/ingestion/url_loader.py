import logging
from typing import List
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain_core.documents import Document

logger = logging.getLogger(__name__)

def load_url(url: str) -> List[Document]:
    """Load a web page and return list of Documents with metadata.

    Args:
        url: The full URL to load (must be publically accessible).

    Returns:
        List of Document objects extracted from the page.

    Raises:
        ValueError: If the URL is empty or invalid.
        RuntimeError: If the page could not be fetched or parsed.
    """
    if not url or not url.startswith("http"):
        raise ValueError(f"Invalid URL: {url}")
    
    logger.info(f"Loading URL: {url}")

    try:
        loader = UnstructuredURLLoader(
            urls=[url],
            headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"}
        )
        documents = loader.load()
        
    except Exception as e:
        logger.error(f"Failed to load URL {url}: {e}")
        raise RuntimeError(f"Could not load URL: {url}") from e
    
    if not documents:
        logger.warning(f"No content extracted from URL: {url}")
        return []
    
    if "robot policy" in documents[0].page_content.lower():
            raise RuntimeError(f"Site blocked the request (bot protection): {url}")
    
    # Enrich metadata
    for i, doc in enumerate(documents):
        doc.metadata.update({
            "source": url,
            "url": url,
            "doc_type": "url",
            "page": 1,
        })

    logger.info(f"Loaded {len(documents)} document (s) from {url}")
    return documents


if __name__=="__main__":
    print(load_url("https://en.wikipedia.org/wiki/Swiggy"))