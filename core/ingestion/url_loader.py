import logging
from pathlib import Path
from typing import List, Literal
from langchain_community.document_loaders import (
     UnstructuredURLLoader,
     UnstructuredHTMLLoader,
     SeleniumURLLoader
)
from langchain_core.documents import Document

logger = logging.getLogger(__name__)

LoadMode = Literal["auto","static","js","html"]

def load_url(url: str, mode: LoadMode="auto") -> List[Document]:
    """Load a web page and return list of Documents with metadata.

    Args:
        url: The full URL to load (must be publically accessible).
        mode: Loading strategy -
            'auto' -> tries first, falls back to JS if empty
            'static -> fast, no JS execution (plain HTML only)
            'js -> Selenium headless Chrome (JS-rendered pages)
            'html -> local .html filepath passed to url arg

    Returns:
        List of Document objects extracted from the page.

    Raises:
        ValueError: If the URL is empty or invalid.
        RuntimeError: If the page could not be fetched or parsed.
    """
    if not url or not url.startswith("http"):
        raise ValueError(f"Invalid URL: {url}")
    
    logger.info(f"Loading URL [{mode}]: {url}")

    if mode == "auto":
        docs = _load_static(url)

        if _is_blocked(docs):
            logger.info("Static load blocked — retrying with Selenium...")
            docs = _load_js(url)
        elif _is_shallow_content(docs):
            logger.info("Static load got shell content — retrying with Selenium...")
            docs = _load_js(url)
        else:
            logger.info("Static load got real content — Selenium not needed.")

    elif mode == "static":
        docs = _load_static(url)
    elif mode == "js":
        docs = _load_js(url)
    else:
        raise ValueError(f"Unknown mode: {mode}. Use 'auto', 'static', or 'js'.")
    
    if not docs:
        logger.warning(f"No content extracted from URL: {url}")
        return []
    
    if _is_blocked(docs):
        raise RuntimeError(f"Site blocked the request (bot protection): {url}")
    
    for i, doc in enumerate(docs):
        doc.metadata.update({
            "source": url,
            "url": url,
            "doc_type": "url",
            "page": i,
            "load_mode": mode,
        })

    logger.info(f"Loaded {len(docs)} document(s) from {url}")
    return docs

def load_html_file(file_path: str, source_name: str = None) -> List[Document]:
    """Load a locally saved HTML file as a fallback for URLs.

    Args:
        file_path: Path to the .html file saved from a browser.
        source_name: Human-readable source label for metadata.

    Returns:
        List of Document objects.

    Raises:
        FileNotFoundError: If the HTML file does not exist.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"HTML file not found: {file_path}")
    
    logger.info(f"Loading local HTML file: {path.name}")

    loader = UnstructuredHTMLLoader(str(path))
    documents = loader.load()

    for i, doc in enumerate(documents):
        doc.metadata.update({
            "source": source_name or path.name,
            "file_path": str(path.absolute()),
            "doc_type": "url",
            "page": i,
            "load_mode": "html",
        })

    logger.info(f"Loaded {len(documents)} document(s) from {path.name}")
    return documents

def _load_static(url: str) -> List[Document]:
    """Fast load - no JS execution. Works for plain HTML pages."""
    try:
        loader = UnstructuredURLLoader(
            urls=[url],
            headers = {
                "User-Agent": (
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/120.0.0.0 Safari/537.36"
                )
            }
        )
        return loader.load()
    except Exception as e:
        logger.warning(f"Static load failed for {url}: {e}")
        return []
    
def _load_js(url: str) -> List[Document]:
    """Selenium headless Chrome - works for JS-rendered pages."""
    try:
        from selenium.webdriver.chrome.options import Options
        from webdriver_manager.chrome import ChromeDriverManager

        browser_args = [
            "--headless",
            "--no-sandbox",
            "--disable-dev-shm-usage",
            "--disable-gpu",
            "--window-size=1920,1080",
            # Bypass bot detection
            "--disable-blink-features=AutomationControlled",
            "--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36",
        ]

        loader = SeleniumURLLoader(
            urls=[url],
            browser = "chrome",
            headless=True,
            arguments=browser_args,
        )
        return loader.load()
    except Exception as e:
        logger.error(f"Selenium load failed for {url}: {e}")
        return []
    
def _is_blocked(docs: List[Document]) -> bool:
    """Detect common bot-block messages in loaded content."""
    if not docs:
        return False
    content = docs[0].page_content.lower()
    block_signal = [
        "robot policy",
        "enable javascript",
        "please enable js",
        "access denied",
        "403 forbidden",
        "captcha",
        "cloudflare",
        "checking your browser",
    ]
    return any(signal in content for signal in block_signal)

def _is_shallow_content(docs: List[Document]) -> bool:
    """Detect if static load got real content or just a JS shell."""
    if not docs:
        return True
    
    content = docs[0].page_content.strip()

    if len(content) < 200:
        return True
    
    shell_signal = [
        "sign in", "login", "0cart", "search",
        "enable javascript", "loading...",
        "please wait", "checking your browser",
    ]
    signal_count = sum(1 for s in shell_signal if s in content.lower())
    if signal_count >= 2:
        return True
    
    return False

if __name__=="__main__":
    import sys
    url = sys.argv[1] if len(sys.argv) > 1 else "https://razorpay.com/docs/payments/faqs/"

    docs = load_url("https://www.swiggy.com/support", mode="auto")
    print(f"\nLoaded {len(docs)} doc(s)")
    if docs:
        print(f"First 300 chars:\n{docs[0].page_content[:300]}")