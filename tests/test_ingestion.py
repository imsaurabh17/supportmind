import pytest
from core.ingestion.pdf_loader import load_pdf
from core.ingestion.chunker import chunk_documents
from core.vectorstore.chroma_store import add_document, similarity_search

def test_load_pdf_raises_on_missing_file():
    with pytest.raises(FileNotFoundError):
        load_pdf("data/raw/nonexistent.pdf")

def test_chunk_documents_produces_chunks():
    docs = load_pdf("D:/supportmind/supportmind/data/raw/razorpay_refund_policy.pdf")
    chunks = chunk_documents(docs)
    assert len(chunks) > 0
    assert len(chunks) >= len(docs)
    assert all('chunk_id' in c.metadata for c in chunks)
    assert all('source' in c.metadata for c in chunks)

def test_similarity_search_returns_results():
    results = similarity_search("how to get a refund", k=3)
    assert len(results) >= 1
    assert all(len(r.page_content) > 0 for r in results)
    assert all('source' in r.metadata for r in results)