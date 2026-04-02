from typing import List, Dict
from langchain_core.documents import Document

def format_citation(source_documents: List[Document]) -> List[Dict]:
    """Extract unique citations from source documents.

    Args:
        source_documents: List of Document objects returned by the RAG chain.

    Returns:
        List of dicts: [{sourcce, page, chunk_id}], deduplicated by (source, page).
    
    """
    seen = set()
    citations = []
    for doc in source_documents:
        meta = doc.metadata
        key = (meta.get('source','unknown'), meta.get('page','?'))

        if key not in seen:
            seen.add(key)
            citations.append({
                'source' : meta.get('source', 'Unknown Document'),
                'page': meta.get('page', 'N/A'),
                'chunk_id': meta.get('chunk_id', ''),
            })

    return citations
    
def format_citation_as_text(citations: List[Dict]) -> str:
    """Convert citation dicts to a readable string for display.

    Example output:
        [Source: razorpay_refund_policy.pdf, Page 2]
        [Sourcce: swiggy_faq.pdf, Page 5]
    """
    if not citations:
        return 'No sources found.'
    return '\n'.join(
        f"[Source: {c['source']}, Page {c['page']}]" 
        for c in citations
    )