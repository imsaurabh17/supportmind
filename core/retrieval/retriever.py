import logging
from typing import Dict, List, Any

from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.runnables import Runnable

from core.llm.groq_client import get_llm
from core.llm.prompts import build_qa_prompt, build_condense_prompt, TONE_INSTRUCTIONS
from core.vectorstore.chroma_store import get_vectorstore
from core.sentiment.detector import detect_sentiment
from core.retrieval.citation_formatter import format_citation

logger = logging.getLogger(__name__)


def build_chain(collection_name: str = 'supportmind_docs') -> Runnable:
    """Build and return a composable RAG chain using LCEL.

    Replaces the legacy ConversationalRetrievalChain with two composable parts:
    - history_aware_retriever: condenses follow-up questions using chat history
    - retrieval_chain: generates answers grounded in retrieved documents

    Args:
        collection_name: ChromaDB collection to retrieve from.

    Returns:
        A configured LCEL Runnable (retrieval chain).
    """
    logger.info(f"Building RAG chain for collection: {collection_name}")

    vectorstore = get_vectorstore(collection_name)

    retriever = vectorstore.as_retriever(
        search_type='similarity',
        search_kwargs={'k': 5},
    )

    # Handles follow-up question condensation using chat history
    history_aware_retriever = create_history_aware_retriever(
        get_llm(), retriever, build_condense_prompt()
    )

    logger.info('RAG chain components built successfully')
    return {
        'llm': get_llm(),
        'history_aware_retriever': history_aware_retriever,
    }


def ask(
        chain_component: Dict,
        question: str,
        chat_history: List = None,
) -> Dict[str, Any]:
    """Ask a question. Detects sentiment, injects tone, returns structured response.

    Tone is injected at runtime as a standard input variable — no prompt surgery needed.

    Args:
        chain: The LCEL Runnable from build_chain().
        question: The user's raw message text.
        chat_history: List of (human, ai) string tuples from previous turns.

    Returns:
        Dict with keys: answer, citations, sentiment, source_documents.
    """
    if chat_history is None:
        chat_history = []

    # Step 1: detect sentiment, resolve tone instruction
    sentiment = detect_sentiment(question)
    logger.info(f"Detected sentiment: {sentiment}")

    # Step 2: convert history tuples → LangChain message objects
    history_messages = []
    for human_msg, ai_msg in chat_history:
        history_messages.append(HumanMessage(content=human_msg))
        history_messages.append(AIMessage(content=ai_msg))

    # Rebuild document chain with correct tone for this specific message
    tone = TONE_INSTRUCTIONS[sentiment]
    document_chain = create_stuff_documents_chain(
        chain_component['llm'],
        build_qa_prompt(tone_instruction=tone)
    )
    rag_chain = create_retrieval_chain(
        chain_component['history_aware_retriever'],
        document_chain
    )

    # Step 3: invoke — tone_instruction is a standard variable, not prompt surgery
    result = rag_chain.invoke({
        'input': question,
        'chat_history': history_messages,
    })

    # Step 4: format citations from retrieved context docs
    source_docs = result.get('context', [])
    citations = format_citation(source_docs)

    return {
        'answer': result['answer'],
        'citations': citations,
        'sentiment': sentiment,
        'source_documents': source_docs,
    }


if __name__ == "__main__":
    from dotenv import load_dotenv; load_dotenv()
    from core.retrieval.retriever import build_chain, ask

    chain = build_chain()

    # Test 1: basic question
    result = ask(chain, '"How are refunds processed and how long do they take?"')
    print('=== ANSWER ===')
    print(result['answer'])
    print()
    print('=== SENTIMENT ===')
    print(result['sentiment'])
    print()
    print('=== CITATIONS ===')
    for c in result['citations']:
        print(f"[Source: {c['source']}, Page {c['page']}]")