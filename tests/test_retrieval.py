import pytest
from core.retrieval.retriever import build_chain, ask

@pytest.fixture(scope="module")
def chain_components():
    """Build chain once for all tests in this module."""
    return build_chain()

def test_chain_returns_answer_and_sources(chain_components):
    """Test that a basic question returns an answer with source documents."""
    result = ask(chain_components, "What is the refund policy?")

    assert "answer" in result, "Result must have 'answer' key"
    assert isinstance(result["answer"], str), "Answer must be a string"
    assert len(result["answer"]) > 0, "Answer must not be empty"

    assert "source_documents" in result, "Result must have 'source_documents' key"
    assert len(result["source_documents"]) > 0, "Must retrieve at least one source document"

def test_citations_are_formatted(chain_components):
    """Test that citations are extracted and formatted from source documents."""
    result = ask(chain_components, "How long does a refund take?")

    assert "citations" in result, "Result must have 'citations' key"
    assert isinstance(result["citations"], list), "Citations must be a list"

    for citation in result["citations"]:
        assert "source" in citation, "Each citation must have 'source'"
        assert "page" in citation, "Each citation must have 'page'"

def test_memory_follow_up_question(chain_components):
    """Test that follow-up questions use context from previous turn."""
    first_question = "What is the refund window?"
    first_result = ask(chain_components, first_question)
    first_answer = first_result["answer"]

    follow_up_result = ask(
        chain_components,
        "What about sale items - does the same policy apply?",
        chat_history=[(first_question, first_answer)],
    )

    assert "answer" in follow_up_result, "Follow-up must return an answer"
    assert len(follow_up_result['answer']) > 0, "Follow-up answer must not be empty"

    answer_lower = follow_up_result["answer"].lower()
    assert any(
        word in answer_lower for word in ["sale", "refund", "policy", "item"]
    ), "Follow-up answer should reference refund/sale context from memory"