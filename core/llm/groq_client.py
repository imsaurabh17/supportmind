import os
import logging
from functools import lru_cache
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

@lru_cache(maxsize=1)
def get_llm() -> ChatGroq:
    """Singleton: get ChatGroq LLM instance."""
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise EnvironmentError("Groq api key not found in environment variables")
    
    logger.info("Initializing Groq LLM: llama-3.1-8b-instant")
    return ChatGroq(
        model='llama-3.3-70b-versatile',
        api_key=api_key,
        temperature=0.1,
        max_tokens=1024,
        timeout=60
    )

def test_llm_collection() -> bool:
    """Quick ping to verify Groq API is reachable. Use during setup only."""
    try:
        llm = get_llm()
        response = llm.invoke([HumanMessage(content="Reply with the single work: OK")])
        logger.info(f"Groq connection test passed: {response.content}")
        return True
    except Exception as e:
        logger.error(f"Groq connection test failed: {e}")
        return False