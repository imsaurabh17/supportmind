from langchain_core.prompts import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)

TONE_INSTRUCTIONS = {
    'frustrated' : (
        'The user seems frustrated or upset. Respond with extra empathy.'
        'start with acknowledgment (e.g., "I completely understand your frustration.").'
        'Be step-by-step and thorogh. Do not be dismissive.'
    ),
    'neutral' : (
        'Respond in a professional, clear, and helpful tone.'
    ),
    'happy': (
        'The user seems happy. match their energy - be warm, friendly, and concise.'
    ),
}

SYSTEM_TEMPLATE = """You are SupportMind, an AI customer support agent.
Use ONLY the context provided below to answer questions.
If the answer is not in the context, say: 'I don't have that information in my \
knowledge base. Please contact support directly.'

TONE INSTRUCTION: {tone_instruction}

CONTEXT:
{context}

CITATION RULE: Always end your response with a 'Sources:' section listing the
document names and page numbers you used. Format: [Source: filename, Page X]
"""

def build_qa_prompt(tone_instruction: str = "") -> ChatPromptTemplate:
    """Build the main QA prompt template with tone + context injection."""
    system = f"""You are SupportMind, an AI customer support agent.
    Use ONLY the context provided below to answer questions.
    If the answer is not in the context, say: 'I don't have that information in my \
        knowledge base. Please contact support directly.'
        
        TONE INSTRUCTION: {tone_instruction}

CONTEXT:
{{context}}

CITATION RULE: Always end your response with a 'Sources:' section listing the
document names and page numbers you used. Format: [Source: filename, Page X]
"""
    return ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(system),
        HumanMessagePromptTemplate.from_template('{input}'),
    ])


CONDENSE_TEMPLATE = """Given the following conversation history and a follow-up question,
rephrase the follow-up question to be a standalone question.

Chat History:
{chat_history}

Follow Up Question: {input}
Standalone question:"""

def build_condense_prompt() -> ChatPromptTemplate:
    """Condense prompt - handles chat history as plain text, no type issues."""
    return ChatPromptTemplate.from_template(CONDENSE_TEMPLATE)