from langchain_core.prompts import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
)

TONE_INSTRUCTIONS = {
    'frustrated' : (
        'The user seems frustrated or upset. Respnd with extra empathy.'
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

def build_qa_prompt() -> ChatPromptTemplate:
    """Build the main QA prompt template with tone + context injection."""
    return ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(SYSTEM_TEMPLATE),
        MessagesPlaceholder(variable_name='chat_history'),
        HumanMessagePromptTemplate.from_template('{question}'),
    ])