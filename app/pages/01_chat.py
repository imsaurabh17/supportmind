import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import streamlit as st
from core.retrieval.retriever import build_chain, ask
from core.sentiment.detector import SENTIMENT_EMOJI,SENTIMENT_COLOR

st.set_page_config(
    page_title ="SupportMind - Chat",
    page_icon = "💬",
    layout = "wide",
    initial_sidebar_state="expanded",
)

st.markdown('''
<style>
.stApp { background-color: #0D1117; color: #E6EDF3; }
.stChatMessage { background-color: #161B22; border-radius: 8px; }
.citation-badge {
  background: #21262D; border-left: 3px solid #58A6FF;
  padding: 8px 12px; border-radius: 4px; font-size: 0.85em;
  color: #8B949E; margin-top: 8px;
}
.sentiment-badge {
  display: inline-block; padding: 2px 10px;
  border-radius: 12px; font-size: 0.8em; font-weight: bold;
}
</style>
''', unsafe_allow_html=True)

with st.sidebar:
    st.markdown("### 🧠 SupportMind")
    st.markdown("---")
    st.markdown("💬 **Chat**")
    st.page_link("pages/02_upload.py", label="📄 Upload docs")
    st.markdown("---")

    # collection selector
    collection = st.session_state.get("collection_name", "supportmind_docs")
    st.caption(f"Active collection: `{collection}`")

    if st.button("🧹 Clear Chat History", use_container_width=True):
        st.session_state.messages = []
        st.session_state.chain = None
        st.rerun()

if "messages" not in st.session_state:
    st.session_state.messages = []

if "collection_name" not in st.session_state:
    st.session_state.collection_name = "supportmind_docs"

if st.session_state.get("chain") is None:
    with st.spinner("Loading knowledge base..."):
        st.session_state.chain = build_chain(
            st.session_state.collection_name
        )

st.markdown("### Chat with your documents")
st.caption("Ask anything about your uploaded docs. Sentiment-aware responses with citations.")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

        if msg["role"] == "assistant" and "sentiment" in msg:
            sentiment = msg["sentiment"]
            emoji = SENTIMENT_EMOJI[sentiment]
            color = SENTIMENT_COLOR[sentiment]

            st.markdown(
                f'<span class="sentiment-badge" style="background:{color}22; color:{color};">'
                f'{emoji} {sentiment}</span>',
                unsafe_allow_html=True,
            )

        if msg["role"] == "assitant" and msg.get("citations"):
            citation_text = "  |  ".join(
                f"📎 {c['source']}, Page {c['page']}"
                for c in msg["citations"]
            )
            st.markdown(
                f'<div class="citation-badge">{citation_text}</div>',
                unsafe_allow_html=True,
            )

if prompt := st.chat_input("Ask a question about your documents..."):

    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            result = ask(st.session_state.chain, prompt)

        answer = result["answer"]
        sentiment = result["sentiment"]
        citations = result["citations"]
        emoji = SENTIMENT_EMOJI[sentiment]
        color = SENTIMENT_COLOR[sentiment]

        st.markdown(answer)

        st.markdown(
            f'<span class="sentiment-badge" style="background:{color}22; color:{color};">'
            f'{emoji} {sentiment}</span>',
            unsafe_allow_html=True,
        )

        if citations:
            citation_text = "  |  ".join(
                f"📎 {c['source']}, Page {c['page']}"
                for c in citations
            )
            st.markdown(
                f'<div class="citation-badge">{citation_text}</div>',
                unsafe_allow_html=True,
            )

    # Save assitant mesage to history
    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "sentiment": sentiment,
        "citations": citations,
    })