import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import logging
import os
from pathlib import Path
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO)

st.set_page_config(
    page_title='SupportMind',
    page_icon='🧠',
    layout='wide',
    initial_sidebar_state='expanded',
)

# Dark theme CSS
st.markdown('''
<style>
.stApp {background-color: #0D1117; color: #E6EDF3; }
.stChatMessage {background-color: #161B22; border-radius: 8px;}
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

st.title('🧠 SupportMind')
st.caption('AI Customer Support Agent with RAG | Sentiment-Aware Responses')

if 'chain' not in st.session_state:
    st.session_state.chain = None

if 'messages' not in st.session_state:
    st.session_state.messages = []

if 'collection_name' not in st.session_state:
    st.session_state.collection_name = 'supportmind_docs'

st.info('👈 Use the sidebar to upload documents, then start chatting below.')