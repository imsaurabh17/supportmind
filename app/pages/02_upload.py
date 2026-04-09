import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import logging
import tempfile
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

from core.ingestion.pipeline import ingest

load_dotenv()
logger = logging.getLogger(__name__)

st.set_page_config(
    page_title="Upload Documents – SupportMind",
    page_icon="📄",
    layout="wide",
)

st.markdown("""
<style>
.stApp { background-color: #0D1117; color: #E6EDF3; }
.success-banner {
    background: #0F2A1D;
    border-left: 4px solid #3FB950;
    padding: 12px 16px;
    border-radius: 4px;
    color: #3FB950;
    margin-bottom: 8px;
}
.error-banner {
    background: #2A0F0F;
    border-left: 4px solid #F78166;
    padding: 12px 16px;
    border-radius: 4px;
    color: #F78166;
    margin-bottom: 8px;
}
</style>
""", unsafe_allow_html=True)

st.title("📄 Upload Documents")
st.caption("Add PDFs or URLs to your SupportMind knowledge base")

# Collection selector
collection_name = st.text_input(
    "collection name",
    value=st.session_state.get("collection_name", "supportmind_docs"),
    help="Documents are stored in named collections. Use the same in the Chat page.",
    )
st.session_state["collection_name"] = collection_name

st.divider()

tab_pdf, tab_url = st.tabs(["📁 Upload PDF", "🌐 Paste URL"])

# PDF Upload
with tab_pdf:
    st.subheader("Upload PDF files")
    uploaded_files = st.file_uploader(
        "Drag and drop PDFs here, r click to browse",
        type=["pdf"],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )

    if uploaded_files:
        st.info(f"{len(uploaded_files)} file(s) selected: "
                f"{', '.join(f.name for f in uploaded_files)}")
        
        if st.button("⚡ Ingest PDFs", type="primary", use_container_width=True):
            progress_bar = st.progress(0, text="Starting ingestion...")
            status_box=st.empty()
            results=[]

            for i, uploaded_file in enumerate(uploaded_files):
                status_box.markdown(f"**Processing:** '{uploaded_file.name}'")
                progress_bar.progress(
                    int((i / len(uploaded_files)) * 80),
                    text=f"Loading {uploaded_file.name}...",
                )

                try:
                    # Save to temp file - PyPDFLoader needs a real file path
                    with tempfile.NamedTemporaryFile(
                        delete=False, suffix=".pdf"
                    ) as tmp:
                        tmp.write(uploaded_file.read())
                        tmp_path = tmp.name

                    progress_bar.progress(
                        int((i / len(uploaded_files)) * 90),
                        text=f"Embedding {uploaded_file.name}...",
                    )

                    result = ingest(
                        file_paths=[tmp_path],
                        collection_name=collection_name,
                    )
                    count = result["total_chunks"]
                    results.append({"file": uploaded_file.name, "chunks": count, "ok": True})

                except Exception as e:
                    logger.error(f"Failed to ingest {uploaded_file.name}: {e}")
                    results.append({"file": uploaded_file.name, "error": str(e), "ok": False})

                finally:
                    Path(tmp_path).unlink(missing_ok=True)

                progress_bar.progress(100, text="Done!")
                status_box.empty()

                for r in results:
                    if r["ok"]:
                        st.markdown(
                            f'<div class="success-banner">✅ <b>{r["file"]}</b> — '
                            f'{r["chunks"]} chunks ingested into <code>{collection_name}</code></div>',
                            unsafe_allow_html=True,
                        )
                    else:
                        st.markdown(
                            f'<div class="error-banner">❌ <b>{r["file"]}</b> — {r["error"]}</div>',
                            unsafe_allow_html=True,
                        )
                
                # Reset chain so Chat page picks up new docs on next load
                st.session_state["chain"] = None
                st.success("💬 Go to the Chat page to start asking questions!")

# URL Ingestion
with tab_url:
    st.subheader("Ingest from URL")
    url_input = st.text_area(
        "Paste one URL per line",
        placeholder = "https://razorpay.com/docs/payments/\nhttps://support.swiggy.com/...",
        height=120,
    )

    if st.button("⚡ Ingest URLs", type="primary", use_container_width=True):
        urls = [u.strip() for u in url_input.splitlines() if u.strip()]

        if not urls:
            st.warning("Please paste at least one URL.")
        else:
            progress_bar = st.progress(0, text="Starting URL ingestion...")
            status_box = st.empty()

            for i, url in enumerate(urls):
                status_box.markdown(f"**Fetching:** `{url}`")
                progress_bar.progress(
                    int((i/len(urls))*90),
                    text=f"Processing {url[:60]}...",
                )

                try:
                    result = ingest(
                        urls=[url],
                        collection_name=collection_name,
                    )
                    count = result["total_chunks"]
                    st.markdown(
                        f'<div class="success-banner">✅ <b>{url}</b> — '
                        f'{count} chunks ingested into <code>{collection_name}</code></div>',
                        unsafe_allow_html=True,
                    )
                except Exception as e:
                    logger.error(f"Failed to ingest URL {url}: {e}")
                    st.markdown(
                        f'<div class="error-banner">❌ <b>{url}</b> — {str(e)}</div>',
                        unsafe_allow_html=True,
                    )

            progress_bar.progress(100, text="Done!")
            status_box.empty()
            st.session_state["chain"] = None
            st.success("💬 Go to the Chat page to start asking questions!")

# Knolwdge base status
st.divider()
st.subheader("📊 Knowledge Base Status")

try:
    from core.vectorstore.chroma_store import get_vectorstore
    store = get_vectorstore(collection_name)
    count = store._collection.count()
    col1, col2 = st.columns(2)
    col1.metric("Total chunks stored", count)
    col2.metric("Active collection", collection_name)
    if count == 0:
        st.warning("No documents ingested yet. Upload a PDF or add a URL above.")
except Exception:
    st.info("Upload documents to see knowledge base stats.")