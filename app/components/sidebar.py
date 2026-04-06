import logging
import streamlit as st
from pathlib import Path
from core.ingestion.pipeline import run_pipeline
from core.vectorstore.chroma_store import list_collections

logger = logging.getLogger(__name__)

def render_sidebar() -> None:
    """Render the sidebar with file upload, collection selector, and controls.
    
    Side effects:
        - Updates st.session_state.collection_name on collection change
        - Updates st.session_state.chain when new docs are ingested
        - Clears st.session_state.messages on 'Clear History' click
        """
    
    with st.sidebar:
        st.title("⚙️ SupportMind")
        st.divider()

        # 1. Collection selector
        _render_collection_selector()

        st.divider()

        # 2. Document uploader
        _render_uploader()

        st.divider()

        # 3. Session controls

        _render_controls()

        st.divider()

        # 4. Status info
        _render_status()

def _render_collection_selector() -> None:
    """Let the user pick which ChromaDB collection to chat with."""
    st.subheader("📚 Knowledge Base")

    try:
        store = list_collections()
        collections = store.list_collections() # returns list[str]
    except Exception as e:
        logger.warning(f"Could not load collections: {e}")
        collections = []

    if not collections:
        st.info("No collections yet. Upload a document below.")

        if "collection_name" not in st.session_state:
            st.session_state.collection_name = "supportmind_docs"
        return
    
    selected = st.selectbox(
        label = "Active collection",
        options = collections,
        index = collections.index(st.session_state.get("collection_name", collections[0]))
        if st.session_state.get("collection_name") in collections
        else 0,
        help = "Switch between different document sets",
    )

    if selected != st.session_state.get("collection_name"):
        st.session_state.collection_name = selected
        st.session_state.chain = None # force rebuild in chat page
        st.session_state.messages = []
        st.rerun()

def _render_uploader() -> None:
    """File uploader + URL input that triggers the ingestion pipeline."""
    st.subheader("📎 Add Documents")

    # PDF upload
    uploaded_files = st.file_uploader(
        label="Upload PDFs",
        type=["pdf"],
        accept_multiple_files=True,
        help="Upload one or more PDF files to ingest",
    )

    # URL input
    url_input = st.text_area(
        label="Or paste URLs (one per line)",
        placeholder="https://razorpay.com/docs/...\nhttps://swiggy.com/faq/...",
        height=80,
    )

    # Collection name for new ingestion
    new_collection = st.text_input(
        label="Save to collection",
        value=st.session_state.get("collection_name", "supportmind_docs"),
        help="Docs will be added to this collection in ChromaDB",
    )

    if st.button("🚀 Ingest Documents", use_container_width=True):
        _run_ingestion(uploaded_files, url_input, new_collection)

def _run_ingestion(uploaded_files, url_input: str, collection_name: str) -> None:
    """Save uploaded files to disk, then call the ingestion pipeline."""
    pdf_paths = []
    urls = [u.strip() for u in url_input.splitlines() if u.strip()]

    if not uploaded_files and not urls:
        st.warning("Please upload at least one PDF or paste a URL.")
        return
    
    raw_dir = Path("data/raw")
    raw_dir.mkdir(parents=True, exist_ok=True)

    for file in uploaded_files or []:
        dest = raw_dir / file.name
        dest.write_bytes(file.read())
        pdf_paths.append(str(dest))
        logger.info(f"Saved uploaded file: {dest}")

    # Run pipeline with progress bar
    progress = st.progress(0, text="Starting ingestion...")
    status = st.empty()

    try:
        steps = [
            (0.15, "📄 Loading documents…"),
            (0.40, "✂️  Chunking text…"),
            (0.65, "🧮 Generating embeddings…"),
            (0.85, "💾 Storing in ChromaDB…"),
            (1.00, "✅ Done!"),
        ]

        # Kick off pipeline (blocking - runs in same thread)
        # we update the bar at fixed checkpoints to give visual feedback
        for pct, msg in steps[:-1]:
            progress.progress(pct, text=msg)
            status.caption(msg)

        metadata = run_pipeline(
            pdf_paths=pdf_paths,
            urls=urls,
            collection_name=collection_name,
        )

        progress.progress(1.0, text=steps[-1][1])
        status.empty()

        # update session state
        st.session_state.collection_name = collection_name
        st.session_state.chain = None # force chain rebuild with new docs
        st.session_state.messages = []

        n_chunks = metadata.get("total_chunks", "?")
        st.success(f"Ingested **{n_chunks} chunks** into '{collection_name}'")
        logger.info(f"Ingestion complete: {metadata}")

        st.rerun()

    except Exception as e:
        progress.empty()
        status.empty()
        logger.error(f"Ingestion failed: {e}", exc_info=True)
        st.error(f"Ingestion failed: {e}")

def _render_controls() -> None:
    """Clear chat history button."""
    st.subheader("🧹 Session")

    if st.button('Clear Chat History', use_container_width=True):
        st.session_state.messages = []
        st.session_state.chain = None # also resets memory in the chain
        st.success("Chat history cleared.")
        st.rerun()

def _render_status() -> None:
    """Show a compact summary at the bottom of the sidebar."""
    collection = st.session_state.get("collection_name", "--")
    msg_count = len(st.session_state.get("messages", []))
    chain_ready = "Ready" if st.session_state.get("chain") else "⏳ Not loaded"

    st.caption("**Status**")
    st.caption(f"Collection: '{collection}'")
    st.caption(f"Messages: {msg_count}")
    st.caption(f"Chain: {chain_ready}")

if __name__=="__main__":
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).resolve().parent.parent))

    render_sidebar()