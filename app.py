"""
app.py — Streamlit Web Interface for Document Intelligence RAG.

HOW TO RUN:
    streamlit run app.py
"""

import streamlit as st
from pathlib import Path
import tempfile
import sys

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).parent))
from src.config import DATA_DIR
from src.vectorstore import VectorStore
from src.document_loader import (
    load_pdf_file,
    load_text_file,
    ingest_documents_into_vectorstore,
)
from src.rag import RAGPipeline
from src.llm import LLMClient


# ── Page Configuration ─────────────────────────────────────────────
st.set_page_config(
    page_title="Document Intelligence RAG",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("📚 Document Intelligence RAG System")
st.caption("Chat with your PDFs and documents using Local Embeddings + ChromaDB + Ollama")


# ── Initialize Session State ───────────────────────────────────────
# Streamlit re-runs the entire script on every user interaction.
# st.session_state preserves variables (chat history, pipeline objects) across re-runs!

if "vector_store" not in st.session_state:
    st.session_state.vector_store = VectorStore()

if "llm_client" not in st.session_state:
    st.session_state.llm_client = LLMClient()

if "rag_pipeline" not in st.session_state:
    st.session_state.rag_pipeline = RAGPipeline(
        vector_store=st.session_state.vector_store,
        llm_client=st.session_state.llm_client,
    )

if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        {
            "role": "assistant",
            "content": "Hello! Upload a PDF or text document in the sidebar, or ask a question about documents already in the database.",
            "sources": [],
        }
    ]


# ── Sidebar: Document Upload & Management ──────────────────────────
with st.sidebar:
    st.header("📁 Document Management")

    uploaded_files = st.file_uploader(
        "Upload documents (.pdf, .txt)",
        type=["pdf", "txt"],
        accept_multiple_files=True,
    )

    if st.button("📥 Ingest Uploaded Documents", type="primary"):
        if not uploaded_files:
            st.warning("Please select at least one file to upload.")
        else:
            total_new_chunks = 0
            with st.spinner("Processing and embedding documents..."):
                for uploaded_file in uploaded_files:
                    # Save temporarily to disk to process with pypdf
                    suffix = Path(uploaded_file.name).suffix.lower()
                    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                        tmp.write(uploaded_file.getvalue())
                        tmp_path = Path(tmp.name)

                    # Extract chunks
                    if suffix == ".pdf":
                        chunks = load_pdf_file(tmp_path)
                    else:
                        chunks = load_text_file(tmp_path)

                    # Fix metadata source to original filename
                    for c in chunks:
                        c.metadata["source"] = uploaded_file.name

                    # Ingest into ChromaDB
                    added = ingest_documents_into_vectorstore(
                        chunks, st.session_state.vector_store
                    )
                    total_new_chunks += added

                    # Clean up temp file
                    try:
                        tmp_path.unlink()
                    except Exception:
                        pass

            st.success(f"Successfully ingested {total_new_chunks} chunks into ChromaDB!")

    st.divider()

    # System Status Metrics
    st.subheader("📊 System Status")
    doc_count = st.session_state.vector_store.count()
    st.metric("Total Records in ChromaDB", doc_count)

    ollama_online = st.session_state.llm_client.is_available()
    if ollama_online:
        st.success("🟢 Local LLM (Ollama): Online")
        models = st.session_state.llm_client.list_installed_models()
        if models:
            st.caption(f"Available models: {', '.join(models)}")
    else:
        st.warning("🟡 Local LLM (Ollama): Offline")
        st.caption("Retrieved chunks will be shown in fallback mode.")

    st.divider()
    if st.button("🗑️ Clear Chat History"):
        st.session_state.chat_history = []
        st.rerun()


# ── Main Chat Interface ────────────────────────────────────────────
for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])
        if msg.get("sources"):
            with st.expander("🔍 View Retrieved Sources"):
                for i, src in enumerate(msg["sources"]):
                    st.markdown(
                        f"**Source {i + 1}:** `{src['source']}` (Page {src['page']}, Chunk {src['chunk_index']}) — *Similarity: {src['similarity']}*"
                    )
                    st.info(src["preview"])

# Chat input bar at the bottom
if user_question := st.chat_input("Ask a question about your documents..."):
    # 1. Display user message immediately
    st.session_state.chat_history.append({"role": "user", "content": user_question})
    with st.chat_message("user"):
        st.write(user_question)

    # 2. Generate response via RAG pipeline
    with st.chat_message("assistant"):
        with st.spinner("Retrieving relevant context and generating answer..."):
            response = st.session_state.rag_pipeline.answer(user_question)
            st.write(response.answer)

            if response.sources:
                with st.expander("🔍 View Retrieved Sources"):
                    for i, src in enumerate(response.sources):
                        st.markdown(
                            f"**Source {i + 1}:** `{src['source']}` (Page {src['page']}, Chunk {src['chunk_index']}) — *Similarity: {src['similarity']}*"
                        )
                        st.info(src["preview"])

    # 3. Save assistant response to session history
    st.session_state.chat_history.append(
        {
            "role": "assistant",
            "content": response.answer,
            "sources": response.sources,
        }
    )
