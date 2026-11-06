"""
Configuration for the Document Intelligence RAG System.

This file centralizes all settings so you can tweak the system
from one place instead of hunting through multiple files.
"""

from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "documents"
CHROMA_DB_DIR = PROJECT_ROOT / "chroma_db"
SQLITE_DB_PATH = PROJECT_ROOT / "metadata.db"

# ── Embedding Model ───────────────────────────────────────────────
# This model converts text into 384-dimensional vectors.
# It's small (~80MB) and runs fast on CPU.
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
EMBEDDING_DIMENSION = 384

# ── ChromaDB ──────────────────────────────────────────────────────
COLLECTION_NAME = "documents"

# ── Document Chunking ─────────────────────────────────────────────
CHUNK_SIZE = 500        # Characters per chunk
CHUNK_OVERLAP = 50      # Overlap between chunks (for context continuity)

# ── LLM (Ollama) ─────────────────────────────────────────────────
OLLAMA_MODEL = "mistral"    # Change to "llama3.2" if you prefer
OLLAMA_BASE_URL = "http://localhost:11434"

# ── RAG Settings ──────────────────────────────────────────────────
TOP_K_RESULTS = 3       # Number of chunks to retrieve per query
