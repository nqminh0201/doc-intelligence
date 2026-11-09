"""
vectorstore.py — Store and search embeddings using ChromaDB.

WHAT THIS FILE DOES:
    In embeddings.py, we generated vectors but they lived only in RAM.
    The moment the script ended, they were gone.

    This file connects to ChromaDB — a vector database that:
    1. Saves embeddings to disk (persistent storage)
    2. Indexes them with HNSW for fast similarity search
    3. Stores metadata alongside each vector (source file, page number, etc.)
    4. Lets you add, query, update, and delete documents

KEY CONCEPTS:
    - "Collection" = like a table in SQL. Groups related documents together.
    - "Document"   = the original text chunk.
    - "Embedding"  = the vector representation of that chunk.
    - "Metadata"   = extra info (filename, page, timestamp) stored with each chunk.
    - "ID"         = unique identifier for each chunk (like a primary key in SQL).
"""

# ── Imports ────────────────────────────────────────────────────────
import chromadb                          # The vector database
from sentence_transformers import SentenceTransformer  # To generate embeddings
from datetime import datetime            # For timestamps in metadata

# Import our centralized settings
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.config import (
    EMBEDDING_MODEL_NAME,
    CHROMA_DB_DIR,
    COLLECTION_NAME,
)


class VectorStore:
    """
    A wrapper around ChromaDB that handles:
    - Creating/connecting to a persistent vector database
    - Adding documents with their embeddings and metadata
    - Searching by semantic similarity
    - Listing, updating, and deleting documents

    Think of this class as a "smart filing cabinet" that:
    - Stores your documents
    - Understands what each document MEANS
    - Can instantly find the most relevant documents for any question
    """

    def __init__(self):
        """
        Initialize the VectorStore:
        1. Load the embedding model (for converting text → vectors)
        2. Connect to ChromaDB on disk (creates the database if it doesn't exist)
        3. Get or create a "collection" (like a table in SQL)
        """
        # ── Load the embedding model ───────────────────────────────
        # This is the same model from embeddings.py.
        # We load it once and reuse it for all operations.
        print("Loading embedding model...")
        self.model = SentenceTransformer(EMBEDDING_MODEL_NAME)
        print(f"Model '{EMBEDDING_MODEL_NAME}' loaded!")

        # ── Connect to ChromaDB ────────────────────────────────────
        # PersistentClient saves data to disk at the given path.
        # If the folder already exists, it reconnects to the existing database.
        # If not, it creates a new one.
        #
        # Alternative: chromadb.Client() — in-memory only (lost on restart)
        print(f"Connecting to ChromaDB at: {CHROMA_DB_DIR}")
        self.client = chromadb.PersistentClient(path=str(CHROMA_DB_DIR))

        # ── Get or create a collection ─────────────────────────────
        # A "collection" is like a table in a SQL database.
        # All our document chunks will go into this one collection.
        #
        # get_or_create_collection:
        #   - If a collection with this name exists → returns it
        #   - If not → creates a new one
        self.collection = self.client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"description": "Document chunks for RAG system"}
        )
        doc_count = self.collection.count()
        print(f"Collection '{COLLECTION_NAME}' ready! ({doc_count} documents stored)")

    def add_documents(self, texts: list[str], metadatas: list[dict] = None):
        """
        Add text documents to the vector database.

        Parameters:
            texts     — List of text strings to store
                        e.g., ["chunk 1 text...", "chunk 2 text...", ...]
            metadatas — Optional list of dicts with extra info per chunk
                        e.g., [{"source": "report.pdf", "page": 1}, ...]

        What happens inside:
            1. Each text is converted to a 384-dim vector by the embedding model
            2. A unique ID is generated for each chunk
            3. The text, vector, metadata, and ID are stored in ChromaDB
            4. ChromaDB automatically indexes the vectors with HNSW
        """
        if not texts:
            print("No texts to add!")
            return

        # Generate embeddings for all texts at once (batch = faster)
        print(f"Generating embeddings for {len(texts)} documents...")
        embeddings = self.model.encode(texts).tolist()
        # .tolist() converts numpy array → Python list (ChromaDB expects lists)

        # Generate unique IDs for each document
        # Format: "doc_0", "doc_1", ... offset by how many docs already exist
        existing_count = self.collection.count()
        ids = [f"doc_{existing_count + i}" for i in range(len(texts))]

        # If no metadata was provided, create default metadata
        if metadatas is None:
            metadatas = [
                {"source": "manual_input", "added_at": datetime.now().isoformat()}
                for _ in texts
            ]
        else:
            # Add timestamp to each metadata dict
            for meta in metadatas:
                meta["added_at"] = datetime.now().isoformat()

        # ── Add to ChromaDB ────────────────────────────────────────
        # This single call does everything:
        #   - Stores the original text (documents)
        #   - Stores the vector (embeddings)
        #   - Stores the metadata (metadatas)
        #   - Indexes everything for fast search
        self.collection.add(
            documents=texts,        # The original text
            embeddings=embeddings,  # The 384-dim vectors
            metadatas=metadatas,    # Extra info (source file, page, etc.)
            ids=ids,                # Unique identifiers
        )
        print(f"Added {len(texts)} documents! Total in database: {self.collection.count()}")

    def search(self, query: str, top_k: int = 3) -> dict:
        """
        Search for documents similar to a query.

        Parameters:
            query — The question or search text
            top_k — How many results to return (default: 3)

        Returns:
            A dict with keys: 'documents', 'metadatas', 'distances', 'ids'

        What happens inside:
            1. The query text is embedded into a 384-dim vector
            2. ChromaDB uses HNSW to find the top_k nearest vectors
            3. Returns the original text, metadata, and distance for each match

        Note on "distance" vs "similarity":
            ChromaDB returns DISTANCE (lower = more similar)
            This is the opposite of cosine SIMILARITY (higher = more similar)
            Distance = 1 - similarity
            So distance 0.0 = perfect match, distance 2.0 = completely opposite
        """
        print(f"Searching for: \"{query}\"")

        # Embed the query
        query_embedding = self.model.encode(query).tolist()

        # Query ChromaDB
        # ChromaDB handles the HNSW search internally — this is O(log N)!
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )

        return results

    def list_all(self) -> dict:
        """
        List all documents currently in the database.
        Useful for debugging and verification.
        """
        return self.collection.get(
            include=["documents", "metadatas"]
        )

    def delete(self, ids: list[str]):
        """
        Delete documents by their IDs.

        Parameters:
            ids — List of document IDs to delete, e.g., ["doc_0", "doc_2"]
        """
        self.collection.delete(ids=ids)
        print(f"Deleted {len(ids)} documents. Remaining: {self.collection.count()}")

    def count(self) -> int:
        """Return the total number of documents in the collection."""
        return self.collection.count()


# ── Demo: Run this file directly to see ChromaDB in action ─────────
def main():
    print("=" * 60)
    print("CHROMADB VECTOR DATABASE DEMO")
    print("=" * 60)

    # ── Step 1: Initialize the vector store ────────────────────────
    store = VectorStore()

    # ── Step 2: Add some documents ─────────────────────────────────
    # Imagine these are chunks extracted from different documents
    documents = [
        "Machine learning is a subset of artificial intelligence that "
        "enables systems to learn and improve from experience.",

        "Python is widely used in data science and machine learning "
        "because of its simple syntax and powerful libraries.",

        "Neural networks are computing systems inspired by biological "
        "neural networks in the human brain.",

        "The stock market experienced significant volatility today "
        "due to unexpected changes in interest rates.",

        "Climate change is causing rising sea levels and more frequent "
        "extreme weather events around the world.",

        "Transfer learning allows models trained on one task to be "
        "adapted for a different but related task.",
    ]

    # Metadata tells us WHERE each chunk came from
    metadatas = [
        {"source": "ai_textbook.pdf", "page": 1, "topic": "ML basics"},
        {"source": "python_guide.pdf", "page": 15, "topic": "Python for ML"},
        {"source": "ai_textbook.pdf", "page": 42, "topic": "Neural networks"},
        {"source": "finance_news.txt", "page": 1, "topic": "Finance"},
        {"source": "climate_report.pdf", "page": 7, "topic": "Climate"},
        {"source": "ai_textbook.pdf", "page": 89, "topic": "Transfer learning"},
    ]

    # Only add if the collection is empty (avoid duplicates on re-run)
    if store.count() == 0:
        store.add_documents(documents, metadatas)
    else:
        print(f"Collection already has {store.count()} documents, skipping insert.")

    # ── Step 3: Search! ────────────────────────────────────────────
    queries = [
        "How do machines learn from data?",
        "What programming language should I use for AI?",
        "What is happening with global warming?",
    ]

    for query in queries:
        print("\n" + "-" * 60)
        results = store.search(query, top_k=3)

        print(f"  Top 3 results:\n")
        for i in range(len(results["documents"][0])):
            doc = results["documents"][0][i]
            meta = results["metadatas"][0][i]
            dist = results["distances"][0][i]
            similarity = 1 - dist  # Convert distance → similarity

            print(f"    {i+1}. [similarity: {similarity:.4f}]")
            print(f"       Source: {meta.get('source', 'unknown')} (page {meta.get('page', '?')})")
            print(f"       Text: \"{doc[:80]}...\"")
            print()

    # ── Step 4: Show what's stored ─────────────────────────────────
    print("=" * 60)
    print(f"DATABASE SUMMARY: {store.count()} documents stored")
    print(f"Database location: {CHROMA_DB_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
