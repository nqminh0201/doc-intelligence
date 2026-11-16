"""
test_loader.py — End-to-end test for document loading and vector ingestion.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.document_loader import load_documents_from_directory, ingest_documents_into_vectorstore
from src.vectorstore import VectorStore


def test_ingestion_and_search():
    print("=" * 60)
    print("TEST: INGESTING DOCUMENTS & SEMANTIC RETRIEVAL")
    print("=" * 60)

    # 1. Load all documents from data/documents/
    chunks = load_documents_from_directory()
    assert len(chunks) > 0, "Expected at least 1 chunk from documents folder"
    print(f"\n[OK] Loaded {len(chunks)} chunks from directory.")

    # 2. Ingest into ChromaDB
    store = VectorStore()
    ingest_documents_into_vectorstore(chunks, store)
    print(f"[OK] Ingested into ChromaDB. Total database records: {store.count()}")

    # 3. Test semantic query
    query = "Why do language models hallucinate and how does RAG fix it?"
    print(f"\nSearching query: \"{query}\"")
    results = store.search(query, top_k=2)

    docs = results["documents"][0]
    metas = results["metadatas"][0]
    distances = results["distances"][0]

    for i in range(len(docs)):
        similarity = 1 - distances[i]
        print(f"\n  Result #{i + 1} (similarity: {similarity:.4f}):")
        print(f"    Source: {metas[i].get('source')} (chunk {metas[i].get('chunk_index')})")
        print(f"    Text: \"{docs[i][:150]}...\"")

    print("\n" + "=" * 60)
    print("TEST PASSED SUCCESSFULLY!")
    print("=" * 60)


if __name__ == "__main__":
    test_ingestion_and_search()
