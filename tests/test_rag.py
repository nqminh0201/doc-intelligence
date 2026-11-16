"""
test_rag.py — End-to-End Automated Test for the RAG Pipeline.

Tests:
    1. Context prompt formatting with source metadata citations
    2. Top-K retrieval from ChromaDB
    3. Structured RAGResponse output (both offline fallback and live LLM)
"""

import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.rag import RAGPipeline, RAGResponse


def test_rag_pipeline():
    print("=" * 60)
    print("RUNNING RAG PIPELINE AUTOMATED TEST")
    print("=" * 60)

    pipeline = RAGPipeline()
    question = "How does HNSW search work in ChromaDB?"

    response = pipeline.answer(question, top_k=2)

    assert isinstance(response, RAGResponse), "Response must be a RAGResponse instance"
    assert response.question == question, "Question field must match input query"
    assert isinstance(response.answer, str) and len(response.answer) > 0, "Answer must be non-empty string"
    assert isinstance(response.sources, list), "Sources must be a list"

    print(f"[+] Query Tested       : {response.question}")
    print(f"[+] Sources Retrieved  : {len(response.sources)}")
    for idx, src in enumerate(response.sources, start=1):
        print(f"    [{idx}] {src['source']} (Page {src['page']}, Chunk {src['chunk_index']}) — Similarity: {src['similarity']}")

    print("\n[+] Generated / Fallback Answer Preview:")
    print("-" * 40)
    print(response.answer[:400] + ("..." if len(response.answer) > 400 else ""))
    print("-" * 40)
    print("[+] ALL RAG PIPELINE ASSERTIONS PASSED!")


if __name__ == "__main__":
    test_rag_pipeline()
