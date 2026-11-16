"""
rag.py — The Complete Retrieval-Augmented Generation (RAG) Pipeline.

WHAT THIS FILE DOES:
    Ties all components together:
    1. User Question -> Embedded into vector
    2. VectorStore (ChromaDB) -> Retrieves top-k matching document chunks
    3. Context Builder -> Formats chunks with source file and page citations
    4. Prompt Engineering -> Instructs LLM to stay grounded and avoid hallucination
    5. Local LLM (Ollama) -> Generates fact-grounded answer with source citations
"""

from typing import Dict, Any, List, Optional
from dataclasses import dataclass

# Project modules
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.config import TOP_K_RESULTS
from src.vectorstore import VectorStore
from src.llm import LLMClient


@dataclass
class RAGResponse:
    """
    Structured output returned by the RAG pipeline.
    
    Attributes:
        question — Original user question
        answer   — Generated answer from the LLM
        sources  — List of source citations used to generate the answer
    """
    question: str
    answer: str
    sources: List[Dict[str, Any]]


class RAGPipeline:
    """
    Orchestrates retrieval from ChromaDB and generation via Ollama.
    """

    SYSTEM_PROMPT = (
        "You are an expert Document Intelligence Assistant. Your task is to answer "
        "user questions accurately using ONLY the provided context.\n\n"
        "Guidelines:\n"
        "1. Base your answer strictly on the facts in the Context. Do not invent details.\n"
        "2. If the context does not contain the answer, politely state: "
        "'I do not have enough information in the provided documents to answer this question.'\n"
        "3. Always reference the sources (document name and page/chunk) where your answer came from.\n"
        "4. Be concise, well-structured, and helpful."
    )

    def __init__(
        self,
        vector_store: Optional[VectorStore] = None,
        llm_client: Optional[LLMClient] = None
    ):
        """
        Initialize the RAG pipeline.
        """
        self.vector_store = vector_store or VectorStore()
        self.llm = llm_client or LLMClient()

    def build_context_prompt(self, query: str, retrieved_chunks: Dict[str, Any]) -> str:
        """
        Format retrieved document chunks and user question into an augmented prompt.
        """
        documents = retrieved_chunks.get("documents", [[]])[0]
        metadatas = retrieved_chunks.get("metadatas", [[]])[0]

        context_blocks = []
        for i, (doc, meta) in enumerate(zip(documents, metadatas)):
            source = meta.get("source", "Unknown Document")
            page = meta.get("page", "N/A")
            chunk_idx = meta.get("chunk_index", i)
            context_blocks.append(
                f"[Source {i + 1}: {source} (Page {page}, Chunk {chunk_idx})]\n{doc}"
            )

        formatted_context = "\n\n".join(context_blocks)

        prompt = (
            f"--- CONTEXT ---\n"
            f"{formatted_context}\n\n"
            f"--- USER QUESTION ---\n"
            f"{query}\n\n"
            f"--- ANSWER ---"
        )
        return prompt

    def answer(self, question: str, top_k: int = TOP_K_RESULTS) -> RAGResponse:
        """
        Full RAG workflow:
        1. Retrieve top_k chunks from ChromaDB
        2. Build grounded prompt
        3. Query local LLM
        4. Return structured answer + sources
        """
        # 1. Retrieve relevant chunks
        search_results = self.vector_store.search(question, top_k=top_k)
        documents = search_results.get("documents", [[]])[0]
        metadatas = search_results.get("metadatas", [[]])[0]
        distances = search_results.get("distances", [[]])[0]

        # Extract sources list
        sources = []
        for doc, meta, dist in zip(documents, metadatas, distances):
            sources.append({
                "source": meta.get("source"),
                "page": meta.get("page"),
                "chunk_index": meta.get("chunk_index"),
                "similarity": round(1 - dist, 4),
                "preview": doc[:120] + "..." if len(doc) > 120 else doc
            })

        # 2. Check if vector store has any documents
        if not documents:
            return RAGResponse(
                question=question,
                answer="No documents are currently ingested in the database. Please add documents first.",
                sources=[]
            )

        # 3. Check if Ollama is available
        if not self.llm.is_available():
            # Graceful fallback: return retrieved context so user can still see retrieval works!
            fallback_answer = (
                "[NOTE: Ollama server is not running on localhost:11434.]\n\n"
                "Here are the most relevant document chunks retrieved by ChromaDB:\n\n"
            )
            for i, src in enumerate(sources):
                fallback_answer += (
                    f"{i + 1}. Source: {src['source']} (Page {src['page']}, Chunk {src['chunk_index']})\n"
                    f"   Similarity: {src['similarity']}\n"
                    f"   Text: \"{src['preview']}\"\n\n"
                )
            fallback_answer += "Once Ollama is started, the LLM will synthesize these into a full answer."
            return RAGResponse(question=question, answer=fallback_answer, sources=sources)

        # 4. Build prompt and generate answer via LLM
        prompt = self.build_context_prompt(question, search_results)
        answer_text = self.llm.generate(
            prompt=prompt,
            system_prompt=self.SYSTEM_PROMPT,
            temperature=0.2
        )

        return RAGResponse(
            question=question,
            answer=answer_text,
            sources=sources
        )


# ── Demo / Quick Test ──────────────────────────────────────────────
def main():
    print("=" * 60)
    print("RAG PIPELINE END-TO-END DEMO")
    print("=" * 60)

    pipeline = RAGPipeline()
    question = "Why do language models hallucinate and how does RAG fix it?"

    print(f"\nUser Question:\n  \"{question}\"\n")
    response = pipeline.answer(question)

    print("=" * 60)
    print("RAG ANSWER:")
    print("=" * 60)
    print(response.answer)

    print("\n" + "=" * 60)
    print("SOURCES CITED:")
    print("=" * 60)
    for i, src in enumerate(response.sources):
        print(f"  [{i + 1}] {src['source']} (Page: {src['page']}, Chunk: {src['chunk_index']})")
        print(f"      Similarity Score: {src['similarity']}")
        print(f"      Excerpt: \"{src['preview']}\"\n")


if __name__ == "__main__":
    main()
