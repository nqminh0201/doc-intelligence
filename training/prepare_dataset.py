"""
prepare_dataset.py — Prepare an Instruction-Tuning Dataset for LoRA Fine-Tuning.

WHAT THIS FILE DOES:
    1. Reads document chunks from your local knowledge base (or ChromaDB).
    2. Formats Question-Context-Answer triples into the standard "Instruction-Input-Output"
       chat template used to fine-tune Large Language Models.
    3. Splits the data into a Training set (80%) and a Validation/Test set (20%)
       and saves them as JSONL (.jsonl) files.

KEY CONCEPTS:
    - RAG vs. Fine-Tuning:
        * RAG gives the model *new facts* at runtime (like an open-book exam).
        * Fine-Tuning teaches the model *how to behave*, *domain terminology*, and
          *formatting style* (like studying to become a specialist).
    - Instruction Tuning Dataset:
        Models learn from examples structured as:
        ### Instruction: Answer the question based on the context...
        ### Context: <retrieved document text>
        ### Question: <user question>
        ### Response: <ideal grounded answer with citation>
    - JSONL (JSON Lines):
        The industry-standard file format for LLM datasets where each line is a
        valid, self-contained JSON object. This allows streaming huge datasets
        line-by-line without loading gigabytes into RAM at once.
"""

import json
import random
from pathlib import Path
from typing import List, Dict

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
TRAINING_DIR = Path(__file__).parent
DATA_OUTPUT_DIR = TRAINING_DIR / "data"


def get_curated_qa_examples() -> List[Dict[str, str]]:
    """
    Curated domain-specific Question-Context-Answer pairs based on our
    AI Engineering & Document Intelligence domain.

    In a large production system, you can generate thousands of these pairs
    automatically by feeding each document chunk from ChromaDB into a strong
    teacher LLM (e.g., Llama 3 / GPT-4) with the prompt:
    "Generate 3 factual question-and-answer pairs from this text chunk."
    """
    return [
        {
            "context": (
                "[Source: ai_engineering_handbook.txt (Page 1, Chunk 0)]\n"
                "Retrieval-Augmented Generation (RAG) is an architectural pattern that improves "
                "Large Language Model (LLM) responses by grounding them in external, verifiable "
                "documents. Standard LLMs suffer from two major limitations: knowledge cutoffs "
                "and hallucinations."
            ),
            "question": "What are the two main limitations of standard LLMs that RAG solves?",
            "answer": (
                "Based on the provided context, standard LLMs suffer from two main limitations: "
                "1) Knowledge cutoffs (no awareness of private or recent data after training), and "
                "2) Hallucinations (confidently generating plausible-sounding but false facts). "
                "[Source: ai_engineering_handbook.txt, Page 1, Chunk 0]"
            ),
        },
        {
            "context": (
                "[Source: ai_engineering_handbook.txt (Page 1, Chunk 1)]\n"
                "Vector embeddings are dense numerical representations of text. A neural network "
                "called an embedding model (such as all-MiniLM-L6-v2) maps sentences into a "
                "384-dimensional vector space where semantically similar sentences are positioned "
                "close together."
            ),
            "question": "How many dimensions does the all-MiniLM-L6-v2 embedding model output?",
            "answer": (
                "The all-MiniLM-L6-v2 embedding model maps sentences into a 384-dimensional "
                "vector space where texts with similar meanings are placed close together. "
                "[Source: ai_engineering_handbook.txt, Page 1, Chunk 1]"
            ),
        },
        {
            "context": (
                "[Source: ai_engineering_handbook.txt (Page 1, Chunk 2)]\n"
                "Cosine similarity measures the cosine of the angle between two vectors in high-dimensional "
                "space. A score of 1.0 indicates identical direction (meaning), 0.0 indicates orthogonal "
                "(unrelated) concepts, and -1.0 indicates opposite meaning."
            ),
            "question": "What does a cosine similarity score of 1.0 vs 0.0 mean?",
            "answer": (
                "A cosine similarity score of 1.0 indicates that two vectors point in the exact same "
                "direction (identical semantic meaning), whereas a score of 0.0 means the vectors are "
                "orthogonal (completely unrelated concepts). "
                "[Source: ai_engineering_handbook.txt, Page 1, Chunk 2]"
            ),
        },
        {
            "context": (
                "[Source: ai_engineering_handbook.txt (Page 1, Chunk 3)]\n"
                "ChromaDB uses Hierarchical Navigable Small World (HNSW) graphs to perform Approximate "
                "Nearest Neighbor (ANN) search. Instead of comparing a query against every document "
                "one by one in O(N) time, HNSW navigates a multi-layered graph in O(log N) time."
            ),
            "question": "Why is HNSW search faster than brute-force vector search?",
            "answer": (
                "HNSW (Hierarchical Navigable Small World) performs Approximate Nearest Neighbor search "
                "by navigating a multi-layered graph in O(log N) logarithmic time, rather than comparing "
                "the query against every single vector in O(N) linear time. "
                "[Source: ai_engineering_handbook.txt, Page 1, Chunk 3]"
            ),
        },
        {
            "context": (
                "[Source: ai_engineering_handbook.txt (Page 1, Chunk 4)]\n"
                "Sliding-window chunking splits long documents into smaller segments (e.g., 500 characters) "
                "with a fixed overlap (e.g., 50 characters). The overlap prevents critical context from "
                "being severed when a sentence falls across a boundary between two chunks."
            ),
            "question": "Why do we use chunk overlap when splitting documents?",
            "answer": (
                "Chunk overlap (such as 50 overlapping characters between consecutive 500-character chunks) "
                "prevents critical semantic context from being cut in half when an important sentence or idea "
                "straddles the boundary between two chunks. "
                "[Source: ai_engineering_handbook.txt, Page 1, Chunk 4]"
            ),
        },
        {
            "context": (
                "[Source: ai_engineering_handbook.txt (Page 1, Chunk 2)]\n"
                "Cosine similarity measures the cosine of the angle between two vectors in high-dimensional "
                "space. ChromaDB returns cosine distance, which is defined as Distance = 1 - Cosine Similarity."
            ),
            "question": "What is the stock price of Apple today?",
            "answer": (
                "I do not have enough information in the provided documents to answer this question."
            ),
        },
        {
            "context": (
                "[Source: architecture_guide.pdf (Page 3, Chunk 1)]\n"
                "Low-Rank Adaptation (LoRA) freezes the pre-trained model weights and injects trainable "
                "rank decomposition matrices (A and B) into the Transformer attention layers. This reduces "
                "the number of trainable parameters by up to 99%, enabling fine-tuning on consumer hardware."
            ),
            "question": "How does LoRA reduce the memory required for fine-tuning?",
            "answer": (
                "LoRA freezes the original pre-trained model weights and injects small, trainable rank "
                "decomposition matrices (A and B) into the Transformer layers, reducing trainable parameters "
                "by up to 99%. [Source: architecture_guide.pdf, Page 3, Chunk 1]"
            ),
        },
        {
            "context": (
                "[Source: architecture_guide.pdf (Page 4, Chunk 2)]\n"
                "When configuring LLM generation for RAG, a low temperature (between 0.0 and 0.2) is "
                "recommended. Temperature scales the logits before the softmax function: low temperature "
                "concentrates probability mass on the most likely tokens, reducing fabrication."
            ),
            "question": "What temperature setting should be used for RAG and why?",
            "answer": (
                "A low temperature between 0.0 and 0.2 is recommended for RAG because it concentrates "
                "probability on the highest-confidence tokens before softmax, making responses deterministic "
                "and reducing fabrication. [Source: architecture_guide.pdf, Page 4, Chunk 2]"
            ),
        },
        {
            "context": (
                "[Source: architecture_guide.pdf (Page 5, Chunk 0)]\n"
                "Streamlit executes Python scripts reactively from top to bottom on every user interaction. "
                "To prevent expensive resources like database connections and embedding models from reloading "
                "on every keystroke, developers store them in st.session_state."
            ),
            "question": "How do you prevent Streamlit from reloading the embedding model on every click?",
            "answer": (
                "You store persistent objects like database connections and models inside `st.session_state`, "
                "which preserves state across Streamlit's top-to-bottom script re-runs. "
                "[Source: architecture_guide.pdf, Page 5, Chunk 0]"
            ),
        },
        {
            "context": (
                "[Source: architecture_guide.pdf (Page 2, Chunk 1)]\n"
                "Unlike legacy static word embeddings (such as Word2Vec) where every occurrence of a word "
                "receives the exact same vector, modern Transformer encoders use self-attention to compute "
                "context-dependent representations."
            ),
            "question": "How do Transformer embeddings differ from Word2Vec?",
            "answer": (
                "Word2Vec assigns a single static vector to a word regardless of context, whereas Transformer "
                "encoders use self-attention to produce dynamic, context-dependent embeddings based on the "
                "surrounding words. [Source: architecture_guide.pdf, Page 2, Chunk 1]"
            ),
        },
    ]


def format_instruction_prompt(example: Dict[str, str]) -> str:
    """
    Formats a single Q&A dictionary into the unified prompt string that the
    Causal Language Model will learn to complete during training.
    """
    return (
        "### Instruction:\n"
        "You are an expert Document Intelligence Assistant. Answer the question accurately "
        "using ONLY the provided context, and always cite the source.\n\n"
        f"### Context:\n{example['context']}\n\n"
        f"### Question:\n{example['question']}\n\n"
        f"### Response:\n{example['answer']}"
    )


def save_jsonl(records: List[Dict[str, str]], filepath: Path) -> None:
    """Write a list of dictionaries to a JSON Lines (.jsonl) file."""
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def prepare_datasets(train_ratio: float = 0.8) -> None:
    """
    Build formatted training and validation datasets and write them to disk.
    """
    print("=" * 60)
    print("PHASE 7: PREPARING INSTRUCTION-TUNING DATASET")
    print("=" * 60)

    raw_examples = get_curated_qa_examples()

    # Format each example with the full training text + individual fields
    formatted_records = []
    for ex in raw_examples:
        formatted_records.append({
            "question": ex["question"],
            "context": ex["context"],
            "answer": ex["answer"],
            "text": format_instruction_prompt(ex),
        })

    # Shuffle deterministically with a fixed seed so results are reproducible
    random.seed(42)
    random.shuffle(formatted_records)

    split_idx = int(len(formatted_records) * train_ratio)
    train_records = formatted_records[:split_idx]
    val_records = formatted_records[split_idx:]

    train_path = DATA_OUTPUT_DIR / "train.jsonl"
    val_path = DATA_OUTPUT_DIR / "val.jsonl"

    save_jsonl(train_records, train_path)
    save_jsonl(val_records, val_path)

    print(f"[+] Total Q&A examples prepared : {len(formatted_records)}")
    print(f"[+] Training set saved          : {train_path} ({len(train_records)} examples)")
    print(f"[+] Validation set saved        : {val_path} ({len(val_records)} examples)")
    print("\n--- Sample Formatted Training Prompt ---")
    print(train_records[0]["text"])
    print("=" * 60)


if __name__ == "__main__":
    prepare_datasets()
