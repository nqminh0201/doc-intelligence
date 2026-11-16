# 📚 Document Intelligence RAG System

A Retrieval-Augmented Generation (RAG) system that lets you chat with your documents using local LLMs.

## What It Does

1. **Upload** PDFs or text documents
2. **Embed** them into a vector database (ChromaDB)
3. **Ask questions** in natural language
4. **Get answers** with source citations, powered by a local LLM (Ollama)

## Tech Stack

- **Python 3.12** — Core language
- **Ollama** — Local LLM inference (Llama 3.2 / Mistral)
- **sentence-transformers** — Text embeddings (`all-MiniLM-L6-v2`)
- **ChromaDB** — Persistent HNSW vector database
- **PEFT / Transformers** — LoRA fine-tuning pipeline
- **Streamlit** — Interactive Web UI

## Project Structure

```
doc-intelligence/
├── src/                        # Core source code
│   ├── embeddings.py           # Embedding generation & cosine similarity
│   ├── vectorstore.py          # ChromaDB persistent operations
│   ├── document_loader.py      # PDF/text loading & sliding-window chunking
│   ├── llm.py                  # Ollama local LLM client
│   ├── rag.py                  # End-to-end RAG pipeline
│   └── config.py               # Centralized configuration
├── training/                   # LoRA model fine-tuning
│   ├── prepare_dataset.py      # Instruction-tuning JSONL dataset builder
│   ├── train.py                # LoRA adapter training script
│   └── evaluate.py             # Base vs. LoRA perplexity & output evaluation
├── data/documents/             # Local PDF/text documents
├── notebooks/                  # Interactive Jupyter notebooks
├── tests/                      # Automated test suite
│   ├── test_loader.py          # Document chunking & ingestion test
│   └── test_rag.py             # End-to-end RAG retrieval test
└── app.py                      # Streamlit web application
```

## Setup

```bash
# Clone the repo
git clone https://github.com/nqminh0201/doc-intelligence.git
cd doc-intelligence

# Create virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS/Linux

# Install dependencies
pip install -r requirements.txt
```

## Usage

```bash
# Start the Streamlit Web UI
python -m streamlit run app.py

# Run automated tests
python tests/test_loader.py
python tests/test_rag.py

# Run LoRA Fine-Tuning & Evaluation
python training/prepare_dataset.py
python training/train.py
python training/evaluate.py
```

## License

MIT
