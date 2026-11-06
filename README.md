# 📚 Document Intelligence RAG System

A Retrieval-Augmented Generation (RAG) system that lets you chat with your documents using local LLMs.

## What It Does

1. **Upload** PDFs or text documents
2. **Embed** them into a vector database (ChromaDB)
3. **Ask questions** in natural language
4. **Get answers** with source citations, powered by a local LLM (Ollama)

## Tech Stack

- **Python 3.15** — Core language
- **Ollama** — Local LLM inference (Mistral / Llama)
- **sentence-transformers** — Text embeddings
- **ChromaDB** — Vector database
- **SQLite** — Metadata storage
- **Streamlit** — Web UI

## Project Structure

```
doc-intelligence/
├── src/                    # Core source code
│   ├── embeddings.py       # Embedding generation
│   ├── vectorstore.py      # ChromaDB operations
│   ├── database.py         # SQLite metadata
│   ├── document_loader.py  # PDF/text loading & chunking
│   ├── llm.py              # Ollama API wrapper
│   ├── rag.py              # RAG pipeline
│   └── config.py           # Configuration
├── training/               # Model fine-tuning
├── data/documents/         # Your documents go here
├── notebooks/              # Jupyter exploration
├── tests/                  # Unit tests
└── app.py                  # Streamlit web app
```

## Setup

```bash
# Clone the repo
git clone https://github.com/YOUR_USERNAME/doc-intelligence.git
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
# Start the app
streamlit run app.py
```

## License

MIT
