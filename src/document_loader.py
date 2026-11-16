"""
document_loader.py — Load, extract, and chunk real documents for RAG.

WHAT THIS FILE DOES:
    1. Loads files from disk (supports .txt and .pdf files)
    2. Extracts clean text page-by-page from PDFs
    3. Splits long text into manageable overlapping chunks
    4. Attaches rich metadata to each chunk (filename, page number, chunk index)
    5. Ingests all chunks directly into our ChromaDB VectorStore

KEY CONCEPTS:
    - Chunking: Breaking large documents into smaller pieces so embedding models
      don't truncate text and vector representations remain sharp and accurate.
    - Overlap: Leaving 50-100 characters of common context between adjacent chunks
      so ideas or sentences falling on boundaries aren't sliced in two.
    - Metadata: Tracking WHERE every piece of text came from so the LLM can cite
      exact page numbers and file names when answering questions!
"""

from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Any
from pypdf import PdfReader

# Import configuration
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.config import CHUNK_SIZE, CHUNK_OVERLAP, DATA_DIR


@dataclass
class DocumentChunk:
    """
    Represents a single chunk of text ready to be embedded.
    
    Attributes:
        text     — The chunk of text string
        metadata — Dictionary containing source info (file name, page, chunk index)
    """
    text: str
    metadata: Dict[str, Any]


def chunk_text(
    text: str,
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
    base_metadata: Dict[str, Any] = None
) -> List[DocumentChunk]:
    """
    Split a block of text into smaller overlapping chunks.
    
    Why overlap?
        If chunk 1 ends with "...password is" and chunk 2 starts with "12345...",
        neither chunk has the full answer! Overlap ensures the transition
        sentence is preserved in at least one chunk.

    Parameters:
        text          — Full text string to chunk
        chunk_size    — Target length of each chunk in characters (e.g. 500)
        chunk_overlap — Overlap between consecutive chunks in characters (e.g. 50)
        base_metadata — Dictionary with file name, page number, etc.
    
    Returns:
        A list of DocumentChunk objects
    """
    if base_metadata is None:
        base_metadata = {}

    # Clean whitespace (replace multiple spaces/newlines with single ones)
    cleaned_text = " ".join(text.split())
    if not cleaned_text:
        return []

    # If the whole text is smaller than chunk_size, return it as 1 chunk
    if len(cleaned_text) <= chunk_size:
        meta = base_metadata.copy()
        meta["chunk_index"] = 0
        meta["char_length"] = len(cleaned_text)
        return [DocumentChunk(text=cleaned_text, metadata=meta)]

    chunks = []
    start = 0
    chunk_idx = 0

    # Sliding window loop
    while start < len(cleaned_text):
        end = start + chunk_size
        chunk_slice = cleaned_text[start:end]

        # If the remaining slice is small, we're at the very end
        if len(cleaned_text) - start <= chunk_size:
            chunk_slice = cleaned_text[start:].strip()
            if chunk_slice:
                meta = base_metadata.copy()
                meta["chunk_index"] = chunk_idx
                meta["char_length"] = len(chunk_slice)
                chunks.append(DocumentChunk(text=chunk_slice, metadata=meta))
            break

        # Try to avoid cutting words in half if we are not at the very end
        last_space = chunk_slice.rfind(" ")
        if last_space > int(chunk_size * 0.75):
            end = start + last_space
            chunk_slice = cleaned_text[start:end]

        chunk_slice = chunk_slice.strip()
        if chunk_slice:
            meta = base_metadata.copy()
            meta["chunk_index"] = chunk_idx
            meta["char_length"] = len(chunk_slice)
            chunks.append(DocumentChunk(text=chunk_slice, metadata=meta))
            chunk_idx += 1

        # Advance window forward by (chunk_length - overlap)
        step = len(chunk_slice) - chunk_overlap
        if step <= 0:
            step = len(chunk_slice)
        start += step

    return chunks


def load_text_file(filepath: Path) -> List[DocumentChunk]:
    """
    Read a .txt or .md file and split it into chunks.
    """
    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
        content = f.read()

    base_metadata = {
        "source": filepath.name,
        "file_type": "text",
        "page": 1,  # Plain text files are treated as page 1
    }

    return chunk_text(content, base_metadata=base_metadata)


def load_pdf_file(filepath: Path) -> List[DocumentChunk]:
    """
    Read a .pdf file page-by-page using pypdf and chunk each page individually.
    Preserves exact page numbers in the metadata!
    """
    chunks = []
    reader = PdfReader(str(filepath))
    total_pages = len(reader.pages)

    print(f"Reading '{filepath.name}' ({total_pages} pages)...")

    for page_idx, page in enumerate(reader.pages):
        page_number = page_idx + 1
        page_text = page.extract_text() or ""

        if not page_text.strip():
            continue  # Skip blank pages or pages with only images

        base_metadata = {
            "source": filepath.name,
            "file_type": "pdf",
            "page": page_number,
            "total_pages": total_pages,
        }

        page_chunks = chunk_text(page_text, base_metadata=base_metadata)
        chunks.extend(page_chunks)

    return chunks


def load_documents_from_directory(dir_path: Path = DATA_DIR) -> List[DocumentChunk]:
    """
    Scan a directory for all .txt and .pdf files, extract their text,
    and return a flat list of all DocumentChunks.
    """
    if not dir_path.exists():
        print(f"Directory '{dir_path}' does not exist. Creating it.")
        dir_path.mkdir(parents=True, exist_ok=True)
        return []

    all_chunks = []
    supported_extensions = [".txt", ".pdf", ".md"]

    # Gather all files with supported extensions
    files = [
        f for f in dir_path.iterdir()
        if f.is_file() and f.suffix.lower() in supported_extensions
    ]

    if not files:
        print(f"No documents found in '{dir_path}'. Supported formats: {supported_extensions}")
        return []

    print(f"Found {len(files)} document(s) in '{dir_path}':")
    for file in files:
        print(f"  - {file.name}")
        if file.suffix.lower() in [".txt", ".md"]:
            chunks = load_text_file(file)
        elif file.suffix.lower() == ".pdf":
            chunks = load_pdf_file(file)
        else:
            continue

        print(f"    -> Extracted {len(chunks)} chunks")
        all_chunks.extend(chunks)

    print(f"\nTotal chunks extracted across all documents: {len(all_chunks)}")
    return all_chunks


def ingest_documents_into_vectorstore(
    chunks: List[DocumentChunk],
    vector_store: Any
) -> int:
    """
    Take a list of DocumentChunks and insert them into the ChromaDB VectorStore.
    
    Returns:
        Number of chunks added
    """
    if not chunks:
        print("No chunks to ingest!")
        return 0

    texts = [c.text for c in chunks]
    metadatas = [c.metadata for c in chunks]

    vector_store.add_documents(texts=texts, metadatas=metadatas)
    return len(chunks)


# ── Demo / Self-Test ────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("DOCUMENT LOADER & CHUNKING TEST")
    print("=" * 60)

    # 1. Test chunking logic with sample text
    sample_article = (
        "Artificial intelligence is transforming industries across the globe. "
        "From healthcare diagnostics to financial fraud detection, machine learning "
        "models are augmenting human decision-making with unprecedented speed. "
        "However, one of the primary limitations of large language models is hallucination, "
        "where a model confidently states inaccurate information. "
        "Retrieval-Augmented Generation (RAG) addresses this issue directly. "
        "By retrieving verifiable facts from a private knowledge base before generating "
        "answers, RAG systems ensure that model outputs remain grounded in truth and "
        "traceable back to original source documents. Vector databases like ChromaDB "
        "and embedding models like all-MiniLM-L6-v2 make this retrieval process "
        "efficient and accurate at scale."
    )

    print("\n--- 1. Testing Sliding Window Chunking ---")
    chunks = chunk_text(
        sample_article,
        chunk_size=200,      # Small chunk size to demonstrate multiple chunks
        chunk_overlap=30,    # 30 chars overlap
        base_metadata={"source": "sample_ai_overview.txt", "author": "Demo"}
    )

    print(f"Total chunks created: {len(chunks)}\n")
    for i, chunk in enumerate(chunks):
        print(f"Chunk #{i + 1} (length: {len(chunk.text)} chars):")
        print(f"  Metadata: {chunk.metadata}")
        print(f"  Text: \"{chunk.text}\"\n")

    # 2. Test directory scanning
    print("--- 2. Scanning documents folder ---")
    doc_chunks = load_documents_from_directory(DATA_DIR)
    print(f"Chunks found in {DATA_DIR}: {len(doc_chunks)}")


if __name__ == "__main__":
    main()
