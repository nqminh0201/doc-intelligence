"""
embeddings.py — Learn what embeddings are by seeing them in action.

WHAT THIS FILE DOES:
    1. Loads a pre-trained embedding model (all-MiniLM-L6-v2)
    2. Converts sentences into vectors (lists of 384 numbers)
    3. Compares sentences by measuring how similar their vectors are

KEY CONCEPT:
    An "embedding" is a way to represent text as numbers so that
    similar meanings → similar numbers. This lets computers understand
    that "cat" and "kitten" are related, even though the letters are different.
"""

# ── Imports ────────────────────────────────────────────────────────
# SentenceTransformer: the class that loads an embedding model and
#   converts text → vectors. Think of it as a "meaning calculator."
from sentence_transformers import SentenceTransformer, util

# torch: PyTorch, the deep learning framework running under the hood.
#   We only use it here for the similarity calculation.
import torch


def main():
    # ── Step 1: Load the embedding model ───────────────────────────
    # "all-MiniLM-L6-v2" is a small, fast model (~80MB) that produces
    # 384-dimensional vectors. It was trained on over 1 billion sentence
    # pairs to learn what "similar meaning" looks like.
    #
    # The first time you run this, it downloads the model from Hugging Face.
    # After that, it's cached locally.
    print("Loading embedding model...")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    print("Model loaded!\n")

    # ── Step 2: Define some example sentences ──────────────────────
    # We'll use these to see how embeddings capture meaning.
    # Notice: sentences 1 & 2 mean similar things, sentence 3 is different.
    sentences = [
        "The cat sat on the mat",           # About a cat
        "A kitten is resting on a rug",     # Also about a cat (different words!)
        "The stock market crashed today",    # Completely different topic
        "Python is a programming language",  # Tech topic
        "I love coding in Python",          # Also tech, related to Python
    ]

    # ── Step 3: Generate embeddings ────────────────────────────────
    # model.encode() takes a list of strings and returns a list of vectors.
    # Each vector is a list of 384 numbers (floats between -1 and 1).
    #
    # What happens inside:
    #   1. The text is split into tokens (sub-words)
    #   2. Tokens pass through a neural network (transformer)
    #   3. The output is averaged into a single 384-dim vector
    print("Generating embeddings...")
    embeddings = model.encode(sentences, convert_to_tensor=True)
    # convert_to_tensor=True → gives us a PyTorch tensor (efficient math)

    # Let's see what an embedding looks like!
    print(f"Number of sentences: {len(sentences)}")
    print(f"Embedding shape: {embeddings.shape}")
    # Shape will be (5, 384) → 5 sentences, each with 384 numbers
    print(f"\nFirst 10 numbers of sentence 1's embedding:")
    print(f"  {embeddings[0][:10].tolist()}")
    # These numbers look random, but they encode the MEANING of the sentence!

    # ── Step 4: Compare similarities ───────────────────────────────
    # "Cosine similarity" measures the angle between two vectors.
    #   - 1.0 = identical meaning
    #   - 0.0 = completely unrelated
    #   - -1.0 = opposite meaning (rare in practice)
    #
    # Think of it like this: two arrows pointing in the same direction
    # = similar meaning. Two arrows pointing in different directions
    # = different meaning.
    print("\n" + "=" * 60)
    print("SIMILARITY SCORES (1.0 = identical, 0.0 = unrelated)")
    print("=" * 60)

    # Compare every sentence with every other sentence
    for i in range(len(sentences)):
        for j in range(i + 1, len(sentences)):
            # util.cos_sim computes cosine similarity between two vectors
            score = util.cos_sim(embeddings[i], embeddings[j]).item()

            # .item() converts a single-element tensor to a Python float
            label = ""
            if score > 0.7:
                label = "  [+] VERY SIMILAR"
            elif score > 0.4:
                label = "  [~] SOMEWHAT RELATED"
            else:
                label = "  [-] DIFFERENT TOPICS"

            print(f"\n  [{score:.4f}]{label}")
            print(f"    A: \"{sentences[i]}\"")
            print(f"    B: \"{sentences[j]}\"")

    # ── Step 5: Try your own query ─────────────────────────────────
    # This is the foundation of semantic search!
    # You ask a question, we embed it, and find the most similar sentence.
    print("\n" + "=" * 60)
    print("SEMANTIC SEARCH DEMO")
    print("=" * 60)

    query = "What animals are in the house?"
    print(f"\n  Query: \"{query}\"")

    # Embed the query using the same model
    query_embedding = model.encode(query, convert_to_tensor=True)

    # Compare query with all sentences
    scores = util.cos_sim(query_embedding, embeddings)[0]
    # scores is a tensor with 5 values (one similarity per sentence)

    # Sort by similarity (highest first)
    # torch.argsort returns the INDICES that would sort the tensor
    sorted_indices = torch.argsort(scores, descending=True)

    print("  Results (most relevant first):\n")
    for rank, idx in enumerate(sorted_indices):
        score = scores[idx].item()
        print(f"    {rank + 1}. [{score:.4f}] \"{sentences[idx]}\"")

    print("\n  [INFO] Notice: the cat sentences ranked highest, even though")
    print("     the query never mentioned 'cat' — it understood MEANING!")


if __name__ == "__main__":
    main()
