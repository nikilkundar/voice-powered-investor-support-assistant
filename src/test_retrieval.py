
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DB_PATH = PROJECT_ROOT / "data" / "chroma_db"

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
COLLECTION_NAME = "sunrise_faq"

# Load the models and database once, not once per question.
_model = SentenceTransformer(MODEL_NAME, device="cpu")
_client = chromadb.PersistentClient(path=str(DB_PATH))
_collection = _client.get_collection(name=COLLECTION_NAME)


def retrieve_faqs(question: str, n_results: int = 3) -> list[dict]:
    """Return the most relevant FAQ entries for a question."""
    if not question or not question.strip():
        raise ValueError("Question cannot be empty.")

    query_embedding = _model.encode(
        question,
        normalize_embeddings=True,
    )

    results = _collection.query(
        query_embeddings=[query_embedding.tolist()],
        n_results=min(n_results, _collection.count()),
        include=["documents", "metadatas", "distances"],
    )

    entries = []

    for faq_id, document, metadata, distance in zip(
        results["ids"][0],
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        entries.append({
            "faq_id": metadata["faq_id"],
            "document": document,
            "distance": float(distance),
        })

    return entries


def test_retrieval():
    question = (
        "I invested in an equity mutual fund. "
        "If I redeem after 14 months, how are my gains taxed "
        "and will TDS be deducted?"
    )

    results = retrieve_faqs(question)

    print("\nInvestor question:")
    print(question)
    print("\nRetrieved FAQ entries:")

    for item in results:
        print(f"\nSource: {item['faq_id']}")
        print(f"Distance: {item['distance']:.4f}")
        print(f"Content:\n{item['document']}")


if __name__ == "__main__":
    test_retrieval()
