
import re
from pathlib import Path

import chromadb
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer


PROJECT_ROOT = Path(__file__).resolve().parent.parent
PDF_PATH = PROJECT_ROOT / "input" / "SunriseAMC_FAQ.pdf"
DB_PATH = PROJECT_ROOT / "data" / "chroma_db"

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
COLLECTION_NAME = "sunrise_faq"




def clean_faq_text(text: str) -> str:
    """Remove footer text and trailing section headings."""

    footer_markers = [
        "Sunrise Asset Management Co. Ltd.",
        "Mutual Fund investments are subject to market risks.",
    ]

    for marker in footer_markers:
        position = text.find(marker)

        if position != -1:
            text = text[:position]

    # Remove section headings that appear at the end of a FAQ.
    text = re.sub(
        r"\s+\d+\.\s+(?:KYC|SIP|Redemption|Taxation)\s*$",
        "",
        text,
        flags=re.IGNORECASE,
    )

    return re.sub(r"\s+", " ", text).strip()




def extract_faq_chunks():
    """Extract each numbered FAQ question and its answer."""

    if not PDF_PATH.is_file():
        raise FileNotFoundError(f"FAQ PDF not found: {PDF_PATH}")

    reader = PdfReader(str(PDF_PATH))

    if reader.is_encrypted:
        raise ValueError("The FAQ PDF is encrypted.")

    page_text = [
        page.extract_text() or ""
        for page in reader.pages
    ]
    full_text = "\n".join(page_text)

    # Find each numbered FAQ question, such as Q1. or Q10.
    matches = list(
        re.finditer(r"(?m)^\s*Q(\d+)\.\s*", full_text)
    )

    if not matches:
        raise ValueError(
            "No numbered FAQ questions found in the PDF."
        )

    chunks = []

    for index, match in enumerate(matches):
        start = match.start()

        if index + 1 < len(matches):
            end = matches[index + 1].start()
        else:
            end = len(full_text)

        faq_text = full_text[start:end].strip()
        faq_text = clean_faq_text(faq_text)

        faq_id = f"Q{match.group(1)}"

        # Keep the question and answer together.
        chunks.append({
            "id": faq_id,
            "text": faq_text,
            "metadata": {
                "faq_id": faq_id,
                "source": PDF_PATH.name,
            },
        })

    return chunks


def build_knowledge_base():
    print("Extracting FAQ questions...")
    chunks = extract_faq_chunks()

    print(f"Found {len(chunks)} FAQ chunks.")

    print("Loading embedding model...")
    embedding_model = SentenceTransformer(
        EMBEDDING_MODEL,
        device="cpu",
    )

    documents = [chunk["text"] for chunk in chunks]

    print("Creating embeddings...")
    embeddings = embedding_model.encode(
        documents,
        normalize_embeddings=True,
        show_progress_bar=True,
    )

    # Persist the vector database locally.
    client = chromadb.PersistentClient(
        path=str(DB_PATH)
    )

    # Rebuild this small prototype collection from the PDF
    # each time this script runs.
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass

    collection = client.create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    collection.add(
        ids=[chunk["id"] for chunk in chunks],
        documents=documents,
        embeddings=embeddings.tolist(),
        metadatas=[
            chunk["metadata"] for chunk in chunks
        ],
    )

    print("Knowledge base created successfully.")
    print(f"Chunks stored: {collection.count()}")
    print(f"Database location: {DB_PATH}")

    for chunk in chunks:
        print(f"- {chunk['id']}: {chunk['text'][:100]}")


if __name__ == "__main__":
    build_knowledge_base()
