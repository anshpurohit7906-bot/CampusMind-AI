from src.document_loader import DocumentLoader
from src.text_chunker import TextChunker
from src.embedding_service import EmbeddingService
from src.vector_store import VectorStore


def main():
    # Change this to your PDF path
    pdf_path = "data/sample.pdf"

    print("Loading document...")
    loader = DocumentLoader()
    text = loader.extract_text(pdf_path)

    print("Chunking document...")
    chunker = TextChunker()
    chunks = chunker.chunk_text(text)

    print(f"Generated {len(chunks)} chunks.")

    print("Generating embeddings...")
    embedding_service = EmbeddingService()
    embeddings = embedding_service.embed_chunks(chunks)

    print("Storing in ChromaDB...")
    vector_store = VectorStore()

    # Prevent duplicate indexing
    if vector_store.collection.count() == 0:
        vector_store.add_documents(chunks, embeddings)
        print(f"Indexed {len(chunks)} document chunks.")
    else:
        print("Vector database already initialized.")

    print("Done!")


if __name__ == "__main__":
    main()