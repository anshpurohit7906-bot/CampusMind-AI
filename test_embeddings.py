from src.document_loader import DocumentLoader
from src.text_chunker import TextChunker
from src.embedding_service import EmbeddingService

loader = DocumentLoader()
chunker = TextChunker()
embedding_service = EmbeddingService()

text = loader.extract_text("data/sample.pdf")

chunks = chunker.chunk_text(text)

embedding_vectors = embedding_service.embed_chunks(chunks)


print(f"Total Chunks: {len(chunks)}")
print(f"Total Embeddings: {len(embedding_vectors)}")
print(f"Length of First Embedding: {len(embedding_vectors[0])}")