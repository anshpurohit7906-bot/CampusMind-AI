from src.document_loader import DocumentLoader
from src.text_chunker import TextChunker
from src.embedding_service import EmbeddingService
from src.vector_store import VectorStore

# Step 1: Create objects
loader = DocumentLoader()
chunker = TextChunker()
embedding_service = EmbeddingService()
vector_store = VectorStore()

# Step 2: Load PDF
text = loader.extract_text("data/sample.pdf")

# Step 3: Create chunks
chunks = chunker.chunk_text(text)

# Step 4: Generate embeddings
embedding_vectors = embedding_service.embed_chunks(chunks)

# Step 5: Store them in ChromaDB
vector_store.add_documents(chunks, embedding_vectors)

print("✅ Documents successfully stored in ChromaDB!")
print(f"Stored Chunks: {len(chunks)}")
print(f"Stored Embeddings: {len(embedding_vectors)}")