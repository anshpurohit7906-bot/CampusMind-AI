from src.document_loader import DocumentLoader
from src.text_chunker import TextChunker
from src.embedding_service import EmbeddingService
from src.vector_store import VectorStore
from src.retriever import Retriever

# Create objects
loader = DocumentLoader()
chunker = TextChunker()
embedding_service = EmbeddingService()
vector_store = VectorStore()

# Load PDF
text = loader.extract_text("data/sample.pdf")

# Create chunks
chunks = chunker.chunk_text(text)

# Generate embeddings
embedding_vectors = embedding_service.embed_chunks(chunks)

# Store in ChromaDB
vector_store.add_documents(chunks, embedding_vectors)

# Create retriever
retriever = Retriever(
    embedding_service=embedding_service,
    vector_store=vector_store,
)

# Ask a question
question = "What is NumPy?"

results = retriever.retrieve(question)

print(f"\nQuestion: {question}\n")

for i, result in enumerate(results, start=1):
    print("=" * 60)
    print(f"Result {i}")
    print("=" * 60)
    print(f"Distance: {result['distance']:.4f}")
    print(result["text"])