from src.document_loader import DocumentLoader
from src.text_chunker import TextChunker

# Create objects
loader = DocumentLoader()
chunker = TextChunker()

# Load PDF
text = loader.extract_text("data/sample.pdf")

# Create chunks
chunks = chunker.chunk_text(text)

# Print results
print(f"Total Chunks: {len(chunks)}")

for i, chunk in enumerate(chunks, start=1):
    print(f"\n{'=' * 50}")
    print(f"Chunk {i}")
    print(f"{'=' * 50}")
    print(chunk)

    # Only print first 2 chunks for testing
    if i == 2:
        break