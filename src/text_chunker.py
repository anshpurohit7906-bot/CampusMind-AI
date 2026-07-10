from langchain_text_splitters import RecursiveCharacterTextSplitter
from typing import List


class TextChunker:
    """
    Splits large text documents into smaller, overlapping chunks
    suitable for embedding and retrieval in a RAG pipeline.

    Uses LangChain's RecursiveCharacterTextSplitter, which tries to
    split on natural boundaries (paragraphs, sentences, words) before
    falling back to hard character splits.
    """

    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        """
        Initialize the text splitter with fixed chunk parameters.

        Args:
            chunk_size (int): Maximum number of characters per chunk.
            chunk_overlap (int): Number of overlapping characters
                between consecutive chunks (helps preserve context
                across chunk boundaries).
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        # RecursiveCharacterTextSplitter tries a prioritized list of
        # separators (paragraph -> line -> sentence -> word) to keep
        # chunks semantically coherent rather than cutting mid-word.
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""],
        )

    def chunk_text(self, text: str) -> List[str]:
        """
        Split a block of text into overlapping chunks.

        Args:
            text (str): The raw input text (e.g., extracted from a PDF
                by DocumentLoader).

        Returns:
            List[str]: A list of text chunks ready for embedding.

        Raises:
            ValueError: If the input text is empty or not a string.
        """
        if not isinstance(text, str):
            raise ValueError("Input to chunk_text must be a string.")

        if not text.strip():
            raise ValueError("Input text is empty; nothing to chunk.")

        chunks = self.splitter.split_text(text)
        return chunks


