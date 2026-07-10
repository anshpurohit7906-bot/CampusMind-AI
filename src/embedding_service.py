import os
from typing import List

from dotenv import load_dotenv
from google import genai

# Load environment variables from .env file (expects GEMINI_API_KEY)
load_dotenv()


class EmbeddingService:
    """
    Generates embeddings for text chunks using Google's Gemini
    embedding model. Single responsibility: text chunks in,
    embeddings out.
    """

    def __init__(self, model_name: str = "gemini-embedding-001"):
        """
        Initialize the embedding service.

        Args:
            model_name (str): Name of the Gemini embedding model to use.

        Raises:
            ValueError: If the API key is missing from the environment.
        """
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY not found in environment (.env file).")

        # Client handles authentication using the API key
        self.client = genai.Client(api_key=api_key)
        self.model_name = model_name

    def embed_chunks(self, chunks: List[str]) -> List[List[float]]:
        """
        Generate embeddings for a list of text chunks.

        Args:
            chunks (List[str]): Text chunks from TextChunker.

        Returns:
            List[List[float]]: One embedding vector per input chunk,
                in the same order.

        Raises:
            ValueError: If input is not a non-empty list of non-empty strings.
        """
        self._validate_input(chunks)

        embeddings: List[List[float]] = []

        for chunk in chunks:
            response = self.client.models.embed_content(
                model=self.model_name,
                contents=chunk,
            )
            # response.embeddings[0].values holds the vector for this chunk
            embeddings.append(response.embeddings[0].values)

        return embeddings

    @staticmethod
    def _validate_input(chunks: List[str]) -> None:
        """
        Ensure input is a non-empty list of non-empty strings.

        Raises:
            ValueError: On any invalid input.
        """
        if not isinstance(chunks, list):
            raise ValueError("chunks must be a list of strings.")

        if len(chunks) == 0:
            raise ValueError("chunks list is empty; nothing to embed.")

        for i, chunk in enumerate(chunks):
            if not isinstance(chunk, str) or not chunk.strip():
                raise ValueError(f"Chunk at index {i} is not a valid non-empty string.")


