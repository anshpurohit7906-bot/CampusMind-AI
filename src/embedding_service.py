import os
from typing import List

from dotenv import load_dotenv
from google import genai

# Load environment variables
load_dotenv()


class EmbeddingService:
    """
    Generates embeddings for text chunks and user queries
    using Google's Gemini embedding model.

    Responsibilities:
    - Convert document chunks into embeddings.
    - Convert user questions into embeddings.
    """

    def __init__(self, model_name: str = "gemini-embedding-001"):
        """
        Initialize the embedding service.
        """
        api_key = os.getenv("GEMINI_API_KEY")

        if not api_key:
            raise ValueError(
                "GEMINI_API_KEY not found. Check your .env file."
            )

        self.client = genai.Client(api_key=api_key)
        self.model_name = model_name

    def embed_chunks(self, chunks: List[str]) -> List[List[float]]:
        """
        Generate embeddings for document chunks.

        Args:
            chunks: List of text chunks.

        Returns:
            List of embedding vectors.
        """
        self._validate_chunks(chunks)

        embeddings = []

        for chunk in chunks:
            response = self.client.models.embed_content(
                model=self.model_name,
                contents=chunk,
            )

            embeddings.append(list(response.embeddings[0].values))

        return embeddings

    def embed_query(self, question: str) -> List[float]:
        """
        Generate an embedding for a user's question.

        Args:
            question: User's query.

        Returns:
            Embedding vector.
        """
        self._validate_query(question)

        response = self.client.models.embed_content(
            model=self.model_name,
            contents=question,
        )

        return list(response.embeddings[0].values)

    @staticmethod
    def _validate_chunks(chunks: List[str]) -> None:
        """
        Validate chunk input.
        """
        if not isinstance(chunks, list):
            raise ValueError("chunks must be a list.")

        if len(chunks) == 0:
            raise ValueError("chunks cannot be empty.")

        for i, chunk in enumerate(chunks):
            if not isinstance(chunk, str) or not chunk.strip():
                raise ValueError(
                    f"Chunk at index {i} is invalid."
                )

    @staticmethod
    def _validate_query(question: str) -> None:
        """
        Validate query input.
        """
        if not isinstance(question, str) or not question.strip():
            raise ValueError(
                "Question must be a non-empty string."
            )