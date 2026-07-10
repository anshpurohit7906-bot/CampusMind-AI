from typing import List, Dict, Any

from src.embedding_service import EmbeddingService
from src.vector_store import VectorStore


class Retriever:
    """
    Retrieves the most relevant document chunks for a user's question.

    Responsibilities:
    - Convert a user question into an embedding.
    - Search the VectorStore.
    - Return the most relevant chunks.

    It does NOT:
    - Generate answers.
    - Build prompts.
    - Load PDFs.
    """

    def __init__(
        self,
        embedding_service: EmbeddingService,
        vector_store: VectorStore,
    ):
        """
        Initialize the Retriever.

        Args:
            embedding_service: Instance of EmbeddingService.
            vector_store: Instance of VectorStore.
        """
        self.embedding_service = embedding_service
        self.vector_store = vector_store

    def retrieve(
        self,
        question: str,
        top_k: int = 3,
    ) -> List[Dict[str, Any]]:
        """
        Retrieve the most relevant chunks for a user's question.

        Args:
            question: User's natural language question.
            top_k: Number of chunks to retrieve.

        Returns:
            A list of matching chunks with similarity scores.
        """
        self._validate_input(question, top_k)

        # Convert question into embedding
        query_embedding = self.embedding_service.embed_query(question)

        # Search VectorStore
        results = self.vector_store.search(
            query_embedding=query_embedding,
            top_k=top_k,
        )

        return results

    @staticmethod
    def _validate_input(
        question: str,
        top_k: int,
    ) -> None:
        """
        Validate Retriever inputs.
        """
        if not isinstance(question, str) or not question.strip():
            raise ValueError(
                "Question must be a non-empty string."
            )

        if not isinstance(top_k, int) or top_k <= 0:
            raise ValueError(
                "top_k must be a positive integer."
            )