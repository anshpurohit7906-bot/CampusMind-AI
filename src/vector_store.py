from typing import List, Dict, Any
import chromadb


class VectorStore:
    """
    Stores text chunks together with their embeddings in ChromaDB,
    and allows similarity search over them. Single responsibility:
    persist and retrieve (chunk, embedding) pairs — nothing else.
    """

    def __init__(self, collection_name: str = "campusmind_docs", persist_path: str = "./chroma_db"):
        """
        Initialize a persistent ChromaDB client and get/create a collection.

        Args:
            collection_name (str): Name of the ChromaDB collection to use.
            persist_path (str): Local directory where ChromaDB will store data.
        """
        # PersistentClient saves data to disk so it survives restarts
        self.client = chromadb.PersistentClient(path=persist_path)

        # get_or_create_collection avoids errors if it already exists
        self.collection = self.client.get_or_create_collection(name=collection_name)

    def add_documents(self, chunks: List[str], embeddings: List[List[float]]) -> None:
        """
        Add text chunks and their corresponding embeddings to the store.

        Args:
            chunks (List[str]): Text chunks from TextChunker.
            embeddings (List[List[float]]): Embedding vectors from
                EmbeddingService, in the same order as chunks.

        Raises:
            ValueError: If inputs are invalid or mismatched in length.
        """
        self._validate_add_inputs(chunks, embeddings)

        # ChromaDB requires a unique string ID for each entry.
        # We generate simple sequential IDs based on current collection size.
        start_id = self.collection.count()
        ids = [str(start_id + i) for i in range(len(chunks))]

        self.collection.add(
            ids=ids,
            documents=chunks,
            embeddings=embeddings,
        )

    def search(self, query_embedding: List[float], top_k: int = 3) -> List[Dict[str, Any]]:
        """
        Find the most similar stored chunks to a given query embedding.

        Args:
            query_embedding (List[float]): Embedding vector of the user's query.
            top_k (int): Number of top matching chunks to return.

        Returns:
            List[Dict[str, Any]]: A list of results, each containing the
                matched text chunk and its similarity distance.

        Raises:
            ValueError: If query_embedding is invalid or top_k is not positive.
        """
        self._validate_search_inputs(query_embedding, top_k)

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
        )

        # Chroma returns nested lists (one per query); we only sent one query,
        # so we unpack the first (and only) result set.
        documents = results["documents"][0]
        distances = results["distances"][0]

        return [
            {"text": doc, "distance": dist}
            for doc, dist in zip(documents, distances)
        ]

    @staticmethod
    def _validate_add_inputs(chunks: List[str], embeddings: List[List[float]]) -> None:
        """
        Validate inputs for add_documents.

        Raises:
            ValueError: On any invalid input condition.
        """
        if not isinstance(chunks, list) or not isinstance(embeddings, list):
            raise ValueError("chunks and embeddings must both be lists.")

        if len(chunks) == 0 or len(embeddings) == 0:
            raise ValueError("chunks and embeddings must not be empty.")

        if len(chunks) != len(embeddings):
            raise ValueError("chunks and embeddings must be the same length.")

        for i, chunk in enumerate(chunks):
            if not isinstance(chunk, str) or not chunk.strip():
                raise ValueError(f"Chunk at index {i} is not a valid non-empty string.")

        for i, emb in enumerate(embeddings):
            if not isinstance(emb, list) or len(emb) == 0:
                raise ValueError(f"Embedding at index {i} is not a valid non-empty vector.")

    @staticmethod
    def _validate_search_inputs(query_embedding: List[float], top_k: int) -> None:
        """
        Validate inputs for search.

        Raises:
            ValueError: On any invalid input condition.
        """
        if not isinstance(query_embedding, list) or len(query_embedding) == 0:
            raise ValueError("query_embedding must be a non-empty list of floats.")

        if not isinstance(top_k, int) or top_k <= 0:
            raise ValueError("top_k must be a positive integer.")


    