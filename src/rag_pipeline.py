import os

from dotenv import load_dotenv
from google import genai

from src.embedding_service import EmbeddingService
from src.vector_store import VectorStore
from src.retriever import Retriever
from src.prompt_builder import PromptBuilder

load_dotenv()

class RAGPipeline:
    """
    Complete Retrieval-Augmented Generation pipeline.

    Responsibilities:
    - Retrieve relevant chunks.
    - Build a prompt.
    - Generate an answer using Gemini.
    """

    def __init__(self):
        self.embedding_service = EmbeddingService()

        self.vector_store = VectorStore()

        self.retriever = Retriever(
            self.embedding_service,
            self.vector_store,
        )

        self.prompt_builder = PromptBuilder()

        api_key = os.getenv("GEMINI_API_KEY")

        if not api_key:
            raise ValueError(
                "GEMINI_API_KEY not found."
            )

        self.client = genai.Client(api_key=api_key)

        self.model_name = "gemini-2.5-flash"

    def ask(self, question: str) -> str:
        retrieved_chunks = self.retriever.retrieve(question)
        prompt = self.prompt_builder.build_prompt(
            question,
            retrieved_chunks,
        )
        print("=" * 80)
        print(prompt)
        print("=" * 80)

        try:
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt,
            )
            return response.text

        except Exception as e:
            return f"Error generating response: {e}"