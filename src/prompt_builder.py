from typing import List, Dict, Any


class PromptBuilder:
    """
    Builds a prompt for Gemini using the user's question
    and the retrieved document chunks.
    """

    def build_prompt(
        self,
        question: str,
        retrieved_chunks: List[Dict[str, Any]],
    ) -> str:
        """
        Build the final prompt for Gemini.

        Args:
            question: User's question.
            retrieved_chunks: Relevant chunks returned by Retriever.

        Returns:
            A formatted prompt string.
        """

        self._validate_input(question, retrieved_chunks)

        # Combine all retrieved chunk texts into one context
        context = "\n\n".join(
            chunk["text"] for chunk in retrieved_chunks
        )

        prompt = f"""
You are CampusMind-AI.

Answer the user's question ONLY using the context below.

If the answer cannot be found in the context, reply:

"I couldn't find that information in the provided documents."

-------------------------
Context:

{context}

-------------------------

Question:
{question}

Answer:
"""

        return prompt

    @staticmethod
    def _validate_input(
        question: str,
        retrieved_chunks: List[Dict[str, Any]],
    ) -> None:

        if not isinstance(question, str) or not question.strip():
            raise ValueError(
                "Question must be a non-empty string."
            )

        if not isinstance(retrieved_chunks, list):
            raise ValueError(
                "retrieved_chunks must be a list."
            )

        if len(retrieved_chunks) == 0:
            raise ValueError(
                "No retrieved chunks were provided."
            )
        for chunk in retrieved_chunks:
            if "text" not in chunk:
             raise ValueError(
            "Each retrieved chunk must contain a 'text' field."
        )