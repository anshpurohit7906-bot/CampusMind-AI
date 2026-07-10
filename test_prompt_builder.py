from src.prompt_builder import PromptBuilder

# Create PromptBuilder object
builder = PromptBuilder()

# Sample question
question = "What is NumPy?"

# Sample retrieved chunks (simulating Retriever output)
retrieved_chunks = [
    {
        "text": "NumPy is a Python library for numerical computing.",
        "distance": 0.8019,
    },
    {
        "text": "It provides support for multidimensional arrays and mathematical operations.",
        "distance": 0.9033,
    },
    {
        "text": "NumPy is widely used in data science and machine learning.",
        "distance": 0.9363,
    },
]

# Build the prompt
prompt = builder.build_prompt(
    question=question,
    retrieved_chunks=retrieved_chunks,
)

# Print the generated prompt
print("=" * 60)
print("Generated Prompt")
print("=" * 60)
print(prompt)