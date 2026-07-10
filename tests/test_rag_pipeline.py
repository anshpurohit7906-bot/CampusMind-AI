from src.rag_pipeline import RAGPipeline

rag = RAGPipeline()

question = question = "What topics are covered in Week 2?"

answer = rag.ask(question)

print("=" * 60)
print("Question:")
print(question)
print("=" * 60)

print("\nAnswer:\n")
print(answer)