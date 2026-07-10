from src.rag_pipeline import RAGPipeline

def main():
    rag = RAGPipeline()

    print("CampusMind-AI")
    print("Type 'exit' to quit.\n")

    while True:
        question = input("Ask: ")

        if question.lower() == "exit":
            break

        answer = rag.ask(question)

        print("\nAnswer:")
        print(answer)
        print("-" * 60)


if __name__ == "__main__":
    main()