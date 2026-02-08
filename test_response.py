from src.service.inference import get_rag_chain

if __name__ == "__main__":
    # Example usage
    while True:
        question = input("Enter your question: ")
        if question.lower() in ["exit", "quit"]:
            break
        rag_chain = get_rag_chain()
        result = rag_chain.stream({"question": question})
        for chunk in result:
            if "answer" in chunk:
                print(chunk["answer"], end="", flush=True)
        print("\nInference completed.\n")