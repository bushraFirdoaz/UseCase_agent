import ollama


MODEL_NAME = "nomic-embed-text"


def create_embedding(text):
    response = ollama.embed(
        model=MODEL_NAME,
        input=text
    )

    return response["embeddings"][0]

if __name__ == "__main__":
    text = "Machine learning allows computers to learn from data."

    embedding = create_embedding(text)

    print("Embedding created successfully!")
    print("Number of values:", len(embedding))
    print("First 5 values:", embedding[:5])