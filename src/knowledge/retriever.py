from src.knowledge.embeddings import create_embedding
from src.knowledge.vector_store import search_documents
import ollama


MODEL_NAME = "llama3.2"


def retrieve_documents(query, top_k=3):
    query_embedding = create_embedding(query)

    results = search_documents(
        query_embedding=query_embedding,
        top_k=top_k
    )

    return results


def generate_answer(query):
    results = retrieve_documents(query)

    documents = results["documents"][0]

    context = "\n\n".join(documents)

    prompt = f"""
You are a helpful AI assistant with access to a set of knowledge documents.

Use the provided document context to answer the user's question.

DOCUMENT CONTEXT:
{context}

USER QUESTION:
{query}

Instructions:
- Answer naturally and clearly.
- Explain the concept in your own words.
- Do not simply copy the document chunks.
- You may summarize and combine information from multiple chunks.
- If the answer is not available in the provided context, say:
  "I couldn't find this information in the provided documents."
"""

    response = ollama.chat(
        model=MODEL_NAME,
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    return response["message"]["content"]


if __name__ == "__main__":
    query = "What is machine learning?"

    answer = generate_answer(query)

    print("\nANSWER:")
    print("========")
    print(answer)