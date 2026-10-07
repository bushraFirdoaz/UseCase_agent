import chromadb
from src.knowledge.document_loader import load_all_documents
from src.knowledge.embeddings import create_embedding

client = chromadb.PersistentClient(path="./chroma_db")

collection = client.get_or_create_collection(
    name="documents"
)


def add_documents(chunks, embeddings, file_name):
    for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):

        collection.add(
            ids=[f"{file_name}_{i}"],
            documents=[chunk],
            embeddings=[embedding],
            metadatas=[{"source": file_name}]
        )


def search_documents(query_embedding, top_k=3):
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k
    )

    return results

def index_documents():
    documents = load_all_documents()

    for document in documents:
        file_name = document["file_name"]
        chunks = document["chunks"]

        embeddings = []

        for chunk in chunks:
            embedding = create_embedding(chunk)
            embeddings.append(embedding)

        add_documents(
            chunks=chunks,
            embeddings=embeddings,
            file_name=file_name
        )

        print(f"Indexed: {file_name}")
        print(f"Chunks: {len(chunks)}")


if __name__ == "__main__":
    index_documents()