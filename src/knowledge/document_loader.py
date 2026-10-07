from pathlib import Path
from pypdf import PdfReader


def load_pdf(file_path):
    reader = PdfReader(file_path)

    text = ""

    for page in reader.pages:
        page_text = page.extract_text()

        if page_text:
            text += page_text + "\n"

    return text
def split_into_chunks(text, chunk_size=1000, overlap=200):
    chunks = []

    start = 0

    while start < len(text):
        end = start + chunk_size

        chunk = text[start:end]
        chunks.append(chunk)

        start += chunk_size - overlap

    return chunks

def load_all_documents(folder_path="documents"):
    documents = []

    folder = Path(folder_path)

    for pdf_file in folder.glob("*.pdf"):
        text = load_pdf(pdf_file)

        chunks = split_into_chunks(text)

        documents.append({
            "file_name": pdf_file.name,
            "text": text,
            "chunks": chunks
        })

    return documents


if __name__ == "__main__":
    documents = load_all_documents()

    for document in documents:
        print("\n==============================")
        print("FILE:", document["file_name"])
        print("TOTAL CHUNKS:", len(document["chunks"]))
        print("==============================")

        if document["chunks"]:
            print(document["chunks"][0])